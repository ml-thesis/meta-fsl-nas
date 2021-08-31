import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


"""GRU Actor Critic implementation for discrete action space"""


class GRUCategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation):
        super().__init__()
        self.activation = activation()

        self.linear1 = nn.Linear(obs_dim+3, hidden_dim)
        self.gru1 = nn.GRU(6, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, act_dim)

    def act(self, obs, prev_action, prev_reward, hidden_in):

        # print(obs.shape, prev_action.shape, prev_reward.shape)

        action_logits, hidden_out = self.forward(
            obs, prev_action, prev_reward, hidden_in)

        # Greedy action selection
        greedy_actions = torch.argmax(action_logits, dim=-1)

        return greedy_actions, hidden_out

    def sample(self, obs, prev_action, prev_reward, hidden_in):

        action_logits, hidden_out = self.forward(
            obs, prev_action, prev_reward, hidden_in)

        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs, hidden_out

    def forward(self, obs, prev_action, prev_reward, hidden_in):

        # print(obs.shape, prev_action.shape, prev_reward.shape, hidden_in.shape)

        prev_reward = prev_reward.unsqueeze(
            0).permute(1, 2, 0)
        prev_action = prev_action.unsqueeze(
            0).permute(1, 2, 0)

        # print(obs.shape, prev_action.shape, prev_reward.shape, hidden_in.shape)

        concat = torch.cat([obs, prev_action, prev_reward], -1)
        # gru_branch = self.activation(self.linear1(concat))
        # print(gru_branch)
        hidden_in = torch.zeros(
            1, concat.shape[1], 256, dtype=torch.float32).cuda()
        gru_branch, hidden_out = self.gru1(concat, hidden_in)
        action_logits = self.activation(self.linear2(gru_branch))

        # print(action_logits.shape)
        # TODO: Permutations
        action_logits = action_logits.permute(1, 0, 2)

        return action_logits, hidden_out


class GRUQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim, activation):
        super().__init__()
        self.activation = activation()

        # obs_dim + 2 for prev action and prev reward
        # self.linear1 = nn.Linear(obs_dim+3, hidden_dim)
        self.gru1 = nn.GRU(6, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, prev_action, prev_reward, hidden_in):

        # TODO: One-hot encode actions?

        # print(obs.shape, prev_action.shape, prev_reward.shape, hidden_in.shape)

        prev_reward = prev_reward.unsqueeze(
            0).permute(1, 2, 0)
        prev_action = prev_action.unsqueeze(
            0).permute(1, 2, 0)

        # print(obs.shape, prev_action.shape, prev_reward.shape, hidden_in.shape)

        concat = torch.cat([obs, prev_action,
                            prev_reward], -1)
        # print(concat.shape[1])

        hidden_in = torch.zeros(
            1, concat.shape[1], 256, dtype=torch.float32).cuda()

        # gru_branch = self.activation(self.linear1(concat))
        gru_branch, hidden_out = self.gru1(concat, hidden_in)
        x = self.activation(self.linear2(gru_branch))

        # TODO: X permute?
        # print(x.shape)
        x = x.permute(1, 0, 2)

        return x, hidden_out


class GRUActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_size=[256, 256],
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        self.pi = GRUCategoricalPolicy(
            obs_dim, act_dim, hidden_size[0], activation)
        self.q1 = GRUQNetwork(obs_dim, act_dim, hidden_size[0], activation)
        self.q2 = GRUQNetwork(obs_dim, act_dim, hidden_size[0], activation)

    def explore(self, obs, prev_a, prev_reward, hidden_in):
        # action selection using softmax
        with torch.no_grad():

            action, _, _, hidden_out = self.pi.sample(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                torch.as_tensor(prev_a, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                torch.as_tensor(prev_reward, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                hidden_in)
        return action.item(), hidden_out

    def act(self, obs, prev_a, prev_reward, hidden_in):
        # Greedy action selection by the policy, argmax
        with torch.no_grad():
            action, hidden_out = self.pi.act(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                torch.as_tensor(prev_a, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                torch.as_tensor(prev_reward, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0).cuda(),
                hidden_in)
        return action.item(), hidden_out


"""MLP Actor Critic implementation for discrete action space"""


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        pi_sizes = [obs_dim] + hidden_size + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh).cuda()

    def act(self, obs):
        action_logits = self.pi(obs)
        greedy_actions = torch.argmax(action_logits, dim=-1)
        return greedy_actions

    def sample(self, obs):
        policy = self.pi(obs)
        action_probs = F.softmax(policy, dim=-1)

        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        self.q = mlp([obs_dim] + hidden_size + [act_dim], activation)

    def forward(self, obs):
        q = self.q(obs)
        return q


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_size=[256, 256],
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        self.pi = CategoricalPolicy(obs_dim, act_dim, hidden_size, activation)
        self.q1 = MLPQNetwork(obs_dim, act_dim, hidden_size, activation)
        self.q2 = MLPQNetwork(obs_dim, act_dim, hidden_size, activation)

    def explore(self, obs):
        with torch.no_grad():
            action, _, _ = self.pi.sample(
                torch.as_tensor(obs, dtype=torch.float32).cuda())
        return action.item()

    def act(self, obs):
        # Greedy action selection by the policy
        with torch.no_grad():
            action = self.pi.act(
                torch.as_tensor(obs, dtype=torch.float32).cuda())
        return action.item()
