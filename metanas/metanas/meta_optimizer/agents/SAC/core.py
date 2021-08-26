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
        self.activation = activation

        self.linear1 = nn.Linear(obs_dim+2, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, act_dim)

    def act(self, obs, prev_action, prev_reward, hidden_in):

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

        # TODO: Permutations of variables

        concat = torch.cat([obs, prev_action, prev_reward], -1)
        gru_branch = self.activation(self.linear1(concat))
        gru_branch, hidden_out = self.gru1(gru_branch, hidden_in)
        action_logits = self.activation(self.linear2(gru_branch))

        # TODO: Permutations

        return action_logits, hidden_out


class GRUQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim, activation):
        super().__init__()
        self.activation = activation

        # obs_dim + 2 for prev action and prev reward
        self.linear1 = nn.Linear(obs_dim+2, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, prev_action, prev_reward, hidden_in):

        # TODO: One-hot encode actions?

        concat = torch.cat([obs, prev_action, prev_reward], -1)
        gru_branch = self.activation(self.linear1(concat))
        gru_branch, hidden_out = self.gru1(gru_branch, hidden_in)
        x = self.activation(self.linear2(gru_branch))

        # TODO: X permute?

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
