import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def create_one_hot(act, act_dim):
    index = torch.eye(act_dim).cuda()
    return index[act.long()]


"""GRU Actor Critic implementation for discrete action space"""


class GRUCategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64], activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.act_dim = act_dim
        self.hidden_size = hidden_sizes[0]

        self.Linear1 = nn.Linear(obs_dim, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size+act_dim+1,  # +1 for the reward
                          self.hidden_size,
                          batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_size, act_dim)

    def sample(self, obs, prev_act, prev_rew, hid):

        action_logits, hidden_out = self.forward(
            obs, prev_act, prev_rew, hid)

        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)  # pi
        actions = action_dist.sample()

        log_action_probs = action_dist.log_prob(actions)

        # inline with spinningup:
        # return act, pi, logp_a, hid_out
        return actions, action_probs, log_action_probs, hidden_out

    def step(self, obs, prev_act, prev_rew, hid, act):

        action_logits, hidden_out = self.forward(
            obs, prev_act, prev_rew, hid)

        pi = Categorical(logits=action_logits)  # pi
        logp_a = pi.log_prob(act)

        return pi, logp_a

    def forward(self, obs, prev_act, prev_rew, hid):

        self.gru.flatten_parameters()

        prev_act = create_one_hot(prev_act, self.act_dim)
        gru_input = self.activation(self.Linear1(obs))
        gru_input = torch.cat(
            [
                gru_input,
                prev_act,
                prev_rew
            ],
            dim=2,
        )

        # unroll the GRU network
        gru_out, hid_out = self.gru(gru_input.float(), hid.float())
        q = self.Linear2(gru_out)

        return q, hid_out


class GRUQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=[64], activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        self.act_dim = act_dim
        self.hidden_size = hidden_sizes[0]

        self.Linear1 = nn.Linear(obs_dim, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size+act_dim+1,  # +1 for the reward
                          self.hidden_size,
                          batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_size, 1)

    def forward(self, obs, prev_act, prev_rew, hid):

        self.gru.flatten_parameters()

        prev_act = create_one_hot(prev_act, self.act_dim)
        gru_input = self.activation(self.Linear1(obs))
        gru_input = torch.cat(
            [
                gru_input,
                prev_act,
                prev_rew
            ],
            dim=2,
        )

        # unroll the GRU network
        gru_out, hid_out = self.gru(gru_input.float(), hid.float())
        v = self.Linear2(gru_out)
        return v, hid_out


class GRUActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=[256, 256],
                 activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # policy builder depends on action space
        assert isinstance(action_space, Discrete), \
            "Current implementation only works for Discrete action space."

        self.pi = GRUCategoricalPolicy(
            obs_dim, action_space.n, hidden_sizes, activation)
        # build value function
        self.v = GRUQNetwork(obs_dim, act_dim, hidden_sizes, activation)

    def step(self, obs, prev_act, prev_rew, hid):
        obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
        prev_act = torch.as_tensor(prev_act, dtype=torch.float32).cuda()
        prev_rew = torch.as_tensor(prev_rew, dtype=torch.float32).cuda()

        with torch.no_grad():
            a, pi, logp_a, h = self.pi.sample(obs, prev_act, prev_rew, hid)
            v, _ = self.v(obs, prev_act, prev_rew, hid)
        return a.cpu().item(), v.cpu().numpy(), logp_a.cpu().numpy(), h


"""MLP Actor Critic implementations"""


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +
                          [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
