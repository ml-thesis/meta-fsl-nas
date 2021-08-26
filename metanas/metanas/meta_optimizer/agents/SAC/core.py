import random
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


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# TODO: Set in config
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GRUReplayBuffer:
    """A FIFO experience replay buffer for GRU policies.
    """

    def __init__(self, size):
        self.capacity = size
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def store(self, last_act, obs, act, rew, next_obs, hid_in,
              hid_out, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            hid_in, hid_out, obs, act, last_act, rew, next_obs, done)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample_batch(self, batch_size=32):
        o_lst, a_lst, a2_lst, r_lst, o2_lst, hi_lst, \
            ho_lst, d_lst = [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)

        for sample in batch:
            h_in, h_out, obs, act, act2, reward, obs2, done = sample
            o_lst.append(obs)
            o2_lst.append(obs2)
            a_lst.append(act)
            a2_lst.append(act2)
            r_lst.append(reward)
            d_lst.append(done)

            # Hidden states dimensions
            hi_lst.append(h_in)  # shape (1, batch_size, hidden_size)
            ho_lst.append(h_out)  # shape (1, batch_size, hidden_size)

        # concatenate along the batch dim
        hi_lst = torch.cat(hi_lst, dim=-2)
        ho_lst = torch.cat(ho_lst, dim=-2)

        batch = dict(
            hid_in=hi_lst,
            hid_out=ho_lst,
            act2=a2_lst,
            obs=o_lst,
            obs2=o2_lst,
            act=a_lst,
            rew=r_lst,
            done=d_lst)

        return {k: torch.tensor(v, dtype=torch.float32).cuda()
                if type(v) != tuple else v
                for k, v in batch.items()}


class GRUActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size, activation,
                 action_range=1., init_w=3e-3):
        super().__init__()

        self.action_range = action_range
        self.activation = activation

        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(obs_dim+act_dim, hidden_size)
        self.lstm1 = nn.GRU(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, act_dim)

    def get_action(self, obs, last_action, hidden_in, deterministic=False):
        # increase 2 dims to match with training data
        obs = torch.Tensor(obs).unsqueeze(0).unsqueeze(0).cuda()
        last_action = torch.Tensor(
            last_action).unsqueeze(0).unsqueeze(0).cuda()

        # mean, log_std, hidden_out = self.forward(
        #     obs, last_action, hidden_in)

        return action[0][0], hidden_out

    def evaluate(self, obs, last_action, hidden_in, epsilon=1e-6):
        mean, log_std, hidden_out = self.forward(obs, last_action, hidden_in)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients
        # flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        # TanhNormal distribution as actions; reparameterization trick
        action_0 = torch.tanh(mean + std * z.cuda())
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features
        # instead of 1 dim probability, needs sum up across the features
        # dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def forward(self, obs, last_action, hidden_in,
                deterministic=False, with_logprob=False):

        obs = obs.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch 1
        fc_branch = F.relu(self.linear1(obs))
        # branch 2
        lstm_branch = torch.cat([obs, last_action], -1)
        print(lstm_branch.shape, obs.shape,
              last_action.shape, self.linear2.weight.shape)

        lstm_branch = F.relu(self.linear2(lstm_branch))
        # (h_0, c_0)
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1, 0, 2)  # back to same axes as input
        # lstm_hidden is actually tuple: (hidden, cell)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN,
                              LOG_STD_MAX)
        return mean, log_std, lstm_hidden


class GRUQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.linear1 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.linear2 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.lstm1 = nn.GRU(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)

    def forward(self, obs, action, last_action, hidden_in):
        """
        obs shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size,
        state_dim)
        """
        obs = obs.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        # branch 1
        fc_branch = torch.cat([obs, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([obs, last_action], -1)
        # linear layer for 3d input only applied on the last dim
        lstm_branch = self.activation(self.linear2(lstm_branch))
        # lstm_branch, lstm_hidden = self.lstm1(
        # lstm_branch, hidden_in)  # no activation after lstm
        lstm_branch, lstm_hidden = self.lstm1(
            lstm_branch, hidden_in)  # no activation after lstm

        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        # lstm_hidden is actually tuple: (hidden, cell)
        return x, lstm_hidden


class GRUActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_size=(256, 256), activation=nn.ReLU()):
        super().__init__()

        # TODO: Adjust Actor Critic for discrete action space
        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = GRUActor(obs_dim, act_dim, hidden_size,
                           activation).cuda()

        self.q1 = GRUQFunction(
            obs_dim, act_dim, hidden_size, activation=activation).cuda()
        self.q2 = GRUQFunction(
            obs_dim, act_dim, hidden_size, activation=activation).cuda()

    def act(self, obs, last_action, hidden, deterministic=False):
        with torch.no_grad():
            a, hidden_out = self.pi.get_action(
                obs, last_action, hidden, deterministic)
            return a, hidden_out


"""MLP Actor Critic implementation for discrete action space and testing purposes"""


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        pi_sizes = [obs_dim] + hidden_size + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh).cuda()

    def act(self, states):
        action_logits = self.pi(states)
        greedy_actions = torch.argmax(action_logits, dim=-1, keepdim=True)

        # if len(action_logits.shape) == 1:
        #     greedy_actions = torch.argmax(action_logits, dim=0, keepdim=True)
        # else:
        #     greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)

        # greedy_actions = torch.argmax(
        #     action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):

        policy = self.pi(states)
        action_probs = F.softmax(policy, dim=-1)

        # if len(policy.shape) == 1:
        #     action_probs = F.softmax(policy, dim=0)
        # else:
        #     action_probs = F.softmax(policy, dim=1)

        action_dist = Categorical(action_probs)
        # TODO: actions can be omitted
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size, activation):
        super().__init__()
        # TODO: This network should end up having an
        # rnn
        # + act_dim
        self.q = mlp([obs_dim] + hidden_size + [act_dim], activation)

        # self.a = mlp([obs_dim] + [hidden_size] + [act_dim], activation)
        # self.v = mlp([obs_dim] + [hidden_size] + [1], activation)

    def forward(self, obs):  # , act):
        # q = self.q(torch.cat([obs, act], dim=-1))
        q = self.q(obs)
        # a = self.a(obs)
        # v = self.v(obs)

        # return v + a - a.mean(1, keepdim=True)
        # print(q.shape)
        return q  # torch.squeeze(q)  # Critical to ensure q has right shape.


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

    def explore(self, o):
        with torch.no_grad():
            action, _, _ = self.pi.sample(
                torch.as_tensor(o, dtype=torch.float32).cuda())
        return action.item()

    def act(self, o):
        # Greedy action selection by the policy
        with torch.no_grad():
            action = self.pi.act(
                torch.as_tensor(o, dtype=torch.float32).cuda())
        return action.item()
