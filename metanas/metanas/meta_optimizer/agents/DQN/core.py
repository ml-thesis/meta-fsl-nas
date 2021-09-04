import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=256, layers=2,
                 activation=nn.ReLU):
        super().__init__()
        self.a = mlp([obs_dim] + [hidden_size]*layers + [act_dim], activation)
        self.v = mlp([obs_dim] + [hidden_size]*layers + [1], activation)

    def forward(self, obs):
        a = self.a(obs)
        v = self.v(obs)
        return v + a - a.mean()


class LSTMQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super().__init__()
        self.activation = activation

        self.Linear1 = nn.Linear(obs_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.Linear2 = nn.Linear(hidden_size, act_dim)

#         self.adv = nn.Linear(256, out_features=act_dim)
#         self.val = nn.Linear(256, 1)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = self.Linear2(x)
        return x, new_h, new_c
