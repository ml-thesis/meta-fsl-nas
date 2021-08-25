import torch
import torch.nn as nn

import numpy as np


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


# class MLPQNetwork(nn.Module):

#     def __init__(self, obs_dim, act_dim, hidden_size=(256, 256), activation=nn.ReLU):
#         super().__init__()
#         # TODO: This network should end up having an
#         # rnn
#         # self.a = mlp([obs_dim] + hidden_size + [act_dim], activation)
#         self.v = mlp([obs_dim] + hidden_size + [1], activation)

#     def forward(self, obs):
#         # a = self.a(obs)
#         v = self.v(obs)
#         # print(a.shape, v.shape)
#         # print(a.mean())
#         return v  # + a - a.mean()


class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super(MLPQNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, act_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
