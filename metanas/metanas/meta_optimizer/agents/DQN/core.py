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


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.a = mlp([obs_dim] + [hidden_size] + [act_dim], activation)
        self.v = mlp([obs_dim] + [hidden_size] + [1], activation)

    def forward(self, obs):
        a = self.a(obs)
        v = self.v(obs)
        return v + a - a.mean()


class LSTMQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, activation=nn.ReLU):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation()

        self.linear1 = nn.Linear(obs_dim, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=1, batch_first=True)
        self.adv = nn.Linear(256, out_features=act_dim)
        self.val = nn.Linear(256, 1)

    def forward(self, data, batch_size, time_step, hidden_state, cell_state):
        # print(data.shape)
        data = data.unsqueeze(1)
        data = data.view(batch_size*time_step, 1, self.obs_dim)

        x = self.activation(self.linear1(data))
        x = x.view(batch_size, time_step, 256)

        lstm_output = self.lstm(x, (hidden_state, cell_state))
        out = lstm_output[0][:, time_step-1, :]
        h_n = lstm_output[1][0]
        c_n = lstm_output[1][1]

        adv_out = self.adv(out)
        val_out = self.val(out)

        qout = val_out.expand(batch_size, self.act_dim) + (
            adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size, self.act_dim))

        return qout, (h_n, c_n)
