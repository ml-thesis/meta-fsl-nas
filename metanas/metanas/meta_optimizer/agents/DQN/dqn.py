from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import copy
import time
import random

from metanas.meta_optimizer.agents.agent import RL_agent
from metanas.meta_optimizer.agents.DQN.core import MLPQNetwork, count_vars


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obs, action, new_obs, reward, done):
        transition = (obs, action, new_obs, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


class DQN(RL_agent):
    def __init__(self, env, test_env, max_ep_len=500, steps_per_epoch=4000,
                 epochs=100, gamma=0.99, lr=3e-3, batch_size=32,
                 num_test_episodes=10, logger_kwargs=dict(), seed=42,
                 save_freq=1, qnet_kwargs=dict(),
                 replay_size=int(1e6), update_after=8000,
                 update_every=5, update_target=3000):
        super().__init__(env, test_env, max_ep_len, steps_per_epoch,
                         epochs, gamma, lr, batch_size, seed,
                         num_test_episodes, logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.save_freq = save_freq
        self.update_every = update_every
        self.update_after = update_after
        self.update_target = update_target

        self.hidden_size = qnet_kwargs["hidden_size"]

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        self.online_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        self.target_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_size)

        self.optimizer = optim.Adam(
            params=self.online_network.parameters(), lr=lr)

        # Decaying eps-greedy
        self.final_epsilon = 0.1
        self.epsilon_decay = 10000
        self.epsilon = 1.0

        # Track when to update
        self.update_counter = 0

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.online_network,
                           self.target_network])
        self.logger.log(
            '\nNumber of parameters: \t q1: %d, \t q2: %d\n' % var_counts)

    def get_action(self, obs):
        """Selects action from the learned Q-function
        """
        if torch.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                obs = torch.Tensor(obs).to(self.device)
                act = self.online_network(obs)
                act = torch.max(act, 0)[1]
                act = act.item()
        else:
            act = self.env.action_space.sample()
        return act

    def update(self, batch_size=32):
        """Perform the update rule on the function approximator,
           the Neural Network
        """
        if (len(self.replay_buffer.memory) < batch_size):
            return

        obs, action, new_obs, reward, done = self.replay_buffer.sample(
            batch_size)

        # Push tensors to device
        obs = torch.Tensor(obs).to(self.device)
        new_obs = torch.Tensor(new_obs).to(self.device)
        reward = torch.Tensor([reward]).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        done = torch.Tensor(done).to(self.device)

        n_obs_indexes = self.online_network(new_obs).detach()
        max_values_index = torch.max(n_obs_indexes, 1)[1]

        new_obs_values = self.target_network(new_obs).detach()
        max_new_obs_values = new_obs_values.gather(1,
                                                   max_values_index.unsqueeze(
                                                       1)
                                                   ).squeeze(1)

        target_value = reward + (1 - done) * self.gamma * max_new_obs_values
        predicted_value = self.online_network(obs).gather(
            1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        for param in self.online_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.update_counter % self.update_target == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict())

        self.update_counter += 1

    def _calculate_epsilon(self, frames):
        self.epsilon = self.epsilon_target + (self.epsilon - self.epsilon_target) * \
            np.exp(-1. * frames / self.epsilon_decay)

    def test_agent(self):
        # TODO: Adjust get_action to obtain deterministic action
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                a = self.get_action(o)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self):
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(self.total_steps):

            a = self.get_action(o)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            self.replay_buffer.push(o, a, o2, r, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, Eps=self.epsilon)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update epsilon
            if t % 500 == 0 and self.epsilon > self.final_epsilon:

                self.epsilon = self.final_epsilon + (self.epsilon - self.final_epsilon) * \
                    np.exp(-1. * (t/self.epsilon_decay))

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                self.update()

            if self.total_steps % self.update_target == 0:
                self.target_network.load_state_dict(
                    self.online_network.state_dict())

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the
                # agent.
                self.test_agent()

                # # Log info about epoch

                log_perf_board = ['EpRet', 'EpLen', 'TestEpRet',
                                  'TestEpLen']
                #   , 'Q2Vals',
                #   'Q1Vals', 'LogPi']
                # log_loss_board = ['LossQ']
                log_board = {'Performance': log_perf_board}  # ,
                #  'Loss': log_loss_board}

                # Update tensorboard
                for key, value in log_board.items():
                    for val in value:
                        mean, std = self.logger.get_stats(val)

                        if key == 'Performance':
                            self.summary_writer.add_scalar(
                                key+'/Average'+val, mean, t)
                            self.summary_writer.add_scalar(
                                key+'/Std'+val, std, t)
                        else:
                            self.summary_writer.add_scalar(
                                key+'/'+val, mean, t)

                epoch = (t+1) // self.steps_per_epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('Epsilon', self.epsilon)
                self.logger.log_tabular('TotalEnvInteracts', t)
                # self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                # self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                # self.logger.log_tabular('LogPi', with_min_and_max=True)
                # self.logger.log_tabular('LossPi', average_only=True)
                # self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
