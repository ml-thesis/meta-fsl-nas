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
from metanas.meta_optimizer.agents.DQN.core import LSTMQNetwork, count_vars


class ReplayBuffer:
    """Memory for experience replay"""

    def __init__(self,  device, size):

        self.memory = deque(maxlen=size)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, episode):
        self.memory.append(episode)
        self.size += 1

    def sample_batch(self, time_step, batch_size=32):
        sampled_epsiodes = random.sample(self.memory, batch_size)
        batch = []
        for episode in sampled_epsiodes:
            length = len(episode)+1-time_step  # if len(episode) + \
            # 1-time_step > 0 else len(episode)
            point = np.random.randint(0, length)
            batch.append(episode[point:point+time_step])
        return batch


class DRQN(RL_agent):
    def __init__(self, env, test_env, max_ep_len=500, steps_per_epoch=4000,
                 epochs=100, gamma=0.99, lr=3e-4, batch_size=64,
                 num_test_episodes=10, logger_kwargs=dict(), seed=42,
                 save_freq=1, qnet_kwargs=dict(),
                 replay_size=int(1e6), update_after=5000,
                 update_every=5, update_target=2000, time_step=16):
        super().__init__(env, test_env, max_ep_len, steps_per_epoch,
                         epochs, gamma, lr, batch_size, seed,
                         num_test_episodes, logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.save_freq = save_freq
        self.update_every = update_every
        self.update_after = update_after
        self.update_target = update_target

        self.time_step = time_step
        self.hidden_size = qnet_kwargs["hidden_size"]

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        self.online_network = LSTMQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        self.target_network = LSTMQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.device, replay_size)

        self.optimizer = optim.Adam(
            params=self.online_network.parameters(), lr=lr)

        # Decaying eps-greedy
        self.final_epsilon = 0.02
        self.epsilon_decay = 10000
        self.epsilon = 0.9

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.online_network,
                           self.target_network])
        self.logger.log(
            '\nNumber of parameters: \t q1: %d, \t q2: %d\n' % var_counts)

    def get_action(self, o, hidden_state, cell_state):
        """Selects action from the learned Q-function
        """
        o = torch.Tensor(o).to(self.device)
        if torch.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                act, hidden_state = self.online_network(
                    o, 1, 1, hidden_state, cell_state)
                act = torch.argmax(act).item()
        else:
            with torch.no_grad():
                _, hidden_state = self.online_network(
                    o, 1, 1, hidden_state, cell_state)
            act = self.env.action_space.sample()
        return act, hidden_state

    def update(self):
        h, c = self.init_hidden_states(self.batch_size)

        batch = self.replay_buffer.sample_batch(
            time_step=self.time_step, batch_size=self.batch_size)

        current_states = []
        acts = []
        rewards = []
        next_states = []

        for b in batch:
            cs, ac, rw, ns = [], [], [], []
            for element in b:
                cs.append(element[0])
                ac.append(element[1])
                rw.append(element[2])
                ns.append(element[3])
            current_states.append(cs)
            acts.append(ac)
            rewards.append(rw)
            next_states.append(ns)

        current_states = np.array(current_states)
        acts = np.array(acts)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        torch_current_states = torch.from_numpy(
            current_states).float().to(self.device)
        torch_acts = torch.from_numpy(acts).long().to(self.device)
        torch_rewards = torch.from_numpy(rewards).float().to(self.device)
        torch_next_states = torch.from_numpy(
            next_states).float().to(self.device)

        Q_next, _ = self.target_network.forward(
            torch_next_states, batch_size=self.batch_size, time_step=self.time_step,
            hidden_state=h, cell_state=c)
        Q_next_max, __ = Q_next.detach().max(dim=1)
        target_values = torch_rewards[:,
                                      self.time_step-1] + (self.gamma * Q_next_max)

        Q_s, _ = self.online_network.forward(torch_current_states, self.batch_size,
                                             time_step=self.time_step, hidden_state=h, cell_state=c)
        Q_s_a = Q_s.gather(
            dim=1, index=torch_acts[:, self.time_step-1].unsqueeze(dim=1)).squeeze(dim=1)

        loss = nn.MSELoss()(Q_s_a, target_values)

        #  save performance measure
        # loss_stat.append(loss.item())

        # make previous grad zero
        self.optimizer.zero_grad()

        # backward
        loss.backward()

        # update params
        self.optimizer.step()

    def test_agent(self):
        h, c = self.init_hidden_states(batch_size=1)

        # TODO: Adjust get_action to obtain deterministic action
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                a, (h, c) = self.get_action(o, h, c)
                o, r, d, _ = self.test_env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self):

        current_episode = []

        h, c = self.init_hidden_states(batch_size=1)

        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        for t in range(self.total_steps):

            a, (h, c) = self.get_action(o, h, c)
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            current_episode.append((o, a, r, o2))

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):

                if len(current_episode) >= self.time_step:
                    self.replay_buffer.store(current_episode)
                current_episode = []

                h, c = self.init_hidden_states(batch_size=1)

                self.logger.store(EpRet=ep_ret, EpLen=ep_len, Eps=self.epsilon)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update epsilon
            if t % 500 == 0 and self.epsilon > self.final_epsilon:

                self.epsilon = self.final_epsilon + (self.epsilon - self.final_epsilon) * \
                    np.exp(-1. * (t/self.epsilon_decay))

                # self.epsilon -= (self.init_epsilon -
                #                  self.final_epsilon)/self.total_steps

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
