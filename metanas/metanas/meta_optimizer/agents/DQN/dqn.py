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


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# class ReplayBuffer:
#     """Memory for experience replay"""

#     def __init__(self, obs_dim, act_dim, device, size):
#         self.obs_buf = np.zeros(combined_shape(
#             size, obs_dim), dtype=np.float32)
#         self.obs2_buf = np.zeros(combined_shape(
#             size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(combined_shape(
#             size, act_dim), dtype=np.float32)
#         self.rew_buf = np.zeros(size, dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.ptr, self.size, self.max_size = 0, 0, size
#         self.device = device

#     def store(self, obs, act, rew, next_obs, done):
#         self.obs_buf[self.ptr] = obs
#         self.obs2_buf[self.ptr] = next_obs
#         self.act_buf[self.ptr] = act
#         self.rew_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr+1) % self.max_size
#         self.size = min(self.size+1, self.max_size)

#     def sample_batch(self, batch_size=32):
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         batch = dict(obs=self.obs_buf[idxs],
#                      obs2=self.obs2_buf[idxs],
#                      act=self.act_buf[idxs],
#                      rew=self.rew_buf[idxs],
#                      done=self.done_buf[idxs])
#         return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
#                 for k, v in batch.items()}


class DQN(RL_agent):
    def __init__(self, env, test_env, max_ep_len=1000, steps_per_epoch=4000,
                 epochs=200, gamma=0.99, lr=1e-3, batch_size=100,
                 num_test_episodes=10, logger_kwargs=dict(), seed=42,
                 save_freq=1, qnet_kwargs=dict(),
                 replay_size=int(1e6), update_after=1000,
                 update_every=50):
        super().__init__(env, test_env, max_ep_len, steps_per_epoch,
                         epochs, gamma, lr, batch_size, seed,
                         num_test_episodes, logger_kwargs)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.save_freq = save_freq
        self.update_every = update_every
        self.update_after = update_after

        # TODO: Adjust Actor Critic for discrete action space
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        self.online_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        self.target_network = MLPQNetwork(
            obs_dim, act_dim, **qnet_kwargs).to(self.device)

        # self.target_network = copy.deepcopy(
        #     self.online_network).to(self.device)

        # self.replay_buffer = ReplayBuffer(obs_dim, act_dim,
        #                                   self.device, replay_size)

        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(
            params=self.online_network.parameters(), lr=lr)

        # Decaying eps-greedy
        self.epsilon = 0.2
        # self.epsilon_target = 0.02
        # self.epsilon_decay = 200

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.online_network, self.target_network])
        self.logger.log(
            '\nNumber of parameters: \t q1: %d, \t q2: %d\n' % var_counts)

    # def _calculate_epsilon(self, frames):
    #     self.epsilon = self.epsilon_target + (self.epsilon - self.epsilon_target) * \
    #         np.exp(-1. * frames / self.epsilon_decay)

    def get_action(self, o):
        """Selects action from the learned Q-function
        """
        if torch.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                o = torch.Tensor(o).to(self.device)
                act = self.online_network(o)
                act = torch.max(act, 0)[1]
                act = act.item()
        else:
            act = self.env.action_space.sample()
        return act

    def update(self, o, a, r, o2, d):
        obs = torch.Tensor(o).to(self.device)
        new_obs = torch.Tensor(o2).to(self.device)
        reward = torch.Tensor([r]).to(self.device)

        if d:
            target_value = reward
        else:
            n_obs_values = self.online_network(new_obs).detach()
            max_values = torch.max(n_obs_values)
            target_value = reward + self.gamma * max_values

        predicted_value = self.online_network(obs)[a]

        loss = nn.MSELoss()(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # batch = self.replay_buffer.sample_batch(self.batch_size)
        # o, o2 = batch['obs'],  batch['obs2']
        # a, r, d = batch['act'].unsqueeze(1), batch['rew'].unsqueeze(
        #     1), batch['done'].unsqueeze(1)

        # with torch.no_grad():
        #     target_q = self.target_network(o2)
        #     online_q = self.online_network(o2)

        #     online_max_a = torch.argmax(online_q, dim=1, keepdim=True)
        #     target_value = r + (1-d) * self.gamma * \
        #         target_q.gather(1, online_max_a.long())

        # loss = F.mse_loss(self.online_network(
        #     o).gather(1, online_max_a.long()), target_value)

        # self.optimizer.zero_grad()
        # loss.backward()

        # # Gradient clipping
        # for param in self.online_network.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # self.optimizer.step()

        # self.target_network.load_state_dict(
        #     self.online_network.state_dict())

    def test_agent(self):
        # TODO: Adjust get_action to obtain deterministic action
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o))
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

            # TODO: Part of discrete SAC
            # Clip reward to [-1.0, 1.0].
            # clipped_reward = max(min(reward, 1.0), -1.0)

            # self._calculate_epsilon(t)

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            # self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            # if  and t % self.update_every == 0:
            #     # for _ in range(self.update_every):
            self.update(o, a, r, o2, d)

            # if t >= self.update_after % self.update_every == 0:
            #     self.update()

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
                self.logger.log_tabular('TotalEnvInteracts', t)
                # self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                # self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                # self.logger.log_tabular('LogPi', with_min_and_max=True)
                # self.logger.log_tabular('LossPi', average_only=True)
                # self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
