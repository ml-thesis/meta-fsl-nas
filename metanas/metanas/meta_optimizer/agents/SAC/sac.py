import copy
import time
import itertools
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from metanas.meta_optimizer.agents.utils.logx import EpochLogger
from metanas.meta_optimizer.agents.SAC.core import MLPActorCritic, count_vars


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, device, size):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
                for k, v in batch.items()}


class SAC:
    def __init__(self, env, test_env, ac_kwargs=dict(),
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6),
                 gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100,
                 start_steps=10000, update_after=1000, update_every=50,
                 num_test_episodes=10, logger_kwargs=dict(), save_freq=1,
                 seed=42):

        # TODO: Set in inheritance class
        self.env = env
        self.test_env = test_env
        # TODO: Adjust this to be a configuration
        self.max_ep_len = 1000
        self.save_freq = save_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.update_after = update_after
        self.num_test_episodes = num_test_episodes

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.polyak = polyak
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs
        self.update_every = update_every
        self.start_steps = start_steps

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # The online and target networks
        self.ac = MLPActorCritic(env.observation_space,
                                 env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = copy.deepcopy(self.ac)

        # SpinngingUp logging & Tensorboard
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        self.summary_writer = SummaryWriter(
            log_dir=logger_kwargs['output_dir'], flush_secs=1)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),
                                        self.ac.q2.parameters())

        # Set replay buffer
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape[0], self.env.action_space.n,
            self.device, replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

        # TODO: Optimize the alpha value
        self.alpha = alpha

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log(
            '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def compute_critic_loss(self, batch):
        o, o2 = batch['obs'],  batch['obs2']
        a, r, d = batch['act'], batch['rew'], batch['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        with torch.no_grad():
            # Target actions come from *current* policy
            _, a2, logp_a2 = self.ac.pi.sample(o2)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)

            # Next Q-value
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            next_q = (a2 * (q_pi_targ - self.alpha * logp_a2)
                      ).sum(dim=1, keepdims=True)
            r = r.unsqueeze(1)

            backup = r + self.gamma * (1 - d) * next_q

        # MSE loss against Bellman backup
        loss_q1 = ((q1.mean() - backup).pow(2)).mean()
        loss_q2 = ((q2.mean() - backup).pow(2)).mean()
        loss_q = loss_q1 + loss_q2

        # TODO: Include calculating TD errors for PER weights

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())  # .mean().item())

        return loss_q, q_info

    def compute_policy_loss(self, batch):
        o = batch['obs']

        # (Log of) probabilities to calculate expectations of Q and entropies.
        # action probability, log action probabilities
        _, pi, logp_pi = self.ac.pi.sample(o)

        with torch.no_grad():
            q1_pi = self.ac.q1(o, pi)
            q2_pi = self.ac.q2(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())  # .mean().item())

        return loss_pi, pi_info

    def update(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_critic_loss(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Recording Q-values
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_policy_loss(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Recording policy values
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update
                # target params, as opposed to "mul" and "add", which would
                # make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32))

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

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = self.ac.explore(o)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # TODO: Part of discrete SAC
            # Clip reward to [-1.0, 1.0].
            # clipped_reward = max(min(reward, 1.0), -1.0)

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    self.update()

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the
                # agent.
                self.test_agent()

                # Log info about epoch

                log_perf_board = ['EpRet', 'EpLen', 'TestEpRet',
                                  'TestEpLen', 'Q2Vals',
                                  'Q1Vals', 'LogPi']
                log_loss_board = ['LossPi', 'LossQ']
                log_board = {'Performance': log_perf_board,
                             'Loss': log_loss_board}

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
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)

                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()
