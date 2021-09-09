import copy
import time
import scipy.signal
import itertools
import random
import collections
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from metanas.meta_optimizer.agents.utils.logx import EpochLogger
from metanas.meta_optimizer.agents.PPO.core import GRUActorCritic, count_vars, combined_shape, discount_cumsum
from metanas.meta_optimizer.agents.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, setup_pytorch_for_mpi, sync_params, mpi_avg_grads


class EpisodicReplayBuffer:
    def __init__(self, replay_size=int(1e6), max_ep_len=500,
                 batch_size=1, time_step=8, device=None,
                 random_update=False):

        if random_update is False and batch_size > 1:
            raise AssertionError(
                "Cant apply sequential updates with different sequence sizes in a batch")

        self.random_update = random_update
        self.max_ep_len = max_ep_len
        self.batch_size = batch_size
        self.time_step = time_step
        self.device = device

        self.memory = collections.deque(maxlen=replay_size)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_eps = []

        if self.random_update:
            sampled_episodes = random.sample(self.memory, self.batch_size)

            min_step = self.max_ep_len
            # get minimum time step possible
            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))

            for episode in sampled_episodes:
                if min_step > self.time_step:
                    idx = np.random.randint(0, len(episode)-self.time_step+1)
                    sample = episode.sample(
                        time_step=self.time_step,
                        idx=idx)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1)
                    sample = episode.sample(
                        time_step=min_step,
                        idx=idx)
                sampled_eps.append(sample)
        else:
            idx = np.random.randint(0, len(self.memory))
            sampled_eps.append(self.memory[idx].sample())

        return self._sample_to_tensor(sampled_eps, len(sampled_eps[0]['obs']))

    def _sample_to_tensor(self, sample, seq_len):
        obs = [sample[i]["obs"] for i in range(self.batch_size)]
        act = [sample[i]["act"] for i in range(self.batch_size)]
        ret = [sample[i]["ret"] for i in range(self.batch_size)]
        adv = [sample[i]["adv"] for i in range(self.batch_size)]
        logp = [sample[i]["logp"] for i in range(self.batch_size)]
        prev_act = [sample[i]["prev_act"] for i in range(self.batch_size)]
        prev_rew = [sample[i]["prev_rew"] for i in range(self.batch_size)]

        obs = torch.FloatTensor(obs).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        act = torch.LongTensor(act).reshape(
            self.batch_size, seq_len).to(self.device)
        ret = torch.FloatTensor(ret).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        adv = torch.FloatTensor(adv).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        logp = torch.FloatTensor(logp).reshape(
            self.batch_size, seq_len, -1).to(self.device)
        prev_act = torch.FloatTensor(prev_act).reshape(
            self.batch_size, seq_len).to(self.device)
        prev_rew = torch.FloatTensor(prev_rew).reshape(
            self.batch_size, seq_len, -1).to(self.device)

        return (obs, act, ret, adv, logp, prev_act, prev_rew), seq_len


class EpisodeMemory:
    """Tracks the transitions within an episode
    """

    def __init__(self, random_update, gamma=0.99, lam=0.95):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []

        self.prev_act_buf = []
        self.prev_rew_buf = []

        self.val_buf = []
        self.logp_buf = []

        # Calculated at the end of episode
        self.adv_buf = []
        self.ret_buf = []

        self.gamma, self.lam = gamma, lam
        self.random_update = random_update

    def put(self, transition):
        self.obs_buf.append(transition[0])
        self.act_buf.append(transition[1])
        self.prev_act_buf.append(transition[2])
        self.prev_rew_buf.append(transition[3])
        self.rew_buf.append(transition[4])
        self.val_buf.append(transition[5])
        self.logp_buf.append(transition[6])

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        rews = np.append(np.array(self.rew_buf), last_val)
        vals = np.append(np.array(self.val_buf), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf = discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for
        # the value function
        self.ret_buf = discount_cumsum(rews, self.gamma)[:-1]

    def sample(self):
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf,
                    prev_rew=self.prev_rew_buf, prev_act=self.prev_act_buf)

        return data


class PPO:
    def __init__(self, env, test_env, max_ep_len=500, steps_per_epoch=4000,
                 epoch=100, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, batch_size=1, num_test_episodes=10,
                 logger_kwargs=dict(), seed=42, save_freq=1,
                 train_pi_iters=80, train_v_iters=80, lam=0.97,
                 target_kl=0.01, replay_size=int(1e6),
                 ac_kwargs=dict(), hidden_size=128):

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.test_env = env
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epoch
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_freq = save_freq

        self.lam = lam
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

        self.random_update = False

        self.hidden_size = hidden_size
        self.num_test_episodes = num_test_episodes

        self.clip_ratio = clip_ratio

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape

        self.summary_writer = SummaryWriter(
            log_dir=logger_kwargs['output_dir'], flush_secs=1)

        # Create actor-critic module
        self.ac = GRUActorCritic(env.observation_space,
                                 env.action_space, **ac_kwargs).to(self.device)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        # SpinngingUp logging & Tensorboard
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.episode_buffer = EpisodicReplayBuffer(
            replay_size=int(1e6), max_ep_len=max_ep_len,
            batch_size=1, time_step=8, device=self.device,
            random_update=False)

        # Sync params across processes
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(count_vars(module)
                           for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' %
                        var_counts)

    def init_hidden_states(self, batch_size):
        h = torch.zeros([1, batch_size, self.hidden_size]).to(self.device)
        return h

    # Set up function for computing PPO policy loss

    def compute_actor_loss(self, data):
        obs, act, _, adv, logp_old, prev_act, prev_rew = data

        # Policy loss, adjusted for RL2
        h = self.init_hidden_states(self.batch_size)
        pi, logp = self.ac.pi.step(obs, prev_act, prev_rew, h, act)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio,
                               1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_critic_loss(self, data):
        obs, _, ret, _, _, prev_act, prev_rew = data

        h = self.init_hidden_states(self.batch_size)
        v, _ = self.ac.v(obs, prev_act, prev_rew, h)

        return ((v - ret)**2).mean()

    def update(self):
        data, seq_len = self.episode_buffer.sample()

        pi_l_old, pi_info_old = self.compute_actor_loss(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_critic_loss(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_actor_loss(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log(
                    'Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_critic_loss(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                          KL=kl, Entropy=ent, ClipFrac=cf,
                          DeltaLossPi=(loss_pi.item() - pi_l_old),
                          DeltaLossV=(loss_v.item() - v_l_old))

    def get_action(self, o, prev_act, prev_rew, h):
        o = torch.Tensor(o).float().to(self.device).unsqueeze(0).unsqueeze(0)
        prev_act = torch.Tensor([prev_act]).float().to(
            self.device).unsqueeze(0)
        prev_rew = torch.Tensor([prev_rew]).float().to(
            self.device).unsqueeze(0).unsqueeze(-1)

        a, v, logp, h = self.ac.step(o, prev_act, prev_rew, h)
        return a, v, logp, h

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0

            h = self.init_hidden_states(1)
            a2 = self.test_env.action_space.sample()
            r2 = 0

            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                a, _, _, h = self.get_action(o, a2, r2, h)
                o, r, d, _ = self.test_env.step(a)

                a2 = a
                r2 = r

                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train_agent(self):
        # Prepare for interaction with environment
        start_time = time.time()

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(1, self.epochs):
            o, ep_ret, ep_len = self.env.reset(), 0, 0

            episode_record = EpisodeMemory(self.random_update)
            h = self.init_hidden_states(1)
            a2 = self.env.action_space.sample()
            r2 = 0

            for t in range(self.local_steps_per_epoch):
                a, v, logp, h = self.get_action(o, a2, r2, h)

                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                episode_record.put([o, a, a2, r2, r, v, logp])
                self.logger.store(VVals=v)

                # Update obs (critical!)
                o = o2

                # Set previous action and reward
                r2 = r
                a2 = a

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print(
                            'Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _, _ = self.get_action(o, a2, r2, h)
                    else:
                        v = 0
                    episode_record.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)

                    self.episode_buffer.put(episode_record)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0
                    episode_record = EpisodeMemory(self.random_update)
                    h = self.init_hidden_states(1)

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                self.logger.save_state({'env': self.env}, None)

            # Perform PPO update!
            for i in range(32):
                self.update()

            # Test the performance of the deterministic version of the
            # agent.
            self.test_agent()

            # Log info about epoch

            log_perf_board = ['EpRet', 'EpLen', 'TestEpRet',
                              'TestEpLen', 'KL',
                              'ClipFrac']
            log_loss_board = ['LossPi', 'LossV']
            log_board = {'Performance': log_perf_board,
                         'Loss': log_loss_board}

            # Update tensorboard
            for key, value in log_board.items():
                for val in value:
                    mean, std = self.logger.get_stats(val)

                    if key == 'Performance':
                        self.summary_writer.add_scalar(
                            key+'/Average'+val, mean, (epoch)*self.steps_per_epoch)
                        self.summary_writer.add_scalar(
                            key+'/Std'+val, std,  (epoch)*self.steps_per_epoch)
                    else:
                        self.summary_writer.add_scalar(
                            key+'/'+val, mean,  (epoch)*self.steps_per_epoch)

            self.logger.log_tabular('Epoch', (epoch))
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular(
                'TotalEnvInteracts', (epoch)*self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
