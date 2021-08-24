import copy
import time
import itertools
import numpy as np

import torch
from torch.optim import Adam

from metanas.meta_optimizer.agents.SAC.core import GRUActorCritic, GRUReplayBuffer
from metanas.meta_optimizer.agents.agent import NAS_agent


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class RL2_SAC(NAS_agent):
    def __init__(self, config, meta_model, env):
        super().__init__(config, meta_model, env)

        # TODO: Set these variables from config
        self.hidden_size = 512
        self.alpha = 0.
        self.gamma = 0.99

        replay_size = int(1e6)
        self.target_entropy = -2
        self.alpha_constant = copy.deepcopy(self.alpha)

        self.polyak = 0.995
        self.lr = 3e-4
        self.auto_entropy = False

        self.update_after = 1000
        self.update_every = 200
        self.start_steps = 10000
        self.steps_per_epoch = 4000

        self.batch_size = 5
        self.max_ep_len = 200

        # Create actor-critic module and target networks
        self.ac = GRUActorCritic(env.observation_space,
                                 env.action_space, self.hidden_size)

        self.ac_targ = copy.deepcopy(self.ac)

        # TODO: Format these variables for alpha annealling.
        # If alpha SGD is enabled for exploration, also add location
        # of this value in the paper.

        # Take gradient of alpha to balance exploitation vs exploration
        # TODO: How does this work in meta-learning setting?
        # Garage proposes different alphas for every task, (multi-task
        # learning)
        # https://garage.readthedocs.io/en/latest/user/algo_mtsac.html
        log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device="cuda")
        self.alpha_optimizer = Adam([log_alpha], lr=self.lr)

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(self.ac.q1.parameters(),
                                   self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(q_params, lr=self.lr)

        # Experience buffer
        # Note, this replay buffer stores entire trajectories because of the
        # recurrent policies
        self.replay_buffer = GRUReplayBuffer(size=replay_size)

        # TODO: Add # of variables of agent

    def compute_q_loss(self, data):
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        # Hidden layers of the RNN layer
        hid_in, hid_out = data['hid_in'], data['hid_out']

        r = torch.unsqueeze(r, -1)
        d = torch.unsqueeze(d, -1)

        q1, _ = self.ac.q1(o, a, a2, hid_in)
        q2, _ = self.ac.q2(o, a, a2, hid_in)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, _, _, _ = self.ac.pi.evaluate(o2, a, hid_out)

            # Target Q-values
            # Careful, hiden are tuples (a, b)
            q1_pi_targ, _ = self.ac_targ.q1(o2, a2, a, hid_out)
            q2_pi_targ, _ = self.ac_targ.q2(o2, a2, a, hid_out)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * \
                (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())
        return loss_q, q_info

    def compute_pi_loss(self, data):
        o, r, o2, d = data['obs'], data['rew'], data['obs2'], data['done']
        a, a2 = data['act'], data['act2']
        # Hidden layers of the LSTM layer
        hid_in, hid_out = data['hid_in'], data['hid_out']

        pi, logp_pi, _, _, _, _ = self.ac.pi.evaluate(o, a2, hid_in)
        q1_pi, _ = self.ac.q1(o, pi, a2, hid_in)
        q2_pi, _ = self.ac.q2(o, pi, a2, hid_in)
        q_pi = torch.min(q1_pi, q2_pi)

        # TODO: Possibility of adding decaying alpha
        # Could apply alpha auto entropy as trade-off between
        # exploration (max entropy) and exploitation (max Q)
        if self.auto_entropy is True:
            alpha_loss = -(self.log_alpha * (logp_pi +
                                             self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            # keep a constant alpha
            alpha = self.alpha_constant
            alpha_loss = 0

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())
        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_q_loss(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_pi_loss(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to
                # update target params, as opposed to "mul" and "add",
                # which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, a2, hidden, deterministic=False):
        """Obtains actions from the correct actor

        returns (action, hidden_in)
        """
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32),
                           a2, hidden, deterministic)

    def act_on_test_env(self, test_env):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

        # Recurrent shape
        hidden_out = torch.zeros([1, 1, self.hidden_size],
                                 dtype=torch.float).cuda()
        a2 = test_env.action_space.sample()

        # Only do done check here?
        while not(d or (ep_len == self.max_ep_len)):
            hidden_in = hidden_out
            # TODO: Obtain this action from the policy
            # deterministic=True doesn't fully work yet
            a, hidden_out = self.get_action(o, a2, hidden_in, True)

            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1

        # self.log_episode(ep_ret, ep_len)
        test_env.reset()

        # Final darts evaluation step
        # task_info = test_env.darts_evaluation_step(
        #     self.config.sparsify_input_alphas,
        #     self.config.limit_skip_connections)

        # return task_info

    def act_on_env(self, env):

        # Variables for episodic replay buffer
        e_a, e_a2, e_o, e_o2, e_d, e_r = [], [], [], [], [], []

        # Prepare for interaction with environment
        # total_steps = steps_per_epoch * epochs
        start_time = time.time()

        o, d, ep_ret, ep_len = env.reset(), False, 0, 0

        # Recurrent shape
        hidden_out = torch.zeros([1, 1, self.hidden_size],
                                 dtype=torch.float).cuda()
        a2 = env.action_space.sample()

        # Prepare for interaction with environment
        epochs = 5
        total_steps = self.steps_per_epoch * epochs
        max_ep_len = self.steps_per_epoch

        # step counter
        t = 0

        for t in range(total_steps):
            # Set hidden_in
            hidden_in = hidden_out

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            # TODO: Obtain this action from the policy
            if t > self.start_steps:
                a, hidden_out = self.get_action(o, a2, hidden_in)
            else:
                a = env.action_space.sample()

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            if t == 0:
                init_hid_in = hidden_in
                init_hid_out = hidden_out

            # Episodic replay buffer
            e_a.append(a)
            e_a2.append(a2)
            e_o.append(o)
            e_o2.append(o2)
            e_d.append(d)
            e_r.append(r)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2
            a2 = a

            # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # Store experience to replay buffer
            e_a = np.asarray(e_a)
            e_a2 = np.asarray(e_a2)
            e_o = np.asarray(e_o)
            e_o2 = np.asarray(e_o2)
            e_d = np.asarray(e_d)
            e_r = np.asarray(e_r)
            self.replay_buffer.store(e_a2, e_o, e_a, e_r, e_o2,
                                     init_hid_in, init_hid_out, e_d)
            e_a, e_a2, e_o, e_o2, e_d, e_r = [], [], [], [], [], []

            # logger.store(EpRet=ep_ret, EpLen=ep_len)

            # Sample MDP from distribution of environments, RL2
            # env.set_task(random.choice(ml1.train_tasks))

            o, ep_ret, ep_len = env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0 and \
                    len(self.replay_buffer) > self.batch_size:
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data=batch)

            t += 1

        # End of epoch handling
        # TODO: Add this logging

        # self.log_episode(ep_ret, ep_len)

        # Final darts evaluation step
        # task_info = env.darts_evaluation_step(
        #     self.config.sparsify_input_alphas,
        #     self.config.limit_skip_connections)

        env.reset()

        # return task_info

    def log_episode(self, ep_ret, ep_len, ep_loss=None):
        self.logger.add_scalar("Return", ep_ret, self.task_iter)
        self.logger.add_scalar("Episode Length", ep_len, self.task_iter)
        if ep_loss is not None:
            self.logger.add_scalar("Loss", ep_loss, self.task_iter)
