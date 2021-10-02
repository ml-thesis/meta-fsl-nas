from abc import ABC

import os
import gym

import torch

from torch.utils.tensorboard import SummaryWriter
from metanas.meta_optimizer.agents.utils.logx import EpochLogger

# TODO: Refactor after having a working version on metaNAS


class NAS_agent(ABC):
    def __init__(self, config, meta_model, env):
        self.meta_model = meta_model
        self.config = config
        self.env = env

        # TensorBoard logger
        # logger_path = os.path.join(config.path, config.agent)
        # self.logger = SummaryWriter(logger_path, flush_secs=1)

        # self.task_iter = 0

    def act_on_test_env(self, env: gym.Env) -> dict():
        """Agent mutates the DARTS alphas based on the given
        environment and test task

        Args:
            env (Gym.Env): the NAS environment

        Returns:
            dict: Task and environment info
        """
        return

    def act_on_env(self, env: gym.Env) -> dict():
        """Agent mutates the DARTS alphas based on the given task
        an entire episode.

        Args:
            env (Gym.Env): the NAS environment

        Returns:
            dict: Task and environment info
        """
        return


class RL_agent(ABC):
    def __init__(self, env, test_env, max_ep_len=1000, steps_per_epoch=4000,
                 epochs=100, gamma=0.99, lr=1e-3, batch_size=100, seed=42,
                 num_test_episodes=10, logger_kwargs=dict()):

        self.env = env
        self.test_env = test_env
        self.max_ep_len = max_ep_len

        self.lr = lr
        self.seed = seed
        self.gamma = gamma

        self.epochs = epochs
        self.num_test_episodes = num_test_episodes
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * epochs

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # SpinngingUp logging & Tensorboard
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.summary_writer = SummaryWriter(
            log_dir=logger_kwargs['output_dir'], flush_secs=1)
