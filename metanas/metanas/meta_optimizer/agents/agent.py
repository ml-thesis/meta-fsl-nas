from abc import ABC

import os
import gym

from torch.utils.tensorboard import SummaryWriter


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
