import numpy as np

from metanas.meta_optimizer.agent import NAS_agent


class RandomAgent(NAS_agent):
    def _init_(self, meta_model, config):
        super().__init(meta_model, config)

    def act_on_test_env(self, test_env):
        _, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

        while not(d or (ep_len == test_env.max_steps)):
            a = test_env.action_space.sample()

            _, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1

        self.log_episode(ep_ret, ep_len)
        test_env.reset()

        # Final darts evaluation step
        task_info = test_env.darts_evaluation_step(
            self.config.sparsify_input_alphas,
            self.config.limit_skip_connections)

        return task_info

    def act_on_env(self, env):
        _, d, ep_ret, ep_len = env.reset(), False, 0, 0

        while not(d or (ep_len == env.max_steps)):
            a = env.action_space.sample()
            _, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

        self.log_episode(ep_ret, ep_len)
        env.reset()

        # Final darts evaluation step
        task_info = env.darts_evaluation_step(
            self.config.sparsify_input_alphas,
            self.config.limit_skip_connections)

        return task_info

    def log_episode(self, ep_ret, ep_len, ep_loss=None):
        self.logger.add_scalar("Return", ep_ret, self.task_iter)
        self.logger.add_scalar("Episode Length", ep_len, self.task_iter)
        if ep_loss is not None:
            self.logger.add_scalar("Loss", ep_loss, self.task_iter)
