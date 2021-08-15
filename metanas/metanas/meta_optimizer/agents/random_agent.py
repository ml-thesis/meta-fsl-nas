import numpy as np

from metanas.meta_optimizer.agent import NAS_agent


class RandomAgent(NAS_agent):
    def _init_(self, meta_model, config):
        self.meta_model = meta_model
        self.config = config

    def test_agent(self, test_env):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

        while not(d or (ep_len == test_env.max_steps)):
            a = test_env.action_space.sample()

            # TODO: Test if it works
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
            # TODO: Add logger tensorboard
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

        test_env.reset()
        return ep_ret, ep_len

    def act_on_episode(self, env):
        o, d, ep_ret, ep_len = env.reset(), False, 0, 0

        while not(d or (ep_len == env.max_steps)):
            # a = env.action_space.sample()
            # a = np.random.choice(
            #     6, size=1, p=[0.05, 0.05, 0.1, 0.6, 0.1, 0.1])[0]
            a = 5

            # TODO: Test if it works
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
        # TODO: Add logger tensorboard
        # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

        env.reset()

        # TODO: Include these in the dict
        return ep_ret, ep_len
