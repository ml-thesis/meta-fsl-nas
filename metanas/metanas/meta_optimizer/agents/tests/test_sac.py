import argparse

from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs
from metanas.meta_optimizer.agents.SAC.sac import SAC
# from metanas.meta_optimizer.agents.SAC.rl2_sac import RL2_SAC

import gym
from gym.wrappers import AtariPreprocessing

if __name__ == "__main__":
    # TODO: Logging for testing?
    parser = argparse.ArgumentParser()

    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")

    logger_kwargs = setup_logger_kwargs("test_exp", seed=42)

    ac_kwargs = dict(hidden_sizes=[256]*2)
    # Ignore the meta-model and config input
    agent = SAC(env, test_env, logger_kwargs=logger_kwargs)
    agent.train_agent()
