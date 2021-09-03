import argparse

from metanas.meta_optimizer.agents.utils.run_utils import setup_logger_kwargs
from metanas.meta_optimizer.agents.DQN.dqn import DQN
# from metanas.meta_optimizer.agents.SAC.rl2_sac import RL2_SAC

import gym
from gym.wrappers import AtariPreprocessing

if __name__ == "__main__":
    # TODO: Logging for testing?
    parser = argparse.ArgumentParser()

    env = gym.make("CartPole-v1")
    test_env = gym.make("CartPole-v1")

    logger_kwargs = setup_logger_kwargs("DQN", seed=42)

    qnet_kwargs = dict(hidden_size=256)
    # Ignore the meta-model and config input
    agent = DQN(env, test_env,  qnet_kwargs=qnet_kwargs,
                logger_kwargs=logger_kwargs)
    agent.train_agent()
