import argparse

from metanas.meta_optimizer.agents.SAC.rl2_sac import RL2_SAC

import gym
from gym.wrappers import AtariPreprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # env = gym.make("Pong-v0")

    env = gym.make("Pong-v0")
    env = AtariPreprocessing(env, frame_skip=1)

    # Ignore the meta-model and config input
    agent = RL2_SAC(None, None, env)
    agent.act_on_env(env)
