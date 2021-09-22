# Reinforcement Learning Agents
The recurrent reinforcement learning agents to find a better initialization for the alpha values of DARTS in metaNAS.

# Implemented Agents
All agents employ RL2 policy and value function networks, which use recurrent networks and take the current observation, last action and last reward as input. 

- [Discrete Soft-Actor Critic (SAC)](https://arxiv.org/abs/1910.07207)
- [Duelling DDQN](https://arxiv.org/abs/1511.06581)

The PPO agent is not implemented as an existing implementation of RL2 for PPO and TPRO already exist in the Garage framework.