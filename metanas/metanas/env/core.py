import torch
import torch.nn.functional as F
import numpy as np

import time
import gym
from gym import spaces

from metanas.utils import utils
import metanas.utils.genotypes as gt


"""Wrapper for the RL agent to interact with the meta-model in the outer-loop
utilizing the OpenAI gym interface
"""


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, meta_model, task_optimizer=None,
                 test_phase=False, cell_type="normal",
                 reward_estimation=False,
                 max_steps=100, test_env=None):
        super().__init__()
        self.config = config
        self.test_env = test_env
        self.cell_type = cell_type
        self.primitives = config.primitives
        self.reward_estimation = reward_estimation

        self.test_phase = test_phase
        self.meta_model = meta_model
        # self.task_optimizer = task_optimizer
        # self.meta_predictor
        self.meta_epoch = 0

        # The task is set in the meta-loop
        self.current_task = None
        self.max_steps = max_steps
        self.reward_range = (-1, 1)

        # Initialize the step counter
        self.step_count = 0
        self.terminate_episode = False

        # TODO: Set seed?

        # Set baseline accuracy to scale the reward
        self.baseline_acc = 0

        # Initialize State/Observation space
        # Intermediate + input nodes
        self.n_nodes = self.config.nodes + 2

        # Adjacency matrix
        self.A = np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

        # Remove the 2 input nodes from A
        self.A[0, 1] = 0
        self.A[1, 0] = 0

        self.initialize_observation_space()

        # Initialize action space
        # |A| + 2*|O| + 1, the +1 for the termination
        action_size = len(self.A) + 2*len(self. primitives) + 1
        self.action_space = spaces.Discrete(action_size)

    def reset(self):
        """Reset the environment state"""
        assert self.current_task is not None, \
            "A task needs to be set before evaluation"

        # Initialize the step counters
        self.step_count = 0
        self.terminate_episode = False

        self.update_states()

        # Set starting edge for agent
        self.set_start_state()

        # Set baseline accuracy to scale the reward
        self.baseline_acc = self.compute_reward()

        return self.current_state

    def set_task(self, task):
        """The meta-loop passes the task for the environment to solve"""
        self.current_task = task
        self.reset()

    def initialize_observation_space(self):
        # Generate the internal states of the graph
        self.update_states()

        # Set starting edge for agent
        self.set_start_state()

        self.observation_space = spaces.Box(
            0, 1,
            shape=self.current_state.shape,
            dtype=np.int32)

    def update_states(self):
        """Set all the state variables for the environment on
        reset and updates.

        Raises:
            RuntimeError: On passing invalid cell types
        """
        s_idx = 0
        self.states = []
        self.edge_to_index = {}
        self.edge_to_alpha = {}

        # Define (normalized) alphas
        if self.cell_type == "normal":
            # Idea of letting RL observe the normalized alphas,
            # and mutate the actual alpha values
            self.normalized_alphas = [
                F.softmax(alpha, dim=-1).detach().cpu()
                for alpha in self.meta_model.alpha_normal]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_normal]

        elif self.cell_type == "reduce":
            self.normalized_alphas = [
                F.softmax(alpha, dim=-1).detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

        for i, edges in enumerate(self.normalized_alphas):
            # edges: Tensor(n_edges, n_ops)
            edge_max, _ = torch.topk(edges[:, :], 1)
            # selecting the top-k input nodes, k=2
            _, topk_edge_indices = torch.topk(edge_max.view(-1), k=2)

            for j, edge in enumerate(edges):
                self.edge_to_index[(j, i+2)] = s_idx
                self.edge_to_index[(i+2, j)] = s_idx+1

                self.edge_to_alpha[(j, i+2)] = (i, j)
                self.edge_to_alpha[(i+2, j)] = (i, j)

                # For undirected edge we add the edge twice
                self.states.append(
                    np.concatenate((
                        [j],
                        [i+2],
                        [int(j in topk_edge_indices)],
                        self.A[i+2],
                        edge.detach().numpy())))

                self.states.append(
                    np.concatenate((
                        [i+2],
                        [j],
                        [int(j in topk_edge_indices)],
                        self.A[j],
                        edge.detach().numpy())))
                s_idx += 2

        self.states = np.array(self.states)

    def set_start_state(self):
        # TODO: Add probability to the starting edge?
        self.current_state_index = 0
        self.current_state = self.states[
            self.current_state_index]

    def update_meta_model(self, value, row_idx, edge_idx, op_idx):
        """Adjust alpha value of the meta-model for a given element
        and value

        Raises:
            RuntimeError: On passing invalid cell types
        """

        if self.cell_type == "normal":
            with torch.no_grad():
                self.meta_model.alpha_normal[
                    row_idx][edge_idx][op_idx] += value

        elif self.cell_type == "reduce":
            with torch.no_grad():
                self.meta_model.alpha_reduce[
                    row_idx][edge_idx][op_idx] += value

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

    def render(self, mode='human'):
        """Render the environment, according to the specified mode."""
        for row in self.states:
            print(row)

    def step(self, action):
        start = time.time()

        # Mutates the meta_model and the local state
        action_info, reward = self._perform_action(action)

        # The final step time
        end = time.time()
        running_time = int(end - start)

        self.step_count += 1

        # Conditions to terminate the episode
        done = self.step_count == self.max_steps or \
            self.terminate_episode

        info_dict = {
            "step_count": self.step_count,
            "action_id": action,
            "action": action_info,
            "reward": reward,
            "done": done,
            "running_time": running_time
        }

        return self.current_state, reward, done, info_dict

    def close(self):
        return NotImplemented

    def _perform_action(self, action):
        """Perform the action on both the meta-model and local state"""

        action_info = ""
        reward = 0

        # denotes the current edge it is on
        cur_node = int(self.current_state[0])
        next_node = int(self.current_state[1])

        # Adjacancy matrix A, navigating to the next node
        if action in np.arange(len(self.A)):

            # Determine if agent is allowed to traverse
            # the edge
            if self.A[next_node][action] > 0:
                # Legal action
                cur_node = next_node
                next_node = action

                s_idx = self.edge_to_index[(cur_node, next_node)]
                self.current_state = self.states[s_idx]

                action_info = f"Legal move from {next_node} to {action}"

            elif self.A[next_node][action] < 1:
                # Illegal next_node is not connected the action node
                # return reward -1, and stay in the same edge
                reward = -1

                action_info = f"Illegal move from {next_node} to {action}"

        # Increasing the alpha for the given operation
        if action in np.arange(len(self.A),
                               len(self.A)+len(self.primitives)):
            # Adjust action indices to fit the operations
            action = action - len(self.A)

            # Find the current edge to mutate
            row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
            s_idx = self.edge_to_index[(cur_node, next_node)]

            increase_val = torch.sum(
                torch.abs(self.alphas[row_idx][edge_idx]))

            self.update_meta_model(increase_val,
                                   row_idx,
                                   edge_idx,
                                   action)

            # Update the local state after increasing the alphas
            self.update_states()

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Compute reward after updating
            reward = self.compute_reward()

            loc = f"({row_idx}, {edge_idx}, {action})"
            action_info = f"Increase alpha {loc} by {increase_val}"

        # Decreasing the alpha for the given operation
        if action in np.arange(len(self.A)+len(self.primitives),
                               len(self.A)+2*len(self.primitives)):
            # Adjust action indices to fit the operations
            action = action - len(self.A) - len(self.primitives)

            # Find the current edge to mutate
            row_idx, edge_idx = self.edge_to_alpha[(cur_node, next_node)]
            s_idx = self.edge_to_index[(cur_node, next_node)]

            decrease_val = -torch.sum(
                torch.abs(self.alphas[row_idx][edge_idx]))

            self.update_meta_model(decrease_val,
                                   row_idx,
                                   edge_idx,
                                   action)

            # Update the local state after increasing the alphas
            self.update_states()

            # Set current state again!
            self.current_state = self.states[s_idx]

            # Compute reward after updating
            reward = self.compute_reward()

            loc = f"({row_idx}, {edge_idx}, {action})"
            action_info = f"Decrease alpha {loc} by {decrease_val}"

        # Terminate the episode
        if action in np.arange(len(self.A)+2*len(self.primitives),
                               len(self.A)+2*len(self.primitives)+1,
                               ):
            self.terminate_episode = True
            action_info = f"Terminate the episode at step {self.step_count}"

        return action_info, reward

    # Calculation/Estimations of the reward

    def compute_reward(self):
        # For testing env
        if self.test_env is not None:
            return np.random.uniform(low=-1, high=1, size=(1,))[0]

        if self.reward_estimation:
            acc = self._meta_predictor_estimation(self.current_task)
        else:
            acc = self._darts_estimation(self.current_task)

        # Scale reward to (-1, 1) range
        reward = self.scale_reward(acc)
        return reward

    def scale_reward(self, accuracy):
        """
        Map the accuracy of the network to [-1, 1] for
        the environment.

        Mapping the accuracy in [s1, s2] to [b1, b2]

        for s in [s1, s2] to obtain the reward we compute
        reward = b1 + ((s-a1)*(b2-b1)) / (a2-a1)
        """
        # Map accuracies greater than the baseline to
        # [0, 1]
        reward = 0

        if self.baseline_acc <= accuracy:
            a1, a2 = self.baseline_acc, 1.0
            b1, b2 = 0.0, 1.0

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
        # Map accuracies smaller than the baseline to
        # [-1, 0]
        elif self.baseline_acc >= accuracy:
            a1, a2 = 0.0, self.baseline_acc
            b1, b2 = -1, 0.0

            reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
        # Else, the reward is 0, baseline_acc == accuracy

        return reward

    def _darts_estimation(self, task):
        # TODO: Appropriate refactor of the accuracy calculation,
        # Only estimating on the training set, no training (yet).
        self.meta_model.eval()

        train_acc = []

        # with torch.no_grad():
        for _, (train_X, train_y) in enumerate(task.train_loader):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)

            logits = self.meta_model(train_X)

            prec1, _ = utils.accuracy(logits, train_y, topk=(1, 5))
            train_acc.append(prec1.item())

        reward = sum(train_acc) / len(train_acc)
        return reward

    def _meta_predictor_estimation(self, task):
        return NotImplemented
