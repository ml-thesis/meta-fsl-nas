import torch
import torch.nn as nn
import numpy as np

import time
import copy
from collections import OrderedDict, namedtuple

import gym
from gym import error, spaces
from gym.utils import seeding

from metanas.models.search_cnn import SearchCNNController
from metanas.task_optimizer.darts import Architect
import metanas.utils.genotypes as gt
from metanas.utils import utils

"""Wrapper for the RL agent to interact with the meta-model in the outer-loop
utilizing the OpenAI gym interface
"""


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, meta_model, task_optimizer,
                 test_phase=False, cell_type="normal",
                 max_steps=100):
        super().__init__()
        self.config = config
        self.cell_type = cell_type
        self.test_phase = test_phase
        self.meta_model = meta_model
        self.meta_epoch = 0

        # The task is set in the meta-loop
        self.current_task = None
        self.max_steps = max_steps
        self.reward_range = (-1, 1)

        # Initialize the step counter
        self.step_count = 0
        self.terminate_episode = False

        # when test_phase is True the environment is used for
        # meta-testing
        # TODO: Training the model/obtain the reward

        # Initialize State/Observation space

        # Adjacency matrix
        self.A = np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

        # Remove the 2 input nodes from A
        self.A[0, 1] = 0
        self.A[1, 0] = 0

        # Set starting edge for agent
        self.set_start_edge()

        # Intermediate + input nodes
        self.n_nodes = self.config + 2
        self.initialize_observation_space()

        # Initialize Action space
        # TODO: |A| + 2*|O| + 1, the +1 for the termination
        self.action_space = spaces.Discrete(6)

    def reset(self):
        """Reset the environment state"""
        # Initialize the step counters
        self.step_count = 0
        self.terminate_episode = False

        # Set starting edge for agent
        self.set_start_edge()

        # TODO: return observation

    def set_start_edge(self):
        # TODO: Add probability to the starting edge?
        self.cur_node = 0
        self.next_node = 2

    def initialize_observation_space(self):

        # Define (normalized) alphas
        if self.cell_type == "normal":
            # Idea of letting RL observe the normalized alphas,
            # and mutate the actual alpha values
            self.normalized_alphas = [
                self.meta_model.apply_normalizer(
                    alpha).detach().cpu()
                for alpha in self.meta_model.alpha_normal]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_normal]

        elif self.cell_type == "reduce":
            self.normalized_alphas = [
                self.meta_model.apply_normalizer(
                    alpha).detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

            self.alphas = [
                alpha.detach().cpu()
                for alpha in self.meta_model.alpha_reduce]

        else:
            raise RuntimeError(f"Cell type {self.cell_type} is not supported.")

        # Generate the internal states of the graph
        self.update_states()

        self.observation_space = spaces.Box(0, 1,
                                            shape=self.state[0].shape,
                                            dtype=np.int32)

    def update_states(self):
        # Index for every state
        s_i = 0
        state_variables = []

        # TODO: Pick normalized alphas or regular alphas
        for i, edges in enumerate(self.normalized_alphas):
            # edges: Tensor(n_edges, n_ops)
            edge_max, _ = torch.topk(edges[:, :], 1)
            # selecting the top-k input nodes, k=2
            _, topk_edge_indices = torch.topk(edge_max.view(-1), k=2)

            for j, edge in enumerate(edges):
                state_variables.append(
                    np.concatenate((
                        [j],
                        [i+2],
                        self.A[s_i],
                        [int(j in topk_edge_indices)],
                        edge.detach().numpy())))
            s_i += 1

        self.states = np.array(state_variables)
        self.current_state = self.states[0]

    def render(self, mode='human'):
        """Render the environment, according to the specified mode."""
        for row in self.state:
            print(row)

    def step(self, action):
        start = time.time()
        # Mutates the meta_model and the local state
        action_info = self._perform_action(action)

        # Calculate a new reward when the alphas are changed
        # Reward in the range 0, 1
        if action in [0, 1, 2]:
            reward = self._darts_step(
                self.current_task,
                self.meta_epoch,
                self.test_phase)
        else:
            reward = 0.0

        # The final step time
        end = time.time()
        running_time = int(end - start)

        self.step_count += 1

        # Conditions to terminate the episode
        done = self.step_count == self.max_steps or \
            self.train_step == self.max_darts_steps or \
            self.terminate_episode

        info_dict = {
            "step_count": self.step_count,
            "action_id": action,
            "action": action_info,
            "reward": reward,
            "done": done,
            "running_time": running_time
        }

        return self.state, reward, done, info_dict

    def close(self):
        return NotImplemented

    def set_task(self, task, meta_epoch):
        """The meta-loop passes the task for the environment to solve"""
        self.current_task = task
        self.meta_epoch = meta_epoch

        self.reset()

    def _perform_action(self, action):
        """Perform the action on both the meta-model and local state"""

        action_info = ""

        # First two actions increase/decrease alpha
        if action == 0:
            decrease_val = np.sum(
                np.abs(self.state[self.row_ptr, self.col_ptr]))
            # Adjust local alphas and meta-model alphas

            self.state[self.row_ptr, self.col_ptr] -= decrease_val

            edge_idx, row_idx = gt.find_indice(self.row_ptr, self.n_nodes)
            with torch.no_grad():
                self.meta_model.alpha_normal[
                    edge_idx][row_idx][self.col_ptr] -= decrease_val
            action_info = f"Decrease the alpha value by {decrease_val} at ({self.row_ptr}, {self.col_ptr})"

        if action == 1:
            increase_val = np.sum(
                np.abs(self.state[self.row_ptr, self.col_ptr]))

            # Adjust local alphas and meta-model alphas
            self.state[self.row_ptr,
                       self.col_ptr] += increase_val

            edge_idx, row_idx = gt.find_indice(self.row_ptr, self.n_nodes)
            with torch.no_grad():
                self.meta_model.alpha_normal[
                    edge_idx][row_idx][self.col_ptr] += increase_val
            action_info = f"Increase the alpha value by {increase_val} at ({self.row_ptr}, {self.col_ptr})"

        # Set alpha to zero
        if action == 2:
            self.state[self.row_ptr, self.col_ptr] = 0

            edge_idx, row_idx = gt.find_indice(self.row_ptr, self.n_nodes)
            with torch.no_grad():
                self.meta_model.alpha_normal[
                    edge_idx][row_idx][self.col_ptr] = 0
            action_info = f"Decrease the alpha value to 0 at ({self.row_ptr}, {self.col_ptr})"

        # Move row and column pointers
        if action == 3:
            self.row_ptr -= 1 if self.row_ptr != 0 else 0
            action_info = f"Move the row pointer to {self.row_ptr}"

        if action == 4:
            self.row_ptr += 1 if self.row_ptr < self.state.shape[0]-1 \
                else 0
            action_info = f"Move the row pointer to {self.row_ptr}"

        if action == 5:
            self.col_ptr += 1 if self.col_ptr < self.state.shape[1]-1 \
                else 0
            action_info = f"Move the column pointer to {self.col_ptr}"

        # Terminate the episode
        if action == 6:
            self.terminate_episode = True
            action_info = f"Terminate the episode at step {self.step_count}"
        return action_info

            )

    # TODO: Appropriate refactor of the accuracy calculation
    def _darts_step(
        self,
        task
    ):

        self.meta_model.model.train()

        train_acc=[]

        for step, ((train_X, train_y), (val_X, val_y)) in enumerate(
            zip(task.train_loader, task.valid_loader)
        ):
            train_X, train_y = train_X.to(
                self.config.device), train_y.to(self.config.device)
            val_X, val_y = val_X.to(
                self.config.device), val_y.to(self.config.device)

            logits = self.meta_model(train_X)

            prec1, _ = utils.accuracy(logits, train_y, topk=(1, 5))
            train_acc.append(prec1.item())

        reward = sum(train_acc) / len(train_acc)
        return reward


def scale_reward(accuracy, baseline):
    """
    Map the accuracy of the network to [-1, 1] for
    the environment.

    Mapping the accuracy in [s1, s2] to [b1, b2]

    for s in [s1, s2] to obtain the reward we compute
    reward = b1 + ((s-a1)*(b2-b1)) / (a2-a1)
    """
    # Map accuracies greater than the baseline to
    # [0, 1]
    if baseline <= accuracy:
        a1, a2 = baseline, 1.0
        b1, b2 = 0.0, 1.0

        reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
        return reward
    # Map accuracies smaller than the baseline to
    # [-1, 0]
    elif baseline >= accuracy:
        a1, a2 = 0.0, baseline
        b1, b2 = -1, 0.0

        reward = b1 + ((accuracy-a1)*(b2-b1)) / (a2-a1)
        return reward

    # Else, the reward is 0
    elif baseline == accuracy:
        return 0.0
