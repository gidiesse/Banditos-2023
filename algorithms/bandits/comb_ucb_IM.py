from algorithms.bandits.learner import Learner
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy.typing as npt
from typing import List
import numpy as np


class ucbIM(Learner):
    def __init__(self, n_arms, edge_indexes, n_nodes, mc_it):
        super().__init__(n_arms)
        self.confidence = np.ones(n_arms) * 1e3
        self.arm_indexes = edge_indexes
        self.n_nodes = n_nodes
        self.prob_matrix = np.zeros(shape=(n_nodes, n_nodes))
        self.mc_it = mc_it
        self.n_samples = np.zeros(n_arms)
        self.credits = np.zeros(shape=(n_nodes, n_nodes))
        self.occur_v_active = np.zeros(shape=(n_nodes, n_nodes))

    def select_best_seeds(self, n_seeds):
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] += self.confidence[arm]
        selector = GreedySeedsSelection(self.prob_matrix, self.mc_it, n_max_steps=self.n_nodes)
        return selector.select_seeds(n_seeds)

    def update(self, pull_arm, reward, new_episode: npt.NDArray):
        self.t += 1
        self.update_observations(pull_arm, reward)
        self.compute_upper_bounds(new_episode)
        return self.estimate_probabilities_optimized(new_episode)

    """
    This method is used to compute the estimation of the probability matrix according
    to the *** method. We assume to be in a setting where we can observe only the history
    of past activated nodes and not the activated edges (like in a real scenario). 
    However, since we suppose to know which edges can be activated with a non null 
    probability, we set to zero the probabilities of all the edges that are not among
    the ones that can be activated.
    
    Args:
    dataset: Collection of episodes of influence maximization problems.                 
    """
    def estimate_probabilities(self, dataset: List[npt.NDArray]):
        prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for node_index in range(self.n_nodes):
            # node_index is u
            credits = np.zeros(self.n_nodes)
            occur_v_active = np.zeros(self.n_nodes)
            for episode in dataset:
                # time index -> when node_index was active
                idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)  # t_u_k
                if idx_w_active.size > 0 and idx_w_active > 0:
                    active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
                    credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
                # for every node different from node_index,
                for v in range(0, self.n_nodes):
                    if v != node_index:
                        idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                        if idx_v_active.size > 0 and (idx_w_active.size == 0 or idx_v_active < idx_w_active):
                            occur_v_active[v] += 1
            estimated_prob = credits / occur_v_active
            estimated_prob = np.nan_to_num(estimated_prob, nan=0.0, posinf=0.0, neginf=0.0)
            prob_matrix[:, node_index] = estimated_prob

        self.prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] = prob_matrix[i, j]
        return prob_matrix

    """
    This method is used to compute the estimation of the probability matrix according
    to the *** method. We assume to be in a setting where we can observe only the history
    of past activated nodes and not the activated edges (like in a real scenario). 
    However, since we suppose to know which edges can be activated with a non null 
    probability, we set to zero the probabilities of all the edges that are not among
    the ones that can be activated.

    Args:
    dataset: Collection of episodes of influence maximization problems.                 
    """

    def estimate_probabilities_optimized(self, episode: npt.NDArray):
        prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for node_index in range(self.n_nodes):
            # node_index is u
            # time index -> when node_index was active
            idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)  # t_u_k
            if idx_w_active.size > 0 and idx_w_active > 0:
                active_nodes_in_prev_step = episode[idx_w_active - 1, :].reshape(-1)
                self.credits[node_index] += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)
            # for every node different from node_index,
            for v in range(0, self.n_nodes):
                if v != node_index:
                    idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                    if idx_v_active.size > 0 and (idx_w_active.size == 0 or idx_v_active < idx_w_active):
                        self.occur_v_active[node_index, v] += 1
            estimated_prob = self.credits[node_index] / self.occur_v_active[node_index]
            estimated_prob = np.nan_to_num(estimated_prob, nan=0.0, posinf=0.0, neginf=0.0)
            prob_matrix[:, node_index] = estimated_prob

        self.prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] = prob_matrix[i, j]
        return self.prob_matrix

    """
    This method is used to compute the upper confidence bounds for all the arms.
    1) Count the number of time the edge was observed as the number of times 
       that the starting node for that edge was activated. 
    2) Compute the usual confidence bound for each edge.  
    
    Args:
    dataset: Collection of episodes of influence maximization problems.     
    """
    def compute_upper_bounds(self, new_episode: npt.NDArray):
        for arm in range(self.n_arms):
            observed = False
            for node in np.argwhere(new_episode.sum(axis=0) == 1):
                if node == self.arm_indexes[arm][0] and not observed:
                    self.n_samples[arm] += 1
                    observed = True
        for arm in range(self.n_arms):
            self.confidence[arm] = np.sqrt(np.log(self.t) / (4*self.n_samples[arm])) if self.n_samples[arm] else 1e3


