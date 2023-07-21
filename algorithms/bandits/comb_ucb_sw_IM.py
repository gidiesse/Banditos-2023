from algorithms.bandits.comb_ucb_IM import ucbIM
import numpy as np
import numpy.typing as npt
from typing import List


class ucbSW(ucbIM):
    def __init__(self, n_arms, edge_indexes, n_nodes, mc_it, window_size):
        super().__init__(n_arms, edge_indexes, n_nodes, mc_it)
        self.window_size = window_size
        self.pulled_arms = np.array([])
        self.credits_collected = [[np.array([]) for j in range(n_nodes)] for i in range(n_nodes)]
        self.occur_v_active = [[np.array([]) for j in range(n_nodes)] for i in range(n_nodes)]

    def estimate_probabilities_optimized(self, dataset: List[npt.NDArray]):
        prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for node_index in range(self.n_nodes):
            # node_index is u
            credits = np.zeros(self.n_nodes)
            occur_v_active = np.zeros(self.n_nodes)
            for episode in dataset[-self.window_size:]:
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

    def compute_upper_bounds(self, dataset: List[npt.NDArray]):
        n_samples = np.zeros(self.n_arms)
        for episode in dataset[-self.window_size:]:
            for arm in range(self.n_arms):
                observed = False
                for node in np.argwhere(episode.sum(axis=0) == 1):
                    if node == self.arm_indexes[arm][0] and not observed:
                        n_samples[arm] += 1
                        observed = True
        for arm in range(self.n_arms):
            self.confidence[arm] = np.sqrt(np.log(self.t) / (4 * n_samples[arm])) if n_samples[arm] else 1e3






