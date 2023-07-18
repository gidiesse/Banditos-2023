from algorithms.bandits.cusum import CUSUM
import numpy as np
from algorithms.bandits.comb_ucb_IM import ucbIM
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy.typing as npt

class ucbCUSUM(ucbIM):
    def __init__(self, n_arms, edge_indexes, n_nodes, mc_it,  M=100, eps=0.3, h=20, alpha=0.01):
        super().__init__(n_arms, edge_indexes, n_nodes, mc_it)
        self.change_detection = CUSUM(M, eps, h)
        self.valid_rewards = []
        self.detections = []
        self.alpha = alpha

    def select_best_seeds(self, n_seeds):
        if np.random.binomial(1, 1-self.alpha):
            for arm in range(self.n_arms):
                (i, j) = self.arm_indexes[arm].astype(int)
                self.prob_matrix[i, j] += self.confidence[arm]
            selector = GreedySeedsSelection(self.prob_matrix, self.mc_it, self.n_nodes)
            return selector.select_seeds(n_seeds)
        else:
            all_nodes = np.array(range(n_seeds))
            selected_seeds = np.random.choice(all_nodes, size=n_seeds, replace=False)
            seeds = np.zeros(self.n_nodes)
            seeds[selected_seeds] += 1
            return seeds

    def update(self, pulled_arm, reward, new_episode: npt.NDArray):
        self.t += 1
        if self.change_detection.update(reward):
            self.detections.append(self.t)
            self.n_samples = np.zeros(self.n_arms)
            self.credits = np.zeros(shape=(self.n_nodes, self.n_nodes))
            self.occur_v_active = np.zeros(shape=(self.n_nodes, self.n_nodes))
            self.change_detection.reset()
        self.update_observations(pulled_arm, reward)
        self.compute_upper_bounds(new_episode)
        return self.estimate_probabilities_optimized(new_episode)

    def compute_upper_bounds(self, new_episode: npt.NDArray):
        for arm in range(self.n_arms):
            observed = False
            for node in np.argwhere(new_episode.sum(axis=0) == 1):
                if node == self.arm_indexes[arm][0] and not observed:
                    self.n_samples[arm] += 1
                    observed = True
        for arm in range(self.n_arms):
            self.confidence[arm] = np.sqrt(np.log(self.change_detection.t) / (4 * self.n_samples[arm])) if self.n_samples[arm] else 1e3
