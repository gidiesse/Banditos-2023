from algorithms.bandits.learner import Learner
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy as np
import numpy.typing as npt


class EXP3(Learner):
    def __init__(self, n_arms, edge_indexes, n_nodes, mc_it, T):
        super().__init__(n_arms)
        self.arm_indexes = edge_indexes
        self.n_nodes = n_nodes
        self.mc_it = mc_it
        self.T = T
        self.S = np.zeros(n_arms)
        self.prob_matrix = np.zeros(shape=(n_nodes, n_nodes))
        self.lr = np.sqrt(np.log(self.n_arms) / (20 * self.n_arms * self.T))
        self.w = np.ones(n_arms)
        self.gamma = 0.2

    """
    def select_best_seeds(self, n_seeds):
        self.prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        P = np.exp(self.lr * self.S) / np.sum(np.exp(self.lr * self.S))
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] = P[arm]
        selector = GreedySeedsSelection(self.prob_matrix, self.mc_it, self.n_nodes)
        return selector.select_seeds(n_seeds)
    """

    def select_best_seeds(self, n_seeds):
        self.prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        P = (1 - self.gamma) * self.w / self.w.sum() + self.gamma/self.n_arms
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] = P[arm]
        selector = GreedySeedsSelection(self.prob_matrix, self.mc_it, self.n_nodes)
        return selector.select_seeds(n_seeds)

    """
    def update(self, pulled_arm, reward, new_episode: npt.NDArray):
        self.t += 1
        self.update_observations(reward=reward, pulled_arm=pulled_arm)
        P = np.exp(self.lr * self.S) / np.sum(np.exp(self.lr * self.S))
        for arm in range(self.n_arms):
            observed = False
            activated = False
            for i in np.argwhere(new_episode.sum(axis=0) == 1):
                if i == self.arm_indexes[arm][0]:
                    if not observed:
                        observed = True
                for j in np.argwhere(new_episode.sum(axis=0) == 1):
                    if j == self.arm_indexes[arm][1]:
                            activated = True
            if observed:
                self.S[arm] = self.S[arm] + 1 - (1 - activated) / P[arm]
            else:
                self.S[arm] = self.S[arm] + 1
    """

    def update(self, pulled_arm, reward, new_episode: npt.NDArray):
        self.t += 1
        self.update_observations(reward=reward, pulled_arm=pulled_arm)
        P = (1 - self.gamma) * self.w / self.w.sum() + self.gamma/self.n_arms
        x_hat = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            observed = False
            activated = False
            for i in np.argwhere(new_episode.sum(axis=0) == 1):
                for j in np.argwhere(new_episode.sum(axis=0) == 1):
                    if i == self.arm_indexes[arm][0]:
                        if not observed:
                            observed = True
                        if j == self.arm_indexes[arm][1]:
                            activated = True
            if observed:
                x_hat[arm] = int(activated)
        self.w = self.w * np.exp(self.gamma * x_hat / self.n_arms)
