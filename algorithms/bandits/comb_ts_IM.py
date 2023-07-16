from algorithms.bandits.learner import Learner
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy as np
import numpy.typing as npt

class TSLearner(Learner):
    def __init__(self, n_arms, edge_indexes, n_nodes, mc_it):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.arm_indexes = edge_indexes
        self.n_nodes = n_nodes
        self.prob_matrix = np.zeros(shape=(n_nodes, n_nodes))
        self.mc_it = mc_it

    def select_best_seeds(self, n_seeds):
        self.prob_matrix = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for arm in range(self.n_arms):
            (i, j) = self.arm_indexes[arm].astype(int)
            self.prob_matrix[i, j] = np.random.beta(self.beta_parameters[arm, 0], self.beta_parameters[arm, 1])
        selector = GreedySeedsSelection(self.prob_matrix, self.mc_it, self.n_nodes)
        return selector.select_seeds(n_seeds)

    def update(self, pulled_arm, reward, new_episode: npt.NDArray):
        self.t += 1
        self.update_observations(reward=reward, pulled_arm=pulled_arm)
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
                self.beta_parameters[arm, 0] += int(activated)
                self.beta_parameters[arm, 1] += 1.0 - int(activated)





