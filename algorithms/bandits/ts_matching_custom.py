from algorithms.bandits.learner import Learner
from scipy.optimize import linear_sum_assignment
import numpy as np

"""
WORK IN PROGRESS, THIS CLASS DOESN'T WORK!
"""

class TSMatchingCustom(Learner):
    def __init__(self, n_products, n_units, n_cc):
        super().__init__(n_products * n_cc)
        self.corr_m = np.array([])
        self.n_products = n_products
        self.n_units = n_units
        self.n_cc = n_cc
        self.empirical_means = np.ones(n_products*n_cc)
        self.precision = np.ones(n_products*n_cc) * 1e-4

    def init_matrix(self, activated_customers):
        corr_matrix = np.zeros(shape=(len(activated_customers), self.n_products*self.n_units))
        for p in range(self.n_products * self.n_units):
            for c in range(len(activated_customers)):
                corr_matrix[c, p] = p // self.n_units + self.n_products * activated_customers[c]
        self.corr_m = corr_matrix.astype(int)

    def pull_arm(self, activated_customers):
        self.init_matrix(activated_customers)
        rew_matrix = np.zeros(shape=(self.corr_m.shape[0], self.corr_m.shape[1]))
        for i in range(self.corr_m.shape[0]):
            for j in range(self.corr_m.shape[1]):
                rew_matrix[i, j] = np.random.normal(self.empirical_means[self.corr_m[i, j].astype(int)])
        row_ind, col_ind = linear_sum_assignment(-rew_matrix)
        return row_ind, col_ind

    def update(self, pulled_arms, rewards):
        # self.t += 1
        pulled_arms_flat = np.array([])
        for i, j in zip(pulled_arms[0], pulled_arms[1]):
            pulled_arms_flat = np.append(pulled_arms_flat, self.corr_m[i, j])
        for pulled_arm, reward in zip(pulled_arms_flat.astype(int), rewards):
            self.update_observations(pulled_arm, reward)
            n_samples = len(self.rewards_per_arm[pulled_arm])
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (n_samples - 1) + reward) / n_samples



