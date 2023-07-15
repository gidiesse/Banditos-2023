import numpy as np
from scipy.optimize import linear_sum_assignment

class EnvironmentGaussian:
    def __init__(self, n_products, n_units, n_cc, gaussian_means, sigma=1):
        self.n_arms = n_products * n_cc
        self.means = gaussian_means.reshape(-1)
        self.n_cc = n_cc
        self.n_products = n_products
        self.n_units = n_units
        self.corr_m = np.array([])
        self.sigma = np.ones(self.n_arms) * sigma

    def init_matrix(self, activated_customers):
        corr_matrix = np.zeros(shape=(len(activated_customers), self.n_products * self.n_units))
        for p in range(self.n_products * self.n_units):
            for c in range(len(activated_customers)):
                corr_matrix[c, p] = p // self.n_units + self.n_products * activated_customers[c]
        self.corr_m = corr_matrix

    def optimal_matching(self, activated_customers):
        self.init_matrix(activated_customers)
        rew_matrix = np.zeros(shape=(self.corr_m.shape[0], self.corr_m.shape[1]))
        for i in range(rew_matrix.shape[0]):
            for j in range(rew_matrix.shape[1]):
                rew_matrix[i, j] = self.means[self.corr_m[i, j].astype(int)]
        opt_matching = linear_sum_assignment(-rew_matrix)
        return rew_matrix[opt_matching].sum()

    def round(self, pulled_arms):
        reward = np.random.normal(loc=self.means[self.corr_m[pulled_arms].astype(int)],
                                  scale=self.sigma[self.corr_m[pulled_arms].astype(int)])
        return reward
