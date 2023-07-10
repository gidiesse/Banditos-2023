import numpy as np
from algorithms.bandits.learner import Learner


class Exp3(Learner):
    def __init__(self, n_arms, T):
        super().__init__(n_arms)
        self.T = T
        self.S = np.zeros(n_arms)
        self.lr = np.sqrt(2*np.log(self.n_arms) / (self.n_arms * self.T))

    def pull_arm(self):
        P = np.exp(self.lr*self.S) / np.sum(np.exp(self.lr*self.S))
        pulled_arm = np.random.multinomial(1, P)
        return np.where(pulled_arm == 1)[0][0]

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        P = np.exp(self.lr * self.S) / np.sum(np.exp(self.lr * self.S))
        indicator = np.zeros(self.n_arms)
        indicator[pulled_arm] = 1
        self.S = self.S + np.ones(self.n_arms) - (indicator*(1-reward)) / P # update all arms
        # self.S[pulled_arm] = self.S[pulled_arm] + 1 - (1-reward)/P[pulled_arm] # update only pulled arm
