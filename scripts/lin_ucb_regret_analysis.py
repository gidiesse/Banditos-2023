import numpy as np
import matplotlib.pyplot as plt
from algorithms.environments.linear_mab_environment import *
from algorithms.bandits.lin_ucb_learner import *

"""
Regret analysis for the linear UCB algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph. 
The outputs are 4 plots: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

n_nodes = 30
n_arms = 50
T = 365
n_experiments = 100
lin_ucb_rewards_per_experiment = []

env = LinearMabEnvironment(n_arms=n_arms, dim=2, n_nodes=n_nodes)

for e in range(0, n_experiments):
    lin_ucb_learner = LinearUcbLearner(arms_features=env.arms_features)
    for t in range(0,T):
        pulled_arm = lin_ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        lin_ucb_learner.update(pulled_arm, reward)
    lin_ucb_rewards_per_experiment.append(lin_ucb_learner.collected_rewards)

# Cumulative regret of the algorithm
opt = env.opt()
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret = np.cumsum(opt - np.array(lin_ucb_rewards_per_experiment), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(n_experiments)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(lin_ucb_rewards_per_experiment), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret = (opt - np.array(lin_ucb_rewards_per_experiment))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(cum_regret, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Instantaneous reward of the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward = np.array(lin_ucb_rewards_per_experiment)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()





