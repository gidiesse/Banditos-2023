from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import matplotlib.pyplot as plt
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.comb_ts_IM import TSLearner
import numpy as np

"""
Regret analysis for the TS algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph. 
The outputs are 4 plots: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

n_nodes = 30
n_arms = 50
T = 100
n_experiments = 10
n_steps_max = n_nodes
n_seeds = 3
ts_rewards_per_experiment = []

env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T)
clairvoyant_sel = GreedySeedsSelection(env.prob_matrix, n_experiments*20, n_arms)
opt_seeds = clairvoyant_sel.select_seeds(n_seeds)
opt = 0
for e in range(0, n_experiments*20):
    opt += env.simulate_episode(opt_seeds).sum()
opt = opt / (n_experiments*20)
print(opt)
for e in range(0, n_experiments):
    print(e)
    ts_learner = TSLearner(n_arms, env.arms, n_nodes, mc_it=n_experiments*5)
    for t in range(0, T):
        seeds_estimated = ts_learner.pull_arm(n_seeds)
        simulation = env.simulate_episode(seeds_estimated)
        reward = simulation.sum()
        ts_learner.update(1, reward, new_episode=simulation)
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret = np.cumsum(opt - np.array(ts_rewards_per_experiment), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend(["TS", ".95 CI"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(ts_rewards_per_experiment), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend(["TS", ".95 CI"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret = (opt - np.array(ts_rewards_per_experiment))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(cum_regret, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.legend(["TS", ".95 CI"])
plt.show()

# Instantaneous reward of the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward = np.array(ts_rewards_per_experiment)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.legend(["TS", ".95 CI"])
plt.show()





