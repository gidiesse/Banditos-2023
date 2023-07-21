from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.exp3 import EXP3
from algorithms.bandits.comb_ucb_sw_IM import ucbSW
from algorithms.bandits.ucb_cusum_IM import ucbCUSUM
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy as np
import matplotlib.pyplot as plt

"""
STEP 6
Regret analysis for the two non stationary flavours of the UCB algorithm and EXP3 applied to learning 
the probability matrix associated to the activations of the edges of a graph in a non stationary 
environment.
Three approaches: sliding window UCB, cusum UCB, EXP3 
You can set the hf flag and n_phases to establish the degree of non stationarity of the environment.
The outputs are 4 plots: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

# here we establish if we are in the case of high-frequency phase changes or not
hf = False

np.random.seed(7)
n_nodes = 30
n_arms = 50
T = 365
n_phases = 3
n_experiments = 20
n_seeds = 1

# Sliding window parameter
window_size = 6 * int(T**0.5)

# CUSUM parameters
M = 120
# h has to be modified in the grid search
h = 2 * np.log(T)
# eps has to be modified in the grid search
eps = 0.25


exp3_rewards_per_experiment = []
ucb_sw_rewards_per_experiment = []
ucb_cusum_rewards_per_experiment = []
opt_reward_per_day = []

env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases, hf=hf)

# Now we have to select the optimal seeds for each of the phases
opt_seeds = []
for phase in range(n_phases):
    clairvoyant_sel = GreedySeedsSelection(env.prob_matrix[phase], n_experiments*50, n_max_steps=n_nodes)
    opt_seeds.append(clairvoyant_sel.select_seeds(n_seeds))

# Now we have different optimal values for each of the phases
opt = [0 for i in range(n_phases)]
for phase in range(n_phases):
    for e in range(0, n_experiments*20):
        opt[phase] += env.simulate_episode(opt_seeds[phase], phase).sum()
    opt[phase] = opt[phase] / (n_experiments*20)
print(opt)

opt_reward_per_day = []
for t in range(T):
    opt_reward_per_day.append(opt[env.select_phase(t)])

for e in range(0, n_experiments):
    print(e)
    exp3 = EXP3(n_arms, env.arms, n_nodes, n_experiments*5, T)
    ucb_sw = ucbSW(n_arms, env.arms, n_nodes, mc_it=n_experiments * 5, window_size=window_size)
    ucb_cusum = ucbCUSUM(n_arms, env.arms, n_nodes, mc_it=n_experiments * 5, M=M, h=h, eps=eps)
    # We need the whole dataset of the episodes for the sliding window approach
    dataset_episodes = []
    for t in range(0, T):
        seeds_estimated_exp3 = exp3.select_best_seeds(n_seeds)
        seeds_estimated_ucb_sw = ucb_sw.select_best_seeds(n_seeds)
        seeds_estimated_ucb_cusum = ucb_cusum.select_best_seeds(n_seeds)
        phase = env.select_phase(t)
        simulation_exp3 = env.simulate_episode(seeds_estimated_exp3, phase)
        simulation_ucb_sw = env.simulate_episode(seeds_estimated_ucb_sw, phase)
        simulation_ucb_cusum = env.simulate_episode(seeds_estimated_ucb_cusum, phase)
        dataset_episodes.append(simulation_ucb_sw)
        reward_ucb = simulation_exp3.sum()
        reward_ucb_sw = simulation_ucb_sw.sum()
        reward_ucb_cusum = simulation_ucb_cusum.sum()
        exp3.update(1, reward_ucb, simulation_exp3)
        ucb_sw.update(1, reward_ucb_sw, dataset_episodes)
        ucb_cusum.update(1, reward_ucb_cusum, simulation_ucb_cusum)
    exp3_rewards_per_experiment.append(exp3.collected_rewards)
    ucb_sw_rewards_per_experiment.append(ucb_sw.collected_rewards)
    ucb_cusum_rewards_per_experiment.append(ucb_cusum.collected_rewards)


# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret_exp3 = np.cumsum(opt_reward_per_day - np.array(exp3_rewards_per_experiment), axis=1)
cum_regret_ucb_sw = np.cumsum(opt_reward_per_day - np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_regret_ucb_cusum = np.cumsum(opt_reward_per_day - np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_regret_exp3 = np.mean(cum_regret_exp3, axis=0)
mean_cum_regret_ucb_sw = np.mean(cum_regret_ucb_sw, axis=0)
mean_cum_regret_ucb_cusum = np.mean(cum_regret_ucb_cusum, axis=0)
std_cum_regret_exp3 = np.std(cum_regret_exp3, axis = 0) / np.sqrt(n_experiments)
std_cum_regret_ucb_sw = np.std(cum_regret_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_cum_regret_ucb_cusum = np.std(cum_regret_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_regret_exp3)
plt.fill_between(range(len(mean_cum_regret_exp3)), mean_cum_regret_exp3-1.96*std_cum_regret_exp3, mean_cum_regret_exp3+1.96*std_cum_regret_exp3)
plt.plot(mean_cum_regret_ucb_sw)
plt.fill_between(range(len(mean_cum_regret_ucb_sw)), mean_cum_regret_ucb_sw-1.96*std_cum_regret_ucb_sw, mean_cum_regret_ucb_sw+1.96*std_cum_regret_ucb_sw)
plt.plot(mean_cum_regret_ucb_cusum)
plt.fill_between(range(len(mean_cum_regret_ucb_cusum)), mean_cum_regret_ucb_cusum-1.96*std_cum_regret_ucb_cusum, mean_cum_regret_ucb_cusum+1.96*std_cum_regret_ucb_cusum)
plt.legend(["EXP3", ".95 CI EXP3", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward_exp3 = np.cumsum(np.array(exp3_rewards_per_experiment), axis=1)
cum_reward_ucb_sw = np.cumsum(np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_reward_ucb_cusum = np.cumsum(np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_reward_exp3 = np.mean(cum_reward_exp3, axis=0)
mean_cum_reward_ucb_sw = np.mean(cum_reward_ucb_sw, axis=0)
mean_cum_reward_ucb_cusum = np.mean(cum_reward_ucb_cusum, axis=0)
std_cum_reward_exp3 = np.std(cum_reward_exp3, axis = 0) / np.sqrt(n_experiments)
std_cum_reward_ucb_sw = np.std(cum_reward_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_cum_reward_ucb_cusum = np.std(cum_reward_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_reward_exp3)
plt.fill_between(range(len(mean_cum_reward_exp3)), mean_cum_reward_exp3-1.96*std_cum_reward_exp3, mean_cum_reward_exp3+1.96*std_cum_reward_exp3)
plt.plot(mean_cum_reward_ucb_sw)
plt.fill_between(range(len(mean_cum_reward_ucb_sw)), mean_cum_reward_ucb_sw-1.96*std_cum_reward_ucb_sw, mean_cum_reward_ucb_sw+1.96*std_cum_reward_ucb_sw)
plt.plot(mean_cum_reward_ucb_cusum)
plt.fill_between(range(len(mean_cum_reward_ucb_cusum)), mean_cum_reward_ucb_cusum-1.96*std_cum_reward_ucb_cusum, mean_cum_reward_ucb_cusum+1.96*std_cum_reward_ucb_cusum)
plt.legend(["EXP3", ".95 CI EXP3", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret_exp3 = (opt_reward_per_day - np.array(exp3_rewards_per_experiment))
inst_regret_ucb_sw = (opt_reward_per_day - np.array(ucb_sw_rewards_per_experiment))
inst_regret_ucb_cusum = (opt_reward_per_day - np.array(ucb_cusum_rewards_per_experiment))
mean_inst_regret_exp3 = np.mean(inst_regret_exp3, axis=0)
mean_inst_regret_ucb_sw = np.mean(inst_regret_ucb_sw, axis=0)
mean_inst_regret_ucb_cusum = np.mean(inst_regret_ucb_cusum, axis=0)
std_inst_regret_exp3 = np.std(inst_regret_exp3, axis = 0) / np.sqrt(n_experiments)
std_inst_regret_ucb_sw = np.std(inst_regret_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_inst_regret_ucb_cusum = np.std(inst_regret_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_regret_exp3)
plt.fill_between(range(len(mean_inst_regret_exp3)), mean_inst_regret_exp3-1.96*std_inst_regret_exp3, mean_inst_regret_exp3+1.96*std_inst_regret_exp3)
plt.plot(mean_inst_regret_ucb_sw)
plt.fill_between(range(len(mean_inst_regret_ucb_sw)), mean_inst_regret_ucb_sw-1.96*std_inst_regret_ucb_sw, mean_inst_regret_ucb_sw+1.96*std_inst_regret_ucb_sw)
plt.plot(mean_inst_regret_ucb_cusum)
plt.fill_between(range(len(mean_inst_regret_ucb_cusum)), mean_inst_regret_ucb_cusum-1.96*std_inst_regret_ucb_cusum, mean_inst_regret_ucb_cusum+1.96*std_inst_regret_ucb_cusum)
plt.legend(["EXP3", ".95 CI EXP3", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous reward collected by the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward_exp3 = np.array(exp3_rewards_per_experiment)
inst_reward_ucb_sw = np.array(ucb_sw_rewards_per_experiment)
inst_reward_ucb_cusum = np.array(ucb_cusum_rewards_per_experiment)
mean_inst_reward_exp3 = np.mean(inst_reward_exp3, axis=0)
mean_inst_reward_ucb_sw = np.mean(inst_reward_ucb_sw, axis=0)
mean_inst_reward_ucb_cusum = np.mean(inst_reward_ucb_cusum, axis=0)
std_inst_reward_exp3 = np.std(inst_reward_exp3, axis = 0) / np.sqrt(n_experiments)
std_inst_reward_ucb_sw = np.std(inst_reward_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_inst_reward_ucb_cusum = np.std(inst_reward_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_reward_exp3)
plt.fill_between(range(len(mean_inst_reward_exp3)), mean_inst_reward_exp3-1.96*std_inst_reward_exp3, mean_inst_reward_exp3+1.96*std_inst_reward_exp3)
plt.plot(mean_inst_reward_ucb_sw)
plt.fill_between(range(len(mean_inst_reward_ucb_sw)), mean_inst_reward_ucb_sw-1.96*std_inst_reward_ucb_sw, mean_inst_reward_ucb_sw+1.96*std_inst_reward_ucb_sw)
plt.plot(mean_inst_reward_ucb_cusum)
plt.fill_between(range(len(mean_inst_reward_ucb_cusum)), mean_inst_reward_ucb_cusum-1.96*std_inst_reward_ucb_cusum, mean_inst_reward_ucb_cusum+1.96*std_inst_reward_ucb_cusum)
plt.legend(["EXP3", ".95 CI EXP3", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()
