from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.ucb_sw import ucbSW
from algorithms.bandits.ucb_cusum import UCBCUSUM
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


"""
STEP 5 - Sensibility analysis 
Regret analysis for the UCB algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph in a non stationary environment.
Two approaches: sliding window UCB and cusum UCB. 
We conduct a grid search on:
[1] the sliding window length, for what concerns the sliding window approach
[2] M, epsilon, h parameters for what concerns the CUSUM approach
We explore 5 different windows length for the sliding window approach.
We explore 3 different values for each of the 3 parameters (M,epsilon,h) of the CUSUM approach.

The outputs are 5 plots of the cumulative regret in the case of analysis of the sliding window, 
27 plots of the cumulative regret in the case of analysis of the CUSUM. 
"""

# Sliding window flag: if set to True grid search on SW, grid search on cusum otherwise
SW = False

# Parameters of the problem
n_nodes = 30
n_arms = 50
T = 365
n_phases = 3
n_experiments = 30

# Sliding window parameter grid
window_size = [int(T**0.5/2), int(T**0.5), int(T**0.5)*2, int(T**0.5)*4, int(T**0.5)*8]

# CUSUM parameters grid -> (M, eps, h) in each tuple
cusum_grid = []
for comb in product([1, 2, 4], [0.05, 0.1, 0.2], [np.log(T)/12, np.log(T)/6, np.log(T)]):
    cusum_grid.append(comb)

# Grid to use in the experiment:
if SW:
    grid = window_size
else:
    grid = cusum_grid

for i in range(len(grid)):
    ucb_rewards_per_experiment = []
    opt_reward_per_experiment = []

    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)

    for e in range(0, n_experiments):
        if SW:
            ucb = ucbSW(n_arms, grid[i])
        else:
            ucb = UCBCUSUM(n_arms, *grid[i])
        opt = np.array([])
        for t in range(0, T):
            pulled_arm_ucb = ucb.pull_arm()
            reward_ucb = env.round(pulled_arm_ucb, t)
            ucb.update(pulled_arm_ucb, reward_ucb)
            opt = np.append(opt, env.opt(t))
        ucb_rewards_per_experiment.append(ucb.collected_rewards)
        opt_reward_per_experiment.append(opt)

    # Cumulative regret of the algorithm
    plt.figure(i)
    plt.ylabel("Cumulative regret")
    plt.xlabel("t")
    cum_regret_ucb = np.cumsum(opt_reward_per_experiment - np.array(ucb_rewards_per_experiment), axis=1)
    mean_cum_regret_ucb = np.mean(cum_regret_ucb, axis=0)
    std_cum_regret_ucb = np.std(cum_regret_ucb, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_cum_regret_ucb)
    plt.fill_between(range(len(mean_cum_regret_ucb)), mean_cum_regret_ucb-1.96*std_cum_regret_ucb, mean_cum_regret_ucb+1.96*std_cum_regret_ucb)
    if SW:
        plt.legend([f"UCB_SW({grid[i]})", f".95 CI UCB_SW({grid[i]})"])
    else:
        grid[i] = (round(grid[i][0], 2), round(grid[i][1], 2), round(grid[i][2], 2))
        plt.legend([f"UCB_CUSUM({grid[i]})", f".95 CI UCB_CUSUM({grid[i]})"])
    plt.show()
