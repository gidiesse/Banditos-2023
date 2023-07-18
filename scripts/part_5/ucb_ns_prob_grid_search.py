from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.comb_ucb_sw_IM import ucbSW
from algorithms.bandits.ucb_cusum_IM import ucbCUSUM
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
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
We explore 3 different windows length for the sliding window approach.
We explore 3 different values for each of the 3 parameters (M,epsilon,h) of the CUSUM approach.

The outputs are 3 plots of the cumulative regret in the case of analysis of the sliding window, 
27 plots of the cumulative regret in the case of analysis of the CUSUM. 
"""

# Sliding window flag: if set to True grid search on SW, grid search on cusum otherwise
SW = True

# Parameters of the problem
n_nodes = 30
n_arms = 50
T = 365
n_phases = 3
n_experiments = 20
n_seeds = 1

# Sliding window parameter grid
window_size = [int(T**0.5), 2*int(T**0.5),4*int(T**0.5), 6*int(T**0.5), 8*int(T**0.5)]


# CUSUM parameters grid -> (M, eps, h) in each tuple
cusum_grid = []
for comb in product([60, 120, 180], [0.1, 0.5, 1], [np.log(T), 3*np.log(T), 6*np.log(T)]):
    cusum_grid.append(comb)

# Grid to use in the experiment:
if SW:
    grid = window_size
else:
    grid = cusum_grid

for i in range(len(grid)):
    np.random.seed(0)
    ucb_rewards_per_experiment = []
    opt_reward_per_experiment = []

    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)
    # Now we have to select the optimal seeds for each of the phases
    opt_seeds = []
    for phase in range(n_phases):
        clairvoyant_sel = GreedySeedsSelection(env.prob_matrix[phase], n_experiments * 50, n_max_steps=n_nodes)
        opt_seeds.append(clairvoyant_sel.select_seeds(n_seeds))

    # Now we have different optimal values for each of the phases
    opt = [0 for i in range(n_phases)]
    for phase in range(n_phases):
        for e in range(0, n_experiments * 20):
            opt[phase] += env.simulate_episode(opt_seeds[phase], phase).sum()
        opt[phase] = opt[phase] / (n_experiments * 20)
    print(opt)
    opt_reward_per_day = []
    for t in range(T):
        opt_reward_per_day.append(opt[env.select_phase(t)])

    for e in range(0, n_experiments):
        print(e)
        if SW:
            ucb = ucbSW(n_arms, env.arms, n_nodes, n_experiments*5, grid[i])
            dataset_episodes = []
        else:
            ucb = ucbCUSUM(n_arms, env.arms, n_nodes, n_experiments*5, *grid[i])
        opt = np.array([])
        for t in range(0, T):
            seeds_estimated_ucb = ucb.select_best_seeds(n_seeds)
            phase = env.select_phase(t)
            simulation_ucb = env.simulate_episode(seeds_estimated_ucb, phase)
            reward_ucb = simulation_ucb.sum()
            if SW:
                dataset_episodes.append(simulation_ucb)
                ucb.update(1, reward_ucb, dataset_episodes)
            else:
                ucb.update(1, reward_ucb, simulation_ucb)
        ucb_rewards_per_experiment.append(ucb.collected_rewards)

    # Cumulative regret of the algorithm
    plt.figure(i)
    plt.ylabel("Cumulative regret")
    plt.xlabel("t")
    cum_regret_ucb = np.cumsum(opt_reward_per_day - np.array(ucb_rewards_per_experiment), axis=1)
    mean_cum_regret_ucb = np.mean(cum_regret_ucb, axis=0)
    std_cum_regret_ucb = np.std(cum_regret_ucb, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_cum_regret_ucb, 'r')
    plt.fill_between(range(len(mean_cum_regret_ucb)), mean_cum_regret_ucb-1.96*std_cum_regret_ucb, mean_cum_regret_ucb+1.96*std_cum_regret_ucb)
    if SW:
        plt.legend([f"UCB_SW({grid[i]})", f".95 CI UCB_SW({grid[i]})"])
    else:
        grid[i] = (round(grid[i][0], 2), round(grid[i][1], 2), round(grid[i][2], 2))
        plt.legend([f"UCB_CUSUM({grid[i]})", f".95 CI UCB_CUSUM({grid[i]})"])
    plt.savefig(f"plot_{i}")
    plt.show()
