from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.comb_ucb_sw_IM import ucbSW
from algorithms.bandits.ucb_cusum_IM import ucbCUSUM
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from utils.plots import plot_cumulative_regret
import numpy as np
from itertools import product
import argparse
import concurrent.futures

"""
STEP 5 - Sensibility analysis 
Regret analysis for the UCB algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph in a non stationary environment.
Two approaches: sliding window UCB and cusum UCB. 
We conduct a grid search on:
[1] the sliding window length, for what concerns the sliding window approach
[2] M, epsilon, h parameters for what concerns the CUSUM approach
We explore 5 different windows length for the sliding window approach.
We explore 27 different combinations of values for each of the 3 parameters (M,epsilon,h) of the CUSUM approach.

The outputs are 3 plots of the cumulative regret in the case of analysis of the sliding window, 
27 plots of the cumulative regret in the case of analysis of the CUSUM. 
"""

# Fixed parameters of the problem
n_nodes = 30
n_arms = 50
T = 365
n_phases = 3
n_experiments = 20
n_seeds = 1
techniques_name = ["SW", "CUSUM"]
rand_seed = 0
max_processes = 8

# Sliding window parameter grid
window_size = [int(T ** 0.5), 2 * int(T ** 0.5), 4 * int(T ** 0.5), 6 * int(T ** 0.5), 8 * int(T ** 0.5)]

# CUSUM parameters grid -> (M, eps, h) in each tuple
cusum_grid = []
for comb in product([60, 120, 180], [0.1, 0.5, 1], [np.log(T), 3 * np.log(T), 6 * np.log(T)]):
    cusum_grid.append(comb)


def run_sw_simulations(size):
    print("WS:", size)
    np.random.seed(rand_seed)
    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)
    ucb_rewards_per_experiment = []
    for e in range(0, n_experiments):
        ucb = ucbSW(n_arms, env.arms, n_nodes, n_experiments * 5, size)
        dataset_episodes = []
        for t in range(0, T):
            seeds_estimated_ucb = ucb.select_best_seeds(n_seeds)
            phase = env.select_phase(t)
            simulation_ucb = env.simulate_episode(seeds_estimated_ucb, phase)
            reward_ucb = simulation_ucb.sum()
            dataset_episodes.append(simulation_ucb)
            ucb.update(1, reward_ucb, dataset_episodes)
        ucb_rewards_per_experiment.append(ucb.collected_rewards)
    return size, ucb_rewards_per_experiment


def run_cusum_simulations(params):
    print("CUSUM:", str(params))
    np.random.seed(rand_seed)
    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)
    ucb_rewards_per_experiment = []
    for e in range(0, n_experiments):
        ucb = ucbCUSUM(n_arms, env.arms, n_nodes, n_experiments * 5, *params)
        for t in range(0, T):
            seeds_estimated_ucb = ucb.select_best_seeds(n_seeds)
            phase = env.select_phase(t)
            simulation_ucb = env.simulate_episode(seeds_estimated_ucb, phase)
            reward_ucb = simulation_ucb.sum()
            ucb.update(1, reward_ucb, simulation_ucb)
        ucb_rewards_per_experiment.append(ucb.collected_rewards)
    return params, ucb_rewards_per_experiment


def get_optimal_reward():
    np.random.seed(rand_seed)
    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)
    # Now we have to select the optimal seeds for each of the phases
    opt_seeds = []
    for phase in range(n_phases):
        clairvoyant_sel = GreedySeedsSelection(env.prob_matrix[phase], n_experiments * 50, n_max_steps=n_nodes)
        opt_seeds.append(clairvoyant_sel.select_seeds(n_seeds))
    # Now we have different optimal values for each of the phases
    # opt = [0 for i in range(n_phases)]
    opt = np.zeros(n_phases)
    for phase in range(n_phases):
        for e in range(0, n_experiments * 20):
            opt[phase] += env.simulate_episode(opt_seeds[phase], phase).sum()
        opt[phase] = opt[phase] / (n_experiments * 20)
    opt_reward_per_day = []
    for t in range(T):
        opt_reward_per_day.append(opt[env.select_phase(t)])
    return opt_reward_per_day


def main(technique):
    opt_reward = get_optimal_reward()
    if technique == techniques_name[0]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            for result in executor.map(run_sw_simulations, window_size):
                plot_cumulative_regret(result[1], opt_reward, n_experiments,
                                       technique + str(result[0]) + "_exp" + str(n_experiments) + "_seed" + str(n_seeds),
                                       savefig=True, rel_path="grid_search")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            for result in executor.map(run_cusum_simulations, cusum_grid):
                rounded = tuple(round(element, 2) for element in result[0])
                plot_cumulative_regret(result[1], opt_reward, n_experiments,
                                       technique + str(rounded) + "_exp" + str(n_experiments) + "_seed" + str(n_seeds),
                                       savefig=True, rel_path="grid_search")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--technique', type=str, default=techniques_name[0], choices=techniques_name,
                    help='Name of the technique. Default is SW')
    args = ap.parse_args()

    print('Selected command line options:')
    print('\t--technique :', args.technique)

    main(args.technique)
