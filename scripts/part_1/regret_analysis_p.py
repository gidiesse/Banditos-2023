from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from utils.plots import *
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.comb_ts_IM import TSLearner
from algorithms.bandits.comb_ucb_IM import ucbIM
import numpy as np
import time
import concurrent.futures
import argparse

"""
Regret analysis for the selected algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph. 
The outputs are 4 plots: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

np.random.seed(17)
bandits_name = ["TS", "UCB"]


def run_experiment(n, env, bandit, n_nodes, n_arms, T, n_experiments, n_seeds):
    np.random.seed(n + 666)
    # print("start exp", n)
    if bandit == "TS":
        learner = TSLearner(n_arms, env.arms, n_nodes, mc_it=n_experiments * 5)
    else:
        learner = ucbIM(n_arms, env.arms, n_nodes, mc_it=n_experiments * 5)
    for t in range(0, T):
        seeds_estimated = learner.select_best_seeds(n_seeds)
        simulation = env.simulate_episode(seeds_estimated)
        reward = simulation.sum()
        learner.update(1, reward, new_episode=simulation)
    return learner.collected_rewards


def main(bandit, n_nodes, n_arms, T, n_experiments, n_seeds, max_processes):
    env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T)
    clairvoyant_sel = GreedySeedsSelection(env.prob_matrix, n_experiments * 20, n_arms)
    opt_seeds = clairvoyant_sel.select_seeds(n_seeds)
    opt = 0
    for e in range(0, n_experiments * 20):
        opt += env.simulate_episode(opt_seeds).sum()
    opt = opt / (n_experiments * 20)
    print("Optimal: ", opt)

    rewards = []
    start_t = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        for result in executor.map(run_experiment, range(0, n_experiments), [env] * n_experiments,
                                   [bandit] * n_experiments, [n_nodes] * n_experiments, [n_arms] * n_experiments,
                                   [T] * n_experiments, [n_experiments] * n_experiments, [n_seeds] * n_experiments):
            rewards.append(result)
    finish_t = time.perf_counter()
    execution_time = finish_t - start_t
    print("Execution time:", execution_time, "s")

    plot_cumulative_regret(rewards, opt, n_experiments, bandit)
    plot_cumulative_reward(rewards, n_experiments, bandit)
    plot_instantaneous_regret(rewards, opt, n_experiments, bandit)
    plot_instantaneous_reward(rewards, opt, n_experiments, bandit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bandit', type=str, default=bandits_name[0], choices=bandits_name, help='Bandit algorithm. '
                                                                                                    'Default is TS')
    ap.add_argument('-n', '--nodes', type=int, default=30, help='Number of nodes. Default is 30')
    ap.add_argument('-a', '--arms', type=int, default=50, help='Number of arms. Default is 50')
    ap.add_argument('-p', '--period', type=int, default=365, help='Period of days. Default is 365')
    ap.add_argument('-e', '--experiments', type=int, default=16, help='Number of experiments. Default is 16')
    ap.add_argument('-s', '--seeds', type=int, default=1, help='Number of seeds. Default is 1')
    ap.add_argument('-m', '--max_processes', type=int, default=8, help='Maximum number of concurrent processes. '
                                                                       'Default is 8')
    args = ap.parse_args()

    print('Selected command line options:')
    print('\t--bandit :', args.bandit)
    print('\t--nodes :', args.nodes)
    print('\t--arms :', args.arms)
    print('\t--period :', args.period)
    print('\t--experiments :', args.experiments)
    print('\t--seeds :', args.seeds)
    print('\t--max_processes :', args.max_processes)

    main(args.bandit, args.nodes, args.arms, args.period, args.experiments, args.seeds, args.max_processes)
