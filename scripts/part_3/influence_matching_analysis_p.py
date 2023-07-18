from algorithms.environments.environment_matching_gaussian import EnvironmentGaussian
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.ucb_matching_custom import UCBMatchingCustom
from algorithms.bandits.comb_ucb_IM import ucbIM
from algorithms.bandits.ts_matching_custom import TSMatchingCustom
from algorithms.bandits.comb_ts_IM import TSLearner
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from utils.plots import *
import numpy as np
import time
import concurrent.futures
import argparse

"""
PART 3: 
We apply jointly two bandits for (1) the maximization of the number of activated customers and
(2) the maximization of the reward given by the matching of the customers with our products.  
"""

np.random.seed(17)
# Fixed attributes
n_prods = 3
n_units = 3
n_cc = 3
sigma = 18
bandits_name = ["TS", "UCB"]


def run_mc_it(index, arg):
    np.random.seed(index + 333)
    # Unpack arguments
    env_influence = arg["env_influence"]
    env_matching = arg["env_matching"]
    bandit = arg["bandit"]
    n_nodes = arg["n_nodes"]
    n_edges = arg["n_edges"]
    mc_it = arg["mc_it"]
    n_seeds = arg["n_seeds"]
    T = arg["T"]

    print("It:", index)

    if bandit == "TS":
        learner_matching = TSMatchingCustom(n_products=n_prods, n_units=n_units, n_cc=n_cc, sigma=sigma)
        learner_influence_maximization = TSLearner(n_arms=n_edges, edge_indexes=env_influence.arms,
                                                   n_nodes=n_nodes, mc_it=mc_it * 5)
    else:
        learner_matching = UCBMatchingCustom(n_products=n_prods, n_units=n_units, n_cc=n_cc)
        learner_influence_maximization = ucbIM(n_arms=n_edges, edge_indexes=env_influence.arms,
                                               n_nodes=n_nodes, mc_it=mc_it * 5)
    matching_reward_it = []
    for t in range(0, T):
        # STEP 1 (Influence maximization):
        # We use the bandit (either UCB or TS) estimation for the edge activation probabilities.
        # We select the best seeds accordingly, then we simulate an episode of influence
        # maximization, and we collect the reward, i.e. the number of activated customers
        seeds_estimated = learner_influence_maximization.select_best_seeds(n_seeds)
        simulation = env_influence.simulate_episode(seeds_estimated)
        reward_influence = simulation.sum()
        learner_influence_maximization.update(1, reward_influence, new_episode=simulation)

        # STEP 2 (Matching):
        # We match the activated customers with our products, and use a bandit to estimate the
        # rewards of the matching and hence to perform the optimal match at each round.
        influenced_nodes = simulation.sum(axis=0)
        cc = env_influence.customer_class()
        matching_customers = np.array([])
        for act, c in zip(influenced_nodes, cc):
            if act == 1:
                matching_customers = np.append(matching_customers, c)
        pulled_arms = learner_matching.pull_arm(matching_customers)
        rewards = env_matching.round(pulled_arms, matching_customers)
        learner_matching.update(pulled_arms, rewards)
        matching_reward_it.append(rewards.sum())

    return index, learner_influence_maximization.collected_rewards, matching_reward_it


def main(bandit, n_nodes, n_edges, T, mc_it, n_seeds, max_processes):
    # This is the Environment used for the edge activation probabilities between the customers
    # Environment used for influence maximization step
    env_influence = MabEnvironmentNS(n_arms=n_edges, dim=2, n_nodes=n_nodes, T=T)

    # This is the Environment used for getting the matching rewards.
    gaussian_means = np.array([[10, 0, 0],
                               [5, 40, 30],
                               [5, 30, 40]])
    env_matching = EnvironmentGaussian(n_prods, n_units, n_cc, gaussian_means, sigma=sigma)

    # Find the optimum of the influence maximization problem and matching problem
    # by running the clairvoyant algorithm on the true probability matrix and
    # on the true means of the matching problem
    clairvoyant_sel = GreedySeedsSelection(env_influence.prob_matrix, mc_it * 20, n_edges)
    opt_seeds = clairvoyant_sel.select_seeds(n_seeds)
    opt_number_activations = 0
    opt_matching_reward = 0
    for e in range(0, mc_it * 20):
        simulation = env_influence.simulate_episode(opt_seeds)
        opt_number_activations += simulation.sum()
        influenced_nodes = simulation.sum(axis=0)
        cc = env_influence.customer_class()
        matching_customers = np.array([])
        for act, c in zip(influenced_nodes, cc):
            if act == 1:
                matching_customers = np.append(matching_customers, c)
        opt_matching_reward += env_matching.optimal_matching(matching_customers)
    opt_number_activations = opt_number_activations / (mc_it * 20)
    opt_matching_reward = opt_matching_reward / (mc_it * 20)
    print("Opt activations:", opt_number_activations)
    print("Opt reward:", opt_matching_reward)

    learner_influence_rewards_per_experiment = np.zeros(shape=(mc_it, T))
    learner_matching_rewards_per_experiment = np.zeros(shape=(mc_it, T))

    start_t = time.perf_counter()
    # We conduct mc_it Montecarlo iterations
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        arg = {
            "env_influence": env_influence,
            "env_matching": env_matching,
            "bandit": bandit,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "mc_it": mc_it,
            "n_seeds": n_seeds,
            "T": T
        }
        for result in executor.map(run_mc_it, range(mc_it), [arg] * mc_it):
            learner_influence_rewards_per_experiment[result[0], :] = result[1]
            learner_matching_rewards_per_experiment[result[0], :] = result[2]
    finish_t = time.perf_counter()
    execution_time = finish_t - start_t
    print("MC execution time:", execution_time, "s")

    plot_cumulative_regret(rewards=learner_matching_rewards_per_experiment, opt=opt_matching_reward,
                           n_experiments=mc_it, algorithm_name=bandit, opt_title="- Matching bandit")
    plot_cumulative_reward(rewards=learner_matching_rewards_per_experiment, n_experiments=mc_it,
                           algorithm_name=bandit, opt_title="- Matching bandit")
    plot_instantaneous_regret(rewards=learner_matching_rewards_per_experiment, opt=opt_matching_reward,
                              n_experiments=mc_it, algorithm_name=bandit, opt_title="- Matching bandit")
    plot_instantaneous_reward(rewards=learner_matching_rewards_per_experiment, opt=opt_matching_reward,
                              n_experiments=mc_it, algorithm_name=bandit, opt_title="- Matching bandit")
    plot_cumulative_regret(rewards=learner_influence_rewards_per_experiment, opt=opt_number_activations,
                           n_experiments=mc_it, algorithm_name=bandit, opt_title="- Influence bandit")
    plot_cumulative_reward(rewards=learner_influence_rewards_per_experiment, n_experiments=mc_it,
                           algorithm_name=bandit, opt_title="- Influence bandit")
    plot_instantaneous_regret(rewards=learner_influence_rewards_per_experiment, opt=opt_number_activations,
                              n_experiments=mc_it, algorithm_name=bandit, opt_title="- Influence bandit")
    plot_instantaneous_reward(rewards=learner_influence_rewards_per_experiment, opt=opt_number_activations,
                              n_experiments=mc_it, algorithm_name=bandit, opt_title="- Influence bandit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bandit', type=str, default=bandits_name[0], choices=bandits_name, help='Bandit algorithm. '
                                                                                                    'Default is TS')
    ap.add_argument('-n', '--nodes', type=int, default=30, help='Number of nodes. Default is 30')
    ap.add_argument('-a', '--arms', type=int, default=50, help='Number of arms. Default is 50')
    ap.add_argument('-p', '--period', type=int, default=365, help='Period of days. Default is 365')
    ap.add_argument('-i', '--iterations', type=int, default=10, help='Number of MC iterations. Default is 10')
    ap.add_argument('-s', '--seeds', type=int, default=1, help='Number of seeds. Default is 1')
    ap.add_argument('-m', '--max_processes', type=int, default=8, help='Maximum number of concurrent processes. '
                                                                       'Default is 8')
    args = ap.parse_args()

    print('Selected command line options:')
    print('\t--bandit :', args.bandit)
    print('\t--nodes :', args.nodes)
    print('\t--arms :', args.arms)
    print('\t--period :', args.period)
    print('\t--iterations :', args.iterations)
    print('\t--seeds :', args.seeds)
    print('\t--max_processes :', args.max_processes)

    main(bandit=args.bandit, n_nodes=args.nodes, n_edges=args.arms, T=args.period, mc_it=args.iterations,
         n_seeds=args.seeds, max_processes=args.max_processes)
