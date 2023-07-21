from algorithms.environments.environment_matching_gaussian import EnvironmentGaussian
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from algorithms.bandits.ts_matching_custom import TSMatchingCustom
from algorithms.bandits.ucb_matching_custom import UCBMatchingCustom
from algorithms.optimization.context_generation import contextGeneration
from algorithms.bandits.comb_ucb_IM import ucbIM
from algorithms.bandits.comb_ts_IM import TSLearner
import numpy as np
import matplotlib.pyplot as plt

"""
PART 4: we learn jointly influence probabilities and rewards for the optimal matching when the true context
for dividing the customers is unknown. We observe two features and we decide in 52 days what is the best possible
context for classifying the customers based on a greedy approach.
"""


np.random.seed(15)
n_nodes = 30
n_arms = 50
mc_it = 20
T = 365
n_steps_max = n_nodes
n_seeds = 1
n_prods = 3
n_units = 3
n_cc = 3
sigma = 20
test_time = 14
break_points = range(0, test_time*7, test_time)

# Flag True if the regret analysis has to be carried on TS algorithm, otherwise UCB
TS = False
if TS:
    legend = "Combinatorial TS"
else:
    legend = "Combinatorial UCB"

env_influence = MabEnvironmentNS(n_arms=n_arms, dim=2, n_nodes=n_nodes, T=T)
selector = GreedySeedsSelection(env_influence.prob_matrix, mc_it, n_steps_max)
optimal_seeds = selector.select_seeds(k=n_seeds)

gaussian_means = np.array([[10, 0, 0],
                           [5, 40, 30],
                           [5, 30, 40]])
env_matching = EnvironmentGaussian(n_prods, n_units, n_cc, gaussian_means, sigma)

learner_influence_rewards_per_experiment = np.zeros(shape=(mc_it, T))
learner_matching_rewards_per_experiment = np.zeros(shape=(mc_it, T))

# Find the optimum of the influence maximization problem and matching problem
# by running the clairvoyant algorithm on the true probability matrix and
# on the true means of the matching problem

clairvoyant_sel = GreedySeedsSelection(env_influence.prob_matrix, mc_it*20, n_arms)
opt_seeds = clairvoyant_sel.select_seeds(n_seeds)
opt_number_activations = 0
opt_matching_reward = 0
for e in range(0, mc_it*20):
    simulation = env_influence.simulate_episode(opt_seeds)
    opt_number_activations += simulation.sum()
    influenced_nodes = simulation.sum(axis=0)
    cc = env_influence.customer_class()
    matching_customers = np.array([])
    for act, c in zip(influenced_nodes, cc):
        if act == 1:
            matching_customers = np.append(matching_customers, c)
    opt_matching_reward += env_matching.optimal_matching(matching_customers)
opt_number_activations = opt_number_activations / (mc_it*20)
opt_matching_reward = opt_matching_reward / (mc_it*20)


split_chosen = np.zeros(8)
for it in range(mc_it):
    print(it)
    matching_rew_it = []
    context_generation = contextGeneration(nodes_features=env_influence.nodes_features,
                                           test_time=test_time)
    if TS:
        learner_influence_maximization = TSLearner(n_arms=n_arms, edge_indexes=env_influence.arms,
                                                   n_nodes=n_nodes, mc_it=mc_it * 5)
    else:
        learner_influence_maximization = ucbIM(n_arms=n_arms, edge_indexes=env_influence.arms,
                                               n_nodes=n_nodes, mc_it=mc_it * 5)
    for t in range(T):
        # STEP 1 (Influence maximization):
        # We use the bandit (either UCB or TS) estimation for the edge activation probabilities.
        # We select the best seeds accordingly, then we simulate an episode of influence
        # maximization and we collect the reward, i.e. the number of activated customers
        seeds_estimated = learner_influence_maximization.select_best_seeds(n_seeds)
        simulation = env_influence.simulate_episode(seeds_estimated)
        reward_influence = simulation.sum()
        learner_influence_maximization.update(1, reward_influence, new_episode=simulation)
        influenced_nodes = simulation.sum(axis=0)

        # Estimation of the context when using a greedy algorithm
        cc_estimated, n_cc_estimated = context_generation.choose_split()
        matching_customers_estimated = np.array([])
        for act, c in zip(influenced_nodes, cc_estimated):
            if act == 1:
                matching_customers_estimated = np.append(matching_customers_estimated, c)

        # Computation of the true context (clairvoyant)
        cc = env_influence.customer_class()
        matching_customers = np.array([])
        for act, c in zip(influenced_nodes, cc):
            if act == 1:
                matching_customers = np.append(matching_customers, c)

        # We instantiate the bandit if we are at a breakpoint for the phases
        if t in break_points:
            if TS:
                learner_matching = TSMatchingCustom(n_prods, n_units, n_cc_estimated, sigma)
            else:
                learner_matching = UCBMatchingCustom(n_prods, n_units, n_cc_estimated)

        pulled_arms = learner_matching.pull_arm(matching_customers_estimated)
        rewards = env_matching.round(pulled_arms, matching_customers)
        learner_matching.update(pulled_arms, rewards)
        context_generation.collected_rewards = np.append(context_generation.collected_rewards, rewards.sum())
        matching_rew_it.append(rewards.sum())

    learner_matching_rewards_per_experiment[it, :] = matching_rew_it
    learner_influence_rewards_per_experiment[it, :] = learner_influence_maximization.collected_rewards
    split_chosen[context_generation.best_split()] += 1

print(f"The number of times we chose split i is: \n {split_chosen}")

# Cumulative regret of the matching bandit
plt.figure(0)
plt.ylabel("Cumulative regret matching")
plt.xlabel("t")
cum_regret = np.cumsum(opt_matching_reward - np.array(learner_matching_rewards_per_experiment), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Cumulative reward collected by the matching bandit
plt.figure(1)
plt.ylabel("Cumulative reward matching")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(learner_matching_rewards_per_experiment), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous regret of the matching bandit
plt.figure(2)
plt.ylabel("Instantaneous regret matching")
plt.xlabel("t")
inst_regret = (opt_matching_reward - np.array(learner_matching_rewards_per_experiment))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(inst_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.axhline(y=0, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous reward of the matching bandit
plt.figure(3)
plt.ylabel("Instantaneous reward matching")
plt.xlabel("t")
inst_reward = np.array(learner_matching_rewards_per_experiment)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.axhline(y=opt_matching_reward, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Cumulative regret of the influence maximization bandit
plt.figure(4)
plt.ylabel("Cumulative regret influence maximization")
plt.xlabel("t")
cum_regret = np.cumsum(opt_number_activations - np.array(learner_influence_rewards_per_experiment), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Cumulative reward collected by influence maximization bandit
plt.figure(5)
plt.ylabel("Cumulative reward influence maximization")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(learner_influence_rewards_per_experiment), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous regret of the influence maximization bandit
plt.figure(6)
plt.ylabel("Instantaneous regret influence maximization")
plt.xlabel("t")
inst_regret = (opt_number_activations - np.array(learner_influence_rewards_per_experiment))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(inst_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.axhline(y=0, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous reward of the influence maximization bandit
plt.figure(7)
plt.ylabel("Instantaneous reward influence maximization")
plt.xlabel("t")
inst_reward = np.array(learner_influence_rewards_per_experiment)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.axhline(y=opt_number_activations, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()


