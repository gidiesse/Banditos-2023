from algorithms.environments.environment_matching_gaussian import EnvironmentGaussian
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from algorithms.bandits.ts_matching_custom import TSMatchingCustom
from algorithms.bandits.ucb_matching_custom import UCBMatchingCustom
from algorithms.optimization.context_generation import contextGeneration
import numpy as np
import matplotlib.pyplot as plt

# We are working for Tiffany & co, our aim is to carry out an influence maximization campaign
# on Instagram to raise awareness about our products and entice new customers.
# We offer a wide range of products with prince ranging from 200 Euros to 50,000 Euros.
# In a simplified scenario, we imagine having 3 types of products (of 3 units each) which are
# representative of our price range and customer taste.
# We have divided our customer base into 3 categories:
# 1) C1 = Lower income (young and old)
# 2) C2 = Higher income young
# 3) C3 = Higher income old
# Our products are:
# 1) D1 = Silver jewellery
# 2) D2 = New gold designs
# 3) D3 = Classic designs (pearls and diamonds)
# We assume that the optimal matching are: (C1, D1), (C2, D2) and (C3, D3)
# Our assumptions for the gaussian means for the matching reward are as follows:
# 1) If we match a lower income customer with either a product from D2 or D3, our mean reward is 0
# 2) If we match a lower income customer with a product from D1, our mean reward is 10.
# 3) If we match C2 with D2 or C3 with D3, our mean reward is 40.
# 4) If we match C2 with D3 or C3 with D2, our mean reward is 30.
# 5) If we match C2 or C3 with D1, our mean reward is 5.

# TODO: implement the script as in part 3 (not part 2)

np.random.seed(15)
n_nodes = 30
n_arms = 50
mc_it = 50
T = 365
n_steps_max = n_nodes
n_seeds = 3
n_prods = 3
n_units = 3
n_cc = 3
sigma = 20
test_time = 14
break_points = range(0, test_time*7, test_time)

# Flag True if the regret analysis has to be carried on TS algorithm, otherwise UCB
TS = True
if TS:
    legend = "Combinatorial TS"
else:
    legend = "Combinatorial UCB"

env_influence = MabEnvironmentNS(n_arms=n_arms, dim=2, n_nodes=n_nodes, T=T)
selector = GreedySeedsSelection(env_influence.prob_matrix, mc_it, n_steps_max)
optimal_seeds = selector.select_seeds(k=n_seeds)

gaussian_means = np.array([[10, 0, 0], [5, 40, 30], [5, 30, 40]])
env_matching = EnvironmentGaussian(n_prods, n_units, n_cc, gaussian_means, sigma)

learner_rew = np.zeros(shape=(mc_it, T))
opt_rew = 0
split_chosen = np.zeros(8)
for it in range(mc_it):
    learner_rew_it = []
    opt_rew_it = 0
    context_generation = contextGeneration(nodes_features=env_influence.nodes_features,
                                           test_time=test_time)
    for e in range(T):
        influenced_nodes = selector.simulate_episode(env_influence.prob_matrix, optimal_seeds)

        # Estimation of the context when by using a greedy algorithm
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
        if e in break_points:
            if TS:
                learner = TSMatchingCustom(n_prods, n_units, n_cc_estimated, sigma)
            else:
                learner = UCBMatchingCustom(n_prods, n_units, n_cc_estimated)

        opt_rew_it += env_matching.optimal_matching(matching_customers)

        pulled_arms = learner.pull_arm(matching_customers_estimated)
        rewards = env_matching.round(pulled_arms, matching_customers)
        learner.update(pulled_arms, rewards)
        context_generation.collected_rewards = np.append(context_generation.collected_rewards, rewards.sum())
        learner_rew_it.append(rewards.sum())

    opt_rew += opt_rew_it/T
    learner_rew[it, :] = learner_rew_it
    split_chosen[context_generation.best_split()] += 1

print(f"The number of times we chose split i is: \n {split_chosen}")
opt_rew /= mc_it

# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret = np.cumsum(opt_rew - np.array(learner_rew), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(learner_rew), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret = (opt_rew - np.array(learner_rew))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(inst_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.axhline(y=0, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()

# Instantaneous reward of the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward = np.array(learner_rew)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.axhline(y=opt_rew, color='black', linestyle='-')
plt.legend([f"{legend}", ".95 CI"])
plt.show()





