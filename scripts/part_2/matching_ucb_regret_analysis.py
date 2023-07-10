from algorithms.bandits.ucb_matching import *
from algorithms.environments.environment_matching_gaussian import EnvironmentGaussian
from algorithms.environments.linear_mab_environment import LinearMabEnvironment
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from algorithms.bandits.ucb_matching_custom import UCBMatchingCustom
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

n_nodes = 30
n_arms = 50
mc_it = 300
n_exp = 10
n_steps_max = 100
n_seeds = 3
n_prods = 3
n_units = 3
n_cc = 3

env_influence = LinearMabEnvironment(n_arms=n_arms, dim=2, n_nodes=n_nodes)
selector = GreedySeedsSelection(env_influence.prob_matrix, mc_it, n_steps_max)
optimal_seeds = selector.select_seeds(k=n_seeds)

gaussian_means = np.array([[10, 0, 0], [5, 40, 30], [5, 30, 40]])
env_matching = EnvironmentGaussian(n_prods, n_units, n_cc, gaussian_means)

ucb_rew = np.zeros(shape=(mc_it, n_exp))
opt_rew = np.zeros(shape=(mc_it, n_exp))
for it in range(mc_it):
    ucb_learner = UCBMatchingCustom(n_prods, n_units, n_cc)
    ucb_rew_it = []
    opt_rew_it = []
    for e in range(n_exp):
        influenced_nodes = selector.simulate_episode(env_influence.prob_matrix, optimal_seeds)
        cc = env_influence.customer_class()
        matching_customers = np.array([])
        for act, c in zip(influenced_nodes, cc):
            if act == 1:
                matching_customers = np.append(matching_customers, c)
        opt_rew_it.append(env_matching.optimal_matching(matching_customers))

        pulled_arms = ucb_learner.pull_arm(matching_customers)
        rewards = env_matching.round(pulled_arms)
        ucb_learner.update(pulled_arms, rewards)
        ucb_rew_it.append(rewards.sum())

    opt_rew[it, :] = opt_rew_it
    ucb_rew[it, :] = ucb_rew_it

# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret = np.cumsum(np.array(opt_rew) - np.array(ucb_rew), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend(["Combinatorial UCB", ".95 CI"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(ucb_rew), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend(["Combinatorial UCB", ".95 CI"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret = (np.array(opt_rew) - np.array(ucb_rew))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.legend(["Combinatorial UCB", ".95 CI"])
plt.show()

# Instantaneous reward of the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward = np.array(ucb_rew)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.legend(["Combinatorial UCB", ".95 CI"])
plt.show()





