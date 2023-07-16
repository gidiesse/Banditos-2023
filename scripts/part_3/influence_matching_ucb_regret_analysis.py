from outdated.ucb_matching import *
from algorithms.environments.environment_matching_gaussian import EnvironmentGaussian
from outdated.linear_mab_environment import LinearMabEnvironment
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
from algorithms.bandits.ucb_matching_custom import UCBMatchingCustom
from outdated.lin_ucb_learner import LinearUcbLearner
import matplotlib.pyplot as plt

"""
PART 3: 
We apply jointly two bandits for the estimation of (1) the influence probabilities and (2) the mean of the rewards
we obtain for the matching. 
"""

n_nodes = 30
n_edges = 50
mc_it = 100
n_exp = 10
n_steps_max = 100
n_seeds = 3
n_prods = 3
n_units = 3
n_cc = 3
T = 365

# This is the Environment used for the edge activation probabilities between the customers
# Environment used for influence maximization step
env_influence = LinearMabEnvironment(n_arms=n_edges, dim=2, n_nodes=n_nodes)

# This is the Environment used for getting the matching rewards.
gaussian_means = np.array([[10, 0, 0], [5, 40, 30], [5, 30, 40]])
env_matching = EnvironmentGaussian(n_prods, n_units, n_cc, gaussian_means)

lin_ucb_rewards_per_experiment = np.zeros(shape=(mc_it, T))
ucb_rew = np.zeros(shape=(mc_it, n_exp))
opt_rew = np.zeros(shape=(mc_it, n_exp))

# We conduct mc_it Montecarlo iterations
for it in range(0, mc_it):
    # STEP 1 (Influence maximization): we let the bandit learn for 365 days the probabilities
    # of activation of the influence graph
    lin_ucb_learner = LinearUcbLearner(arms_features=env_influence.arms_features)
    for t in range(0, T):
        pulled_arm = lin_ucb_learner.pull_arm()
        reward = env_influence.round(pulled_arm)
        lin_ucb_learner.update(pulled_arm, reward)
    lin_ucb_rewards_per_experiment[it, :] = lin_ucb_learner.collected_rewards
    estimated_prob_matrix_ucb = np.zeros(shape=(n_nodes, n_nodes))
    p_ucb = lin_ucb_learner.compute_ucbs()
    for a in range(n_edges):
        [i, j] = env_influence.arms[a]
        estimated_prob_matrix_ucb[i, j] = p_ucb[a]

    # STEP 2 (Selection of the best set of seeds): We use the probability matrix that was estimated
    # by the bandit to select the best set of seed with the Greedy Algorithm
    selector = GreedySeedsSelection(estimated_prob_matrix_ucb, mc_it, n_steps_max)
    optimal_seeds = selector.select_seeds(k=n_seeds)

    # STEP 3 (Matching): By using the set of seeds we previously selected, we simulate n_exp
    # rounds of influence maximization (using the true probability matrix).
    # Then we match the activated customers with our products, and use a bandit to estimate the
    # rewards of the matching and hence to perform the optimal match at each round.
    matching_ucb_learner = UCBMatchingCustom(n_prods, n_units, n_cc)
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
        pulled_arms = matching_ucb_learner.pull_arm(matching_customers)
        rewards = env_matching.round(pulled_arms)
        matching_ucb_learner.update(pulled_arms, rewards)
        ucb_rew_it.append(rewards.sum())
    opt_rew[it, :] = opt_rew_it
    ucb_rew[it, :] = ucb_rew_it

# Cumulative regret of the matching bandit
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

# Cumulative reward collected by the matching bandit
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

# Instantaneous regret of the matching bandit
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

# Instantaneous reward of the matching bandit
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

# Cumulative regret of the linear bandit (influence maximization)
opt = env_influence.opt()
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret = np.cumsum(opt - np.array(lin_ucb_rewards_per_experiment), axis=1)
mean_cum_regret = np.mean(cum_regret, axis=0)
std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_regret, 'r')
plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret-1.96*std_cum_regret, mean_cum_regret+1.96*std_cum_regret)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Cumulative reward collected by the linear bandit (influence maximization)
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward = np.cumsum(np.array(lin_ucb_rewards_per_experiment), axis=1)
mean_cum_reward = np.mean(cum_reward, axis=0)
std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_cum_reward, 'r')
plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward-1.96*std_cum_reward, mean_cum_reward+1.96*std_cum_reward)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Instantaneous regret of the linear bandit (influence maximization)
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret = (opt - np.array(lin_ucb_rewards_per_experiment))
mean_inst_regret = np.mean(inst_regret, axis=0)
std_inst_regret = np.std(cum_regret, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_regret, 'r')
plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret-1.96*std_inst_regret, mean_inst_regret+1.96*std_inst_regret)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()

# Instantaneous reward of the linear bandit (influence maximization)
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward = np.array(lin_ucb_rewards_per_experiment)
mean_inst_reward = np.mean(inst_reward, axis=0)
std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(mc_it)
plt.plot(mean_inst_reward, 'r')
plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward-1.96*std_inst_reward, mean_inst_reward+1.96*std_inst_reward)
plt.legend(["Linear UCB", ".95 CI"])
plt.show()











