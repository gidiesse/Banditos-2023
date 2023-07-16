from algorithms.optimization.greedy_seeds_selection import *
from outdated.linear_mab_environment import *
from outdated.lin_ucb_learner import *
from outdated.ts_learner import TSLearner

"""
This is a script for selecting the best set of seeds for an influence maximisation problem.
The greedy algorithm is run two times. Once with the probability matrix estimated using a linear 
UCB bandit algorithm (the probabilities are the upper confidence bound of each arm once the 
bandit has finished learning, so after 365 days) and once with the probability matrix estimated 
using a TS bandit algorithm (where the probabilities are obtained as the expected value of the 
beta distribution associated with each arm after the bandit has finished learning, so always after 
365 days). 
"""

n_nodes = 30
n_arms = 50
T = 365
mc_it = 500
n_steps_max = 100
n_seeds = 3

env = LinearMabEnvironment(n_arms=n_arms, dim=2, n_nodes=n_nodes)
lin_ucb_learner = LinearUcbLearner(arms_features=env.arms_features)
ts_learner = TSLearner(n_arms=n_arms)

for t in range(0, T):
    pulled_arm_ucb = lin_ucb_learner.pull_arm()
    pulled_arm_ts = ts_learner.pull_arm()
    if pulled_arm_ts == pulled_arm_ucb:
        reward_ts = env.round(pulled_arm_ts)
        reward_ucb = reward_ts
    else:
        reward_ts = env.round(pulled_arm_ts)
        reward_ucb = env.round(pulled_arm_ucb)
    lin_ucb_learner.update(pulled_arm_ucb, reward_ucb)
    ts_learner.update(pulled_arm_ts, reward_ts)

p_ucb = lin_ucb_learner.compute_ucbs()
estimated_prob_matrix_ucb = np.zeros(shape=(n_nodes, n_nodes))

p_ts = ts_learner.beta_parameters[:,0] / (ts_learner.beta_parameters[:,0] + ts_learner.beta_parameters[:,1])
estimated_prob_matrix_ts = np.zeros(shape=(n_nodes, n_nodes))

for a in range(n_arms):
    [i, j] = env.arms[a]
    estimated_prob_matrix_ts[i, j] = p_ts[a]
    estimated_prob_matrix_ucb[i, j] = p_ucb[a]

print(f"True prob pmatrix: \n {env.prob_matrix} \n")
print(f"UCB prob pmatrix: \n {estimated_prob_matrix_ucb} \n")
print(f"TS prob pmatrix: \n {estimated_prob_matrix_ts} \n")

sel_ts = GreedySeedsSelection(estimated_prob_matrix_ts, mc_it, n_steps_max)
sel_ucb = GreedySeedsSelection(estimated_prob_matrix_ucb, mc_it, n_steps_max)

optimal_seeds_ts = sel_ts.select_seeds(k=n_seeds)
optimal_seeds_ucb = sel_ucb.select_seeds(k=n_seeds)

print(f"Selected seeds with probability matrix estimated with linear UCB are: \n {optimal_seeds_ucb}")
print(f"Selected seeds with probability matrix estimated with TS are: \n {optimal_seeds_ts}")





