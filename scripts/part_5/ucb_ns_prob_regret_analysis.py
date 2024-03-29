from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.comb_ucb_IM import ucbIM
from algorithms.bandits.comb_ucb_sw_IM import ucbSW
from algorithms.bandits.ucb_cusum_IM import ucbCUSUM
from algorithms.optimization.greedy_seeds_selection import GreedySeedsSelection
import numpy as np
import matplotlib.pyplot as plt

"""
STEP 5
Regret analysis for the UCB algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph in a non stationary environment.
Three approaches: stationary UCB, sliding window UCB, cusum UCB 
Why SW approach is not working? We have 50 arms and only 365 days. If we have a sliding window 
that contains less than 50 samples, we will constantly be in the exploration phase (there will be
always 50-SW non-played arms, and these are the arms that we'll play next rounds). 
If we have more than 50 samples in the SW, then we can play the most rewarding arm; still we have to 
deal with 50 arms, so if the SW length is too big we won't be able to detect changes, if it's too short 
we'll suffer from the same problem of exploring too much. 
The outputs are 4 plots, comparing the performances of the 3 approaches: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

np.random.seed(7)
n_nodes = 30
n_arms = 50
T = 365
n_phases = 3
n_experiments = 10
n_seeds = 1

# Sliding window parameter
window_size = 6 * int(T**0.5)


# CUSUM parameters
M = 120  # Number of rounds to wait for making work the CUSUM
# h has to be modified in the grid search
h = 3 * np.log(T)


ucb_rewards_per_experiment = []
ucb_sw_rewards_per_experiment = []
ucb_cusum_rewards_per_experiment = []
opt_reward_per_day = []

env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases)

# Now we have to select the optimal seeds for each of the phases
opt_seeds = []
for phase in range(n_phases):
    clairvoyant_sel = GreedySeedsSelection(env.prob_matrix[phase], n_experiments*50, n_max_steps=n_nodes)
    opt_seeds.append(clairvoyant_sel.select_seeds(n_seeds))

# Now we have different optimal values for each of the phases
opt = [0 for i in range(n_phases)]
for phase in range(n_phases):
    for e in range(0, n_experiments*20):
        opt[phase] += env.simulate_episode(opt_seeds[phase], phase).sum()
    opt[phase] = opt[phase] / (n_experiments*20)
print(opt)

opt_reward_per_day = []
for t in range(T):
    opt_reward_per_day.append(opt[env.select_phase(t)])

for e in range(0, n_experiments):
    print(e)
    ucb = ucbIM(n_arms, env.arms, n_nodes, mc_it=n_experiments*5)
    ucb_sw = ucbSW(n_arms, env.arms, n_nodes, mc_it=n_experiments*5, window_size=window_size)
    ucb_cusum = ucbCUSUM(n_arms, env.arms, n_nodes, mc_it=n_experiments * 5, M=M, h=h)
    # We need the whole dataset of the episodes for the sliding window approach
    dataset_episodes = []
    for t in range(0, T):
        seeds_estimated_ucb = ucb.select_best_seeds(n_seeds)
        seeds_estimated_ucb_sw = ucb_sw.select_best_seeds(n_seeds)
        seeds_estimated_ucb_cusum = ucb_cusum.select_best_seeds(n_seeds)
        phase = env.select_phase(t)
        simulation_ucb = env.simulate_episode(seeds_estimated_ucb, phase)
        simulation_ucb_sw = env.simulate_episode(seeds_estimated_ucb_sw, phase)
        simulation_ucb_cusum = env.simulate_episode(seeds_estimated_ucb_cusum, phase)
        dataset_episodes.append(simulation_ucb_sw)
        reward_ucb = simulation_ucb.sum()
        reward_ucb_sw = simulation_ucb_sw.sum()
        reward_ucb_cusum = simulation_ucb_cusum.sum()
        ucb.update(1, reward_ucb, simulation_ucb)
        ucb_sw.update(1, reward_ucb_sw, dataset_episodes)
        ucb_cusum.update(1, reward_ucb_cusum, simulation_ucb_cusum)
    ucb_rewards_per_experiment.append(ucb.collected_rewards)
    ucb_sw_rewards_per_experiment.append(ucb_sw.collected_rewards)
    ucb_cusum_rewards_per_experiment.append(ucb_cusum.collected_rewards)
    if not e:
        print(ucb_cusum.detections)



# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret_ucb = np.cumsum(opt_reward_per_day - np.array(ucb_rewards_per_experiment), axis=1)
cum_regret_ucb_sw = np.cumsum(opt_reward_per_day - np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_regret_ucb_cusum = np.cumsum(opt_reward_per_day - np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_regret_ucb = np.mean(cum_regret_ucb, axis=0)
mean_cum_regret_ucb_sw = np.mean(cum_regret_ucb_sw, axis=0)
mean_cum_regret_ucb_cusum = np.mean(cum_regret_ucb_cusum, axis=0)
std_cum_regret_ucb = np.std(cum_regret_ucb, axis=0) / np.sqrt(n_experiments)
std_cum_regret_ucb_sw = np.std(cum_regret_ucb_sw, axis=0) / np.sqrt(n_experiments)
std_cum_regret_ucb_cusum = np.std(cum_regret_ucb_cusum, axis=0) / np.sqrt(n_experiments)
plt.plot(mean_cum_regret_ucb)
plt.fill_between(range(len(mean_cum_regret_ucb)), mean_cum_regret_ucb-1.96*std_cum_regret_ucb, mean_cum_regret_ucb+1.96*std_cum_regret_ucb)
plt.plot(mean_cum_regret_ucb_sw)
plt.fill_between(range(len(mean_cum_regret_ucb_sw)), mean_cum_regret_ucb_sw-1.96*std_cum_regret_ucb_sw, mean_cum_regret_ucb_sw+1.96*std_cum_regret_ucb_sw)
plt.plot(mean_cum_regret_ucb_cusum)
plt.fill_between(range(len(mean_cum_regret_ucb_cusum)), mean_cum_regret_ucb_cusum-1.96*std_cum_regret_ucb_cusum, mean_cum_regret_ucb_cusum+1.96*std_cum_regret_ucb_cusum)
plt.legend(["UCB", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward_ucb = np.cumsum(np.array(ucb_rewards_per_experiment), axis=1)
cum_reward_ucb_sw = np.cumsum(np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_reward_ucb_cusum = np.cumsum(np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_reward_ucb = np.mean(cum_reward_ucb, axis=0)
mean_cum_reward_ucb_sw = np.mean(cum_reward_ucb_sw, axis=0)
mean_cum_reward_ucb_cusum = np.mean(cum_reward_ucb_cusum, axis=0)
std_cum_reward_ucb = np.std(cum_reward_ucb, axis=0) / np.sqrt(n_experiments)
std_cum_reward_ucb_sw = np.std(cum_reward_ucb_sw, axis=0) / np.sqrt(n_experiments)
std_cum_reward_ucb_cusum = np.std(cum_reward_ucb_cusum, axis=0) / np.sqrt(n_experiments)
plt.plot(mean_cum_reward_ucb)
plt.fill_between(range(len(mean_cum_reward_ucb)), mean_cum_reward_ucb-1.96*std_cum_reward_ucb, mean_cum_reward_ucb+1.96*std_cum_reward_ucb)
plt.plot(mean_cum_reward_ucb_sw)
plt.fill_between(range(len(mean_cum_reward_ucb_sw)), mean_cum_reward_ucb_sw-1.96*std_cum_reward_ucb_sw, mean_cum_reward_ucb_sw+1.96*std_cum_reward_ucb_sw)
plt.plot(mean_cum_reward_ucb_cusum)
plt.fill_between(range(len(mean_cum_reward_ucb_cusum)), mean_cum_reward_ucb_cusum-1.96*std_cum_reward_ucb_cusum, mean_cum_reward_ucb_cusum+1.96*std_cum_reward_ucb_cusum)
plt.legend(["UCB", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret_ucb = (opt_reward_per_day - np.array(ucb_rewards_per_experiment))
inst_regret_ucb_sw = (opt_reward_per_day - np.array(ucb_sw_rewards_per_experiment))
inst_regret_ucb_cusum = (opt_reward_per_day - np.array(ucb_cusum_rewards_per_experiment))
mean_inst_regret_ucb = np.mean(inst_regret_ucb, axis=0)
mean_inst_regret_ucb_sw = np.mean(inst_regret_ucb_sw, axis=0)
mean_inst_regret_ucb_cusum = np.mean(inst_regret_ucb_cusum, axis=0)
std_inst_regret_ucb = np.std(inst_regret_ucb, axis=0) / np.sqrt(n_experiments)
std_inst_regret_ucb_sw = np.std(inst_regret_ucb_sw, axis=0) / np.sqrt(n_experiments)
std_inst_regret_ucb_cusum = np.std(inst_regret_ucb_cusum, axis=0) / np.sqrt(n_experiments)
plt.plot(mean_inst_regret_ucb)
plt.fill_between(range(len(mean_inst_regret_ucb)), mean_inst_regret_ucb-1.96*std_inst_regret_ucb, mean_inst_regret_ucb+1.96*std_inst_regret_ucb)
plt.plot(mean_inst_regret_ucb_sw)
plt.fill_between(range(len(mean_inst_regret_ucb_sw)), mean_inst_regret_ucb_sw-1.96*std_inst_regret_ucb_sw, mean_inst_regret_ucb_sw+1.96*std_inst_regret_ucb_sw)
plt.plot(mean_inst_regret_ucb_cusum)
plt.fill_between(range(len(mean_inst_regret_ucb_cusum)), mean_inst_regret_ucb_cusum-1.96*std_inst_regret_ucb_cusum, mean_inst_regret_ucb_cusum+1.96*std_inst_regret_ucb_cusum)
plt.axhline(y=0, color='black', linestyle='-')
plt.legend(["UCB", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous reward collected by the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward_ucb = np.array(ucb_rewards_per_experiment)
inst_reward_ucb_sw = np.array(ucb_sw_rewards_per_experiment)
inst_reward_ucb_cusum = np.array(ucb_cusum_rewards_per_experiment)
mean_inst_reward_ucb = np.mean(inst_reward_ucb, axis=0)
mean_inst_reward_ucb_sw = np.mean(inst_reward_ucb_sw, axis=0)
mean_inst_reward_ucb_cusum = np.mean(inst_reward_ucb_cusum, axis=0)
std_inst_reward_ucb = np.std(inst_reward_ucb, axis=0) / np.sqrt(n_experiments)
std_inst_reward_ucb_sw = np.std(inst_reward_ucb_sw, axis=0) / np.sqrt(n_experiments)
std_inst_reward_ucb_cusum = np.std(inst_reward_ucb_cusum, axis=0) / np.sqrt(n_experiments)
plt.plot(mean_inst_reward_ucb)
plt.fill_between(range(len(mean_inst_reward_ucb)), mean_inst_reward_ucb-1.96*std_inst_reward_ucb, mean_inst_reward_ucb+1.96*std_inst_reward_ucb)
plt.plot(mean_inst_reward_ucb_sw)
plt.fill_between(range(len(mean_inst_reward_ucb_sw)), mean_inst_reward_ucb_sw-1.96*std_inst_reward_ucb_sw, mean_inst_reward_ucb_sw+1.96*std_inst_reward_ucb_sw)
plt.plot(mean_inst_reward_ucb_cusum)
plt.fill_between(range(len(mean_inst_reward_ucb_cusum)), mean_inst_reward_ucb_cusum-1.96*std_inst_reward_ucb_cusum, mean_inst_reward_ucb_cusum+1.96*std_inst_reward_ucb_cusum)
plt.legend(["UCB", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()
