from algorithms.environments.linear_mab_environment_ns import LinearMabEnvironmentNS
from algorithms.environments.mab_environment_ns import MabEnvironmentNS
from algorithms.bandits.exp3 import Exp3
from algorithms.bandits.ucb_sw import ucbSW
from algorithms.bandits.ucb_cusum import UCBCUSUM
import numpy as np
import matplotlib.pyplot as plt

"""
STEP 6
Regret analysis for the UCB algorithm applied to the learning of the probability matrix
associated to the activations of the edges of a graph in a non stationary environment.
Three approaches: stationary UCB, sliding window UCB, cusum UCB 
Why SW approach is not working? We have 50 arms and only 365 days. If we have a sliding window 
that contains less than 50 samples, we will constantly be in the exploration phase (there will be
always 50-SW non-played arms, and these are the arms that we'll play next rounds). 
If we have more than 50 samples in the SW, then we can play the most rewarding arm; still we have to 
deal with 50 arms, so if the SW length is too big we won't be able to detect changes, if it's too short 
we'll suffer from the same problem of exploring too much. 
The outputs are 4 plots: 
1) Cumulative regret
2) Cumulative reward
3) Instantaneous regret
4) Instantaneous reward
"""

# here we establish if we are in the case of high-frequency phase changes or not
hf = True

n_nodes = 30
n_arms = 50
T = 365
n_phases = 5
n_experiments = 100

# Sliding window parameter
window_size = int(T**0.5) + n_arms

# CUSUM parameters
M = 2
# h has to be modified in the grid search
h = np.log(T) / 12


exp3_rewards_per_experiment = []
ucb_sw_rewards_per_experiment = []
ucb_cusum_rewards_per_experiment = []
opt_reward_per_experiment = []

env = MabEnvironmentNS(n_arms=n_arms, n_nodes=n_nodes, T=T, n_phases=n_phases, hf=hf)

for e in range(0, n_experiments):
    exp3 = Exp3(n_arms, T)
    ucb_sw = ucbSW(n_arms, window_size)
    ucb_cusum = UCBCUSUM(n_arms, M=M, h=h)
    opt = np.array([])
    for t in range(0, T):
        pulled_arm_exp3 = exp3.pull_arm()
        pulled_arm_ucb_sw = ucb_sw.pull_arm()
        pulled_arm_ucb_cusum = ucb_cusum.pull_arm()
        reward_exp3 = env.round(pulled_arm_exp3, t)
        reward_ucb_sw = env.round(pulled_arm_ucb_sw, t)
        reward_ucb_cusum = env.round(pulled_arm_ucb_cusum, t)
        exp3.update(pulled_arm_exp3, reward_exp3)
        ucb_sw.update(pulled_arm_ucb_sw, reward_ucb_sw)
        ucb_cusum.update(pulled_arm_ucb_cusum, reward_ucb_cusum)
        opt = np.append(opt, env.opt(t))
    exp3_rewards_per_experiment.append(exp3.collected_rewards)
    ucb_sw_rewards_per_experiment.append(ucb_sw.collected_rewards)
    ucb_cusum_rewards_per_experiment.append(ucb_cusum.collected_rewards)
    opt_reward_per_experiment.append(opt)
    if not e:
        print(opt)

# Cumulative regret of the algorithm
plt.figure(0)
plt.ylabel("Cumulative regret")
plt.xlabel("t")
cum_regret_exp3 = np.cumsum(opt_reward_per_experiment - np.array(exp3_rewards_per_experiment), axis=1)
cum_regret_ucb_sw = np.cumsum(opt_reward_per_experiment - np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_regret_ucb_cusum = np.cumsum(opt_reward_per_experiment - np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_regret_exp3 = np.mean(cum_regret_exp3, axis=0)
mean_cum_regret_ucb_sw = np.mean(cum_regret_ucb_sw, axis=0)
mean_cum_regret_ucb_cusum = np.mean(cum_regret_ucb_cusum, axis=0)
std_cum_regret_exp3 = np.std(cum_regret_exp3, axis = 0) / np.sqrt(n_experiments)
std_cum_regret_ucb_sw = np.std(cum_regret_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_cum_regret_ucb_cusum = np.std(cum_regret_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_regret_exp3)
plt.fill_between(range(len(mean_cum_regret_exp3)), mean_cum_regret_exp3-1.96*std_cum_regret_exp3, mean_cum_regret_exp3+1.96*std_cum_regret_exp3)
plt.plot(mean_cum_regret_ucb_sw)
plt.fill_between(range(len(mean_cum_regret_ucb_sw)), mean_cum_regret_ucb_sw-1.96*std_cum_regret_ucb_sw, mean_cum_regret_ucb_sw+1.96*std_cum_regret_ucb_sw)
plt.plot(mean_cum_regret_ucb_cusum)
plt.fill_between(range(len(mean_cum_regret_ucb_cusum)), mean_cum_regret_ucb_cusum-1.96*std_cum_regret_ucb_cusum, mean_cum_regret_ucb_cusum+1.96*std_cum_regret_ucb_cusum)
plt.legend(["EXP3", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Cumulative reward collected by the algorithm
plt.figure(1)
plt.ylabel("Cumulative reward")
plt.xlabel("t")
cum_reward_exp3 = np.cumsum(np.array(exp3_rewards_per_experiment), axis=1)
cum_reward_ucb_sw = np.cumsum(np.array(ucb_sw_rewards_per_experiment), axis=1)
cum_reward_ucb_cusum = np.cumsum(np.array(ucb_cusum_rewards_per_experiment), axis=1)
mean_cum_reward_exp3 = np.mean(cum_reward_exp3, axis=0)
mean_cum_reward_ucb_sw = np.mean(cum_reward_ucb_sw, axis=0)
mean_cum_reward_ucb_cusum = np.mean(cum_reward_ucb_cusum, axis=0)
std_cum_reward_exp3 = np.std(cum_reward_exp3, axis = 0) / np.sqrt(n_experiments)
std_cum_reward_ucb_sw = np.std(cum_reward_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_cum_reward_ucb_cusum = np.std(cum_reward_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_cum_reward_exp3)
plt.fill_between(range(len(mean_cum_reward_exp3)), mean_cum_reward_exp3-1.96*std_cum_reward_exp3, mean_cum_reward_exp3+1.96*std_cum_reward_exp3)
plt.plot(mean_cum_reward_ucb_sw)
plt.fill_between(range(len(mean_cum_reward_ucb_sw)), mean_cum_reward_ucb_sw-1.96*std_cum_reward_ucb_sw, mean_cum_reward_ucb_sw+1.96*std_cum_reward_ucb_sw)
plt.plot(mean_cum_reward_ucb_cusum)
plt.fill_between(range(len(mean_cum_reward_ucb_cusum)), mean_cum_reward_ucb_cusum-1.96*std_cum_reward_ucb_cusum, mean_cum_reward_ucb_cusum+1.96*std_cum_reward_ucb_cusum)
plt.legend(["EXP3", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous regret of the algorithm
plt.figure(2)
plt.ylabel("Instantaneous regret")
plt.xlabel("t")
inst_regret_exp3 = (opt_reward_per_experiment - np.array(exp3_rewards_per_experiment))
inst_regret_ucb_sw = (opt_reward_per_experiment - np.array(ucb_sw_rewards_per_experiment))
inst_regret_ucb_cusum = (opt_reward_per_experiment - np.array(ucb_cusum_rewards_per_experiment))
mean_inst_regret_exp3 = np.mean(inst_regret_exp3, axis=0)
mean_inst_regret_ucb_sw = np.mean(inst_regret_ucb_sw, axis=0)
mean_inst_regret_ucb_cusum = np.mean(inst_regret_ucb_cusum, axis=0)
std_inst_regret_exp3 = np.std(inst_regret_exp3, axis = 0) / np.sqrt(n_experiments)
std_inst_regret_ucb_sw = np.std(inst_regret_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_inst_regret_ucb_cusum = np.std(inst_regret_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_regret_exp3)
plt.fill_between(range(len(mean_inst_regret_exp3)), mean_inst_regret_exp3-1.96*std_inst_regret_exp3, mean_inst_regret_exp3+1.96*std_inst_regret_exp3)
plt.plot(mean_inst_regret_ucb_sw)
plt.fill_between(range(len(mean_inst_regret_ucb_sw)), mean_inst_regret_ucb_sw-1.96*std_inst_regret_ucb_sw, mean_inst_regret_ucb_sw+1.96*std_inst_regret_ucb_sw)
plt.plot(mean_inst_regret_ucb_cusum)
plt.fill_between(range(len(mean_inst_regret_ucb_cusum)), mean_inst_regret_ucb_cusum-1.96*std_inst_regret_ucb_cusum, mean_inst_regret_ucb_cusum+1.96*std_inst_regret_ucb_cusum)
plt.legend(["EXP3", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()

# Instantaneous reward collected by the algorithm
plt.figure(3)
plt.ylabel("Instantaneous reward")
plt.xlabel("t")
inst_reward_exp3 = np.array(exp3_rewards_per_experiment)
inst_reward_ucb_sw = np.array(ucb_sw_rewards_per_experiment)
inst_reward_ucb_cusum = np.array(ucb_cusum_rewards_per_experiment)
mean_inst_reward_exp3 = np.mean(inst_reward_exp3, axis=0)
mean_inst_reward_ucb_sw = np.mean(inst_reward_ucb_sw, axis=0)
mean_inst_reward_ucb_cusum = np.mean(inst_reward_ucb_cusum, axis=0)
std_inst_reward_exp3 = np.std(inst_reward_exp3, axis = 0) / np.sqrt(n_experiments)
std_inst_reward_ucb_sw = np.std(inst_reward_ucb_sw, axis = 0) / np.sqrt(n_experiments)
std_inst_reward_ucb_cusum = np.std(inst_reward_ucb_cusum, axis = 0) / np.sqrt(n_experiments)
plt.plot(mean_inst_reward_exp3)
plt.fill_between(range(len(mean_inst_reward_exp3)), mean_inst_reward_exp3-1.96*std_inst_reward_exp3, mean_inst_reward_exp3+1.96*std_inst_reward_exp3)
plt.plot(mean_inst_reward_ucb_sw)
plt.fill_between(range(len(mean_inst_reward_ucb_sw)), mean_inst_reward_ucb_sw-1.96*std_inst_reward_ucb_sw, mean_inst_reward_ucb_sw+1.96*std_inst_reward_ucb_sw)
plt.plot(mean_inst_reward_ucb_cusum)
plt.fill_between(range(len(mean_inst_reward_ucb_cusum)), mean_inst_reward_ucb_cusum-1.96*std_inst_reward_ucb_cusum, mean_inst_reward_ucb_cusum+1.96*std_inst_reward_ucb_cusum)
plt.legend(["EXP3", ".95 CI UCB", "UCB_SW", ".95 CI UCB_SW", "UCB_CUSUM", ".95 CI UCB_CUSUM"])
plt.show()
