import matplotlib.pyplot as plt
import numpy as np


def plot_cumulative_regret(rewards, opt, n_experiments, algorithm_name, opt_title=""):
    """
    Plot the cumulative regret of the algorithm.

    Parameters:
        rewards (array-like): The array containing the data to be plotted.
        opt (float): The optimal reward value.
        n_experiments (int): The number of experiments.
        algorithm_name (str): The name of the algorithm.
        opt_title (str): Optional string appended to the title. Default is empty.

    Returns:
        None
    """
    plt.ylabel("Cumulative regret "+opt_title)
    plt.xlabel("t")
    cum_regret = np.cumsum(opt - np.array(rewards), axis=1)
    mean_cum_regret = np.mean(cum_regret, axis=0)
    std_cum_regret = np.std(cum_regret, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_cum_regret, 'r')
    plt.fill_between(range(len(mean_cum_regret)), mean_cum_regret - 1.96 * std_cum_regret,
                     mean_cum_regret + 1.96 * std_cum_regret)
    plt.legend([algorithm_name, ".95 CI"])
    plt.show()


def plot_cumulative_reward(rewards, n_experiments, algorithm_name, opt_title=""):
    """
        Plot the cumulative reward collected by the algorithm.

        Parameters:
            rewards (array-like): The array containing the data to be plotted.
            n_experiments (int): The number of experiments.
            algorithm_name (str): The name of the algorithm.
            opt_title (str): Optional string appended to the title. Default is empty.

        Returns:
            None
    """
    plt.ylabel("Cumulative reward "+opt_title)
    plt.xlabel("t")
    cum_reward = np.cumsum(np.array(rewards), axis=1)
    mean_cum_reward = np.mean(cum_reward, axis=0)
    std_cum_reward = np.std(cum_reward, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_cum_reward, 'r')
    plt.fill_between(range(len(mean_cum_reward)), mean_cum_reward - 1.96 * std_cum_reward,
                     mean_cum_reward + 1.96 * std_cum_reward)
    plt.legend([algorithm_name, ".95 CI"])
    plt.show()


def plot_instantaneous_regret(rewards, opt, n_experiments, algorithm_name, opt_title=""):
    """
    Plot the instantaneous regret of the algorithm.

    Parameters:
        rewards (array-like): The array containing the data to be plotted.
        opt (float): The optimal reward value.
        n_experiments (int): The number of experiments.
        algorithm_name (str): The name of the algorithm.
        opt_title (str): Optional string appended to the title. Default is empty.

    Returns:
        None
    """
    plt.ylabel("Instantaneous regret "+opt_title)
    plt.xlabel("t")
    inst_regret = (opt - np.array(rewards))
    mean_inst_regret = np.mean(inst_regret, axis=0)
    std_inst_regret = np.std(inst_regret, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_inst_regret, 'r')
    plt.fill_between(range(len(mean_inst_regret)), mean_inst_regret - 1.96 * std_inst_regret,
                     mean_inst_regret + 1.96 * std_inst_regret)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.legend([algorithm_name, ".95 CI"])
    plt.show()


def plot_instantaneous_reward(rewards, opt, n_experiments, algorithm_name, opt_title=""):
    """
        Plot the instantaneous reward of the algorithm.

        Parameters:
            rewards (array-like): The array containing the data to be plotted.
            opt (float): The optimal reward value.
            n_experiments (int): The number of experiments.
            algorithm_name (str): The name of the algorithm.
            opt_title (str): Optional string appended to the title. Default is empty.

        Returns:
            None
    """
    plt.ylabel("Instantaneous reward "+opt_title)
    plt.xlabel("t")
    inst_reward = np.array(rewards)
    mean_inst_reward = np.mean(inst_reward, axis=0)
    std_inst_reward = np.std(inst_reward, axis=0) / np.sqrt(n_experiments)
    plt.plot(mean_inst_reward, 'r')
    plt.fill_between(range(len(mean_inst_reward)), mean_inst_reward - 1.96 * std_inst_reward,
                     mean_inst_reward + 1.96 * std_inst_reward)
    plt.axhline(y=opt, color='black', linestyle='-')
    plt.legend([algorithm_name, ".95 CI"])
    plt.show()
