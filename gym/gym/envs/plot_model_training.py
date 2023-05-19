import contextlib
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gc
import numpy as np
import matplotlib.pyplot as plt
import time

# SB3
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# REINFORCEMENT LEARNING TRAINING
def test_model(test_timesteps, policy, model_description, model = None, model_name = None, log_dir = None):
    if model:
        if not model_name or not log_dir:
            print("can't test a model without model_name and log_dir parameters")
            return
        log_dir = 'test_' + log_dir

    env = Monitor(fiveG_net(policy), log_dir)
    env.seed(0)
    env.action_space.np_random.seed(123)
    iter = 1  # (HYPER)
    sum_obs_rewards = 0
    sum_steps = 0
    sum_null_actions = 0
    for i in range(iter):
        n_steps = total_timesteps  # (HYPER)
        null_actions = 0
        tot_reward = 0
        prev_step_tot_reward = 0
        obs = env.reset()
        step = -1
        for step in range(n_steps):
            if policy == "noRL":
                # design a deterministic baseline policy
                action = env.action_space.sample()
            elif model is None:
                action = env.action_space.sample()
            else:
                action = model.predict(observation=obs, deterministic=True)
                action = action[0]
            if action == 200:
                null_actions += 1
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            if done:
                break
            if model and (step % 5000 == 0):
                # save csv file for plotting and comparison with training curve
                tot_reward += reward
                ep_info = {"r": round(tot_reward - prev_step_tot_reward, 6), "l": step,
                           "t": round(time.time() - env.t_start, 6)}
                if env.results_writer:
                    env.results_writer.write_row(ep_info)
                prev_step_tot_reward = tot_reward
        sum_obs_rewards += tot_reward
        sum_steps += step + 1
        sum_null_actions += null_actions

    print(model_description, "avg tot steps=", sum_steps / iter, "null_actions=",
          0 if sum_null_actions == 0 else sum_steps / sum_null_actions / iter, 'avg_obs_reward', sum_obs_rewards / sum_steps)
    del env
    gc.collect()


# functions for plotting
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results_return(log_folder, title='Cumulative Reward'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=2)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    # plt.show()
    return plt


def plot_results_multi(log_folder, label, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y /1000, label=label.split('\\')[0])

    # add threshold only once (when adding A2C_Mlp)
    if label == "A2C\_MultiInputPolicy":
        thresh = []
        for el in x:
            thresh.append(-780)
        plt.plot(x,thresh, label="threshold (s. MTD)", color='k')

    # define other elements of the grid
    plt.grid(True)
    plt.xlabel("Number of Episodes")
    plt.ylabel(r"Avg. Reward $\overline{\mathcal{R}}$ (10³)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc= "upper left")
    # plt.show()
    return plt

def plot_results_reward(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = moving_average(y, window=2)
    y2 = []
    for i in range(1, len(y)):
        y2.append((y[i] - y[i-1]) / 5000)
    print(y)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    # plt.show()
    return plt

if __name__ == "__main__":
    agent_types = ["DQN", "A2C", "PPO"]
    policies = ["MultiInputPolicy"]
    log_dirs = []
    label_names = []
    plots = []
    # plot multiple agents
    for agent_type in agent_types:
        for policy in policies:
            # the filename of the model
            model_name = 'model_' + agent_type + '_' + policy + '_' + str(300000)
            # the path for the model
            log_dir = "tmp/" + model_name + "/"
            log_dirs.append(log_dir)
            if(os.path.exists(log_dir)):
                print(log_dir)
                # the label on the legend of the plot
                if agent_type == "A2C_2":
                    ag_type = "A2C"
                elif agent_type == "PPO2":
                    ag_type = "PPO"
                else:
                    ag_type = agent_type
                label_name = ag_type + "\_" + policy.replace("Policy", "")
                label_names.append(label_name)
                plot_results_multi(log_dir, label_name).savefig('./tmp/all_agents.pdf', bbox_inches='tight')
                # plots.append(plot)
            else:
                print(False)