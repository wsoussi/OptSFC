# GYM
import gym
# environment
from fiveg_mdp import fiveG_net
from simulated_testbed import impact_ssla_factors, is_action_possible
# SB3
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
import time
import gc
import asyncio
import json
import requests
import numpy as np
import random


def pick(lst):
    normalized_lst = [x / sum(lst) for x in lst]
    r = random.uniform(0, 1)
    cumulative_sum = 0
    for i, probability in enumerate(normalized_lst):
        cumulative_sum += probability
        if r <= cumulative_sum:
            return i

# TODO define a deterministic baseline policy
def baseline_model(env):
    action = 0 # do nothing
    no_action = not np.any(env.observation['mtd_action'])
    # if there is an MTD action ongoing do nothing
    if not no_action:
        return action
    # decide the VNF according to the SLA impact value
    target_vnf = pick(impact_ssla_factors)
    # decide on the MTD action type according to SSLA
    target_action = pick([env.reinstantiations_per_month, env.migrations_per_month]) + 1 # first
    if target_action == 1:
        action = target_vnf * 2
    else:
        action = target_vnf * 2 + 1
    if is_action_possible(env.observation, action, env):
        return action
    else:
        return env.n_actions - 1 # do nothing


async def real_action(action):
    url = "http://demo_driver/enforce"
    payload = json.dumps({
        "action": 0
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


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
        n_steps = test_timesteps
        null_actions = 0
        tot_reward = 0
        prev_step_tot_reward = 0
        obs = env.reset()
        step = -1
        log_actions = [0] * 201
        for step in range(n_steps):
            if policy == "static":
                # design a deterministic baseline policy
                action = baseline_model(env)
            elif policy == "noRL" and model is None:
                while True:
                    action = env.action_space.sample() % 9
                    action = (env.n_actions - 1) if action == 8 else action
                    if is_action_possible(env.observation, action, env):
                        break
            else:
                action = model.predict(observation=obs, deterministic=True)
                action = action[0]
            log_actions[action] += 1
            if action == env.n_actions - 1:
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
          0 if sum_null_actions == 0 else sum_steps / sum_null_actions / iter, 'avg_obs_reward', sum_obs_rewards / sum_steps,
          "log of actions", log_actions)
    del env
    gc.collect()


if __name__ == '__main__':
    # test model
    total_timesteps = 100000
    agent_type = "static"
    policy = "MultiInputPolicy" #"MultiInputPolicy"
    model_name = 'model_' + agent_type + '_' + policy + '_' + str(total_timesteps)
    log_dir = "./4vnf_test/" + model_name + "/"
    model=None
    # model = PPO.load(log_dir + 'best_model.zip')
    test_model(test_timesteps=1000, policy=policy, model_description=model_name, model=model, model_name=model_name, log_dir=log_dir)