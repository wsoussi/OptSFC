import contextlib
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
import gc
import numpy as np
import json
import matplotlib.pyplot as plt
import time
# GYM
import gym
from gym import spaces
from gym.utils import seeding

#local files
from space_dict import space_dictionary, space_init, reward_init
from simulated_testbed import is_action_possible, get_new_simulated_observation, perform_action, get_rewards, one_step_seconds, update_mtd_constraints
from mo_fiveg_mdp import MOfiveG_net
import mo_gymnasium as mo_gym

# MORL-BASELINES
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
# from morl_baselines.multi_policy.ols.ols import OLS
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.single_policy.ser.mo_ppo import make_env
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning

rewards_coeff = [0.4, 0.3, 0.3]


def scalarization(reward: np.ndarray):
    return float(reward[0] * rewards_coeff[0] + reward[1] * rewards_coeff[1] + reward[2] * rewards_coeff[2])


def eval_mo_reward_conditioned(
    agent,
    env,
    test_timesteps,
    filename,
    scalarization,
    w: np.ndarray = None,
    render: bool = False,
) -> [float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment. This makes the assumption that the agent is conditioned on the accrued reward i.e. for ESR agent.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        scalarization: scalarization function, taking weights and reward as parameters
        w: weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized total reward, scalarized return, vectorized total reward, vectorized return
    """
    obs, _ = env.reset()
    vec_return, disc_vec_return = np.zeros(env.reward_space.shape[0]), np.zeros(env.reward_space.shape[0])
    gamma = 1.0
    n_steps = test_timesteps
    null_actions = 0
    test_tot = 0
    list_cumul_rew = []
    list_null_actions = []
    log_actions = [0] * 201
    for step in range(n_steps):
        if step % 50000 == 0:
            print("had "+str(step) + " steps. Goal: "+ str(n_steps))
            print(test_tot)
        action = agent.eval(obs, disc_vec_return)
        log_actions[action] += 1
        if action == 0:
            null_actions += 1
            list_null_actions.append(step)

        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("done")
            break
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= agent.gamma
        test_tot += info["rew"]
        list_cumul_rew.append(test_tot)

    #save list_cumul_rew and list_null_actions in JSON file
    filename = "./cumul_rew/" + filename
    print("find output in " + filename)
    output = {"list_cumul_rew": list_cumul_rew, "list_null_actions": list_null_actions}
    with open(filename, 'w') as f:
        json.dump(output, f)

    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    print("avg tot steps=", step, "null_actions=",
          0 if null_actions == 0 else step / null_actions, 'avg_obs_reward', test_tot / step, 'avg_discounted_reward', scalarized_return / step,
          "log of actions", log_actions)
    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )



def train_pql():
    env = MOfiveG_net("MlpPolicy")
    ref_point = np.array([0, -25])

    agent = PQL(
        env,
        ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        log=False,
    )

    # Training
    pf = agent.train(num_episodes=1000, log_every=100, action_eval="pareto_cardinality")#"hypervolume")
    assert len(pf) > 0

    # Policy following
    target = np.array(pf.pop())
    tracked = agent.track_policy(target)
    assert np.all(tracked == target)


def train_Envelope(total_timesteps):
    env = MOfiveG_net("MlpPolicy")
    eval_env = MOfiveG_net("MlpPolicy")

    save_replay_buffer = True
    save_dir = "envelope-QL"
    filename = "model_envelope-QL_100"

    agent = Envelope(env)

    # Train the agent
    agent.train(total_timesteps= total_timesteps, total_episodes=1, eval_freq=100)
    # test the agent
    test_timesteps = 100000
    scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo_reward_conditioned(
        agent, env=eval_env, scalarization=scalarization, test_timesteps=test_timesteps, filename= "Envelope"+str(total_timesteps)+".json")



def train_eupg():
    env = MOfiveG_net("MlpPolicy")
    eval_env = MOfiveG_net("MlpPolicy")

    agent = EUPG(env, scalarization=scalarization, gamma=0.99, log=False)
    agent.train(total_timesteps=2000000, eval_env=eval_env, eval_freq=100)
    print("training finished")

    # test the algorithm
    for i in range(10):
        test_timesteps = 100000
        scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo_reward_conditioned(
            agent, env=eval_env, scalarization=scalarization, test_timesteps= test_timesteps, filename= "Eupgg"+str(test_timesteps)+".json")
        if scalar_return / 10000 <= 0.22:
            break
        # print(scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret)


def eval_Envelope(total_timesteps):
    env = MOfiveG_net("MlpPolicy")
    save_replay_buffer = True
    save_dir = "envelope-QL"
    filename = "model_envelope-QL"
    agent = Envelope(env)
    agent.load(path= save_dir+'/'+filename+'.tar' ,load_replay_buffer = save_replay_buffer)

    print("Envelope model loaded \n Model Evaluation:")
    test_timesteps = 1000
    scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo_reward_conditioned(
        agent, env=env, scalarization=scalarization, test_timesteps=total_timesteps)


if __name__ == '__main__':
    # train_eupg()
    train_Envelope(99999)