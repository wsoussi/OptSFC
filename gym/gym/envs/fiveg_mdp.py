import contextlib
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import math
import gc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
# GYM
import gym
from gym import spaces
from gym.utils import seeding
# from gym.wrappers import FlattenObservation

#local files
from space_dict import space_dictionary, space_init, reward_init
from simulated_testbed import is_action_possible, get_new_simulated_observation, perform_action, get_rewards, one_step_seconds, update_mtd_constraints
import copy

# SB3
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
# from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Process



def dict_observation_to_array(observation):
    # Convert the dictionary observation into a numpy array
    obs_array = np.hstack([arr.ravel() for arr in observation.values()])
    return obs_array


def float_to_rgb_pixel(value):
    value = 0 if value == -np.inf or value == np.inf else value
    min_value = -20
    max_value = 999999999
    norm_value = (value - min_value) / (max_value - min_value)
    # Scale the value to the range [0, 255^3]
    scaled_value = int(norm_value * (255 ** 3))
    # Divide the value into three 8-bit integers
    r = scaled_value // (255 ** 2)
    g = (scaled_value % (255 ** 2)) // 255
    b = scaled_value % 255
    # Return the RGB pixel
    return [r, g, b]
def dict_observation_to_image(observation):
    # Convert the dictionary observation into a numpy array
    obs_array = dict_observation_to_array(observation)
    # give the sqrt of obs_array.size
    obs_sqrt = math.ceil(math.sqrt(obs_array.size))
    # for DQN_CNN this is needed to ensure the image is equal or bigger than the kernel
    mult4 = False
    if obs_sqrt < 18:
        mult4 = True
        obs_sqrt *= 3
    # print("the array size is",obs_sqrt,"x",obs_sqrt)

    obs_img = np.zeros(shape=(obs_sqrt, obs_sqrt, 3), dtype=np.uint8)
    for i in range(obs_sqrt):
        for j in range(obs_sqrt):
            if i * obs_sqrt + j >= obs_array.size:
                break
            #increase image by 4 if needed
            if mult4:
                obs_img[i][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+1][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j+1] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i][j+2] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                obs_img[i+2][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
                i += 2
                j += 2
            else:
                obs_img[i][j] = float_to_rgb_pixel(obs_array[i * obs_sqrt + j])
    # obs_img = Image.fromarray(obs_img)
    return obs_img

class fiveG_net(gym.Env):
    metadata = {'render.modes': ['console']}

    # (HYPERPARAMETER) MAX number of network resources *VDUs manageable in the environment
    max_resources = 100
    # n_actions = max_resources * 2 + 1  # + 1 for the 'do nothing' action
    n_actions = space_init['nb_resources'][0] * 2 + 1 # dynamic n_actions count
    rewards_coeff = [0.4, 0.3, 0.3]
    initial_recon_asp = 0.01

    # measure the resouce cost of the operation in near real-time and aggregate it to previous results of the same tuple (action, resource_type / size_unit) to have a better mean
    #       the resource cost unit is $, determined by formula $=intercept + coeffcpu * cpu + coeffram * ram_gb + coeffdisk * disk_gb
    intercept = -0.0820414
    coeff_cpu = 0.03147484
    coeff_ram = 0.00424486
    coeff_disk= 0.000066249

    # attack types
    RECON = 'recon' # recon asp increase at every step with recon_asp_factor
    APT = 'apt' # apt_asp depends on cves and recon_asp
    DOS = 'DoS' # dos_asp depends on cves and recon_asp
    DATA_LEAK = 'data_leak' # data_leak_asp depends on cves and recon_asp
    UNDEFINED = 'undefined'

    # MTD constraints per month per vnf (based on TopoFuzzer computed SSLA)
    migrations_per_month = 60
    reinstantiations_per_month = 330
    constraints_reset = 86400 / one_step_seconds # reset monthly constraints every month (86400=sec/month)

    ''''
    For PROACTIVE part: evaluate attack surface of each resource based on CVEs and the CVSS scores.
     Use CVSS exploitability in real-time and change it based on MTD actions:
     1) if vulnerability needs IP and the MTD action applied changes it than reduce ASP
     2) if vulnerability needs port //             //                //
     3) if vulnerability needs OS   //             //                //
     4) if attack source is blacklisted
     5)
    '''

    def __init__(self, policy, num_envs=1):
        self.policy = policy
        self.observation = copy.deepcopy(space_init)
        self.observation_space = spaces.Dict(space_dictionary)

        if self.policy.startswith("Cnn"):
            image_shape = dict_observation_to_image(self.observation).shape
            self.observation_space = spaces.Box(low=0, high=255, shape=image_shape,
                                                dtype=np.uint8)
        elif self.policy.startswith("Mlp"):
            flat_observation_shape = dict_observation_to_array(self.observation).shape
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat_observation_shape,
                                                   dtype=np.float64)
        self.reward_cumul = 0
        self.constraints_reset_counter = 0
        self.action_space = spaces.Discrete(self.n_actions)
        # init observation to zero for all features
        self.dynamic_asp = [{'recon': self.initial_recon_asp, 'apt': 0, 'dos': 0, 'data_leak': 0, 'undefined': 0}] * self.max_resources

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # reset mtd constraints
        self.constraints_reset_counter += 1
        if self.constraints_reset_counter >= self.constraints_reset:
            update_mtd_constraints(self.observation, self.migrations_per_month, self.reinstantiations_per_month)
            self.constraints_reset_counter = 0
        # reset counters
        self.reward_cumul = 0
        self.reward_noScalar = 0
        self.constraints_reset_counter = 0
        # init observation
        self.observation = copy.deepcopy(space_init)
        # reset asp vector
        self.dynamic_asp = [{'recon': self.initial_recon_asp, 'apt': 0, 'dos': 0, 'data_leak': 0, 'undefined': 0}] * self.max_resources
        self.action_space = spaces.Discrete(self.observation['nb_resources'][0] * 2 + 1) # dynamic action space
        if self.policy.startswith("Cnn"):
            return dict_observation_to_image(self.observation)
        elif self.policy.startswith("Mlp"):
            return dict_observation_to_array(self.observation)
        else:
            return self.observation

    def step(self, action):
        # count the step and if reached the MTD constraint reset do it
        self.constraints_reset_counter += 1
        if self.constraints_reset_counter >= self.constraints_reset:
            update_mtd_constraints(self.observation, self.migrations_per_month, self.reinstantiations_per_month)
            self.constraints_reset_counter = 0

        # DYNAMIC ACTION-SPACE UPDATE
        self.action_space = spaces.Discrete(self.observation['nb_resources'][0] * 2 + 1)
        # message error if the action is invalid
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        valid_action, err = is_action_possible(self.observation, action, self)

        # initialize step reward
        reward_vector = copy.deepcopy(reward_init)

        if not valid_action:
            final_reward = -0.25
            print('action not performed', err)
        else:
            final_reward = 0
            # update network state based on RL agent action
            perform_action(self, self.observation, action, reward_vector)

        # simulate network and update observation
        get_new_simulated_observation(self.observation)

        # compute reward
        get_rewards(self, self.observation, reward_vector)
        # print('resource_reward: ', reward_vector['resource_reward'], 'network_reward: ', reward_vector['network_reward'], 'proactive_security_reward', reward_vector['proactive_security_reward'])
        # print('the action is', action)
        # time.sleep(0.2)
        final_reward += float(reward_vector['resource_reward'] * self.rewards_coeff[0] + reward_vector['network_reward'] * self.rewards_coeff[1] + reward_vector['proactive_security_reward'] * self.rewards_coeff[2])
        self.reward_cumul += final_reward
        self.reward_noScalar = [reward_vector['resource_reward'], reward_vector['network_reward'], reward_vector['proactive_security_reward']]

        # game is done when we reach max number of episodes or reward is really bad
        if self.reward_cumul <= -100000000:
            done = True
        else:
            done = False

        info = {}

        if self.policy.startswith("Cnn"):
            return dict_observation_to_image(self.observation), final_reward, done, info
        elif self.policy.startswith("Mlp"):
            return dict_observation_to_array(self.observation), final_reward, done, info
        else:
            return self.observation, final_reward, done, info


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='console'):
        if self.reward_cumul % 100:
            print('the numul reward is ', self.reward_cumul)
        pass

    def close(self):
        pass

    # MASK FOR STATIC ACTION SPACE
    # def action_mask(self, action_num):
    #     # on every vnf 2 actions can be applied
    #     if action_num > (self.observation['nb_resources'][0] * 2 - 1) and action_num != (self.n_actions - 1):
    #         return False
    #
    #     if action_num == (self.n_actions - 1):
    #         return True
    #
    #     # for the vnf targetted check that it is not under MTD already, that the limit of MTD is not reached and that the amount of cpu, ram and disk needed are available
    #     vnf_index = int(action_num/2)
    #
    #     # check that the limit of MTDs possible is not reached
    #     if action_num % 2 == 0:
    #         # action is a restart
    #         if self.observation['mtd_constraint'][vnf_index][0] == 0:
    #             return False
    #     else:
    #         # action is a migrate
    #         if self.observation['mtd_constraint'][vnf_index][0] == 0:
    #             return False
    #
    #     if self.observation['mtd_action'][vnf_index][0] != 0:
    #         return False
    #     if self.observation['resource_consumption'][vnf_index][0] < self.observation['vim_resources'][vnf_index][0] and self.observation['resource_consumption'][vnf_index][1] < self.observation['vim_resources'][vnf_index][1] and self.observation['resource_consumption'][vnf_index][2] < self.observation['vim_resources'][vnf_index][2]:
    #         return True
    #     else:
    #         return False
    #
    #
    # def action_masks(self) -> [bool]:
    #     bools = []
    #     for action in range(0, self.n_actions):
    #         bools.append(self.action_mask(action))
    #     return bools


    # DYNAMIC ACTION SPACE MASK
    def dyn_action_mask(self, action_num):
        # on every vnf 2 actions can be applied
        if action_num > (self.observation['nb_resources'][0] * 2):
            return False

        if action_num == 0:
            return True

        # for the vnf targetted check that it is not under MTD already, that the limit of MTD is not reached and that the amount of cpu, ram and disk needed are available
        vnf_index = int((action_num-1)/2)

        # check that the limit of MTDs possible is not reached
        if (action_num - 1) % 2 == 0:
            # action is a restart
            if self.observation['mtd_constraint'][vnf_index][0] == 0:
                return False
        else:
            # action is a migrate
            if self.observation['mtd_constraint'][vnf_index][0] == 0:
                return False

        if self.observation['mtd_action'][vnf_index][0] != 0:
            return False
        if self.observation['resource_consumption'][vnf_index][0] < self.observation['vim_resources'][vnf_index][0] and self.observation['resource_consumption'][vnf_index][1] < self.observation['vim_resources'][vnf_index][1] and self.observation['resource_consumption'][vnf_index][2] < self.observation['vim_resources'][vnf_index][2]:
            return True
        else:
            return False


    def action_masks(self) -> [bool]:
        bools = []
        for action in range(0, self.n_actions):
            bools.append(self.dyn_action_mask(action))
        return bools


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str, policy: str, env: Monitor, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.model_name = model_name
        self.policy = policy
        self.env = env
        self.prev_rew = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            #
            ep_rew = sum(self.env.rewards)
            ep_len = len(self.env.rewards)
            ep_info = {"r": round(ep_rew - self.prev_rew, 6), "l": ep_len, "t": round(time.time() - self.env.t_start, 6)}
            if self.env.results_writer:
                self.env.results_writer.write_row(ep_info)
            self.prev_rew = ep_rew

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                  print("Num timesteps: {}".format(self.num_timesteps))
                  # print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                  # # my test
                  # test_model(self.policy, self.model_name, self.model)

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


# REINFORCEMENT LEARNING TRAINING
def train(agent_type, policy, total_timesteps, model_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    # initialize the environment
    # if policy.endswith("LstmPolicy") or policy.endswith("LnLstmPolicy"):
    #     env = DummyVecEnv([lambda: Monitor(fiveG_net(policy), log_dir)])
    # else:

    # env = gym.wrappers.FlattenObservation(fiveG_net(policy))
    env_train = Monitor(fiveG_net(policy), log_dir)
    env_train.seed(0)
    env_train.action_space.np_random.seed(123)

    # DEFINE AND TRAIN MODEL
    # Add some param noise for exploration
    # param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1) # (HYPER)
    # Create the callback: check every 10000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, model_name=model_name, policy=policy, env=env_train) # (HYPER)
    print('start training the ', agent_type, ' agent')
    if agent_type == "DQN":
        model = DQN(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "A2C":
        model = A2C(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "PPO":
        model = PPO(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "DDPG":
        model = DDPG(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "SAC":
        model = SAC(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    elif agent_type == "TD3":
        model = TD3(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    else: #MaskablePPO
        model = MaskablePPO(policy, env_train, verbose=1, tensorboard_log="./tmp/"+model_name+"/")
    with open(log_dir+'Log'+model_name+'.txt','a') as f:
        with contextlib.redirect_stdout(f):
            model.learn(total_timesteps, callback=callback)

    plot_results(log_dir, "OptSFC " + agent_type + " Learning Curve").savefig(log_dir + 'plot_' + model_name + '.pdf')

    del env_train, model
    gc.collect()


# REINFORCEMENT LEARNING TRAINING
def transfer_train(model, total_timesteps, model_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    # initialize the environment
    # if policy.endswith("LstmPolicy") or policy.endswith("LnLstmPolicy"):
    #     env = DummyVecEnv([lambda: Monitor(fiveG_net(policy), log_dir)])
    # else:

    # env = gym.wrappers.FlattenObservation(fiveG_net(policy))
    env_train = Monitor(fiveG_net(policy), log_dir)
    env_train.seed(0)
    env_train.action_space.np_random.seed(123)

    # DEFINE AND TRAIN MODEL
    # Create the callback: check every 10000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, model_name=model_name, policy=policy, env=env_train) # (HYPER)
    print('start training the ', agent_type, ' agent')
    with open(log_dir+'Log'+model_name+'.txt','a') as f:
        with contextlib.redirect_stdout(f):
            model.set_env(env=env_train)
            model.learn(total_timesteps, callback=callback)
            print("training done!")
    plot_results(log_dir, "OptSFC " + agent_type + " Learning Curve").savefig(log_dir + 'plot_' + model_name + '.pdf')
    del env_train, model
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
def plot_results(log_folder, title='Learning Curve'):
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


if __name__ == "__main__":
    env = fiveG_net("MlpPolicy") #MultiInputPolicy
    # flat the env. using FlattenObservation from gym
    env = gym.wrappers.FlattenObservation(env)
    # check_env(env, warn=True)

    # train and save data and best model in /tmp/
    total_timesteps = 1000000  # (HYPER)
    agent_type = "A2C"  # (HYPER)
    policy = "MlpPolicy"
    model_name = 'model_' + agent_type + '_' + policy + '_' + str(total_timesteps)
    log_dir = "./4vnf_less_penalty/" + model_name + "/"
    # Create tmp folder
    os.makedirs(log_dir, exist_ok=True)
    p1 = Process(target=train, args=(agent_type, policy, total_timesteps, model_name, log_dir))  # (HYPER)

    # train and save data and best model in /tmp/
    total_timesteps = 1000000  # (HYPER)
    agent_type = "PPO"  # (HYPER)
    policy = "MlpPolicy"
    model_name = 'model_' + agent_type + '_' + policy + '_' + str(total_timesteps)
    log_dir = "./4vnf_less_penalty/" + model_name + "/"
    # Create tmp folder
    os.makedirs(log_dir, exist_ok=True)
    p2 = Process(target=train, args=(agent_type, policy, total_timesteps, model_name, log_dir))  # (HYPER)

    # train and save data and best model in /tmp/
    total_timesteps = 1000000  # (HYPER)
    agent_type = "MaskablePPO"  # (HYPER)
    policy = "MlpPolicy"
    model_name = 'model_' + agent_type + '_' + policy + '_' + str(total_timesteps)
    log_dir = "./4vnf_less_penalty/" + model_name + "/"
    # Create tmp folder
    os.makedirs(log_dir, exist_ok=True)
    p3 = Process(target=train, args=(agent_type, policy, total_timesteps, model_name, log_dir))  # (HYPER)

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    # TRANSFER LEARNING TEST
    # upload existing model and transfer-learn to improve it and save data and best model in /tmp/
    # old_model_timesteps = 500000
    # total_timesteps = 50000  # (HYPER)
    # agent_type = "A2C"
    # policy = "MultiInputPolicy"
    # model_name = 'model_' + agent_type + '_' + policy + '_' + str(old_model_timesteps)
    # log_dir = "./tmp/" + model_name + "/"
    # model = A2C.load(log_dir + 'best_model.zip')
    # # change log_dir name to save new model separately
    # model_name = 'model_' + agent_type + '_' + policy + '_' + str(old_model_timesteps + total_timesteps)
    # log_dir = "./tmp/" + model_name + "/"
    # os.makedirs(log_dir, exist_ok=True)
    # transfer_train(model, total_timesteps, model_name, log_dir)  # (HYPER)