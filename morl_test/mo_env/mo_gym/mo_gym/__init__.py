from gym.envs.registration import register

register(
    id='basic-v0',
    entry_point='mo_gym.envs:fiveG_net',
)