from gym.envs.registration import register

register(
    id='simple1-v0',
    entry_point='gym_simple1.envs:Simple1',
)