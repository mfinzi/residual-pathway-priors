from gym.envs.registration import register
#from hopper_full import
register(
    id='HopperFull-v0',
    entry_point='full_envs.hopper_full:HopperFull',
    max_episode_steps=1000,
    reward_threshold=3800,
)