from gym.envs.registration import register

register(
    id='bomber-v0',
    entry_point='bomber_env.envs.bomber_env:BomberEnv',
)
