from typing import Callable

from gym import Env

import torch as T

from bomber_env.Memory import Transition, Memory

Strategy = Callable[[T.Tensor], int]


def play(env: Env, strategy: Strategy, memory: Memory = None, log=False) -> float:
    obs, reward, end, _ = env.step(0)
    while not end:
        action = strategy(obs)
        next_obs, reward, end, _ = env.step(action)
        if memory is not None:
            memory.push(Transition(obs, T.tensor([action]), next_obs, T.tensor([reward])))
        obs = next_obs
    if log:
        env.render()
        print(reward)
    return reward
