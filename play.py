from typing import Callable

from gym import Env

import torch as T
import logging

from bomber_env.Memory import Memory
from bomber_env.Transition import Transition

Strategy = Callable[[T.Tensor], int]


def play(env: Env, strategy: Strategy, memory: Memory = None, log=False) -> float:
    env.reset()
    obs, reward, end, _ = env.step(0)
    while not end:
        action = strategy(obs)
        next_obs, reward, end, _ = env.step(action)
        if memory is not None:
            memory.push(Transition(obs, T.tensor([action]), next_obs, T.tensor([reward])))
        obs = next_obs
    if log:
        env.render()
        logging.info(f"played with reward: {reward}")
    return reward
