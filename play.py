from itertools import count

import torch as T


def play(env, net):
    obs, reward, end, _ = env.step(0)
    while not end:
        action = net(obs).argmax()
        obs, reward, end, _ = env.step(action)
    env.render()
    print(reward)
