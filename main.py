from itertools import count

import gym
import logging

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np

from RandomStrategy import RandomStrategy
from bomber_env.Memory import ReplayMemory, MergedMemory
from bomber_env.Transition import Transition
from play import play, Strategy

T.set_default_tensor_type(T.FloatTensor)


logging.root.setLevel(logging.DEBUG)


def net2strat(net: nn.Module) -> Strategy:
    return lambda obs: net(obs).argmax()


BATCH_SIZE = 1024
GAMMA = 0.99999999
EPS_START = 0.9
EPS_END = 0.00
EPS_DECAY = 200
TARGET_UPDATE = 10
HIDDEN_N = 6

policy_net = nn.Sequential(nn.Linear(3, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, 2)
                           )

optimizer = T.optim.Adam(policy_net.parameters(), lr=1e-4)
memoryPositive = ReplayMemory(10000, lambda tr: tr.reward > 0)
memoryNegative = ReplayMemory(10000, lambda tr: tr.reward < 0)
memoryZero = ReplayMemory(10000, lambda tr: tr.reward == 0)

memory = MergedMemory([memoryNegative, memoryPositive, memoryZero])

env = gym.make('bomber-v0')


def optimize_model():
    if len(memoryPositive) < 100:
        return
    state, reward, action, has_next, next_state = memory.sample(BATCH_SIZE)

    state_action_Q = policy_net(state).gather(1, action)
    next_state_Q = T.zeros(BATCH_SIZE)
    next_state_Q[has_next] = policy_net(next_state[has_next]).max(1)[0].detach()
    expected_state_action_Q = (next_state_Q * GAMMA) + reward

    loss = F.mse_loss(state_action_Q, expected_state_action_Q.unsqueeze(1))
    logging.debug(f"Q-training loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_random = 50000
for i in range(num_random):
    strat = RandomStrategy()
    play(env, strat, memory)

for j in range(10000):
    optimize_model()


print(np.array([play(env, net2strat(policy_net), log=True) for i in range(100)]).mean())
