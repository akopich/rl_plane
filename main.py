from itertools import count

import gym
import logging

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np

from RandomStrategy import RandomStrategy
from bomber_env.Memory import ReplayMemory, MergedMemory, Memory
from bomber_env.Transition import Transition
from bomber_env.TransitionHistory import TransitionHistory
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
HIDDEN_N = 10

policy_net = nn.Sequential(nn.Linear(3, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Dropout(p=0.05),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, HIDDEN_N),
                           nn.ReLU(),
                           nn.Linear(HIDDEN_N, 2)
                           )

optimizer = T.optim.Adam(policy_net.parameters(), lr=1e-2)
memoryPositive = ReplayMemory(10000, lambda tr: tr.reward > 0)
memoryNegative = ReplayMemory(10000, lambda tr: tr.reward < 0)
memoryZero = ReplayMemory(10000, lambda tr: tr.reward == 0)

memory = MergedMemory([memoryNegative, memoryPositive, memoryZero])

env = gym.make('bomber-v0')


def optimize_model(history: TransitionHistory, optimize=True):
    if len(memoryPositive) < 100:
        return
    state, reward, action, has_next, next_state = history

    state_action_Q = policy_net(state).gather(1, action)
    next_state_Q = T.zeros(len(history))
    next_state_Q[has_next] = policy_net(next_state[has_next]).max(1)[0].detach()
    expected_state_action_Q = (next_state_Q * GAMMA) + reward

    loss = F.mse_loss(state_action_Q, expected_state_action_Q.unsqueeze(1))

    if optimize:
        logging.debug(f"Q-learning train loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    else:
        logging.debug(f"Q-learning test loss: {loss.item()}")


num_random = 10000
for i in range(num_random):
    strat = RandomStrategy()
    play(env, strat, memory)

memoryPositive2 = ReplayMemory(10000, lambda tr: tr.reward > 0)
memoryNegative2 = ReplayMemory(10000, lambda tr: tr.reward < 0)
memoryZero2 = ReplayMemory(10000, lambda tr: tr.reward == 0)

memory2 = MergedMemory([memoryNegative2, memoryPositive2, memoryZero2])

for i in range(num_random):
    strat = RandomStrategy()
    play(env, strat, memory2)


for j in range(10000+1):
    optimize_model(memory.sample(BATCH_SIZE))
    if j % 1000 == 0:
        logging.info(f"ITER {j}")
        optimize_model(memory.get(), optimize=False)
        optimize_model(memory2.get(), optimize=False)
        logging.info(np.array([play(env, net2strat(policy_net), log=False) for i in range(1000)]).mean())



