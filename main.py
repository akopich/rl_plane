from itertools import count

import gym
from torch import optim

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np

from RandomStrategy import RandomStrategy
from bomber_env.Memory import ReplayMemory, Transition, MergedMemory
from play import play, Strategy

T.set_default_tensor_type(T.DoubleTensor)


def net2strat(net: nn.Module) -> Strategy:
    return lambda obs: net(obs).argmax()


class DQN(nn.Module):
    def __init__(self, hidden_width):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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

#DQN(HIDDEN_N)
# target_net = DQN(HIDDEN_N)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
memoryPositive = ReplayMemory(10000, lambda tr: tr.reward > 0)
memoryNegative = ReplayMemory(10000, lambda tr: tr.reward < 0)
memoryZero = ReplayMemory(10000, lambda tr: tr.reward == 0)

memory = MergedMemory([memoryNegative, memoryPositive, memoryZero])

env = gym.make('bomber-v0')

steps_done = 0


def select_action(state):
    return policy_net(state).argmax()


def optimize_model():
    if len(memoryPositive) < 100:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=T.bool)
    non_final_next_states = T.vstack([next_state for next_state in batch.next_state if next_state is not None])
    state_batch = T.vstack(batch.state)
    action_batch = T.vstack(batch.action)
    reward_batch = T.vstack(batch.reward)

    # Compute Q(s_t, a)
    state_action_Q = policy_net(state_batch).gather(1, action_batch)

    next_state_Q = T.zeros(BATCH_SIZE)
    next_state_Q[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_Q = (next_state_Q * GAMMA) + reward_batch.reshape(-1)

    loss = F.mse_loss(state_action_Q, expected_state_action_Q.unsqueeze(1))
    print(loss.item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_random = 10000
for i in range(num_random):
    strat = RandomStrategy()
    env.reset()
    play(env, strat, memory)

for j in range(5000):
    optimize_model()

for i in range(10):
    env.reset()
    play(env, net2strat(policy_net), None, log=True)
