from itertools import count

import gym
from torch import optim

import torch.nn as nn
import torch.nn.functional as F
import random
import torch as T
import numpy as np

from bomber_env.ReplayMemory import ReplayMemory, Transition
from play import play


T.set_default_tensor_type(T.DoubleTensor)

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


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.00
EPS_DECAY = 200
TARGET_UPDATE = 10
HIDDEN_N = 10

policy_net = DQN(HIDDEN_N)
target_net = DQN(HIDDEN_N)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-2)
memory = ReplayMemory(10000)

env = gym.make('bomber-v0')

steps_done = 0


def select_action(state):
    return policy_net(state).argmax()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=T.bool)
    non_final_next_states = T.vstack([next_state for next_state in batch.next_state if next_state is not None])
    state_batch = T.vstack(batch.state)
    action_batch = T.vstack(batch.action)
    reward_batch = T.vstack(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = T.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.reshape(-1)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 500
for i_episode in range(num_episodes):
    print(i_episode)
    env.reset()
    state, _, _, _ = env.step(0)
    for t in count():
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        if reward != 0:
            print(f"REWARD {reward}")
        reward = T.tensor([reward])
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()
        if done:
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print([transitino.reward for transitino in memory.memory if transitino.reward > 0])
for i in range(10):
    env.reset()
    play(env, target_net)
