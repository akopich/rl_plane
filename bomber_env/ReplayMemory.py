from collections import namedtuple
import random
import torch as T
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def positive(self):
        return T.tensor([transition.reward > 0 for transition in self.memory]).float().sum()

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        rewards = np.array([transition.reward.item() for transition in self.memory])
        if (rewards > 0).sum() <= 0:
            return random.sample(self.memory, batch_size)
        rewards = np.maximum(rewards, rewards[rewards > 0].min()/2)
        p = rewards / rewards.sum()
        indx = np.random.choice(len(self.memory), size=batch_size, p=p)
        return [self.memory[i] for i in indx]

    def __len__(self):
        return len(self.memory)
