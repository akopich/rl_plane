from collections import namedtuple
import random
from itertools import chain
from typing import NamedTuple, Protocol

import torch as T
import numpy as np


class Transition(NamedTuple):
    state: T.Tensor
    action: T.Tensor
    next_state: T.Tensor
    reward: T.Tensor


class Memory(Protocol):
    def positive(self):
        raise NotImplementedError()

    def push(self, transition: Transition):
        raise NotImplementedError()

    def sample(self, batch_size: int):
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class ReplayMemory(Memory):
    def __init__(self, capacity, predicate=lambda x: True):
        self.capacity = capacity
        self.predicate = predicate
        self.memory = []
        self.position = 0

    def positive(self):
        return T.tensor([transition.reward > 0 for transition in self.memory]).float().sum()

    def push(self, transition: Transition):
        if not self.predicate(transition):
            return
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indx = np.random.choice(len(self.memory), batch_size)
        return [self.memory[i] for i in indx]

    def __len__(self):
        return len(self.memory)


class MergedMemory(Memory):
    def __init__(self, memories):
        self.memories = memories

    def positive(self):
        return T.tensor([mem.positive() for mem in self.memories]).sum()

    def push(self, tr):
        for mem in self.memories:
            mem.push(tr)

    def sample(self, batch_size):
        single_size = int(batch_size / len(self.memories))
        last_size = batch_size - single_size * (len(self.memories)-1)
        sizes = [single_size] * (len(self.memories) - 1) + [last_size]
        return list(chain(*[mem.sample(size) for mem, size in zip(self.memories, sizes)]))

    def __len__(self):
        return min([len(mem) for mem in self.memories])
