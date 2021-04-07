from collections import namedtuple
import random
from itertools import chain
from typing import NamedTuple, Protocol, Optional

import torch as T
import numpy as np

STATE_WIDTH = 3
ACTION_WIDTH = 1


class Transition(NamedTuple):
    state: T.Tensor
    action: T.Tensor
    next_state: Optional[T.Tensor]
    reward: T.Tensor

    def __eq__(self, other):
        a = self
        b = other
        next_state_equal = a.next_state == b.next_state
        if not isinstance(next_state_equal, bool):
            next_state_equal = T.all(next_state_equal).item()
        return T.equal(a.state, b.state) \
               and T.equal(a.reward, b.reward) \
               and T.equal(a.action, b.action) \
               and next_state_equal

    def __ne__(self, other):
        return not (self == other)


class Memory(Protocol):
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
        self.state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
        self.next_state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
        self.has_next_state = T.zeros(capacity, dtype=T.bool)
        self.reward = T.zeros(capacity)
        self.action = T.zeros(capacity, ACTION_WIDTH, dtype=T.int64)
        self.counter = 0

    def push(self, transition: Transition):
        if not self.predicate(transition):
            return
        index = self.counter % self.capacity
        self.state[index, :] = transition.state
        self.reward[index] = transition.reward
        self.action[index, :] = transition.action
        if transition.next_state is not None:
            self.has_next_state[index] = True
            self.next_state[index, :] = transition.next_state
        else:
            self.has_next_state[index] = False
        self.counter += 1

    def sample(self, batch_size):
        n = self.capacity if self.counter >= self.capacity else self.counter
        indx = T.randint(n, (batch_size,))

        return self.state[indx, :], self.reward[indx], self.action[indx, :], \
               self.has_next_state[indx], self.next_state[indx, :]

    def __len__(self):
        return self.capacity if self.counter >= self.capacity else self.counter


class MergedMemory(Memory):
    def __init__(self, memories):
        self.memories = memories

    def push(self, tr):
        for mem in self.memories:
            mem.push(tr)

    def sample(self, batch_size):
        single_size = int(batch_size / len(self.memories))
        last_size = batch_size - single_size * (len(self.memories) - 1)
        sizes = [single_size] * (len(self.memories) - 1) + [last_size]

        samples = [mem.sample(size) for mem, size in zip(self.memories, sizes)]

        return tuple([T.cat([sample[i] for sample in samples], 0) for i in range(5)])

    def __len__(self):
        return min([len(mem) for mem in self.memories])
