from collections import namedtuple
import random
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple, Protocol, Optional

import torch as T
import numpy as np

STATE_WIDTH = 3
ACTION_WIDTH = 1


@dataclass(frozen=True)
class TransitionHistory:
    state: T.Tensor
    reward: T.Tensor
    action: T.Tensor
    has_next: T.BoolTensor
    next_state: T.Tensor

    def __iter__(self):
        return iter((self.state, self.reward, self.action, self.has_next, self.next_state))


@dataclass(frozen=True)
class Transition:
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

    def get(self) -> TransitionHistory:
        raise NotImplementedError()

    def sample(self, batch_size: int) -> TransitionHistory:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class ReplayMemory(Memory):
    def __init__(self, capacity, predicate=lambda x: True):
        self.capacity = capacity
        self.predicate = predicate
        self.state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
        self.next_state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
        self.has_next = T.zeros(capacity, dtype=T.bool)
        self.reward = T.zeros(capacity)
        self.action = T.zeros(capacity, ACTION_WIDTH, dtype=T.int64)
        self.counter = 0

    def _tensors(self):
        return self.state, self.reward, self.action, self.has_next, self.next_state

    def push(self, transition: Transition):
        if not self.predicate(transition):
            return
        index = self.counter % self.capacity
        self.state[index, :] = transition.state
        self.reward[index] = transition.reward
        self.action[index, :] = transition.action
        if transition.next_state is not None:
            self.has_next[index] = True
            self.next_state[index, :] = transition.next_state
        else:
            self.has_next[index] = False
        self.counter += 1

    def sample(self, batch_size) -> TransitionHistory:
        n = len(self)
        indx = T.randint(n, (batch_size,))
        return self.__get_by_index(indx)

    def get(self) -> TransitionHistory:
        return self.__get_by_index(T.tensor(range(len(self))))

    def __get_by_index(self, indx: T.Tensor) -> TransitionHistory:
        return TransitionHistory(*[tensor[indx] for tensor in self._tensors()])

    def __len__(self):
        return self.capacity if self.counter >= self.capacity else self.counter


class MergedMemory(Memory):
    def __init__(self, memories):
        self.memories = memories

    def get(self) -> TransitionHistory:
        samples = [list(iter(mem.get())) for mem in self.memories]
        return TransitionHistory(*[T.cat([sample[i] for sample in samples], 0) for i in range(5)])

    def push(self, tr):
        for mem in self.memories:
            mem.push(tr)

    def sample(self, batch_size) -> TransitionHistory:
        single_size = int(batch_size / len(self.memories))
        last_size = batch_size - single_size * (len(self.memories) - 1)
        sizes = [single_size] * (len(self.memories) - 1) + [last_size]

        samples = [list(iter(mem.sample(size))) for mem, size in zip(self.memories, sizes)]

        return TransitionHistory(*[T.cat([sample[i] for sample in samples], 0) for i in range(5)])

    def __len__(self):
        return min([len(mem) for mem in self.memories])
