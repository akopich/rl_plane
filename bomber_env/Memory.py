from typing import Protocol, Tuple, List

import torch as T

from bomber_env.Transition import Transition
from bomber_env.TransitionHistory import TransitionHistory

STATE_WIDTH = 3
ACTION_WIDTH = 1


class Memory(Protocol):
    def push(self, transition: Transition):
        raise NotImplementedError()

    def get(self) -> TransitionHistory:
        raise NotImplementedError()

    def sample(self, batch_size: int) -> TransitionHistory:
        raise NotImplementedError()

    def split(self, p: float) -> Tuple['Memory', 'Memory']:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class ReplayMemory(Memory):
    def __init__(self, capacity, predicate=lambda x: True, history=None, counter=None):
        self.capacity = capacity
        self.predicate = predicate
        if history is None:
            state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
            next_state = T.zeros(capacity, STATE_WIDTH, dtype=T.float)
            has_next = T.zeros(capacity, dtype=T.bool)
            reward = T.zeros(capacity)
            action = T.zeros(capacity, ACTION_WIDTH, dtype=T.int64)
            self.history = TransitionHistory(state, reward, action, has_next, next_state)
            self.counter = 0
        else:
            self.history = history
            self.counter = counter

    def push(self, transition: Transition):
        if not self.predicate(transition):
            return
        index = self.counter % self.capacity
        self.history[index] = transition
        self.counter += 1

    def sample(self, batch_size) -> TransitionHistory:
        n = len(self)
        indx = T.randint(n, (batch_size,))
        return self.history[indx]

    def split(self, p: float) -> Tuple[Memory, Memory]:
        mask1 = T.zeros(self.capacity, dtype=T.bool)
        mask2 = T.zeros(self.capacity, dtype=T.bool)
        n = len(self)
        indx = T.rand(n) < p
        mask1[range(n)] = indx
        mask2[range(n)] = T.logical_not(indx)

        mask2mem = lambda mask: ReplayMemory(mask.sum().item(),
                                             predicate=self.predicate,
                                             history=self.history[mask],
                                             counter=mask.sum().item())
        return mask2mem(mask1), mask2mem(mask2)

    def get(self) -> TransitionHistory:
        return self.history

    def __len__(self):
        return min(self.capacity, self.counter)


class MergedMemory(Memory):
    def __init__(self, memories: List[Memory]):
        self.memories = memories

    def get(self) -> TransitionHistory:
        samples = [list(iter(mem.get())) for mem in self.memories]
        return TransitionHistory(*[T.cat([sample[i] for sample in samples], 0) for i in range(5)])

    def push(self, tr):
        for mem in self.memories:
            mem.push(tr)

    def split(self, p: float) -> Tuple['Memory', 'Memory']:
        mems1: List[Memory] = []
        mems2: List[Memory] = []

        for mem in self.memories:
            mem1, mem2 = mem.split(p)
            mems1 += [mem1]
            mems2 += [mem2]

        return MergedMemory(mems1), MergedMemory(mems2)

    def sample(self, batch_size) -> TransitionHistory:
        single_size = int(batch_size / len(self.memories))
        last_size = batch_size - single_size * (len(self.memories) - 1)
        sizes = [single_size] * (len(self.memories) - 1) + [last_size]

        samples = [mem.sample(size) for mem, size in zip(self.memories, sizes)]

        return TransitionHistory.merge(samples)

    def __len__(self):
        return min([len(mem) for mem in self.memories])
