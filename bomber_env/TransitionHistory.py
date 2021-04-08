from dataclasses import dataclass
from typing import List

import torch as T

from bomber_env.Transition import Transition


@dataclass(frozen=True)
class TransitionHistory:
    state: T.Tensor
    reward: T.Tensor
    action: T.Tensor
    has_next: T.Tensor
    next_state: T.Tensor

    @staticmethod
    def merge(histories: List['TransitionHistory']) -> 'TransitionHistory':
        tensor_tuples = [list(iter(hist)) for hist in histories]
        return TransitionHistory(*[T.cat([tensor_tuple[i] for tensor_tuple in tensor_tuples], 0) for i in range(5)])

    def __tensors(self):
        return self.state, self.reward, self.action, self.has_next, self.next_state

    def __iter__(self):
        return iter((self.state, self.reward, self.action, self.has_next, self.next_state))

    def __getitem__(self, indx: T.Tensor) -> 'TransitionHistory':
        return TransitionHistory(*[tensor[indx] for tensor in self.__tensors()])

    def __setitem__(self, index: int, transition: Transition):
        self.state[index, :] = transition.state
        self.reward[index] = transition.reward
        self.action[index, :] = transition.action
        if transition.next_state is not None:
            self.has_next[index] = True
            self.next_state[index, :] = transition.next_state
        else:
            self.has_next[index] = False

    def __len__(self):
        return self.state.size()[0]

