from dataclasses import dataclass
from typing import Optional

import torch as T


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