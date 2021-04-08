import unittest

from bomber_env.Memory import ReplayMemory, MergedMemory
from bomber_env.Transition import Transition

import torch as T


class MemoryTest(unittest.TestCase):
    def for_mem_of_capacity_3(self, memory):
        to_be_pushed_out = Transition(T.tensor([0., 0., 0.]), T.tensor([0]), T.tensor([10., 20., 30.]), T.tensor(0.))
        transitions = [Transition(T.tensor([1., 2., 3.]), T.tensor([0]), T.tensor([10., 20., 30.]), T.tensor(0.)),
                       Transition(T.tensor([1., 2., 3.]), T.tensor([0]), T.tensor([10., 20., 30.]), T.tensor(0.)),
                       Transition(T.tensor([1., 2., 3.]), T.tensor([0]), None, T.tensor(1.))
                       ]
        memory.push(to_be_pushed_out)
        for tr in transitions:
            memory.push(tr)
        smpl = memory.sample(30)

        for i in range(smpl.state.size()[0]):
            ns = smpl.next_state[i, :] if smpl.has_next[i] else None
            tr = Transition(smpl.state[i, :], smpl.action[i, :], ns, smpl.reward[i])
            self.assertTrue(tr != to_be_pushed_out)
            self.assertTrue(tr in transitions)

    def test(self):
        memory = ReplayMemory(3)
        self.for_mem_of_capacity_3(memory)

    def test_merged(self):
        memory1 = ReplayMemory(3)
        memory2 = ReplayMemory(3)
        memory = MergedMemory([memory1, memory2])
        self.for_mem_of_capacity_3(memory)


if __name__ == '__main__':
    unittest.main()
