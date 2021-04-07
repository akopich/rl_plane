import unittest

from bomber_env.Memory import ReplayMemory, Transition, MergedMemory

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
        state, reward, action, has_next_state, next_state = memory.sample(30)

        for i in range(state.size()[0]):
            ns = next_state[i, :] if has_next_state[i] else None
            tr = Transition(state[i, :], action[i, :], ns, reward[i])
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
