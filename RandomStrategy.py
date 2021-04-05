import torch as T
from numpy.random import uniform


class RandomStrategy:
    def __init__(self):
        self.distance_to_drop = uniform(low=0., high=15000.)

    def __call__(self, obs):
        distance = obs[0]
        if distance > self.distance_to_drop:
            return 0
        else:
            return 1
