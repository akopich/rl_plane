import torch as T
from numpy.random import uniform
from numpy.random import randint

from play import Strategy


class RandomStrategy:
    def __init__(self):
        self.distance_to_drop = uniform(low=0., high=15000.)

    def __call__(self, obs):
        distance = obs[0]
        if distance > self.distance_to_drop:
            return 3
        pitch_increment = randint(-1, 2)
        return pitch_increment
