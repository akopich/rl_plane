import gym
from gym import error, spaces, utils, Space
from gym.spaces import Discrete, Box
from gym.utils import seeding
from numpy.random import uniform
import numpy as np
import torch as T


class BomberEnv(gym.Env):
    action_space = Discrete(2)
    observation_space = Box(low=np.array([0., 0., 0.]),
                            high=np.array([np.inf, np.inf, np.inf]),
                            dtype=np.float64)  # (distance_to_target, altitude, velocity)

    def __init__(self):
        self.reset()

    def step(self, action):
        if action == 1:
            point_of_impact = self.plane_location + np.sqrt(2 * self.alt / self.g) * self.plane_speed
            distance = np.abs(point_of_impact - self.target_location)
            reward = 1/(distance/10 + 1)
            return None, reward, True, {}
        self.plane_location += self.dt * self.plane_speed
        if self.plane_location > self.target_location:
            return None, -10., True, {}

        distance_to_target = self.target_location - self.plane_location
        observation = T.tensor([distance_to_target, self.alt, self.plane_speed])
        return observation, 0, False, {}

    def reset(self):
        self.width = 15000.
        self.g = 9.8
        self.target_location = uniform(low=5000., high=15000.)
        self.plane_location = 0.
        self.plane_speed = 222.
        self.alt = 1000
        self.dt = 1

    def render(self, mode='human'):
        print(f"target_location={self.target_location}, plane_location={self.plane_location}")
