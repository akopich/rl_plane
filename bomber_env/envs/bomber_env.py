import gym
from gym import error, spaces, utils, Space
from gym.spaces import Discrete, Box
from gym.utils import seeding
from numpy.random import uniform
import numpy as np
import torch as T

from bomber_env.envs.Plane import Plane


class BomberEnv(gym.Env):
    action_space = Discrete(2)
    observation_space = Box(low=np.array([0., 0., 0.]),
                            high=np.array([np.inf, np.inf, np.inf]),
                            dtype=np.float64)  # (distance_to_target, altitude, velocity)

    def __init__(self):
        self.reset()

    def step(self, action):
        if action == 1:
            point_of_impact = self.plane.point_of_impact()
            distance = np.abs(point_of_impact - self.target_location)
            if distance < 500:
                reward = 1.
            else:
                reward = -1.
            return None, reward, True, {}
        self.plane.step()
        if self.plane.location > self.target_location:
            return None, -1., True, {}

        distance_to_target = self.target_location - self.plane.location
        observation = T.tensor([distance_to_target, self.plane.altitude, self.plane.speed])
        return observation, 0, False, {}

    def reset(self):
        self.width = 15000.
        self.target_location = uniform(low=5000., high=15000.)

        plane_location = 0.
        plane_speed = uniform(low=100, high=250)
        alt = uniform(low=100, high=1500)
        dt = 1
        g = 9.8
        self.plane = Plane(plane_location, alt, plane_speed, dt, g)

    def render(self, mode='human'):
        print(f"target_location={self.target_location}, plane_location={self.plane.location}, speed={self.plane.speed}, "
              f"CCIP: {self.plane.point_of_impact()}")
