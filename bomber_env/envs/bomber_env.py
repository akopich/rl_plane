import gym
from gym import error, spaces, utils, Space
from gym.spaces import Discrete, Box
from gym.utils import seeding
from numpy.random import uniform
import numpy as np
import torch as T

from bomber_env.envs.Plane import Plane

from enum import Enum


class Action(Enum):
    STRAIGHT = 0
    DOWN = 1
    BOMB = 2


class BomberEnv(gym.Env):
    action_space = Discrete(2)
    observation_space = Box(low=np.array([0., 0., 0.]),
                            high=np.array([np.inf, np.inf, np.inf]),
                            dtype=np.float64)  # (distance_to_target, altitude, velocity)

    def __init__(self):
        self.reset()

    def step(self, action: Action):
        self.cnt += 1
        if self.cnt > 300 or self.plane.location > self.target_location \
                or self.plane.location < 0 or self.plane.altitude < 0:
            return None, -1., True, {}

        if action == Action.BOMB:
            point_of_impact = self.plane.point_of_impact()
            escape_direction = np.sign(self.target_location - point_of_impact)
            target_acceleration = 5
            self.target_location += escape_direction * target_acceleration * self.plane.time_to_impact() ** 2 / 2
            distance = np.abs(point_of_impact - self.target_location)
            if distance < 500:
                reward = 1.
            else:
                reward = -1.
            return None, reward, True, {}

        if action == Action.STRAIGHT:
            pitch = 0
        elif action == Action.DOWN:
            pitch = -1
        else:
            pitch = 1
        self.plane.step(pitch)

        distance_to_target = self.target_location - self.plane.location
        observation = T.tensor([distance_to_target, self.plane.altitude, self.plane.speed, self.plane.pitch], dtype=T.float32)
        return observation, 0, False, {}

    def reset(self):
        self.cnt = 0
        self.width = 15000.
        self.target_location = uniform(low=5000., high=15000.)

        plane_location = 0.
        plane_speed = uniform(low=100, high=250)
        alt = uniform(low=1000, high=2000)
        dt = 1
        g = 9.8
        self.plane = Plane(plane_location, alt, plane_speed, dt, g, 0.)

    def render(self, mode='human'):
        print(f"target_location={self.target_location}, plane_alt={self.plane.altitude}, plane_location={self.plane.location}, speed={self.plane.speed}, "
              f"pitch={self.plane.pitch}, CCIP: {self.plane.point_of_impact()}")
