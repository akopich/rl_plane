from dataclasses import dataclass
import numpy as np

from bomber_env.envs.bomber_env import Action


@dataclass
class DiveStrategy:
    distance_to_drop: float
    expected_pitch: float

    def __call__(self, obs):
        distance, alt, speed, pitch = obs
        if distance <= speed * (1. + np.cos(np.pi/6) + np.cos(np.pi/3)):
            if self.expected_pitch < pitch:
                return Action.DOWN
            return Action.BOMB
        return Action.STRAIGHT



@dataclass
class RandomDiveDiveStrategy:
    distance_to_drop: float
    expected_pitch: float

    def __call__(self, obs):
        distance, alt, speed, pitch = obs
        if distance <= self.distance_to_drop:
            if self.expected_pitch < pitch:
                return Action.DOWN
            return Action.BOMB
        return Action.STRAIGHT
