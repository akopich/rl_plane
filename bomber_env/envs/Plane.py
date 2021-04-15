from dataclasses import dataclass
import numpy as np


class Plane:
    def __init__(self,
                 location: float,
                 altitude: float,
                 speed: float,
                 dt: float,
                 g: float,
                 pitch: float):
        self.pitch = pitch
        self.g = g
        self.dt = dt
        self.speed = speed
        self.altitude = altitude
        self.location = location

    def step(self, pitch_increment: int):
        self.pitch += np.pi / 6 * pitch_increment
        self.location += self.speed * np.cos(self.pitch) * self.dt
        self.altitude += self.speed * np.sin(self.pitch) * self.dt

    def point_of_impact(self) -> float:
        return self.location + self.time_to_impact() * self.speed * np.cos(self.pitch)

    def time_to_impact(self) -> float:
        vy = self.speed * np.sin(self.pitch)
        return (vy + np.sqrt(vy ** 2 + 2 * self.altitude * self.g)) / self.g
