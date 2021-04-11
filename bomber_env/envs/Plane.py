from dataclasses import dataclass
import numpy as np


@dataclass
class Plane:
    location: float
    altitude: float
    speed:    float
    dt:       float
    g:        float

    def step(self):
        self.location += self.speed * self.dt

    def point_of_impact(self) -> float:
        return self.location + self.time_of_impact() * self.speed

    def time_of_impact(self) -> float:
        return np.sqrt(2 * self.altitude / self.g)

