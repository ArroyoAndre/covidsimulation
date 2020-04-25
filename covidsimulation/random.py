import abc
from dataclasses import dataclass, field
from typing import Dict


import numpy as np


@dataclass
class RandomParametersState:
    state: Dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.state.get(key)

    def __setitem__(self, key, value):
        self.state[key] = value
        self.state = {key: value for (key, value) in sorted(self.state.items())}


class RandomParameter(abc.ABC):
    name: str

    def materialize(self, randomness_state: RandomParametersState):
        value = randomness_state[self.name]
        if value is None:
            value = self._get_random()
            randomness_state[self.name] = value
        return value

    @abc.abstractmethod
    def _get_random(self):
        pass


@dataclass
class UniformParameter(RandomParameter):
    name: str
    min: float
    max: float

    def _get_random(self):
        return np.random.uniform(self.min, self.max)


@dataclass
class TriangularParameter(RandomParameter):
    name: str
    min: float
    mode: float
    max: float

    def _get_random(self):
        return np.random.triangular(self.min, self.mode, self.max)

@dataclass
class UniformIntParameter(RandomParameter):
    name: str
    min: int
    max: int

    def _get_random(self):
        return np.random.randint(self.min, self.max)
