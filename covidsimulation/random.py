import abc
from dataclasses import dataclass, field
from typing import Dict, Sequence, Callable, List, Union

import numpy as np


@dataclass
class RandomParametersState:
    state: Dict = field(default_factory=dict)

    def materialize_object(self, obj):
        if isinstance(obj, (int, float, bool, str, Callable)):
            return
        elif isinstance(obj, Dict):
            for key, value in obj.items():
                if isinstance(value, RandomParameter):
                    obj[key] = value.materialize(self)
                else:
                    self.materialize_object(value)
        elif isinstance(obj, (Sequence, np.ndarray)):
            for i, value in enumerate(obj):
                if isinstance(value, RandomParameter):
                    obj[i] = value.materialize(self)
                else:
                    self.materialize_object(value)
        else:
            for key in dir(obj):
                if not key.startswith('_'):
                    value = getattr(obj, key)
                    if isinstance(value, RandomParameter):
                        setattr(obj, key, value.materialize(self))
                    else:
                        self.materialize_object(value)

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

    def __add__(self, other):
        if isinstance(self, ParameterSum):
            return ParameterSum(self.components + [other])
        return ParameterSum([self, other])

    def __radd__(self, other):
        if not other:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(self, ParameterProduct):
            return ParameterProduct(self.components + [other])
        return ParameterProduct([self, other])

    def __rmul__(self, other):
        return self.__mul__(other)


@dataclass
class ParameterSum(RandomParameter):
    components: List[Union[RandomParameter, int, float]]

    def materialize(self, randomness_state: RandomParametersState):
        total = 0
        for parameter in self.components:
            if isinstance(parameter, RandomParameter):
                total += parameter.materialize(randomness_state)
            else:
                total += parameter
        return total

    def _get_random(self):
        raise RuntimeError('Invalid call to _get_random of synthetic random parameter')


@dataclass
class ParameterProduct(RandomParameter):
    components: List[Union[RandomParameter, int, float]]

    def materialize(self, randomness_state: RandomParametersState):
        total = 1.0
        for parameter in self.components:
            if isinstance(parameter, RandomParameter):
                total *= parameter.materialize(randomness_state)
            else:
                total *= parameter
        return total

    def _get_random(self):
        raise RuntimeError('Invalid call to _get_random of synthetic random parameter')


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


@dataclass
class LogUniformParameter(RandomParameter):
    name: str
    min: float
    max: float

    def _get_random(self):
        return np.exp(np.random.uniform(np.log(self.min), np.log(self.max)))
