from typing import Iterable
from dataclasses import dataclass
import numpy as np

from .age_group import AgeGroup


@dataclass
class Population:
    name: str
    age_probabilities: np.ndarray  # Probability of each age group in the population. Must sum 1.0.
    age_groups: AgeGroup
    home_size_probabilities: np.ndarray  # Probability of homes of each size. Must sum 1.0.
    inhabitants: int
    seed_infections: int
    geosocial_displacement: float

    @property
    def isolation_propensity_increase(self):
        pass

    @isolation_propensity_increase.setter
    def isolation_propensity_increase(self, increase: float):
        for age_group in self.age_groups:
            age_group.adesao_isolamento += increase
