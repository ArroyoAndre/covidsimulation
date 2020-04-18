from typing import List
from dataclasses import dataclass
import numpy as np

from .age_group import AgeGroup


@dataclass
class Population:
    name: str
    age_probabilities: np.ndarray  # Probability of each age group in the population. Must sum 1.0.
    age_groups: List[AgeGroup]
    home_size_probabilities: np.ndarray  # Probability of homes of each size. Must sum 1.0.
    inhabitants: int
    seed_infections: int
    geosocial_displacement: float
    isolation_propensity_increase: float = 0.0  # Adjusts isolation_propensity for all age groups
