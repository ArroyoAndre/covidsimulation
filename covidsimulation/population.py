from typing import List, Union
from dataclasses import dataclass
import numpy as np

from .age_group import AgeGroup
from .random import RandomParameter


@dataclass
class Population:
    name: str
    age_probabilities: np.ndarray  # Probability of each age group in the population. Must sum 1.0.
    age_groups: List[AgeGroup]
    home_size_probabilities: np.ndarray  # Probability of homes of each size. Must sum 1.0.
    inhabitants: int
    seed_infections: Union[int, RandomParameter]
    geosocial_displacement: float
    isolation_propensity_increase: Union[float, RandomParameter] = 0.0  # Adjusts isolation_propensity for all age groups
