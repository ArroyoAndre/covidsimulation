from typing import Dict, List, Iterable, Tuple, Optional, Union
from copy import deepcopy
from dataclasses import dataclass, field

from .early_stop import EarlyStop
from .intervention import Intervention
from .simulation import SimulationConstants
from .population import Population
from .random import RandomParametersState, RandomParameter, UniformParameter, TriangularParameter


DEFAULT_D0_INFECTIONS = 2500
DEFAULT_HOME_AGE_COFACTOR = 0.4
DEFAULT_START_DATE = '2020-03-01'
DEFAULT_STREET_TRANSMISSION_SCALE_DAYS = 0.6  # one new person exposed every Exponential(0.3) days
DEFAULT_SOCIAL_GROUP_TRANSMISSION_SCALE_DAYS = 0.6  # one new person exposed every Exponential(0.3) days
DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED = 4  # Initial infections occur on 40+ only
DEFAULT_CAPACITY_VENTILATORS = 2000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_INTENSIVE_CARE = 3000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_BEDS = 20000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_MAX = 80000  # The default might make no sense depending on the size of the population


@dataclass
class Parameters:
    population_segments: Iterable[Population]
    constants: SimulationConstants
    interventions: Iterable[Intervention]
    d0_infections: Union[int, RandomParameter] = DEFAULT_D0_INFECTIONS
    start_date: str = DEFAULT_START_DATE
    home_age_cofactor: float = DEFAULT_HOME_AGE_COFACTOR
    street_transmission_scale_days: RandomParameter = UniformParameter('street_transmission_scale_days', 0.60, 1.25)
    social_group_transmission_scale_difference: RandomParameter = UniformParameter('social_group_transmission_scale_difference', -0.1, 0.1)
    min_age_group_initially_infected: int = DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED
    capacity_ventilators: int = DEFAULT_CAPACITY_VENTILATORS
    capacity_icu: int = DEFAULT_CAPACITY_INTENSIVE_CARE
    capacity_hospital_beds: int = DEFAULT_CAPACITY_HOSPITAL_BEDS
    capacity_hospital_max: int = DEFAULT_CAPACITY_HOSPITAL_MAX  # Maximum overcapacity
    total_inhabitants: Optional[int] = None  # Ignored
    random_parameters_state: RandomParametersState = field(default_factory=RandomParametersState)
    severity_deviation: RandomParameter = TriangularParameter('severity_deviation', -0.6, 0.0, 0.2)
    severity_bias: RandomParameter = UniformParameter('severity_bias', -0.2, 0.2)
    isolation_deviation: RandomParameter = UniformParameter('isolation_deviation', -0.5, 0.5)
    early_stops: Optional[List[EarlyStop]] = None

    def __post_init__(self):
        self.total_inhabitants = sum(p.inhabitants for p in self.population_segments)

    def clone(self):
        return deepcopy(self)
