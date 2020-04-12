
from typing import Dict, List, Iterable, Tuple, Optional
from .simulation import SimulationConstants
from .population import Population
from dataclasses import dataclass


DEFAULT_D0_INFECTIONS = 2500
DEFAULT_HOME_AGE_COFACTOR = 0.4
DEFAULT_START_DATE = '2020-03-01'
DEFAULT_TRANSMISSION_SCALE_DAYS = 0.3  # one new person exposed every Exponential(0.3) days
DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED = 4  # Initial infections occur on 40+ only
DEFAULT_CAPACITY_VENTILATORS = 2000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_INTENSIVE_CARE = 3000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_BEDS = 20000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_MAX = 80000  # The default might make no sense depending on the size of the population


@dataclass
class Parameters:
    population_segments: Iterable[Population]
    constants: SimulationConstants
    distancing: Iterable[Tuple[int, float]]
    d0_infections: int = DEFAULT_D0_INFECTIONS
    start_date: str = DEFAULT_START_DATE
    home_age_cofactor: float = DEFAULT_HOME_AGE_COFACTOR
    transmission_scale_days: float = DEFAULT_TRANSMISSION_SCALE_DAYS
    min_age_group_initially_infected: int = DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED
    capacity_ventilators: int = DEFAULT_CAPACITY_VENTILATORS
    capacity_intensive_care: int = DEFAULT_CAPACITY_INTENSIVE_CARE
    capacity_hospital_beds: int = DEFAULT_CAPACITY_HOSPITAL_BEDS
    capacity_hospital_max: int = DEFAULT_CAPACITY_HOSPITAL_MAX  # Maximum overcapacity
    total_inhabitants: Optional[int] = None  # Ignored

    def __post_init__(self, *args, **kwargs):
        self.total_inhabitants = sum([p.inhabitants for p in self.population_segments])
