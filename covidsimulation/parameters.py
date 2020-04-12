
from typing import Dict, List, Iterable, Tuple
from .simulation import SimulationConstants
from .population import Population


DEFAULT_D0_INFECTIONS = 2500
DEFAULT_HOME_AGE_COFACTOR = 0.4
DEFAULT_START_DATE = '2020-03-01'
DEFAULT_TRANSMISSION_SCALE_DAYS = 0.3  # one new person exposed every Exponential(0.3) days
DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED = 4  # Initial infections occur on 40+ only
DEFAULT_CAPACITY_VENTILATORS = 2000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_INTENSIVE_CARE = 3000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_BEDS = 20000  # The default might make no sense depending on the size of the population
DEFAULT_CAPACITY_HOSPITAL_MAX = 80000  # The default might make no sense depending on the size of the population


class Parameters:
    population_segments: Iterable[Population]
    constants: SimulationConstants
    distancing: Iterable[Tuple[int, float]]
    d0_infections: int
    start_date: str
    home_age_cofactor: float
    transmission_scale_days: float
    min_age_group_initially_infected: int
    capacity_ventilators: int
    capacity_intensive_care: int
    capacity_hospital_beds: int
    capacity_hospital_max: int  # Maximum overcapacity
    total_inhabitants: int

    def __init__(self,
        population_segments,
        constants,
        distancing,
        d0_infections=DEFAULT_D0_INFECTIONS,
        start_date=DEFAULT_START_DATE,
        home_age_cofactor=DEFAULT_HOME_AGE_COFACTOR,
        transmission_scale_days=DEFAULT_TRANSMISSION_SCALE_DAYS,
        min_age_group_initially_infected=DEFAULT_MIN_AGE_GROUP_INITIALLY_INFECTED,
        capacity_ventilators=DEFAULT_CAPACITY_VENTILATORS,
        capacity_intensive_care=DEFAULT_CAPACITY_INTENSIVE_CARE,
        capacity_hospital_beds=DEFAULT_CAPACITY_HOSPITAL_BEDS,
        capacity_hospital_max=DEFAULT_CAPACITY_HOSPITAL_MAX,
    ):
        self.population_segments = population_segments
        self.constants = constants
        self.distancing = distancing
        self.d0_infections = d0_infections
        self.start_date = start_date
        self.home_age_cofactor = home_age_cofactor
        self.transmission_scale_days = transmission_scale_days
        self.min_age_group_initially_infected = min_age_group_initially_infected
        self.capacity_ventilators = capacity_ventilators
        self.capacity_intensive_care = capacity_intensive_care
        self.capacity_hospital_beds = capacity_hospital_beds
        self.capacity_hospital_max = capacity_hospital_max

        self.total_inhabitants = sum([p.inhabitants for p in population_segments])
