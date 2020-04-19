from dataclasses import dataclass
from multiprocessing import Queue
from typing import List, Optional, Dict

import numpy as np
from simpy import Environment, PriorityResource

from .simulation import Person
from .lab import Lab


@dataclass
class SimulationRandomness:
    severity_deviation: float
    severity_bias: float
    isolation_deviation: float
    expositions_interval: float


@dataclass
class SimulationEnvironment:
    env: Environment
    sim_params: 'Parameters'
    duration: int
    sim_number: int
    scaling: float
    simulate_capacity: bool
    randomness: SimulationRandomness
    isolation_factor: float
    attention: PriorityResource
    hospital_bed: PriorityResource
    ventilator: PriorityResource
    icu: PriorityResource
    stats: np.ndarray
    creation_queue: Optional[Queue]
    simulation_queue: Optional[Queue]
    d0: Optional[float] = None
    populations: Dict[str, List[Person]] = None
    people: List[Person] = None
    lab: Optional[Lab] = None
