from dataclasses import dataclass
from multiprocessing import Queue
from typing import List, Optional, Dict

import numpy as np
from simpy import Environment, PriorityResource

from .simulation import Person
from .parameters import Parameters


@dataclass
class SimulationRandomness:
    severity_deviation: float
    severity_bias: float
    isolation_deviation: float
    serial_interval: float


@dataclass
class SimulationEnvironment:
    env: Environment
    sim_params: Parameters
    duration: int
    sim_number: int
    scaling: float
    simulate_capacity: bool
    ramdomness: SimulationRandomness
    isolation_factor: float
    attention: PriorityResource
    hospital_bed: PriorityResource
    ventilator: PriorityResource
    icu: PriorityResource
    populations: Dict[str, List[Person]]
    stats: np.ndarray
    people: List[Person]
    creation_queue: Optional[Queue]
    simulation_queue: Optional[Queue]
    d0: Optional[float] = None
