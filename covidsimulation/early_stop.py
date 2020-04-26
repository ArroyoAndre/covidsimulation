from dataclasses import dataclass

import numpy as np

from .simulation_environment import SimulationEnvironment

@dataclass
class EarlyStop:
    simulation_day: int
    min_num_deaths: int
    max_num_deaths: int


class EarlyStopError(RuntimeError):
    pass


def process_early_stop(senv: SimulationEnvironment, early_stop: EarlyStop):
    day = int(senv.env.now)
    while (not day) or day < early_stop.simulation_day:
        yield senv.env.timeout(1.0)
    if (int(early_stop.min_num_deaths * senv.scaling)
            <= np.array([p.dead and p.diagnosed for p in senv.people]).sum()
            <= early_stop.max_num_deaths * senv.scaling
    ):
        return
    raise EarlyStopError
