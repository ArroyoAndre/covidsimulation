import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from .simulation import logit_transform_value
from .simulation_environment import SimulationEnvironment

@dataclass
class Intervention(abc.ABC):
    simulation_day: int
    parameter: Any

    @abc.abstractmethod
    def apply(self, senv: SimulationEnvironment) -> None:
        pass

    def setup(self, senv: SimulationEnvironment):
        senv.env.process(self._run(senv))

    def _run(self, senv: SimulationEnvironment):
        yield senv.env.timeout(self.simulation_day)
        while senv.d0 is None or self.simulation_day > senv.env.now - senv.d0:
            yield senv.env.timeout(1.0)
        self.apply(senv)

class SocialDistancingChange(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        distancing_factor = self.parameter
        desvio_logit = (senv.randomness.isolation_deviation - 0.5) / 5.0
        isolation_factor = np.power(distancing_factor, 0.65)
        senv.isolation_factor = logit_transform_value(isolation_factor, desvio_logit)
        for person in senv.people:
            person.in_isolation = person.home.isolation_propensity < senv.isolation_factor


class DiagnosisDelayChange(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        diagnosis_max_delay = self.parameter
        for person in senv.people:
            if person.age_group.diagnosis_delay > diagnosis_max_delay:
                person.age_group.diagnosis_delay = diagnosis_max_delay
