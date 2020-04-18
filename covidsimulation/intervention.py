import abc
from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import simpy

from .simulation import logit_transform_value


@dataclass
class Intervention(abc.ABC):
    simulation_day: int
    parameter: Any

    @abc.abstractmethod
    def apply(self, env) -> None:
        pass

    def setup(self, env: simpy.Environment):
        env.process(self._run(env))

    def _run(self, env: simpy.Environment):
        yield env.timeout(self.simulation_day)
        while env.d0 is None or self.simulation_day > env.now - env.d0:
            yield env.timeout(1.0)
        self.apply(env)

class SocialDistancingChange(Intervention):

    def apply(self, env) -> None:
        distancing_factor = self.parameter
        desvio_logit = (env.isolation_deviation - 0.5) / 5.0
        isolation_factor = np.power(distancing_factor, 0.65)
        env.isolation_factor = logit_transform_value(isolation_factor, desvio_logit)
        for person in env.people:
            person.in_isolation = person.home.isolation_propensity < env.isolation_factor


class DiagnosisDelayChange(Intervention):

    def apply(self, env) -> None:
        diagnosis_max_delay = self.parameter
        for person in env.people:
            if person.age_group.diagnosis_delay > diagnosis_max_delay:
                person.age_group.diagnosis_delay = diagnosis_max_delay
