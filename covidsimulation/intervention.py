import abc
from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np

from .simulation import logit_transform_value


@dataclass
class Intervention(abc.ABC):
    simulation_day: int
    parameter: Any

    @abc.abstractmethod
    def apply(self, env) -> None:
        pass

    def setup(self, env):
        yield env.timeout(self.simulation_day)
        while env.d0 is None or self.simulation_day > env.now - env.d0:
            yield env.timeout(1.0)
        self.apply(env)


class SocialDistancingChange(Intervention):

    def apply(self, env) -> None:
        distancing_factor = self.parameter
        desvio_logit = (env.desvio_isolamento - 0.5) / 5.0
        fator_isolamento = np.power(distancing_factor, 0.65)
        env.fator_isolamento = logit_transform_value(fator_isolamento, desvio_logit)
        for pessoa in env.pessoas:
            pessoa.em_isolamento = pessoa.home.isolation_propensity < env.fator_isolamento


class DiagnosisDelayChange(Intervention):

    def apply(self, env) -> None:
        diagnosis_max_delay = self.parameter
        for person in env.pessoas:
            if person.age_group.diagnosis_delay > diagnosis_max_delay:
                person.age_group.diagnosis_delay = diagnosis_max_delay
