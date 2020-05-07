import abc
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .utils import get_date_from_isoformat
from .simulation import logit_transform_value
from .simulation_environment import SimulationEnvironment


@dataclass
class Intervention(abc.ABC):
    simulation_day: Optional[
        Union[int, str]]  # If simulation_day is None, intervention will be applied at start of simulation
    parameter: Any

    @abc.abstractmethod
    def apply(self, senv: SimulationEnvironment) -> None:
        pass

    def setup(self, senv: SimulationEnvironment):
        if isinstance(self.simulation_day, str):
            self.simulation_day = get_simulation_day_from_isoformat_date(self.simulation_day, senv)
        senv.env.process(self._run(senv))

    def _run(self, senv: SimulationEnvironment):
        if self.simulation_day is not None:
            yield senv.env.timeout(self.simulation_day)
            while senv.d0 is None or self.simulation_day > senv.env.now - senv.d0:
                yield senv.env.timeout(1.0)
        self.apply(senv)


class SocialDistancingChange(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        distancing_factor = self.parameter
        senv.isolation_factor = logit_transform_value(distancing_factor,
                                                      senv.sim_params.isolation_deviation)
        for person in senv.people:
            person.in_isolation = person.home.isolation_propensity < senv.isolation_factor


class DiagnosisDelayChange(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        diagnosis_max_delay = self.parameter
        for person in senv.people:
            if person.age_group.diagnosis_delay > diagnosis_max_delay:
                person.age_group.diagnosis_delay = diagnosis_max_delay


class MaskUsage(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        average_adherence = self.parameter
        for person in senv.people:
            if average_adherence:
                person_average_adherence = average_adherence * person.age_group.masks_max_adherence
                person.masks_usage = np.random.beta(
                    person_average_adherence * person.age_group.masks_adherence_shape,
                    (1.0 - person_average_adherence) * person.age_group.masks_adherence_shape,
                )
            else:
                person.masks_usage = 0.0


class HygieneAdoption(Intervention):

    def apply(self, senv: SimulationEnvironment) -> None:
        average_adherence = self.parameter
        for person in senv.people:
            if average_adherence:
                person_average_adherence = average_adherence * person.age_group.hygiene_max_adherence
                person.hygiene_adoption = np.random.beta(
                    person_average_adherence * person.age_group.hygiene_shape,
                    (1.0 - person_average_adherence) * person.age_group.hygiene_shape,
                )
            else:
                person.hygiene_adoption = 0.0


def get_simulation_day_from_isoformat_date(isoformat_date: str, senv: SimulationEnvironment):
    simulation_day = (
            get_date_from_isoformat(isoformat_date) - get_date_from_isoformat(senv.sim_params.start_date)).days
    return max(0, simulation_day)
