# Copyright 2020 André Arroyo and contributors
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
# promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Dict, Iterable, Optional, Callable, Union
import numpy as np
from pathlib import Path
import pickle

from scipy.signal import savgol_filter
from smart_open import open  # Allows for saving into S3 and other cool stuff

from .series import Series
from .random import RandomParametersState

CONFIDENCE_RANGE = (10.0, 90.0)
DEFAULT_START_DATE = '2020-03-01'
DEFAULT_FILTER_WINDOW = 9
DEFAULT_FILTER_ORDER = 3
CONSTANT_METRICS = ['population']  # Metrics that don't change in daily view
EPS = 1e-8  # To avoid division by zero


class Stats:
    stats: np.ndarray
    random_states: List[RandomParametersState]
    measurements: List[str]
    metrics: Dict[str, Tuple[str, str]]
    population_names: Iterable[str]
    age_str: List[str]
    start_date: str
    filter_indices: Optional[Iterable]

    def __init__(
            self,
            stats: np.ndarray,
            random_states: List[RandomParametersState],
            measurements: List[str],
            metrics: Dict[str, Tuple[str, str]],
            population_names: List[str],
            age_str: List[str],
            start_date: str = DEFAULT_START_DATE,
            filter_indices: Optional[Iterable] = None,
    ):
        self.stats = stats
        self.random_states = random_states
        self.measurements = measurements
        self.metrics = metrics
        self.population_names = population_names
        self.age_str = age_str
        self.start_date = start_date
        self.filter_indices = filter_indices

    def _get_measurement(self,
                         metric_name: str,
                         population_names: Optional[Union[str, Iterable[str]]],
                         age_group_names: Optional[Union[str, Iterable[str]]],
                         daily: bool = False,
                         ) -> np.ndarray:
        metric_index = get_index(metric_name, self.measurements)
        if population_names:
            if isinstance(population_names, str):
                population_names = [population_names]
            stats_populacao = np.zeros([self.stats.shape[0], self.stats.shape[3], self.stats.shape[4]])
            for n in population_names:
                indice_populacao = get_index(n, self.population_names)
                stats_populacao += self.stats[:, indice_populacao, metric_index, :, :]
        else:
            stats_populacao = self.stats[:, :, metric_index, :, :].sum(1)
        if age_group_names:
            if isinstance(age_group_names, str):
                age_group_names = [age_group_names]
            stats_faixa = np.zeros([self.stats.shape[0], self.stats.shape[4]])
            for n in age_group_names:
                indice_faixa = get_index(n, self.age_str)
                stats_faixa += stats_populacao[:, indice_faixa, :]
        else:
            stats_faixa = stats_populacao.sum(1)
        if daily:
            if metric_name not in CONSTANT_METRICS:
                stats_faixa = stats_faixa[:, 1:] - stats_faixa[:, :-1]
            else:
                stats_faixa = stats_faixa[:, :-1]
        return stats_faixa

    def _get_metric_raw(self, metric_name, population_names, age_group_names, fatores, daily) -> np.ndarray:
        numerator_name, denominator_name = self.metrics[metric_name]
        estatistica = self._get_measurement(numerator_name, population_names, age_group_names, daily)
        if denominator_name:
            divisor = self._get_measurement(denominator_name, population_names, age_group_names, daily)
            estatistica = estatistica / (divisor + EPS)
        elif not fatores is None:
            estatistica *= fatores
        return estatistica

    def get_metric(
            self,
            metric_name,
            population_names: Optional[Union[str, Iterable[str]]] = None,
            age_group_names: Optional[Union[str, Iterable[str]]] = None,
            normalization_multipliers: np.ndarray = None,
            daily: bool = False,
            confidence_range: Tuple[float, float] = CONFIDENCE_RANGE,
            filter_noise: bool = True,
    ):
        metric = self._get_metric_raw(metric_name, population_names, age_group_names, normalization_multipliers, daily)
        if self.filter_indices is not None:
            metric = metric[self.filter_indices, :]
        if filter_noise:
            filter = partial(savgol_filter, window_length=DEFAULT_FILTER_WINDOW, polyorder=DEFAULT_FILTER_ORDER)
        else:
            filter = lambda x: x
        return MetricResult(
            self,
            Series(filter(np.median(metric, axis=0)), start_date=self.start_date),
            Series(filter(np.percentile(metric, confidence_range[0], axis=0)), start_date=self.start_date),
            Series(filter(np.percentile(metric, confidence_range[1], axis=0)), start_date=self.start_date),
        )

    def save(self, fname):
        if not fname.endswith('.pkl'):
            fname = fname + '.pkl'
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname: str):
        if not fname.endswith('.pkl'):
            fname = fname + '.pkl'
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def filter_best_scores(self, scoring_fn: Callable, fraction_to_keep: float = 0.1):
        scores = []
        for i in range(self.stats.shape[0]):
            stats_i = copy(self)
            stats_i.filter_indices = [i]
            scores.append(scoring_fn(stats_i))
        sorted_indices = np.argsort(np.array(scores))
        num_best = self.stats.shape[0] - int((1.0 - fraction_to_keep) * self.stats.shape[0])
        self.filter_indices = sorted_indices[:num_best]


@dataclass
class MetricResult:
    stats: Stats
    mean: Series
    low: Series
    high: Series

    def __len__(self):
        return 3

    def __getitem__(self, item):
        return (self.mean, self.low, self.high)[item]


def get_index(key, keys):
    for i, value in enumerate(keys):
        if key == value:
            return i
