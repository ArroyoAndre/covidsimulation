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


from typing import List, Tuple, Dict
import numpy as np
import pickle

from smart_open import open  # Allows for saving into S3 and other cool stuff


CONFIDENCE_RANGE = (20, 80)
DEFAULT_START_DATE = '2020-03-01'


class Stats:
    stats: np.ndarray
    measurements: List[str]
    metrics: Dict[str, Tuple[str, str]]
    population_names: List[str]
    age_str: List[str]
    start_date: str

    def __init__(
            self,
            stats: np.ndarray, 
            measurements: List[str], 
            metrics: Dict[str, Tuple[str, str]],
            population_names: List[str],
            age_str: List[str],
            start_date: str=DEFAULT_START_DATE,
        ):
        self.stats = stats
        self.measurements = measurements
        self.metrics = metrics
        self.population_names = population_names
        self.age_str = age_str
        self.start_date = start_date

    def _get_estatistica(self, indice_metrica, nome_populacao, nome_faixa):
        if nome_populacao:
            indice_populacao = get_index(nome_populacao, self.population_names)
            stats_populacao = self.stats[:, indice_populacao, indice_metrica, :, :]
        else:
            stats_populacao = self.stats[:, :, indice_metrica, :, :].sum(1)
        if nome_faixa:
            indice_faixa = get_index(nome_faixa, self.age_str)
            stats_faixa = stats_populacao[:, indice_faixa, :]
        else:
            stats_faixa = stats_populacao.sum(1)
        return stats_faixa

    def get_metric(
            self, 
            nome_metrica, 
            nome_populacao=None, 
            nome_faixa=None, 
            fatores=None, 
            diaria=False,
            confidence_range=CONFIDENCE_RANGE,
        ):
        numerator_name, denominator_name = self.metrics[nome_metrica]
        indice_metrica = get_index(numerator_name, self.measurements)
        estatistica = self._get_estatistica(indice_metrica, nome_populacao, nome_faixa)
        if denominator_name:
            indice_divisor = get_index(denominator_name, self.measurements)
            divisor = self._get_estatistica(indice_divisor, nome_populacao, nome_faixa)
            estatistica = estatistica / divisor
        elif not fatores is None:
            estatistica *= fatores
        if diaria:
            estatistica = estatistica[:, 1:] - estatistica[:, :-1] 
        return (
            self,
            estatistica.mean(0), 
            np.percentile(estatistica, confidence_range[0], axis=0), 
            np.percentile(estatistica, confidence_range[1], axis=0),
        )

    def save(self, fname):
        if not fname.endswith('.pkl'):
            fname = fname + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname: str):
        if not fname.endswith('.pkl'):
            fname = fname + '.pkl'
        with open(fname, 'rb') as f:
            return pickle.load(f)



def get_index(key, keys):
    for i, value in enumerate(keys):
        if key == value:
            return i
