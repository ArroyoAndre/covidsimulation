import numpy as np

from ..age_group import AgeGroup
from ..disease_parameters import OUTCOME_THRESHOLDS
from ..parameters import Parameters
from ..simulation import SimulationConstants


age_structure = {
    '0-9': 0.13,
    '10-19': 0.152,
    '20-29': 0.184,
    '30-39': 0.169,
    '40-49': 0.140,
    '50-59': 0.107,
    '60-69': 0.064,
    '70-79': 0.036,
    '80+': 0.018,
}


total_inhabitants = 20000000


PROPENSAO_ISOLAMENTO_FAIXA = [
    -0.2,  # 0
    -0.4,  # 1
    -0.4,  # 2
    -0.4,  # 3
    0.0,  # 4
    0.4,  # 5
    1.2,  # 6
    2.5,  # 7
    2.5,  # 8
]


PROPENSAO_ISOLAMENTO_PUBLICO_CD = -2.2
PROPENSAO_ISOLAMENTO_PUBLICO_E = -2.8


PROPENSAO_ISOLAMENTO_PRIVADO = [
    0.6,  # 0
    0.0,  # 1
    -0.4,  # 2
    0.0,  # 3
    0.4,  # 4
    1.0,  # 5
    1.5,  # 6
    2.5,  # 7
    2.5,  # 8
]


PUBLICO_E = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i] + PROPENSAO_ISOLAMENTO_PUBLICO_E, 0.7,
               diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'habitantes': total_inhabitants * 1 / 12.0,
    'deslocamento': 1.2,  # deslocamento geográfico
    'infectados_iniciais': 1,
}


PUBLICO_CD = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i] + PROPENSAO_ISOLAMENTO_PUBLICO_CD, 0.9,
               diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'habitantes': total_inhabitants * 6 / 12.0,
    'deslocamento': 0.6,  # deslocamento geográfico
    'infectados_iniciais': 8,
}


PRIVADO = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_PRIVADO[i], 0.95, diagnosis_delay=4.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.3,   # 2p
      0.25,  # 3p
      0.15,  # 4p
    ]),
    'habitantes': total_inhabitants * 5 / 12.0,
    'deslocamento': 0.0,  # deslocamento geográfico
    'infectados_iniciais': 10,
}

population_segments = {'classe_abc+': PRIVADO, 'classe_c-d': PUBLICO_CD, 'publico_e': PUBLICO_E}


distancing = [
    (0, 0.2), # 2020-03-13
    (3, 0.45), # 2020-03-16
    (9, 0.68), # 2020-03-22
    (16, 0.66), # 2020-03-29
    (23, 0.62), # 2020-04-05
]


params = Parameters(
    population_segments,
    SimulationConstants(),
    distancing=distancing,
    d0_infections=20000,
    start_date='2020-03-13',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_intensive_care=4000,
    capacity_ventilators=4000,
    transmission_scale_days=0.3,
    min_age_group_initially_infected=4,
)
