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
    -0.2,  # 2
    -0.3,  # 3
    0.0,  # 4
    0.2,  # 5
    0.8,  # 6
    1.6,  # 7
    1.6,  # 8
]


PROPENSAO_ISOLAMENTO_PUBLICO_CD = -0.4
PROPENSAO_ISOLAMENTO_PUBLICO_E = -1.2


CLASSE_E = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i] + PROPENSAO_ISOLAMENTO_PUBLICO_E, 0.75)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'habitantes': total_inhabitants * 1 / 12.0,
    'deslocamento': 1.6,  # deslocamento geográfico
    'infectados_iniciais': 0,
}


CLASSE_CD = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i] + PROPENSAO_ISOLAMENTO_PUBLICO_CD, 0.92)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'habitantes': total_inhabitants * 8 / 12.0,
    'deslocamento': 0.8,  # deslocamento geográfico
    'infectados_iniciais': 1,
}


CLASSE_AB = {
    'probabilidade_faixas': np.array(list(age_structure.values())),
    'risco_faixas': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i], 0.95)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    'tamanho_casas': np.array([
      0.3,   # 1p
      0.3,   # 2p
      0.25,  # 3p
      0.15,  # 4p
    ]),
    'habitantes': total_inhabitants * 3 / 12.0,
    'deslocamento': 0.0,  # deslocamento geográfico
    'infectados_iniciais': 6,
}

population_segments = {'classe_ab': CLASSE_AB, 'classe_cd': CLASSE_CD, 'classe_e': CLASSE_E}

params = Parameters(
    population_segments,
    SimulationConstants(),
    d0_infections=2500,
    start_date='2020-03-06',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_intensive_care=4000,
    capacity_ventilators=4000,
    transmission_scale_days=0.3,
    min_age_group_initially_infected=4,
)
