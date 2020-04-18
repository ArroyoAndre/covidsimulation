import numpy as np

from ..age_group import AgeGroup
from ..disease_parameters import OUTCOME_THRESHOLDS
from ..intervention import SocialDistancingChange, DiagnosisDelayChange
from ..parameters import Parameters
from ..population import Population
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
    -0.8,  # 0
    -0.9,  # 1
    -0.9,  # 2
    -0.9,  # 3
    -0.5,  # 4
    -0.1,  # 5
    -0.3,  # 6
    1.0,  # 7
    1.0,  # 8
]

PROPENSAO_ISOLAMENTO_PUBLICO_CD = 0.0
PROPENSAO_ISOLAMENTO_PUBLICO_E = -1.2

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

PUBLICO_E = Population(
    name='classe_e',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i] + PROPENSAO_ISOLAMENTO_PUBLICO_E, 0.7,
                 diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([
        0.3,  # 1p
        0.25,  # 2p
        0.25,  # 3p
        0.2,  # 4p
    ]),
    inhabitants=total_inhabitants * 1 / 12.0,
    geosocial_displacement=0.6,  # deslocamento geográfico
    seed_infections=1,
)

PUBLICO_CD = Population(
    name='classe_c-d',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_FAIXA[i], 0.9,
                 diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([
        0.3,  # 1p
        0.25,  # 2p
        0.25,  # 3p
        0.2,  # 4p
    ]),
    inhabitants=total_inhabitants * 6 / 12.0,
    geosocial_displacement=0.2,  # deslocamento geográfico
    seed_infections=6,
)

PRIVADO = Population(
    name='classe_abc+',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], PROPENSAO_ISOLAMENTO_PRIVADO[i], 0.95, diagnosis_delay=4.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([
        0.3,  # 1p
        0.3,  # 2p
        0.25,  # 3p
        0.15,  # 4p
    ]),
    inhabitants=total_inhabitants * 5 / 12.0,
    geosocial_displacement=0.0,  # deslocamento geográfico
    seed_infections=10,
)

population_segments = [PRIVADO, PUBLICO_CD, PUBLICO_E]

interventions = [
    SocialDistancingChange(0, 0.2),  # 2020-03-13
    SocialDistancingChange(3, 0.45),  # 2020-03-16
    SocialDistancingChange(9, 0.68),  # 2020-03-22
    SocialDistancingChange(16, 0.66),  # 2020-03-29
    SocialDistancingChange(23, 0.62),  # 2020-04-05
    DiagnosisDelayChange(18, 14.0),  # Reductions in confirmations queue around 2020-04-6 - 16
    DiagnosisDelayChange(25, 10.0),
    DiagnosisDelayChange(33, 7.0),
]

params = Parameters(
    population_segments,
    SimulationConstants(),
    interventions=interventions,
    d0_infections=20000,
    start_date='2020-03-13',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_intensive_care=4000,
    capacity_ventilators=4000,
    transmission_scale_days=0.3,
    min_age_group_initially_infected=4,
)

sp_official_deaths = [
    (0, 0.0),  # 2020-03-13
    (4, 4.0),  # 2020-03-17
    (9, 22.0),  # 2020-03-22
    (16, 86.0),  # 2020-03-29
    (19, 120.0),  # 2020-04-01
    (23, 251.0),  # 2020-04-05
    (24, 284.0),  # 2020-04-06
    (25, 343.0),  # 2020-04-07
    (26, 392.0),  # 2020-04-08
    (27, 445.0),  # 2020-04-09
    (28, 481.0),  # 2020-04-10
    #    (29, 498.0),  # 2020-04-11
    (30, 524.0),  # 2020-04-12
    #   (31, 539.0),  # 2020-04-13
    (32, 616.0),  # 2020-04-14  * Estimativa
    (33, 673.0),  # 2020-04-15
    (34, 728.0),  # 2020-04-16
    (35, 783.0),  # 2020-04-17
]
