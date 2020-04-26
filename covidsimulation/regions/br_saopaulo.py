from functools import partial

import numpy as np

from ..age_group import AgeGroup
from ..calibrate import score_reported_deaths
from ..disease_parameters import OUTCOME_THRESHOLDS
from ..early_stop import EarlyStop
from ..intervention import SocialDistancingChange, DiagnosisDelayChange
from ..parameters import Parameters
from ..population import Population
from ..random import UniformParameter, UniformIntParameter
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


ISOLATION_PROPENSITY_PER_AGE = [
    0.0,  # 0
    0.0,  # 1
    -0.5,  # 2
    -0.3,  # 3
    0.2,  # 4
    0.7,  # 5
    0.5,  # 6
    2.0,  # 7
    2.0,  # 8
]


ISOLATION_PROPENSITY_SOCIAL_CLASS_CD = -0.9
ISOLATION_PROPENSITY_SOCIAL_CLASS_E = -1.6
ISOLATION_PROPENSITY_SOCIAL_CLASS_ABC = [
    0.4,  # 0
    0.0,  # 1
    -0.6,  # 2
    -0.5,  # 3
    -0.2,  # 4
    0.3,  # 5
    0.8,  # 6
    1.8,  # 7
    1.8,  # 8
]


PUBLICO_E = Population(
    name='classe_e',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i] + ISOLATION_PROPENSITY_SOCIAL_CLASS_E, 0.7,
                 diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.16, 0.23, 0.26, 0.17, 0.12, 0.06]),
    inhabitants=total_inhabitants * 1 / 12.0,
    geosocial_displacement=0.5,  # deslocamento geográfico
    seed_infections=UniformIntParameter('sp_classe_e_seed', 0, 3),
    isolation_propensity_increase=UniformParameter('sp_classe_e_isolation', -1.5, 1.5),
)

PUBLICO_CD = Population(
    name='classe_c-d',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i], 0.9,
                 diagnosis_delay=18.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.19, 0.25, 0.26, 0.16, 0.10, 0.04]),
    inhabitants=total_inhabitants * 6 / 12.0,
    geosocial_displacement=0.25,  # deslocamento geográfico
    seed_infections=UniformIntParameter('sp_classe_c-d_seed', 1, 15),
    isolation_propensity_increase=UniformParameter('sp_classe_c-d_isolation', -1.5, 2.0),
)

PRIVADO = Population(
    name='classe_abc+',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_SOCIAL_CLASS_ABC[i], 0.95, diagnosis_delay=4.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.21, 0.26, 0.26, 0.15, 0.09, 0.03]),
    inhabitants=total_inhabitants * 5 / 12.0,
    geosocial_displacement=0.0,  # deslocamento geográfico
    seed_infections=10,
    isolation_propensity_increase=UniformParameter('sp_classe_abc+_isolation', -1.5, 1.5),
)

population_segments = [PRIVADO, PUBLICO_CD, PUBLICO_E]

interventions = [
    SocialDistancingChange(0, 0.2),  # 2020-03-13
    SocialDistancingChange(3, 0.4),  # 2020-03-16
    SocialDistancingChange(9, 0.68),  # 2020-03-22
    SocialDistancingChange(16, 0.66),  # 2020-03-29
    SocialDistancingChange(23, 0.62),  # 2020-04-05
    SocialDistancingChange(30, 0.60),  # 2020-04-05
    DiagnosisDelayChange(18, 14.0),  # Reductions in confirmations queue around 2020-04-6 - 16
    DiagnosisDelayChange(25, 10.0),
    DiagnosisDelayChange(33, 5.0),

]

params = Parameters(
    population_segments,
    SimulationConstants(),
    interventions=interventions,
    d0_infections=UniformParameter('sp_d0_infections', 12000, 50000),
    start_date='2020-03-13',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_icu=4000,
    capacity_ventilators=4000,
    min_age_group_initially_infected=4,
)

sp_official_deaths = [
    (0, 0.0),  # 2020-03-13
    (4, 4.0),  # 2020-03-17
    (9, 22.0),  # 2020-03-22
    (16, 86.0),  # 2020-03-29
    (19, 120.0),  # 2020-04-01
#    (23, 251.0),  # 2020-04-05
    (24, 284.0),  # 2020-04-06
#    (25, 343.0),  # 2020-04-07
#    (26, 392.0),  # 2020-04-08
#    (27, 445.0),  # 2020-04-09
    (28, 481.0),  # 2020-04-10
    #    (29, 498.0),  # 2020-04-11
#    (30, 524.0),  # 2020-04-12
    #   (31, 539.0),  # 2020-04-13
    (32, 616.0),  # 2020-04-14  * Estimativa
 #   (33, 673.0),  # 2020-04-15
 #   (34, 728.0),  # 2020-04-16
 #   (35, 783.0),  # 2020-04-17
    (36, 840.0),  # 2020-04-18
    (41, 1135.0),  # 2020-04-23
    (42, 1281.0),  # 2020-04-24
]

score_fn_deaths = partial(score_reported_deaths, expected_deaths=sp_official_deaths)

def score_fn(stats):
    return np.log((stats.get_metric('in_intensive_care', 'classe_abc+')[1][19]+1) /
                  (stats.get_metric('in_intensive_care', 'classe_abc+')[1][31]+1)) ** 2 + score_fn_deaths(stats)


early_stops = [EarlyStop(19, 80, 360)]
