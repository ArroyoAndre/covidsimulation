from functools import partial

import numpy as np

from ..age_group import AgeGroup
from ..calibrate import score_reported_deaths
from ..disease_parameters import OUTCOME_THRESHOLDS
from ..early_stop import EarlyStop
from ..intervention import SocialDistancingChange, HygieneAdoption, MaskUsage
from ..parameters import Parameters
from ..population import Population
from ..random import LogUniformParameter, UniformParameter, UniformIntParameter, TriangularParameter
from ..simulation import SimulationConstants

age_structure = {
    '0-9': 0.128,
    '10-19': 0.159,
    '20-29': 0.167,
    '30-39': 0.158,
    '40-49': 0.139,
    '50-59': 0.115,
    '60-69': 0.070,
    '70-79': 0.045,
    '80+': 0.019,
}

total_inhabitants = 11957793

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


ISOLATION_PROPENSITY_SOCIAL_CLASS_CD = -0.5
ISOLATION_PROPENSITY_SOCIAL_CLASS_E = -1.4
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
                 diagnosis_delay=18.0, chance_of_diagnosis_if_moderate=0.0, hygiene_max_adherence=0.33)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.16, 0.23, 0.26, 0.17, 0.12, 0.06]),
    inhabitants=total_inhabitants * 0.13,
    geosocial_displacement=0.5,  # deslocamento geográfico
    seed_infections=0,
    isolation_propensity_increase=UniformParameter('rj_classe_e_isolation', -1.5, 1.5),
)

PUBLICO_CD = Population(
    name='classe_c-d',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i], 0.9,
                 diagnosis_delay=18.0, chance_of_diagnosis_if_moderate=0.15)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.19, 0.25, 0.26, 0.16, 0.10, 0.04]),
    inhabitants=total_inhabitants * 0.54,
    geosocial_displacement=0.25,  # deslocamento geográfico
    seed_infections=UniformIntParameter('rj_classe_c-d_seed', 1, 12),
    isolation_propensity_increase=UniformParameter('rj_classe_c-d_isolation', -1.5, 2.0),
)

PRIVADO = Population(
    name='classe_abc+',
    age_probabilities=np.array(list(age_structure.values())),
    age_groups=[
        AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_SOCIAL_CLASS_ABC[i], 0.95, diagnosis_delay=4.0)
        for i, nome_faixa in enumerate(age_structure.keys())
    ],
    home_size_probabilities=np.array([0.21, 0.26, 0.26, 0.15, 0.09, 0.03]),
    inhabitants=total_inhabitants * 0.33,
    geosocial_displacement=0.0,  # deslocamento geográfico
    seed_infections=10,
    isolation_propensity_increase=UniformParameter('rj_classe_abc+_isolation', -1.5, 1.5),
)

population_segments = [PRIVADO, PUBLICO_CD, PUBLICO_E]

interventions = [
    SocialDistancingChange(0, 0.2),  # 2020-03-14
    SocialDistancingChange(2, 0.4),  # 2020-03-16
    SocialDistancingChange(8, 0.6),  # 2020-03-22
    SocialDistancingChange(10, 0.68),  # 2020-03-24
    SocialDistancingChange(15, 0.66),  # 2020-03-29
    SocialDistancingChange(23, 0.62),  # 2020-04-05
    SocialDistancingChange(30, 0.60),  # 2020-04-05
    SocialDistancingChange(41, 0.55),  # 2020-04-23
#    DiagnosisDelayChange(18, 14.0),  # Reductions in confirmations queue around 2020-04-6 - 16
#    DiagnosisDelayChange(25, 10.0),
#    DiagnosisDelayChange(33, 5.0),
    HygieneAdoption(0, TriangularParameter('hygiene_adoption', 0.5, 0.7, 0.9)),
    MaskUsage(15, TriangularParameter('mask_adoption', 0.5, 0.7, 0.9) * 0.5),
    MaskUsage(39, TriangularParameter('mask_adoption', 0.5, 0.7, 0.9)),
]

params = Parameters(
    population_segments,
    SimulationConstants(),
    interventions=interventions,
    d0_infections=LogUniformParameter('rj_d0_infections', 3000, 10000),
    start_date='2020-03-14',
    capacity_hospital_max=50000,
    capacity_hospital_beds=int(21000 * 0.8),
    capacity_icu=int(3734 * 0.8),
    capacity_ventilators=int(4327 * 0.8),
    min_age_group_initially_infected=4,
)

rj_official_deaths = [
    (0, 0.0),  # 2020-03-14
    (5, 2.0),  # 2020-03-19
    (21, 54.0),  # 2020-03-04
    (27, 128.0),  # 2020-04-10
    (33, 243.0),  # 2020-04-16
    (38, 360.0),  # 2020-04-21
]

score_fn_deaths = partial(score_reported_deaths, expected_deaths=rj_official_deaths)

def score_fn(stats):
    return np.log((stats.get_metric('in_intensive_care', 'classe_abc+')[1][19]+1) /
                  (stats.get_metric('in_intensive_care', 'classe_abc+')[1][31]+1) / 1.2) ** 2 + score_fn_deaths(stats)


early_stops = [EarlyStop(27, 54, 200)]
