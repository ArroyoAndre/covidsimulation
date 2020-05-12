from functools import partial

import numpy as np

from ..age_group import AgeGroup
from ..calibrate import score_reported_deaths
from ..disease_parameters import OUTCOME_THRESHOLDS
from ..early_stop import EarlyStop
from ..intervention import SocialDistancingChange, DiagnosisDelayChange, HygieneAdoption, MaskUsage
from ..parameters import Parameters
from ..population import Population
from ..random import LogUniformParameter, UniformParameter, UniformIntParameter, TriangularParameter
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
                 diagnosis_delay=14.0, chance_of_diagnosis_if_moderate=0.0, hygiene_max_adherence=0.33)
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
                 diagnosis_delay=14.0, chance_of_diagnosis_if_moderate=0.15)
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
    SocialDistancingChange('2020-03-13', 0.2),
    SocialDistancingChange('2020-03-16', 0.4),
    SocialDistancingChange('2020-03-22', 0.68),
    SocialDistancingChange('2020-03-29', 0.66),
    SocialDistancingChange('2020-04-05', 0.62),
    SocialDistancingChange('2020-04-12', 0.60),
    SocialDistancingChange('2020-04-24', 0.55),
    DiagnosisDelayChange('2020-04-06', 10.0),  # Reductions in confirmations queue around 2020-04-6 - 16
    DiagnosisDelayChange('2020-04-15', 8.0),
    DiagnosisDelayChange('2020-04-22', 6.5),
    HygieneAdoption('2020-03-13', TriangularParameter('hygiene_adoption', 0.5, 0.7, 0.9)),
    MaskUsage('2020-03-29', TriangularParameter('mask_adoption', 0.5, 0.7, 0.9) * 0.5),
    MaskUsage('2020-04-24', TriangularParameter('mask_adoption', 0.5, 0.7, 0.9)),
]

params = Parameters(
    population_segments,
    SimulationConstants(),
    interventions=interventions,
    d0_infections=LogUniformParameter('sp_d0_infections', 12000, 50000),
    start_date='2020-03-13',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_icu=4000,
    capacity_ventilators=4000,
    min_age_group_initially_infected=4,
)

sp_official_deaths = [
    ('2020-03-13', 0.0),
    ('2020-03-17', 4.0),
    ('2020-03-22', 22.0),
    ('2020-03-29', 86.0),
    ('2020-04-01', 120.0),
    ('2020-04-06', 284.0),
    ('2020-04-10', 481.0),
    ('2020-04-18', 840.0),
    ('2020-04-23', 1135.0),
    ('2020-04-24', 1281.0),
    ('2020-04-28', 1728.0),
    ('2020-05-05', 2425.0),
]

score_fn_deaths = partial(score_reported_deaths, expected_deaths=sp_official_deaths)


def score_fn(stats):
    return np.log((stats.get_metric('in_intensive_care', 'classe_abc+')[1][19] + 1) /
                  (stats.get_metric('in_intensive_care', 'classe_abc+')[1][31] + 1)) ** 2 + score_fn_deaths(stats)


early_stops = [EarlyStop(24, 150, 500)]
