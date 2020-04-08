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


ISOLATION_PROPENSITY_PER_AGE = [
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


ISOLATION_PROPENSITY_SOCIAL_CLASS_CD = -0.4
ISOLATION_PROPENSITY_SOCIAL_CLASS_E = -1.2


SOCIAL_CLASS_E = {
    'age_probabilities': np.array(list(age_structure.values())),
    'age_risk': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i] + ISOLATION_PROPENSITY_SOCIAL_CLASS_E, 0.75)
        for i, age in enumerate(age_structure.keys())
    ],
    'house_sizes': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'inhabitants': total_inhabitants * 1 / 12.0,
    'shift': 1.6,  # geographic shift
    'initially_infected': 0,
}


SOCIAL_CLASS_CD = {
    'age_probabilities': np.array(list(age_structure.values())),
    'age_risk': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i] + ISOLATION_PROPENSITY_SOCIAL_CLASS_CD, 0.92)
        for i, age in enumerate(age_structure.keys())
    ],
    'house_sizes': np.array([
      0.3,   # 1p
      0.25,  # 2p
      0.25,  # 3p
      0.2,   # 4p
    ]),
    'inhabitants': total_inhabitants * 8 / 12.0,
    'shift': 0.8,
    'initially_infected': 1,
}


SOCIAL_CLASS_AB = {
    'age_probabilities': np.array(list(age_structure.values())),
    'age_risk': [
      AgeGroup(i, OUTCOME_THRESHOLDS[i], ISOLATION_PROPENSITY_PER_AGE[i], 0.95)
        for i, age in enumerate(age_structure.keys())
    ],
    'house_sizes': np.array([
      0.3,   # 1p
      0.3,   # 2p
      0.25,  # 3p
      0.15,  # 4p
    ]),
    'inhabitants': total_inhabitants * 3 / 12.0,
    'shift': 0.0,
    'initially_infected': 6,
}

population_segments = {'class_ab': SOCIAL_CLASS_AB, 'class_cd': SOCIAL_CLASS_CD, 'class_e': SOCIAL_CLASS_E}

params = Parameters(
    population_segments,
    SimulationConstants(),
    d0_infections=3400,
    start_date='2020-03-06',
    capacity_hospital_max=60000,
    capacity_hospital_beds=20000,
    capacity_icu=4000,
    capacity_ventilators=4000,
    transmission_scale_days=0.3,
    min_age_group_initially_infected=4,
)
