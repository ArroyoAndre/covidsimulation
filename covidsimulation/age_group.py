from typing import Optional
from dataclasses import dataclass

import numpy as np


DEFAULT_MASK_TRANSMISSION_REDUCTION = 0.4
DEFAULT_MASK_INFECTION_REDUCTION = 0.2
DEFAULT_MAKS_MAX_ADHERENCE = 0.99
DEFAULT_MASKS_ADHERENCE_SHAPE = 3.0  # Shape of beta distribution of individual adherences
DEFAULT_HYGIENE_EFFECTIVENESS = 0.2
DEFAULT_HYGIENE_MAX_ADHERENCE = 0.99
DEFAULT_HYGIENE_SHAPE = 3.0
DEFAULT_CHANCE_OF_DIAGNOSIS_IF_MODERATE = 0.5

@dataclass
class AgeGroup:
    index: int
    severity: np.array
    isolation_adherence: float
    isolation_effectiveness: float
    diagnosis_delay: Optional[float]  # If set, lab is ignored
    mask_transmission_reduction: float = DEFAULT_MASK_TRANSMISSION_REDUCTION
    mask_infection_reduction: float = DEFAULT_MASK_INFECTION_REDUCTION
    masks_max_adherence: float = DEFAULT_MAKS_MAX_ADHERENCE  # Must be in ]0.0, 1.0[
    masks_adherence_shape: float = DEFAULT_MASKS_ADHERENCE_SHAPE
    hygiene_infection_reduction: float = DEFAULT_HYGIENE_EFFECTIVENESS
    hygiene_max_adherence: float = DEFAULT_HYGIENE_MAX_ADHERENCE
    hygiene_shape: float = DEFAULT_HYGIENE_SHAPE
    chance_of_diagnosis_if_moderate: float = DEFAULT_CHANCE_OF_DIAGNOSIS_IF_MODERATE
