from typing import Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class AgeGroup:
    index: int
    severity: np.array
    isolation_adherence: float
    isolation_effectiveness: float
    diagnosis_delay: Optional[float]  # If set, lab is ignored