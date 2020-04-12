
from typing import Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class AgeGroup:
    indice_faixa: int
    severidades: np.ndarray
    adesao_isolamento: float
    efetividade_isolamento : float
    diagnosis_delay: Optional[float] = None  # if set, lab is ignored
