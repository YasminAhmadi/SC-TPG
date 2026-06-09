# tasks/spec.py 里（或 features/spec.py 都行）
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class UnitObs:
    tag: int
    base_feats: np.ndarray
    hp: float
    cd_norm: float
    enemy_density: float
    bane_threat: float
    ling_threat: float
