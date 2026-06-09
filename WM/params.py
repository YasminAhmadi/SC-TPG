# wm/params.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class WMTokenCodes:
    # token type codes
    TYPE_MARINE: float = 0.0
    TYPE_ZERGLING: float = 1.0
    TYPE_BANELING: float = 2.0

    # token side codes
    SIDE_SELF: float = 0.0
    SIDE_ENEMY: float = 1.0


@dataclass(frozen=True)
class WMRecorderParams:
    max_marines: int = 9
    max_enemies: int = 10
    canonical_side: str = "right"  # "right" or "left"

    hp_marine_max: float = 45.0
    hp_zergling_max: float = 35.0
    hp_baneling_max: float = 30.0

    save_npz: bool = True
    debug: bool = False

    codes: WMTokenCodes = WMTokenCodes()
