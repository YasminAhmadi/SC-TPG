# rewards\params.py
from dataclasses import dataclass

# Reward parameters. Adjust these to shape the reward function as desired.
@dataclass(frozen=True)
class RewardParams:
    R_FRAC_ENEMY: float = 120.0
    R_FRAC_OUR: float = 110.0
    R_DEAD_FINAL: float = 8.0
    R_SURVIVE: float = 1.0
    R_DELDIST: float = 0.3
    R_DMG_DEALT: float = 0.5
    R_DMG_TAKEN: float = 0.10
    R_KILL_ENEMY: float = 10.0
    R_LOSS_MARINE: float = 10.0
    R_EDGE_PEN: float = 0.05
    R_TIMEOUT_BASE: float = 30.0
    R_ALIVE_BONUS: float = 30.0
    R_ALL_DEAD_PEN: float = 40
    R_MOVE_NEAR_ENEMY: float = 0.08
    MOVE_REWARD_DIST_THRESH: float = 0.35
