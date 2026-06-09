# actions/params.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ActionParams:
    order_cooldown: int = 6
    order_cooldown_min: int = 2
    order_cooldown_max: int = 8

    step_size: float = 2.5
    step_size_min: float = 1.8
    step_size_max: float = 4.5

    map_margin: float = 2.0
    wall_repulsion: float = 0.4
    attack_range_approx: float = 5.5
    cd_ready_thresh: float = 0.2