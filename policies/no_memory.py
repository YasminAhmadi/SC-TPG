# policies/no_memory.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

from policies.base import MemoryPolicyBase, DiscreteActionSpace

from tasks.spec import UnitObs


class NoMemoryPolicy(MemoryPolicyBase):
    name = "nomemory"

    def __init__(self, env_n: int, mem_dim: int = 0, base_dim: int = 18):
        super().__init__(
            mem_dim=mem_dim,
            base_dim=base_dim,
            action_space=DiscreteActionSpace(env_n),
            mem_action_space=None, # fixed memory don't have mem_action
        )
        self._prev_hp: Dict[int, float] = {}

    def _reset_extra(self) -> None:
        self._prev_hp.clear()

    def apply_memory_action(self, obs: UnitObs, mem_action: Optional[int]) -> None:
        # never used because mem_dim=0
        return
