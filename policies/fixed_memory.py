# policies/fixed_memory.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

from policies.base import MemoryPolicyBase, DiscreteActionSpace

from tasks.spec import UnitObs


class FixedMemoryPolicy(MemoryPolicyBase):
    name = "fixed"

    def __init__(self, env_n: int, mem_dim: int = 3, base_dim: int = 18):
        super().__init__(
            mem_dim=mem_dim,
            base_dim=base_dim,
            action_space=DiscreteActionSpace(env_n),
            mem_action_space=None, 
        )
        self._prev_hp: Dict[int, float] = {}

    def _reset_extra(self) -> None:
        self._prev_hp.clear()

    def apply_memory_action(self, obs: UnitObs, mem_action: Optional[int]) -> None:

        tag = obs.tag

        hp = float(obs.hp)
        prev_hp = float(self._prev_hp.get(tag, hp))
        delta_hp = max(0.0, prev_hp - hp)
        self._prev_hp[tag] = hp

        mem = self.get_mem(tag)
        # print("mem:", mem)
        panic, cyc, haz = float(mem[0]), float(mem[1]), float(mem[2])

        P_DECAY = 0.9
        P_GAIN_HP = 0.5
        P_GAIN_THREAT = 0.2

        damage_term = min(1.0, delta_hp / 10.0)
        threat_term = float(0.5 * obs.bane_threat + 0.3 * obs.ling_threat)

        panic = P_DECAY * panic + P_GAIN_HP * damage_term + P_GAIN_THREAT * threat_term
        panic = float(np.clip(panic, 0.0, 2.0))

        C_DECAY = 0.5
        cyc = C_DECAY * cyc + (1.0 - C_DECAY) * obs.cd_norm
        cyc = float(np.clip(cyc, 0.0, 1.0))

        H_DECAY = 0.9
        danger_now = (
            0.4 * obs.enemy_density +
            0.4 * obs.bane_threat +
            0.2 * obs.ling_threat +
            0.3 * damage_term
        )
        haz = H_DECAY * haz + (1.0 - H_DECAY) * danger_now
        haz = float(np.clip(haz, 0.0, 2.0))
        
        # print("panic:", panic)
        # print("cyc:", cyc)
        # print("phaz:", haz)

        self.set_mem(tag, np.array([panic, cyc, haz], dtype=np.float32))
