# rewards/beacon.py
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from rewards.params import RewardParams
from utils import marine_units


DEBUG_REWARD = False
BEACON_UNIT_TYPE_ID = 317
ALLIANCE_NEUTRAL = 3

class RewardCalculator:
    def __init__(self, params: RewardParams):
        self.p = params
        self.prev_d = None # prev distance to beacon
        self.init_d = None

    def _find_beacon_pos(self, bot):
        obs = getattr(getattr(bot, "state", None), "observation", None)
        raw = getattr(obs, "raw_data", None) if obs is not None else None
        units = getattr(raw, "units", None) if raw is not None else None
        if units is None:
            return None

        for u in units:
            if int(getattr(u, "alliance", -1)) == ALLIANCE_NEUTRAL and int(getattr(u, "unit_type", -1)) == BEACON_UNIT_TYPE_ID:
                p = getattr(u, "pos", None)
                if p is not None:
                    return Point2((float(p.x), float(p.y)))
        return None

    def reset(self, bot):
        ms = marine_units(bot)
        beacon = self._find_beacon_pos(bot)

        if ms and ms.exists and beacon is not None:
            bx, by = beacon
            d0 = float(ms.center.distance_to(beacon))
            self.prev_d = d0
            self.init_d = d0
        else:
            self.prev_d = None
            self.init_d = None

        if DEBUG_REWARD:
            print(f"[BEACON-REW-RESET] init_d={self.init_d}")

    def calculate_step_reward(self, bot, move_ratio: float) -> float:
        reward = 0.0
        ms = marine_units(bot)
        if not ms or not ms.exists:
            return 0.0

        beacon = self._find_beacon_pos(bot)
        if beacon is None:
            return 0.0

        d_now = float(ms.center.distance_to(beacon))

        prev_d = self.prev_d
        if prev_d is not None:
            progress = float(prev_d - d_now)
            reward += self.p.R_DELDIST * progress
        else:
            progress = 0.0

        self.prev_d = d_now

        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        cx = float(ms.center.x)
        cy = float(ms.center.y)
        dmin = min(cx, cy, W - cx, H - cy) / max(W, H)
        reward += -self.p.R_EDGE_PEN * (1.0 - float(dmin))

        d_norm = float(d_now / max(W, H))
        if d_norm < self.p.MOVE_REWARD_DIST_THRESH and progress > 0.0:
            reward += self.p.R_MOVE_NEAR_ENEMY * float(move_ratio)

        return float(reward)

    def calculate_final_reward(self, bot, reason: str, steps: int) -> float:
        ms = marine_units(bot)
        beacon = self._find_beacon_pos(bot)

        if not ms or not ms.exists or beacon is None:
            return 0.0

        bx, by = beacon
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        d = float(ms.center.distance_to(beacon))
        d_norm = float(d / max(W, H))

        score = 0.0
        if reason == "win":
            # win bonus
            score += float(self.p.R_ALIVE_BONUS)
        elif ("timeout" in reason) or ("force_reset" in reason):
            # Timeout: The further away from the target, the greater the penalty.
            score -= float(self.p.R_TIMEOUT_BASE) * (1.0 + 6.0 * d_norm)

        # distance penalty, to encourage getting closer to beacon even if timeout
        score -= 60.0 * d_norm

        # time penalty, to encourage faster completion
        score -= 0.0015 * float(steps)

        if DEBUG_REWARD:
            print("[BEACON-REW-FINAL]",
                  "reason=", reason,
                  "steps=", steps,
                  "d_norm=", d_norm,
                  "score=", score)

        return float(score)
