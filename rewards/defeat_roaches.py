# rewards/defeat_roaches.py
import numpy as np
from sc2.position import Point2
from rewards.params import RewardParams

from utils import (
    marine_units,
    sum_hp,
    sum_marine_hp,
    count_marines,
)

DEBUG_REWARD = False


def _enemy_types(bot):
    task = getattr(bot, "task", None)
    types = getattr(task, "ENEMY_BIO_TYPES", None) if task is not None else None
    if not types:
        return {}
    return set(types)


def _enemy_units_filtered(bot):
    types = _enemy_types(bot)
    if not types:
        return [u for u in bot.enemy_units if float(getattr(u, "health", 0.0)) > 0.0]
    return [
        u for u in bot.enemy_units
        if (u.type_id in types) and float(getattr(u, "health", 0.0)) > 0.0
    ]


def _sum_enemy_hp_filtered(bot) -> float:
    es = _enemy_units_filtered(bot)
    return float(sum(float(getattr(u, "health", 0.0)) for u in es))


def _count_enemy_filtered(bot) -> int:
    return int(len(_enemy_units_filtered(bot)))


def _enemy_centroid_filtered(bot):
    es = _enemy_units_filtered(bot)
    if not es:
        return None
    xs = np.array([float(u.position.x) for u in es], dtype=np.float32)
    ys = np.array([float(u.position.y) for u in es], dtype=np.float32)
    return Point2((float(xs.mean()), float(ys.mean())))


class RewardCalculator:
    """
    Cleaner DefeatRoaches reward:
    step: damage trade + kill/loss + range-band shaping + stall penalty
    final: progress + outcome + alive bonus
    """
    def __init__(self, params: RewardParams):
        self.p = params

        self.prev_enemy_hp = 0.0
        self.prev_marine_hp = 0.0
        self.prev_enemy_count = 0
        self.prev_marine_count = 0
        self.last_d_to_enemy = None

        self.init_enemy_hp = 0.0
        self.init_marine_hp = 0.0

    def reset(self, bot):
        ms = marine_units(bot)

        self.prev_enemy_hp = float(_sum_enemy_hp_filtered(bot))
        self.prev_marine_hp = float(sum_hp(ms))
        self.prev_enemy_count = int(_count_enemy_filtered(bot))
        self.prev_marine_count = int(ms.amount)
        self.last_d_to_enemy = None

        self.init_enemy_hp = float(self.prev_enemy_hp)
        self.init_marine_hp = float(self.prev_marine_hp)

        if DEBUG_REWARD:
            print(
                f"[REW-RESET-R] init_enemy_hp={self.init_enemy_hp:.3f} "
                f"init_marine_hp={self.init_marine_hp:.3f}"
            )

    def calculate_step_reward(self, bot, move_ratio: float) -> float:
        reward = 0.0
        ms = marine_units(bot)

        cur_e_hp = float(_sum_enemy_hp_filtered(bot))
        cur_m_hp = float(sum_hp(ms))
        cur_e_cnt = int(_count_enemy_filtered(bot))
        cur_m_cnt = int(ms.amount)

        # 1) damage trade
        dealt = max(0.0, self.prev_enemy_hp - cur_e_hp)
        taken = max(0.0, self.prev_marine_hp - cur_m_hp)

        reward += self.p.R_DMG_DEALT * dealt
        reward -= self.p.R_DMG_TAKEN * taken

        # 2) kills / losses
        killed = max(0, self.prev_enemy_count - cur_e_cnt)
        lost = max(0, self.prev_marine_count - cur_m_cnt)

        if killed > 0:
            reward += self.p.R_KILL_ENEMY * float(killed)
        if lost > 0:
            reward -= self.p.R_LOSS_MARINE * float(lost)

        # 3) range-band shaping
        e_cent = _enemy_centroid_filtered(bot)
        if e_cent is not None and ms.exists:
            d_now = float(ms.center.distance_to(e_cent))

            if self.last_d_to_enemy is not None and dealt < 1e-3:
                # https://starcraft.fandom.com/wiki/Marine_(StarCraft_II)
                engage_dist = 6.0  # 5/6. attack range of marine attack roach
                err_prev = abs(self.last_d_to_enemy - engage_dist)
                err_now = abs(d_now - engage_dist)
                reward += self.p.R_DELDIST * float(err_prev - err_now)

            self.last_d_to_enemy = d_now

        # 4) stall penalty
        if self.last_d_to_enemy is not None:
            d_norm = float(
                self.last_d_to_enemy / max(bot.game_info.map_size.x, bot.game_info.map_size.y)
            )
            R_STALL = float(getattr(self.p, "R_STALL", 0.08))
            if (d_norm > 0.45) and (dealt < 1e-3):
                reward -= R_STALL * float(d_norm - 0.45)

        # update prev
        self.prev_enemy_hp = cur_e_hp
        self.prev_marine_hp = cur_m_hp
        self.prev_enemy_count = cur_e_cnt
        self.prev_marine_count = cur_m_cnt

        return float(reward)

    # Using both step and final rewards. Turn this off to only using step reward.
    def calculate_final_reward(self, bot, reason: str, steps: int) -> float:
        snap = getattr(bot, "_rew_ov_final", None)

        if snap is not None:
            cur_enemy_hp = float(snap.get("enemy_hp", 0.0))
            cur_marine_hp = float(snap.get("marine_hp", 0.0))
            alive_marines = int(snap.get("alive_marines", 0))
            mean_d_norm = float(snap.get("mean_d_norm", 0.0))
        else:
            cur_enemy_hp = float(_sum_enemy_hp_filtered(bot))
            cur_marine_hp = float(sum_marine_hp(bot))
            alive_marines = int(count_marines(bot))

            mean_d_norm = 0.0
            e_cent = _enemy_centroid_filtered(bot)
            ms = marine_units(bot)
            if e_cent is not None and ms.exists:
                dists = [u.position.distance_to(e_cent) for u in ms]
                mean_d = float(sum(dists) / max(1, len(dists)))
                max_d = max(float(bot.game_info.map_size.x), float(bot.game_info.map_size.y))
                mean_d_norm = float(mean_d / max_d)

        init_e = float(max(1e-6, self.init_enemy_hp))
        init_m = float(max(1e-6, self.init_marine_hp))

        enemy_hp_lost = max(0.0, init_e - cur_enemy_hp)
        our_hp_lost = max(0.0, init_m - cur_marine_hp)

        frac_enemy = float(enemy_hp_lost / init_e)
        frac_our = float(our_hp_lost / init_m)
        remaining_enemy_frac = float(cur_enemy_hp / init_e)

        # progress term
        R_FRAC_ENEMY = 140.0
        R_FRAC_OUR = 110.0

        combat_score = 0.0
        combat_score += R_FRAC_ENEMY * frac_enemy
        combat_score -= R_FRAC_OUR * frac_our

        if reason == "win":
            combat_score += 20.0
        elif ("timeout" in reason) or ("force_reset_low_army" in reason):
            combat_score -= self.p.R_TIMEOUT_BASE * (0.5 + 1.5 * remaining_enemy_frac)
            if frac_enemy < 0.20:
                combat_score -= 40.0

        if reason == "all_dead":
            all_dead_pen = float(getattr(self.p, "R_ALL_DEAD_PEN", 140.0))
            combat_score -= all_dead_pen * (1.0 - 0.4 * frac_enemy)

        # final spacing penalty
        combat_score -= 15.0 * float(mean_d_norm)

        # alive bonus
        frac_alive = float(alive_marines) / 9.0
        combat_score += self.p.R_ALIVE_BONUS * frac_alive

        # tiny time penalty
        combat_score -= 0.001 * float(steps)

        if DEBUG_REWARD:
            print(
                "[REW-FINAL-R]",
                "reason=", reason,
                "steps=", steps,
                "frac_enemy=", frac_enemy,
                "frac_our=", frac_our,
                "alive=", alive_marines,
                "mean_d_norm=", mean_d_norm,
                "score=", combat_score,
            )

        return float(combat_score)