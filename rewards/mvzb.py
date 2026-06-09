# rewards/mvzb.py
import numpy as np
from rewards.params import RewardParams

from utils import (
    marine_units,
    sum_hp,
    sum_enemy_hp,
    sum_marine_hp,
    count_enemies,
    get_enemy_centroid,
    get_centroid,
    count_marines,
)

DEBUG_REWARD = False  # Debug flag to print detailed reward calculations. Set to False to disable.


class RewardCalculator:
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

        self.prev_enemy_hp = float(sum_enemy_hp(bot))
        self.prev_marine_hp = float(sum_hp(ms))
        self.prev_enemy_count = int(count_enemies(bot))
        self.prev_marine_count = int(ms.amount)
        self.last_d_to_enemy = None

        self.init_enemy_hp = float(self.prev_enemy_hp)
        self.init_marine_hp = float(self.prev_marine_hp)

        if DEBUG_REWARD:
            print(f"[REW-RESET] init_enemy_hp={self.init_enemy_hp:.3f} init_marine_hp={self.init_marine_hp:.3f}")

    def calculate_step_reward(self, bot, move_ratio: float) -> float:
        reward = 0.0
        ms = marine_units(bot)

        cur_e_hp = float(sum_enemy_hp(bot))
        cur_m_hp = float(sum_hp(ms))
        cur_e_cnt = int(count_enemies(bot))
        cur_m_cnt = int(ms.amount)

        dealt = max(0.0, self.prev_enemy_hp - cur_e_hp)
        taken = max(0.0, self.prev_marine_hp - cur_m_hp)
        reward += self.p.R_DMG_DEALT * dealt - self.p.R_DMG_TAKEN * taken

        killed = max(0, self.prev_enemy_count - cur_e_cnt)
        if killed > 0:
            reward += self.p.R_KILL_ENEMY * float(killed)

        lost = max(0, self.prev_marine_count - cur_m_cnt)
        if lost > 0:
            reward -= self.p.R_LOSS_MARINE * float(lost)

        e_cent = get_enemy_centroid(bot)
        if e_cent is not None and ms.exists:
            d_now = float(ms.center.distance_to(e_cent))
            if self.last_d_to_enemy is not None:
                reward += self.p.R_DELDIST * (self.last_d_to_enemy - d_now)
            self.last_d_to_enemy = d_now

        m_cent = get_centroid(marine_units(bot))
        if m_cent is not None:
            W = float(bot.game_info.map_size.x)
            H = float(bot.game_info.map_size.y)
            dmin = min(m_cent.x, m_cent.y, W - m_cent.x, H - m_cent.y) / max(W, H)
            reward += -self.p.R_EDGE_PEN * (1.0 - dmin)

        if self.last_d_to_enemy is not None:
            d_norm = float(self.last_d_to_enemy / max(bot.game_info.map_size.x, bot.game_info.map_size.y))
            if d_norm < self.p.MOVE_REWARD_DIST_THRESH:
                reward += self.p.R_MOVE_NEAR_ENEMY * float(move_ratio)

        self.prev_enemy_hp = cur_e_hp
        self.prev_marine_hp = cur_m_hp
        self.prev_enemy_count = cur_e_cnt
        self.prev_marine_count = cur_m_cnt

        return float(reward)

    def calculate_final_reward(self, bot, reason: str, steps: int) -> float:
        snap = getattr(bot, "_rew_ov_final", None)

        if snap is not None:
            cur_enemy_hp = float(snap.get("enemy_hp", 0.0))
            cur_marine_hp = float(snap.get("marine_hp", 0.0))
            alive_marines = int(snap.get("alive_marines", 0))
            mean_d_norm = float(snap.get("mean_d_norm", 0.0))
        else:
            cur_enemy_hp = float(sum_enemy_hp(bot))
            cur_marine_hp = float(sum_marine_hp(bot))
            alive_marines = int(count_marines(bot))

            mean_d_norm = 0.0
            e_cent = get_enemy_centroid(bot)
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

        frac_enemy = enemy_hp_lost / init_e
        frac_our = our_hp_lost / init_m
        remaining_enemy_frac = cur_enemy_hp / init_e

        # Final reward coefficients. 
        R_FRAC_ENEMY = float(getattr(self.p, "R_FRAC_ENEMY", 120.0))
        R_FRAC_OUR = float(getattr(self.p, "R_FRAC_OUR", 110.0))
        R_DEAD_FINAL = float(getattr(self.p, "R_DEAD_FINAL", 8.0))

        combat_score = 0.0
        combat_score += R_FRAC_ENEMY * frac_enemy
        combat_score -= R_FRAC_OUR * frac_our

        if reason == "win":
            combat_score += 90.0
        elif ("timeout" in reason) or ("force_reset_low_army" in reason):
            combat_score -= self.p.R_TIMEOUT_BASE * (1.0 + 5.0 * remaining_enemy_frac)

        if reason == "all_dead":
            combat_score -= self.p.R_ALL_DEAD_PEN

        # dead marine punishment (final)
        dead_marines = 9 - alive_marines
        combat_score -= R_DEAD_FINAL * float(dead_marines)

        # distance penalty (final)
        combat_score -= 25.0 * float(mean_d_norm)

        # alive bonus (final)
        frac_alive = float(alive_marines) / 9.0
        combat_score += self.p.R_ALIVE_BONUS * frac_alive

        # time penalty (final) important to encourage faster wins
        combat_score -= 0.002 * float(steps)

        if DEBUG_REWARD:
            print("[REW-FINAL]",
                "reason=", reason,
                "steps=", steps,
                "frac_enemy=", frac_enemy,
                "frac_our=", frac_our,
                "alive=", alive_marines,
                "dead=", dead_marines,
                "mean_d_norm=", mean_d_norm,
                "score=", combat_score)

        return float(combat_score)
