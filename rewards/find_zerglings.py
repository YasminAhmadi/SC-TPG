# rewards/find_zerglings.py
from __future__ import annotations
import numpy as np
from rewards.params import RewardParams
from utils import marine_units

DEBUG_REWARD = False

class RewardCalculator:
    def __init__(self, params: RewardParams):
        self.p = params
        self.prev_seen_count: int | None = None
        self.prev_kills: int = 0
        self.prev_deaths: int = 0
        self.prev_mean_enemy_d: float | None = None
        self.prev_marine_hp: float = 0.0

    def reset(self, bot):
        self.prev_seen_count = int(getattr(bot, "_explore_seen_count", 0) or 0)
        self.prev_kills = int(getattr(bot, "_kills_this_ep", 0) or 0)
        self.prev_deaths = int(getattr(bot, "_marine_deaths_this_ep", 0) or 0)

        ms = marine_units(bot)
        self.prev_marine_hp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in ms)) if ms and ms.exists else 0.0
        self.prev_mean_enemy_d = None

    def calculate_step_reward(self, bot, move_ratio: float) -> float:
        reward = 0.0

        # kill and death reward
        kills = int(getattr(bot, "_kills_this_ep", 0) or 0)
        deaths = int(getattr(bot, "_marine_deaths_this_ep", 0) or 0)
        dk = kills - self.prev_kills
        dm = deaths - self.prev_deaths
        self.prev_kills = kills
        self.prev_deaths = deaths

        if dk:
            # reward += float(self.p.R_KILL_ENEMY) * float(dk) # need to adjust the R_KILL_ENEMY
            R_KILL_ENEMY = 20 # adjust the R_KILL_ENEMY, increase the R_KILL_ENEMY
            reward += float(R_KILL_ENEMY) * float(dk)
        if dm:
            reward -= abs(float(self.p.R_LOSS_MARINE)) * float(dm) #

        #coverage
        seen_now = int(getattr(bot, "_explore_seen_count", 0) or 0)
        tot = int(getattr(bot, "_explore_total_cells", 1) or 1)
        if self.prev_seen_count is not None:
            d_seen = int(seen_now - self.prev_seen_count)
            if d_seen > 0 and tot > 0:
                reward += float(self.p.R_DELDIST) * (float(d_seen) / float(tot))
        self.prev_seen_count = seen_now

        # hp loss penalty
        ms = marine_units(bot)
        if ms and ms.exists:
            cur_mhp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in ms))
            taken = max(0.0, self.prev_marine_hp - cur_mhp)
            reward -= float(self.p.R_DMG_TAKEN) * taken
            self.prev_marine_hp = cur_mhp

        # distance to enemy reward
        enemies = list(getattr(bot, "enemy_units", []))
        enemy_visible = (len(enemies) > 0)
        if enemy_visible and ms and ms.exists:
            ex = np.array([float(u.position.x) for u in enemies], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in enemies], dtype=np.float32)

            ds = []
            for m in ms:
                px, py = float(m.position.x), float(m.position.y)
                dx, dy = ex - px, ey - py
                dists = np.sqrt(dx * dx + dy * dy) + 1e-6
                ds.append(float(dists.min()))
            mean_d = float(sum(ds) / max(1, len(ds)))

            if self.prev_mean_enemy_d is not None:
                progress = float(self.prev_mean_enemy_d - mean_d)
                W = float(bot.game_info.map_size.x)
                H = float(bot.game_info.map_size.y)
                d_norm = float(mean_d / max(W, H))
                if d_norm < float(self.p.MOVE_REWARD_DIST_THRESH):
                    reward += float(self.p.R_MOVE_NEAR_ENEMY) * progress
                    if progress > 0:
                        reward += 0.05 * float(move_ratio)

            self.prev_mean_enemy_d = mean_d
        else:
            self.prev_mean_enemy_d = None

        # edge penalty：FindZerglings. Remove it for now, as it may be too punishing for this task.


        return float(reward)

    def calculate_final_reward(self, bot, reason: str, steps: int) -> float:

        ms = marine_units(bot)
        alive = int(ms.amount) if ms and ms.exists else 0

        score = 0.0

        # remove the timeout
        # if "timeout" in reason: 

        # all_dead optional
        if "all_dead" in str(reason):
            score -= float(self.p.R_ALL_DEAD_PEN)

        # survival reward, optional
        max_steps = int(getattr(getattr(bot, "cfg", None), "loop_timeout", 1) or 1)
        survive_frac = float(steps) / float(max_steps)
        score += float(self.p.R_SURVIVE) * survive_frac

        # survival bonus, optional
        score += float(self.p.R_ALIVE_BONUS) * (float(alive) / 3.0)

        return float(score)

# Improvement final reward. Change it to "Main survival reward is only given after killing enemies".
# def calculate_final_reward(self, bot, reason: str, steps: int) -> float:
#     """
#     Final reward for FindZerglings.

#     Important design:
#     - Do NOT give a large unconditional survival bonus.
#     - Survival is rewarded only after task progress, otherwise agents can
#       converge to a passive "hide and survive" local optimum.
#     """
#     ms = marine_units(bot)
#     alive = int(ms.amount) if ms and ms.exists else 0

#     init_m = int(getattr(getattr(bot, "task", None), "INIT_M_ALL", 3))
#     alive_frac = float(alive) / float(max(1, init_m))

#     kills = int(getattr(bot, "_kills_this_ep", 0) or 0)
#     seen_tags = getattr(bot, "_enemy_seen_tags", set())
#     seen_count = len(seen_tags) if seen_tags is not None else 0

#     score = 0.0

#     # 1) all_dead penalty
#     if "all_dead" in str(reason):
#         score -= abs(float(self.p.R_ALL_DEAD_PEN))

#     # 2) very small survival term, gated by progress
#     # No enemy seen and no kill: no survival bonus.
#     if kills > 0:
#         # Full survival bonus only after successful combat progress.
#         score += float(self.p.R_ALIVE_BONUS) * alive_frac
#     elif seen_count > 0:
#         # If the agent at least found enemies, give a small survival reward.
#         # This avoids making early exploration too punishing.
#         score += 0.2 * float(self.p.R_ALIVE_BONUS) * alive_frac
#     else:
#         # No contact, no kill: do not reward passive survival.
#         score += 0.0

#     # 3) Optional tiny time survival, also gated by progress
#     max_steps = int(getattr(getattr(bot, "cfg", None), "loop_timeout", 1) or 1)
#     survive_frac = float(steps) / float(max_steps)

#     if kills > 0:
#         score += float(self.p.R_SURVIVE) * survive_frac
#     elif seen_count > 0:
#         score += 0.2 * float(self.p.R_SURVIVE) * survive_frac

#     return float(score)