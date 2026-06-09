# features/defeat_roaches.py
from __future__ import annotations
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from utils import marine_units, safe_normalize
from tasks.spec import UnitObs

MARINE_HP_MAX = 45.0
ROACH_HP_MAX = 145.0

class RoachFeatureExtractor:
    def __init__(self):
        self.marine_mem = {}
        self.marine_prev_hp = {}

    def reset(self):
        self.marine_mem.clear()
        self.marine_prev_hp.clear()

    def _get_roaches(self, bot):
        # DefeatRoaches roach
        return [u for u in bot.enemy_units if u.type_id == UnitTypeId.ROACH]

    def get_features(self, bot, marine) -> UnitObs:
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)

        px = float(marine.position.x)
        py = float(marine.position.y)

        # 1) self
        x = px / W
        y = py / H
        hp = float(getattr(marine, "health", MARINE_HP_MAX))
        hp_ratio = hp / MARINE_HP_MAX

        cd = float(getattr(marine, "weapon_cooldown", 0.0))
        cd_norm = min(1.0, cd / 10.0)

        dmin = min(px, py, W - px, H - py) / max(W, H)

        # 2) allies
        ms = marine_units(bot)
        ally_dir_x = 0.0
        ally_dir_y = 0.0
        ally_density = 0.0
        if ms.amount > 1:
            others = [u for u in ms if u.tag != marine.tag]
            if others:
                ox = np.array([float(u.position.x) for u in others], dtype=np.float32)
                oy = np.array([float(u.position.y) for u in others], dtype=np.float32)
                cx = float(ox.mean()); cy = float(oy.mean())
                vec = np.array([cx - px, cy - py], dtype=np.float32)
                dir_vec = safe_normalize(vec)
                ally_dir_x = float(dir_vec[0])
                ally_dir_y = float(dir_vec[1])

                dists = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
                R_allies = 3.5
                cnt = float((dists < R_allies).sum())
                ally_density = max(0.0, min(1.0, cnt / 8.0))

        # 3) enemies (general nearest enemy signal + density)
        enemies = list(bot.enemy_units)
        enemy_dir_x = 0.0
        enemy_dir_y = 0.0
        enemy_dist_norm = 1.0
        enemy_density = 0.0

        # 4) roach-specific
        roach_dir_x = 0.0
        roach_dir_y = 0.0
        roach_dist_norm = 1.0
        roach_threat = 0.0

        vis = 0.0

        if enemies:
            vis = 1.0
            ex = np.array([float(u.position.x) for u in enemies], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in enemies], dtype=np.float32)
            dx = ex - px
            dy = ey - py
            dists = np.sqrt(dx * dx + dy * dy) + 1e-6

            j = int(np.argmin(dists))
            dir_vec = np.array([dx[j], dy[j]], dtype=np.float32)
            dir_normed = safe_normalize(dir_vec)
            enemy_dir_x = float(dir_normed[0])
            enemy_dir_y = float(dir_normed[1])

            raw_dist = float(dists[j])
            enemy_dist_norm = float(raw_dist / max(W, H))

            R_enemy = 4.5
            enemy_density = max(0.0, min(1.0, float((dists < R_enemy).sum()) / 6.0))

        roaches = self._get_roaches(bot)
        if roaches:
            rx = np.array([float(u.position.x) for u in roaches], dtype=np.float32)
            ry = np.array([float(u.position.y) for u in roaches], dtype=np.float32)
            rdx = rx - px
            rdy = ry - py
            rdists = np.sqrt(rdx * rdx + rdy * rdy) + 1e-6

            k = int(np.argmin(rdists))
            rvec = np.array([rdx[k], rdy[k]], dtype=np.float32)
            rvec = safe_normalize(rvec)
            roach_dir_x = float(rvec[0])
            roach_dir_y = float(rvec[1])

            raw_rdist = float(rdists[k])
            roach_dist_norm = float(raw_rdist / max(W, H))

            # threat
            R_roach = 4.0
            rcnt = float((rdists < R_roach).sum())
            roach_threat = max(0.0, min(1.0, rcnt / 4.0)) # 4 roaches

        # keep 18D 
        ling_threat = 0.0

        base_feats = np.array(
            [
                x, y, hp_ratio, cd_norm, dmin,
                ally_dir_x, ally_dir_y, ally_density,
                enemy_dir_x, enemy_dir_y, enemy_dist_norm, enemy_density,
                roach_dir_x, roach_dir_y, roach_dist_norm, roach_threat,
                ling_threat, vis,
            ],
            dtype=np.float32,
        )

        return UnitObs(
            tag=int(marine.tag),
            base_feats=base_feats,
            hp=float(hp),
            cd_norm=float(cd_norm),
            enemy_density=float(enemy_density),
            bane_threat=float(roach_threat),
            ling_threat=float(ling_threat),
        )

    # old version. do not use
    def featurize_for_trace(self, bot):
        ms = marine_units(bot)
        if ms:
            return self.get_features(bot, ms.first)
        return None
