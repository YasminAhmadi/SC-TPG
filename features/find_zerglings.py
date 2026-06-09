# features/find_zerglings.py
from __future__ import annotations

import numpy as np

from utils import marine_units, safe_normalize
from tasks.spec import UnitObs

BASE_FEAT_DIM = 18


class BaseFeatureExtractor:
    """
    Pure reactive / no-memory features for FindAndDefeatZerglings.

    18D layout:
    [0..4]   self:
             x, y, hp_ratio, cd_norm, dmin_wall

    [5..7]   ally:
             ally_dir_x, ally_dir_y, ally_density

    [8..11]  visible enemy:
             nearest_enemy_dir_x, nearest_enemy_dir_y,
             nearest_enemy_dist_norm, visible_enemy_density

    [12..15] reactive explore cues (current geometry only):
             north_room, east_room, south_room, west_room

    [16]     wall_threat
    [17]     enemy_visible
    """

    def __init__(self):
        pass

    def reset(self):
        # No temporal state in the no-memory baseline
        pass

    def observe_swarm(self, bot):
        ms = marine_units(bot)
        return [self.get_features(bot, m) for m in ms] if ms else []

    def get_features(self, bot, marine) -> UnitObs:
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        diag = float(np.sqrt(W * W + H * H) + 1e-6)

        px = float(marine.position.x)
        py = float(marine.position.y)


        # 1) self
        x = px / max(W, 1e-6)
        y = py / max(H, 1e-6)

        hp = float(getattr(marine, "health", 45.0))
        hp_ratio = float(np.clip(hp / 45.0, 0.0, 1.0))

        cd = float(getattr(marine, "weapon_cooldown", 0.0))
        cd_norm = float(np.clip(cd / 10.0, 0.0, 1.0))

        dmin_raw = min(px, py, W - px, H - py)
        dmin = float(np.clip(dmin_raw / max(W, H, 1e-6), 0.0, 1.0))


        # 2) ally
        ms = marine_units(bot)
        ally_dir_x = 0.0
        ally_dir_y = 0.0
        ally_density = 0.0

        if ms and ms.amount > 1:
            others = [u for u in ms if u.tag != marine.tag]
            if others:
                ox = np.array([float(u.position.x) for u in others], dtype=np.float32)
                oy = np.array([float(u.position.y) for u in others], dtype=np.float32)

                cx = float(ox.mean())
                cy = float(oy.mean())

                ally_vec = np.array([cx - px, cy - py], dtype=np.float32)
                ally_dir = safe_normalize(ally_vec)
                ally_dir_x = float(ally_dir[0])
                ally_dir_y = float(ally_dir[1])

                dists = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
                R_allies = 3.5
                cnt = float((dists < R_allies).sum())
                ally_density = float(np.clip(cnt / 8.0, 0.0, 1.0))


        # 3) visible enemy only
        nearest_enemy_dir_x = 0.0
        nearest_enemy_dir_y = 0.0
        nearest_enemy_dist_norm = 1.0
        visible_enemy_density = 0.0

        enemies = list(getattr(bot, "enemy_units", []))
        enemy_visible = 1.0 if len(enemies) > 0 else 0.0

        if enemies:
            ex = np.array([float(u.position.x) for u in enemies], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in enemies], dtype=np.float32)

            dx = ex - px
            dy = ey - py
            dists = np.sqrt(dx * dx + dy * dy) + 1e-6

            j = int(np.argmin(dists))
            vec = np.array([dx[j], dy[j]], dtype=np.float32)
            edir = safe_normalize(vec)

            nearest_enemy_dir_x = float(edir[0])
            nearest_enemy_dir_y = float(edir[1])
            nearest_enemy_dist_norm = float(np.clip(dists[j] / diag, 0.0, 1.0))

            # local visible density
            R_enemy = 6.0
            visible_enemy_density = float(np.clip(float((dists < R_enemy).sum()) / 10.0, 0.0, 1.0))


        # 4) purely reactive exploration cues
        north_room = float(np.clip((H - py) / max(H, 1e-6), 0.0, 1.0))
        # print("north_room:", north_room)
        east_room  = float(np.clip((W - px) / max(W, 1e-6), 0.0, 1.0))
        # print("east_room:", east_room)
        south_room = float(np.clip(py / max(H, 1e-6), 0.0, 1.0))
        # print("south_room:", south_room)
        west_room  = float(np.clip(px / max(W, 1e-6), 0.0, 1.0))
        # print("west_room:", west_room)


        # 5) wall threat
        WALL_SAFE = 0.06
        wall_threat = float(np.clip((WALL_SAFE - dmin) / WALL_SAFE, 0.0, 1.0))

        base_feats = np.array(
            [
                x, y, hp_ratio, cd_norm, dmin,
                ally_dir_x, ally_dir_y, ally_density,
                nearest_enemy_dir_x, nearest_enemy_dir_y, nearest_enemy_dist_norm, visible_enemy_density,
                north_room, east_room, south_room, west_room,
                wall_threat, enemy_visible,
            ],
            dtype=np.float32,
        )
        # print("base_feats:", base_feats)
        assert base_feats.shape[0] == BASE_FEAT_DIM

        return UnitObs(
            tag=int(marine.tag),
            base_feats=base_feats,
            hp=float(hp),
            cd_norm=float(cd_norm),
            enemy_density=float(visible_enemy_density),
            bane_threat=float(wall_threat), # keep field for compatibility
            ling_threat=float(enemy_visible), # contact flag for compatibility
        )

    def featurize_for_trace(self, bot) -> np.ndarray:
        ms = marine_units(bot)
        if ms:
            return self.get_features(bot, ms.first).base_feats
        return np.zeros(BASE_FEAT_DIM, dtype=np.float32)