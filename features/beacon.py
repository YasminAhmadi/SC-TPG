# features/beacon.py
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from utils import marine_units, safe_normalize
from tasks.spec import UnitObs

BASE_FEAT_DIM = 18
BEACON_UNIT_TYPE_ID = 317
ALLIANCE_NEUTRAL = 3

class BaseFeatureExtractor:
    """
    MoveToBeacon 18D（reuse）：

    [0..4] self: x, y, hp_ratio, cd_norm, dmin
    [5..7] ally: ally_dir_x, ally_dir_y, ally_density
    [8..11] Obj1: beacon_dir_x, beacon_dir_y, beacon_dist_norm, beacon_closeness
    [12..15] Obj2/Threat: wall_in_dir_x, wall_in_dir_y, wall_dist_norm, wall_threat
    [16] Aux: stuckness
    [17] vis: beacon_found
    """

    def __init__(self):
        # tag -> (prev_x, prev_y)
        self.prev_pos = {}

    def reset(self):
        self.prev_pos.clear()

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

    def observe_swarm(self, bot):
        ms = marine_units(bot)
        return [self.get_features(bot, m) for m in ms] if ms else []

    def get_features(self, bot, marine) -> UnitObs:
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)

        px = float(marine.position.x)
        py = float(marine.position.y)

        # 1) self
        x = px / W
        y = py / H

        hp = float(getattr(marine, "health", 45.0))
        hp_ratio = hp / 45.0

        cd = float(getattr(marine, "weapon_cooldown", 0.0))
        cd_norm = min(1.0, cd / 10.0)

        dmin = min(px, py, W - px, H - py) / max(W, H)

        # 2) ally. No ally for MoveToBeacon
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

                vec = np.array([cx - px, cy - py], dtype=np.float32)
                d = safe_normalize(vec)
                ally_dir_x = float(d[0])
                ally_dir_y = float(d[1])

                dists = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
                R_allies = 3.5
                cnt = float((dists < R_allies).sum())
                ally_density = float(np.clip(cnt / 8.0, 0.0, 1.0))

        # 3) Obj1: beacon
        enemy_dir_x = 0.0
        enemy_dir_y = 0.0
        enemy_dist_norm = 1.0
        enemy_density = 0.0 # beacon_closeness

        vis = 0.0
        beacon_pos = self._find_beacon_pos(bot)
        if beacon_pos is not None:
            vis = 1.0
            bx, by = beacon_pos
            vec = np.array([bx - px, by - py], dtype=np.float32)
            d = safe_normalize(vec)
            enemy_dir_x = float(d[0])
            enemy_dir_y = float(d[1])

            dist = float(np.sqrt((bx - px) ** 2 + (by - py) ** 2) + 1e-6)
            diag = np.sqrt(W*W + H*H)
            enemy_dist_norm = np.clip(dist / diag, 0.0, 1.0)

            R = 3.0  # beacon radius
            enemy_density = float(np.clip((R - dist) / R, 0.0, 1.0))

        # 4) Obj2/Threat: wall hazard
        left = px
        right = W - px
        bottom = py
        top = H - py
        dmin_raw = min(left, right, bottom, top)

        wall_dir = np.zeros(2, dtype=np.float32)
        if dmin_raw == left:
            wall_dir[0] = 1.0
        elif dmin_raw == right:
            wall_dir[0] = -1.0
        elif dmin_raw == bottom:
            wall_dir[1] = 1.0
        else:
            wall_dir[1] = -1.0
        wall_dir = safe_normalize(wall_dir)

        bane_dir_x = float(wall_dir[0])
        bane_dir_y = float(wall_dir[1])
        bane_dist_norm = float(dmin_raw / max(W, H))

        WALL_SAFE = 0.06
        bane_threat = float(np.clip((WALL_SAFE - dmin) / WALL_SAFE, 0.0, 1.0))

        # 5) Aux: stuckness
        prev = self.prev_pos.get(int(marine.tag))
        if prev is None:
            stuck = 0.0
        else:
            dx = px - prev[0]
            dy = py - prev[1]
            step = float(np.sqrt(dx * dx + dy * dy) + 1e-6)
            EPS = 0.15
            stuck = float(np.clip((EPS - step) / EPS, 0.0, 1.0))
        self.prev_pos[int(marine.tag)] = (px, py)
        ling_threat = stuck

        base_feats = np.array(
            [
                x, y, hp_ratio, cd_norm, dmin,
                ally_dir_x, ally_dir_y, ally_density,
                enemy_dir_x, enemy_dir_y, enemy_dist_norm, enemy_density,
                bane_dir_x, bane_dir_y, bane_dist_norm, bane_threat,
                ling_threat, vis,
            ],
            dtype=np.float32,
        )
        assert base_feats.shape[0] == BASE_FEAT_DIM

        return UnitObs(
            tag=int(marine.tag),
            base_feats=base_feats,
            hp=float(hp),
            cd_norm=float(cd_norm),
            enemy_density=float(enemy_density), # beacon_closeness
            bane_threat=float(bane_threat), # wall_threat
            ling_threat=float(ling_threat), # stuckness
        )

    # old version, for trace graph
    def featurize_for_trace(self, bot) -> np.ndarray:
        ms = marine_units(bot)
        if ms:
            return self.get_features(bot, ms.first).base_feats
        return np.zeros(BASE_FEAT_DIM, dtype=np.float32)
