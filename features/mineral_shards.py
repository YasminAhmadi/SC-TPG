# features/mineral_shards.py
import numpy as np
from sc2.position import Point2

from utils import marine_units, safe_normalize
from tasks.spec import UnitObs

BASE_FEAT_DIM = 18
ALLIANCE_NEUTRAL = 3


class BaseFeatureExtractor:
    """
    18D base features, including:

    [0..4] self: x, y, hp_ratio, cd_norm, dmin
    [5..7] ally: ally_dir_x, ally_dir_y, ally_density
    [8..11] Obj1: shard_dir_x, shard_dir_y, shard_dist_norm, shard_closeness
    [12..15] Obj2: wall_in_dir_x, wall_in_dir_y, wall_dist_norm, wall_threat
    [16] Aux: stuckness
    [17] vis: shard_found
    """

    def __init__(self):
        self.prev_pos = {}
        self._cached_loop = -1
        self._cached_shards: list[Point2] = []
        self._shard_type_id: int | None = None

    def reset(self):
        self.prev_pos.clear()
        self._cached_loop = -1
        self._cached_shards = []
        self._shard_type_id = None

    def _raw_units(self, bot):
        obs = getattr(getattr(bot, "state", None), "observation", None)
        raw = getattr(obs, "raw_data", None) if obs is not None else None
        # print("raw:", raw)
        units = getattr(raw, "units", None) if raw is not None else None
        return units

    def _infer_shard_type(self, bot) -> int | None:
        units = self._raw_units(bot)
        if units is None:
            return None

        counts = {}
        for u in units:
            if int(getattr(u, "alliance", -1)) != ALLIANCE_NEUTRAL:
                continue
            tid = int(getattr(u, "unit_type", -1))
            if tid < 0:
                continue
            counts[tid] = counts.get(tid, 0) + 1

        if not counts:
            return None

        # pick neutral unit_type for shard
        best_tid = max(counts.items(), key=lambda kv: kv[1])[0]
        return int(best_tid)

    def _get_shards(self, bot) -> list[Point2]:
        loop = int(getattr(getattr(bot, "state", None), "game_loop", -1))
        if loop == self._cached_loop:
            return self._cached_shards

        units = self._raw_units(bot)
        if units is None:
            self._cached_loop = loop
            self._cached_shards = []
            return []

        if self._shard_type_id is None:
            self._shard_type_id = self._infer_shard_type(bot)

        shards: list[Point2] = []
        if self._shard_type_id is not None:
            for u in units:
                if int(getattr(u, "alliance", -1)) != ALLIANCE_NEUTRAL:
                    continue
                if int(getattr(u, "unit_type", -1)) != int(self._shard_type_id):
                    continue
                p = getattr(u, "pos", None)
                if p is None:
                    continue
                shards.append(Point2((float(p.x), float(p.y))))

        self._cached_loop = loop
        self._cached_shards = shards
        return shards

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

                vec = np.array([cx - px, cy - py], dtype=np.float32)
                d = safe_normalize(vec)
                ally_dir_x = float(d[0])
                ally_dir_y = float(d[1])

                dists = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
                R_allies = 3.5
                cnt = float((dists < R_allies).sum())
                ally_density = float(np.clip(cnt / 8.0, 0.0, 1.0))

        # 3) Obj1: shard
        obj_dir_x = 0.0
        obj_dir_y = 0.0
        obj_dist_norm = 1.0
        obj_close = 0.0

        vis = 0.0
        shards = self._get_shards(bot)
        if shards:
            vis = 1.0
            # closest shard
            ds = np.array([np.hypot(s.x - px, s.y - py) for s in shards], dtype=np.float32)
            j = int(ds.argmin())
            tx, ty = float(shards[j].x), float(shards[j].y)

            vec = np.array([tx - px, ty - py], dtype=np.float32)
            d = safe_normalize(vec)
            obj_dir_x = float(d[0])
            obj_dir_y = float(d[1])

            dist = float(ds[j] + 1e-6)
            diag = float(np.sqrt(W * W + H * H))
            obj_dist_norm = float(np.clip(dist / diag, 0.0, 1.0))

            R = 3.0
            obj_close = float(np.clip((R - dist) / R, 0.0, 1.0))

        # 4) Obj2: wall hazard
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

        base_feats = np.array(
            [
                x, y, hp_ratio, cd_norm, dmin,
                ally_dir_x, ally_dir_y, ally_density,
                obj_dir_x, obj_dir_y, obj_dist_norm, obj_close,
                bane_dir_x, bane_dir_y, bane_dist_norm, bane_threat,
                stuck, vis,
            ],
            dtype=np.float32,
        )
        assert base_feats.shape[0] == BASE_FEAT_DIM

        return UnitObs(
            tag=int(marine.tag),
            base_feats=base_feats,
            hp=float(hp),
            cd_norm=float(cd_norm),
            enemy_density=float(obj_close),
            bane_threat=float(bane_threat),
            ling_threat=float(stuck),
        )

    # old version, for trace graph
    def featurize_for_trace(self, bot) -> np.ndarray:
        ms = marine_units(bot)
        if ms:
            return self.get_features(bot, ms.first).base_feats
        return np.zeros(BASE_FEAT_DIM, dtype=np.float32)
