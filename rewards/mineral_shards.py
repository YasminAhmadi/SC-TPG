# rewards/mineral_shards.py
import numpy as np
from sc2.position import Point2

from rewards.params import RewardParams
from utils import marine_units

ALLIANCE_NEUTRAL = 3
DEBUG_REWARD = False




class RewardCalculator:
    def __init__(self, params: RewardParams):
        self.p = params
        self.prev_score = None 
        self.prev_mean_d = None

        self.prev_shard_tags: set[int] | None = None

        self._cached_loop = -1
        self._shard_type_id: int | None = None
        self._cached_shards: list[Point2] = []
        self._cached_shard_tags: set[int] = set()

    def _score(self, bot) -> float:
        return float(getattr(getattr(bot.state, "score", None), "score", 0.0) or 0.0)
    
    def _raw_units(self, bot):
        obs = getattr(getattr(bot, "state", None), "observation", None)
        raw = getattr(obs, "raw_data", None) if obs is not None else None
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

            # find minerals, filter out others. But won't exist for current mini task, keep for further battlefield map.
            mc = int(getattr(u, "mineral_contents", 0) or 0)
            vc = int(getattr(u, "vespene_contents", 0) or 0)
            if mc > 0 or vc > 0:
                continue

            counts[tid] = counts.get(tid, 0) + 1
        if not counts:
            return None
        return int(max(counts.items(), key=lambda kv: kv[1])[0])

    def _get_shards(self, bot) -> list[Point2]:
        loop = int(getattr(getattr(bot, "state", None), "game_loop", -1))
        if loop == self._cached_loop:
            return self._cached_shards

        units = self._raw_units(bot)
        if units is None:
            self._cached_loop = loop
            self._cached_shards = []
            self._cached_shard_tags = set()
            return []

        if self._shard_type_id is None:
            self._shard_type_id = self._infer_shard_type(bot)

        shards: list[Point2] = []
        tags: set[int] = set()

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
                tags.add(int(getattr(u, "tag", 0)))

        self._cached_loop = loop
        self._cached_shards = shards
        self._cached_shard_tags = tags
        return shards

    def reset(self, bot):
        self.prev_score = self._score(bot)
        self._get_shards(bot)
        self.prev_shard_tags = set(self._cached_shard_tags)

        ms = marine_units(bot)
        shards = self._cached_shards
        if ms and ms.exists and shards:
            ds = []
            for m in ms:
                px, py = float(m.position.x), float(m.position.y)
                dmin = min(float(np.hypot(s.x - px, s.y - py)) for s in shards)
                ds.append(dmin)
            self.prev_mean_d = float(sum(ds) / max(1, len(ds)))
        else:
            self.prev_mean_d = None

    def calculate_step_reward(self, bot, move_ratio: float) -> float:
        reward = 0.0
        ms = marine_units(bot)
        if not ms or not ms.exists:
            return 0.0

        # A) collected
        shards = self._get_shards(bot)
        # print("shards:",shards)
        cur_tags = set(self._cached_shard_tags)
        if self.prev_shard_tags is not None:
            collected = len(self.prev_shard_tags - cur_tags)
            if collected > 0:
                # whether to give reward a bigger score
                reward += float(self.p.R_KILL_ENEMY) * float(collected)
                # reward += float(collected)
        self.prev_shard_tags = cur_tags

        # B) distance shaping, norm
        if shards:
            ds = []
            for m in ms:
                px, py = float(m.position.x), float(m.position.y)
                dmin = min(float(np.hypot(sh.x - px, sh.y - py)) for sh in shards)
                ds.append(dmin)
            mean_d = float(sum(ds) / max(1, len(ds)))

            if self.prev_mean_d is not None:
                W = float(bot.game_info.map_size.x)
                H = float(bot.game_info.map_size.y)
                diag = float(np.hypot(W, H))

                progress = (self.prev_mean_d - mean_d) / max(diag, 1e-6) # normalized
                progress = float(np.clip(progress, -0.02, 0.02))

                reward += float(self.p.R_DELDIST) * progress

                d_norm = mean_d / max(diag, 1e-6)
                if d_norm < float(self.p.MOVE_REWARD_DIST_THRESH) and progress > 0:
                    reward += float(self.p.R_MOVE_NEAR_ENEMY) * float(move_ratio)

            self.prev_mean_d = mean_d

        # C) edge penalty
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        cx = float(ms.center.x)
        cy = float(ms.center.y)
        dmin_edge = min(cx, cy, W - cx, H - cy) / max(W, H)
        reward += -float(self.p.R_EDGE_PEN) * (1.0 - float(dmin_edge))

        return float(reward)

    def calculate_final_reward(self, bot, reason: str, steps: int) -> float:
        # Shards tasks: typically only end with a timeout; final tasks should not be scored repeatedly (the step score has already been accumulated using delta_score).
        return 0.0
    