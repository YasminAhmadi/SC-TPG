# actions/mineral_shards.py
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List

from sc2.position import Point2

from utils import (
    marine_units,
    safe_normalize,
    calculate_wall_repulsion,
    clip_position,
)
from actions.params import ActionParams

ALLIANCE_NEUTRAL = 3


class ActionExecutor:
    """
    Factorized action executor for CollectMineralShards.

    Combined action encoding:
      combined = move_idx * N_SEL + sel_idx

    Movement (N_MOVE=9):
      0: HOLD
      1: TOWARD selected shard
      2: AWAY selected shard
      3: STRAFE_LEFT around selected shard
      4: STRAFE_RIGHT around selected shard
      5: WALL_REPULSE
      6: TOWARD ally
      7: AWAY ally
      8: ORBIT selected shard (deterministic left/right by tag)

    Selector (N_SEL=4):
      0: NEAREST shard
      1: SECOND_NEAREST shard
      2: THIRD_NEAREST shard
      3: FARTHEST shard

    """

    N_MOVE = 9
    N_SEL = 4

    def __init__(self, params: ActionParams):
        self.p = params
        self.order_cooldown = int(getattr(self.p, "order_cooldown", 6))

        # per-unit gating
        self._last_issue_loop_by_tag: Dict[int, int] = {}
        self._last_tgt_by_tag: Dict[int, Point2] = {}

        # shard cache
        self._cached_loop = -1
        self._shard_type_id: Optional[int] = None
        self._cached_shards: List[Point2] = []

    def reset(self):
        self._last_issue_loop_by_tag.clear()
        self._last_tgt_by_tag.clear()

        self._cached_loop = -1
        self._shard_type_id = None
        self._cached_shards = []


    # Raw shard extraction
    def _raw_units(self, bot):
        obs = getattr(getattr(bot, "state", None), "observation", None)
        raw = getattr(obs, "raw_data", None) if obs is not None else None
        units = getattr(raw, "units", None) if raw is not None else None
        return units

    def _infer_shard_type(self, bot) -> Optional[int]:
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

        # need to extract the most frequent neutral unit type as the mineral shard type from raw data
        return int(max(counts.items(), key=lambda kv: kv[1])[0])

    def _get_shards(self, bot) -> List[Point2]:
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

        shards: List[Point2] = []
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


    # Selector logic
    def _select_shard(
        self,
        shards: List[Point2],
        px: float,
        py: float,
        sel_idx: int,
    ) -> Optional[Point2]:
        """
        Selector head:
          0 -> nearest
          1 -> second nearest
          2 -> third nearest
          3 -> farthest
        """
        if not shards:
            return None

        ds = np.array(
            [float(np.hypot(s.x - px, s.y - py)) for s in shards],
            dtype=np.float32,
        )
        order = np.argsort(ds)

        if sel_idx == 0:
            k = 0
        elif sel_idx == 1:
            k = min(1, len(order) - 1)
        elif sel_idx == 2:
            k = min(2, len(order) - 1)
        elif sel_idx == 3:
            k = len(order) - 1
        else:
            k = 0

        return shards[int(order[k])]


    # Main execution
    async def apply_actions(self, bot, marine_actions: Dict[int, int]) -> float:
        ms = marine_units(bot)
        if not ms:
            return 0.0

        shards = self._get_shards(bot)
        # print("shards:", shards)
        if not shards:
            return 0.0

        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        map_margin = float(getattr(self.p, "map_margin", 1.0))
        wall_repulsion_w = float(getattr(self.p, "wall_repulsion", 0.0))
        step_size = float(getattr(self.p, "step_size", 2.5))

        ally_positions = np.array(
            [[float(u.position.x), float(u.position.y)] for u in ms],
            dtype=np.float32,
        )

        total = int(ms.amount)
        moved_count = 0

        for idx, m in enumerate(ms):
            tag = int(m.tag)

            a = int(marine_actions.get(tag, 0))
            a %= (self.N_MOVE * self.N_SEL)

            move_idx = a // self.N_SEL
            sel_idx = a % self.N_SEL

            px = float(m.position.x)
            py = float(m.position.y)
            pos = np.array([px, py], dtype=np.float32)

            # selector: pick reference shard
            target_shard = self._select_shard(shards, px, py, sel_idx)
            if target_shard is None:
                continue

            tx = float(target_shard.x)
            ty = float(target_shard.y)

            # shard frame
            vec_t = np.array([tx - px, ty - py], dtype=np.float32)
            target_dir = safe_normalize(vec_t)
            target_away = -target_dir
            strafe_left = np.array([-target_dir[1], target_dir[0]], dtype=np.float32)
            strafe_right = np.array([target_dir[1], -target_dir[0]], dtype=np.float32)

            # ally frame
            ally_towards = np.zeros(2, dtype=np.float32)
            ally_away = np.zeros(2, dtype=np.float32)
            if total > 1:
                others = np.delete(ally_positions, idx, axis=0)
                c = others.mean(axis=0)
                vec_a = c - pos
                ally_towards = safe_normalize(vec_a)
                ally_away = -ally_towards

            # per-unit wall repulsion
            rep = calculate_wall_repulsion(pos, W, H, map_margin)
            rep = safe_normalize(rep)

            move_vec = self._movement_vec(
                move_idx=move_idx,
                toward=target_dir,
                away=target_away,
                left=strafe_left,
                right=strafe_right,
                ally_towards=ally_towards,
                ally_away=ally_away,
                rep=rep,
                tag=tag,
            )

            # HOLD
            if np.linalg.norm(move_vec) < 1e-6:
                if move_idx == 0 and self._can_issue_unit(tag, bot, m.position):
                    self._issue_hold(m)
                    self._mark_issued_unit(tag, bot, m.position)
                continue

            # Add a small repulsion prior to non-repulse primitives
            if move_idx != 5:
                move_vec = move_vec + wall_repulsion_w * rep

            final_move = safe_normalize(move_vec)
            tgt = self._calc_target(px, py, final_move, W, H, step_size, map_margin)

            if self._can_issue_unit(tag, bot, tgt):
                m.move(tgt)
                self._mark_issued_unit(tag, bot, tgt)
                moved_count += 1

        return moved_count / float(total) if total > 0 else 0.0


    # Movement primitives
    def _movement_vec(
        self,
        move_idx: int,
        toward: np.ndarray,
        away: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        ally_towards: np.ndarray,
        ally_away: np.ndarray,
        rep: np.ndarray,
        tag: int,
    ) -> np.ndarray:
        if move_idx == 0: # HOLD
            return np.zeros(2, dtype=np.float32)
        if move_idx == 1: # TOWARD selected shard
            return toward
        if move_idx == 2: # AWAY selected shard
            return away
        if move_idx == 3: # STRAFE_LEFT
            return left
        if move_idx == 4: # STRAFE_RIGHT
            return right
        if move_idx == 5: # WALL_REPULSE
            return rep
        if move_idx == 6: # TOWARD ally
            return ally_towards
        if move_idx == 7: # AWAY ally
            return ally_away
        if move_idx == 8: # ORBIT selected shard
            return left if (tag % 2 == 0) else right

        return np.zeros(2, dtype=np.float32)


    # helpers
    def _calc_target(
        self,
        px: float,
        py: float,
        move_vec: np.ndarray,
        W: float,
        H: float,
        step_size: float,
        map_margin: float,
    ) -> Point2:
        raw_p = Point2((
            px + float(move_vec[0]) * step_size,
            py + float(move_vec[1]) * step_size,
        ))
        return clip_position(raw_p, W, H, map_margin)

    def _can_issue_unit(self, tag: int, bot, tgt: Point2) -> bool:
        now = int(bot.state.game_loop)
        last_loop = int(self._last_issue_loop_by_tag.get(tag, -100000))
        last_tgt = self._last_tgt_by_tag.get(tag, None)

        if now - last_loop < self.order_cooldown:
            if last_tgt is not None and tgt.distance_to(last_tgt) > 1.2:
                return True
            return False

        if last_tgt is not None and tgt.distance_to(last_tgt) < 0.2:
            return False

        return True

    def _mark_issued_unit(self, tag: int, bot, tgt: Point2):
        self._last_issue_loop_by_tag[tag] = int(bot.state.game_loop)
        self._last_tgt_by_tag[tag] = tgt

    def _issue_hold(self, m):
        if hasattr(m, "hold_position"):
            m.hold_position()
        elif hasattr(m, "stop"):
            m.stop()