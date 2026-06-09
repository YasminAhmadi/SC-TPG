# actions/beacon.py
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from sc2.position import Point2

from utils import (
    marine_units,
    safe_normalize,
    calculate_wall_repulsion,
    clip_position,
)
from actions.params import ActionParams


BEACON_UNIT_TYPE_ID = 317
ALLIANCE_NEUTRAL = 3


class ActionExecutor:
    """
    Factorized action executor for MoveToBeacon.

    Combined action encoding:
      combined = move_idx * N_SCALE + scale_idx

    Movement (N_MOVE=9):
      0: HOLD
      1: TOWARD_BEACON
      2: AWAY_BEACON
      3: STRAFE_LEFT around beacon
      4: STRAFE_RIGHT around beacon
      5: WALL_REPULSE
      6: TOWARD_SPAWN
      7: AWAY_SPAWN
      8: ORBIT_BEACON (deterministic left/right)

    Scale (N_SCALE=4):
      0: SHORT
      1: MEDIUM
      2: LONG
      3: XLONG
    """

    N_MOVE = 9
    N_SCALE = 4

    def __init__(self, params: ActionParams):
        self.p = params
        self.order_cooldown = int(getattr(self.p, "order_cooldown", 4))
        self.step_size = float(getattr(self.p, "step_size", 2.5))
        self.map_margin = float(getattr(self.p, "map_margin", 1.0))
        self.wall_repulsion_w = float(getattr(self.p, "wall_repulsion", 0.15))

        # step multipliers for scale head
        self.scale_multipliers = np.array(
            getattr(self.p, "step_multipliers", (0.5, 1.0, 1.5, 2.0)),
            dtype=np.float32,
        )

        # per-unit gating
        self._last_issue_loop_by_tag: Dict[int, int] = {}
        self._last_tgt_by_tag: Dict[int, Point2] = {}

        # remember spawn point for each marine
        self._spawn_by_tag: Dict[int, np.ndarray] = {}

    def reset(self):
        self._last_issue_loop_by_tag.clear()
        self._last_tgt_by_tag.clear()
        self._spawn_by_tag.clear()

    async def apply_actions(self, bot, marine_actions: Dict[int, int]) -> float:
        ms = marine_units(bot)
        if not ms:
            return 0.0

        beacon_pt = self._find_beacon_pos(bot)
        # print("beacon_pt:", beacon_pt)
        if beacon_pt is None:
            # no valid beacon position -> safest fallback: do nothing
            return 0.0

        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)

        beacon = np.array([float(beacon_pt.x), float(beacon_pt.y)], dtype=np.float32)

        moved_count = 0
        total = int(ms.amount)

        for m in ms:
            tag = int(m.tag)

            if tag not in self._spawn_by_tag:
                self._spawn_by_tag[tag] = np.array(
                    [float(m.position.x), float(m.position.y)],
                    dtype=np.float32,
                )

            # decode combined action
            a = int(marine_actions.get(tag, 0))
            a %= (self.N_MOVE * self.N_SCALE)
            move_idx = a // self.N_SCALE
            scale_idx = a % self.N_SCALE

            px = float(m.position.x)
            py = float(m.position.y)
            pos = np.array([px, py], dtype=np.float32)

            # beacon frame
            beacon_vec = beacon - pos
            beacon_dir = safe_normalize(beacon_vec)
            beacon_away = -beacon_dir
            strafe_left = np.array([-beacon_dir[1], beacon_dir[0]], dtype=np.float32)
            strafe_right = np.array([beacon_dir[1], -beacon_dir[0]], dtype=np.float32)

            # spawn frame
            spawn = self._spawn_by_tag[tag]
            spawn_vec = spawn - pos
            spawn_toward = safe_normalize(spawn_vec)
            spawn_away = -spawn_toward

            # wall repulsion (per-unit, not global centroid)
            rep = calculate_wall_repulsion(pos, W, H, self.map_margin)
            rep = safe_normalize(rep)

            move_vec = self._movement_vec(
                move_idx=move_idx,
                toward_beacon=beacon_dir,
                away_beacon=beacon_away,
                left=strafe_left,
                right=strafe_right,
                wall_repulse=rep,
                toward_spawn=spawn_toward,
                away_spawn=spawn_away,
                tag=tag,
            )

            # HOLD
            if np.linalg.norm(move_vec) < 1e-6:
                if move_idx == 0 and self._can_issue_unit(tag, bot, m.position):
                    self._issue_hold(m)
                    self._mark_issued_unit(tag, bot, m.position)
                continue

            # blend in a small wall-repulse prior unless the primitive itself is already wall repulse
            if move_idx != 5:
                move_vec = move_vec + self.wall_repulsion_w * rep

            final_move = safe_normalize(move_vec)

            step = self.step_size * float(self.scale_multipliers[scale_idx])

            tgt_pt = self._calc_target(
                px=px,
                py=py,
                move_vec=final_move,
                W=W,
                H=H,
                step_size=step,
                map_margin=self.map_margin,
            )

            if self._can_issue_unit(tag, bot, tgt_pt):
                m.move(tgt_pt)
                self._mark_issued_unit(tag, bot, tgt_pt)
                moved_count += 1

        return moved_count / float(total) if total > 0 else 0.0

    
    # Helpers
    def _movement_vec(
        self,
        move_idx: int,
        toward_beacon: np.ndarray,
        away_beacon: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        wall_repulse: np.ndarray,
        toward_spawn: np.ndarray,
        away_spawn: np.ndarray,
        tag: int,
    ) -> np.ndarray:
        if move_idx == 0: # HOLD
            return np.zeros(2, dtype=np.float32)
        if move_idx == 1: # toward beacon
            return toward_beacon
        if move_idx == 2: # away beacon
            return away_beacon
        if move_idx == 3: # strafe left around beacon
            return left
        if move_idx == 4: # strafe right around beacon
            return right
        if move_idx == 5: # wall repulse
            return wall_repulse
        if move_idx == 6: # toward spawn
            return toward_spawn
        if move_idx == 7: # away spawn
            return away_spawn
        if move_idx == 8: # orbit beacon
            return left if (tag % 2 == 0) else right

        return np.zeros(2, dtype=np.float32)

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
    
    # def _get_beacon_point(self, bot) -> Optional[Point2]:
    #     p = getattr(bot, "beacon_pos", None)
    #     if p is not None:
    #         return p
    #     return None