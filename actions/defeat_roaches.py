# actions/defeat_roaches.py
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List

from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId

from utils import (
    marine_units,
    safe_normalize,
    calculate_wall_repulsion,
    clip_position,
)
from actions.params import ActionParams


class ActionExecutor:
    """
    Factorized action executor for DefeatRoaches.

    Combined action encoding:
      combined = move_idx * N_TGT + tgt_idx

    Movement (N_MOVE=9):
      0: HOLD
      1: TOWARD target-frame
      2: AWAY target-frame
      3: STRAFE_LEFT around target-frame
      4: STRAFE_RIGHT around target-frame
      5: WALL_REPULSE
      6: TOWARD ally-center
      7: AWAY ally-center
      8: ORBIT target-frame (deterministic left/right by tag)

    Targeting (N_TGT=4):
      0: NO_ATTACK
      1: ATTACK_NEAREST_TO_SELF
      2: ATTACK_LOWEST_HP
      3: ATTACK_GLOBAL_FOCUS_LOWEST_HP

    """

    N_MOVE = 9
    N_TGT = 4

    def __init__(self, params: ActionParams):
        self.p = params
        self.order_cooldown = int(getattr(self.p, "order_cooldown", 6))

        # per-unit gating
        self._last_issue_loop_by_tag: Dict[int, int] = {}
        self._last_tgt_by_tag: Dict[int, Point2] = {}

    def reset(self):
        self._last_issue_loop_by_tag.clear()
        self._last_tgt_by_tag.clear()

    async def apply_actions(self, bot, marine_actions: Dict[int, int]) -> float:
        ms = marine_units(bot)
        if not ms:
            return 0.0

        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)

        map_margin = float(getattr(self.p, "map_margin", 1.0))
        wall_repulsion_w = float(getattr(self.p, "wall_repulsion", 0.0))
        step_size = float(getattr(self.p, "step_size", 2.5))
        attack_range = float(getattr(self.p, "attack_range_approx", 5.0))

        enemies = list(getattr(bot, "enemy_units", []))
        
        roach_units = [u for u in enemies if getattr(u, "type_id", None) == UnitTypeId.ROACH]
        target_pool: List[object] = roach_units if len(roach_units) > 0 else enemies
        enemy_visible = len(target_pool) > 0

        ally_positions = np.array(
            [[float(u.position.x), float(u.position.y)] for u in ms],
            dtype=np.float32,
        )

        # HP lowest roach
        global_focus_unit = self._select_global_focus_target(target_pool)


        if enemy_visible:
            ex = np.array([float(u.position.x) for u in target_pool], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in target_pool], dtype=np.float32)

        moved_count = 0
        total = int(ms.amount)

        for idx, m in enumerate(ms):
            tag = int(m.tag)

            a = int(marine_actions.get(tag, 0))
            a %= (self.N_MOVE * self.N_TGT)

            move_idx = a // self.N_TGT
            tgt_idx = a % self.N_TGT

            px = float(m.position.x)
            py = float(m.position.y)
            pos = np.array([px, py], dtype=np.float32)

            # ally vectors
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

            if not enemy_visible:

                move_vec = self._movement_vec(
                    move_idx=move_idx,
                    toward=np.zeros(2, dtype=np.float32),
                    away=np.zeros(2, dtype=np.float32),
                    left=np.zeros(2, dtype=np.float32),
                    right=np.zeros(2, dtype=np.float32),
                    ally_towards=ally_towards,
                    ally_away=ally_away,
                    rep=rep,
                    tag=tag,
                )

                if np.linalg.norm(move_vec) < 1e-6:
                    if move_idx == 0 and self._can_issue_unit(tag, bot, m.position):
                        self._issue_hold(m)
                        self._mark_issued_unit(tag, bot, m.position)
                    continue

                if move_idx != 5:
                    move_vec = move_vec + wall_repulsion_w * rep
                final_move = safe_normalize(move_vec)

                tgt_pt = self._calc_target(px, py, final_move, W, H, step_size, map_margin)
                if self._can_issue_unit(tag, bot, tgt_pt):
                    m.move(tgt_pt)
                    self._mark_issued_unit(tag, bot, tgt_pt)
                    moved_count += 1
                continue

            # nearest enemy (for fallback frame)
            dx = ex - px
            dy = ey - py
            dists = np.sqrt(dx * dx + dy * dy) + 1e-6
            j_near = int(np.argmin(dists))
            nearest_enemy = target_pool[j_near]

            # attack target selected by target head
            target_unit, target_dist = self._select_target(
                enemies=target_pool,
                px=px,
                py=py,
                tgt_idx=tgt_idx,
                global_focus_unit=global_focus_unit,
            )

            # movement frame:
            # if target head selected a real target -> use it
            # else fallback to nearest enemy
            frame_unit = target_unit if (target_unit is not None) else nearest_enemy

            fx = float(frame_unit.position.x) - px
            fy = float(frame_unit.position.y) - py
            frame_dir = safe_normalize(np.array([fx, fy], dtype=np.float32))
            frame_away = -frame_dir

            strafe_left = np.array([-frame_dir[1], frame_dir[0]], dtype=np.float32)
            strafe_right = np.array([frame_dir[1], -frame_dir[0]], dtype=np.float32)

            move_vec = self._movement_vec(
                move_idx=move_idx,
                toward=frame_dir,
                away=frame_away,
                left=strafe_left,
                right=strafe_right,
                ally_towards=ally_towards,
                ally_away=ally_away,
                rep=rep,
                tag=tag,
            )


            retreat_like = (move_idx in (2, 7)) # AWAY target-frame / AWAY ally-center
            can_attack = (
                target_unit is not None
                and target_dist <= attack_range
                and tgt_idx != 0
            )

            if can_attack and (not retreat_like):
                if self._can_issue_unit(tag, bot, target_unit.position):
                    m.attack(target_unit)
                    self._mark_issued_unit(tag, bot, target_unit.position)
                continue

            if np.linalg.norm(move_vec) < 1e-6:
                if move_idx == 0 and self._can_issue_unit(tag, bot, m.position):
                    self._issue_hold(m)
                    self._mark_issued_unit(tag, bot, m.position)
                continue

            if move_idx != 5:
                move_vec = move_vec + wall_repulsion_w * rep
            final_move = safe_normalize(move_vec)

            tgt_pt = self._calc_target(px, py, final_move, W, H, step_size, map_margin)
            if self._can_issue_unit(tag, bot, tgt_pt):
                m.move(tgt_pt)
                self._mark_issued_unit(tag, bot, tgt_pt)
                moved_count += 1

        return moved_count / float(total) if total > 0 else 0.0


    # Target selectors
    def _select_global_focus_target(self, enemies) -> Optional[object]:
        """Global focus"""
        if not enemies:
            return None
        return min(enemies, key=lambda u: float(getattr(u, "health", 1e9)))

    def _select_target(
        self,
        enemies,
        px: float,
        py: float,
        tgt_idx: int,
        global_focus_unit: Optional[object],
    ) -> Tuple[Optional[object], float]:
        """
        Return (target_unit, distance)

        tgt_idx:
          0 -> NO_ATTACK
          1 -> nearest to self
          2 -> lowest hp
          3 -> global focus lowest hp
        """
        if tgt_idx == 0 or not enemies:
            return None, 1e9

        if tgt_idx == 1: # nearest to self
            best_u = None
            best_d2 = 1e30
            for u in enemies:
                dx = float(u.position.x) - px
                dy = float(u.position.y) - py
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_u = u
            return best_u, float(np.sqrt(best_d2) + 1e-6)

        if tgt_idx == 2: # lowest hp
            best_u = min(enemies, key=lambda u: float(getattr(u, "health", 1e9)))
            dx = float(best_u.position.x) - px
            dy = float(best_u.position.y) - py
            return best_u, float(np.sqrt(dx * dx + dy * dy) + 1e-6)

        if tgt_idx == 3: # global focus lowest hp
            if global_focus_unit is None:
                return None, 1e9
            dx = float(global_focus_unit.position.x) - px
            dy = float(global_focus_unit.position.y) - py
            return global_focus_unit, float(np.sqrt(dx * dx + dy * dy) + 1e-6)

        return None, 1e9


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
        if move_idx == 1: # TOWARD target-frame
            return toward
        if move_idx == 2: # AWAY target-frame
            return away
        if move_idx == 3: # STRAFE_LEFT
            return left
        if move_idx == 4: # STRAFE_RIGHT
            return right
        if move_idx == 5: # WALL_REPULSE
            return rep
        if move_idx == 6: # TOWARD ally-center
            return ally_towards
        if move_idx == 7: # AWAY ally-center
            return ally_away
        if move_idx == 8: # ORBIT
            return left if (tag % 2 == 0) else right

        return np.zeros(2, dtype=np.float32)


    # Helpers
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