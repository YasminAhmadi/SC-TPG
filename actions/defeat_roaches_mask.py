# actions/defeat_roaches_mask.py
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
    DefeatRoaches executor for noisy env.

      - action side reads the SAME observed roach cache produced by feature side:
            bot._roach_obs_enemies_by_tag[tag]
      - before combat mode: use search obs radius
      - after combat mode: use combat obs radius
      - when local observation is empty, can fall back to virtual frame
    """

    N_MOVE = 9
    N_TGT = 4

    def __init__(self, params: ActionParams):
        self.p = params
        self.order_cooldown = int(getattr(self.p, "order_cooldown", 6))

        self._last_issue_loop_by_tag: Dict[int, int] = {}
        self._last_tgt_by_tag: Dict[int, Point2] = {}

    def reset(self):
        self._last_issue_loop_by_tag.clear()
        self._last_tgt_by_tag.clear()


    # helpers
    def _cfg_val(self, bot, name: str, default):
        if hasattr(self.p, name):
            return getattr(self.p, name)
        cfg = getattr(bot, "cfg", None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        return default

    def _should_activate_combat_mode(self, bot, marines, enemies) -> bool:
        if not marines or not enemies:
            return False

        activate_radius = float(self._cfg_val(bot, "roach_combat_activate_radius", 9.0))
        activate_r2 = activate_radius * activate_radius

        for m in marines:
            px = float(m.position.x)
            py = float(m.position.y)
            for u in enemies:
                dx = float(u.position.x) - px
                dy = float(u.position.y) - py
                d2 = dx * dx + dy * dy
                if d2 <= activate_r2:
                    return True
        return False

    def _fallback_local_observed_enemies(self, bot, marine) -> List[object]:
        """
        Fallback only if feature-side cache is absent.
        Still respects search/combat phase.
        """
        combat_mode = bool(getattr(bot, "_roach_combat_mode", False))

        if not combat_mode:
            marines = list(marine_units(bot))
            enemies_all = [
                u for u in getattr(bot, "enemy_units", [])
                if getattr(u, "type_id", None) == UnitTypeId.ROACH
            ]
            if self._should_activate_combat_mode(bot, marines, enemies_all):
                combat_mode = True
                bot._roach_combat_mode = True

        search_obs_radius = float(self._cfg_val(bot, "roach_search_obs_radius", 999.0))
        combat_obs_radius = float(self._cfg_val(bot, "roach_combat_obs_radius", 11.0))
        obs_radius = combat_obs_radius if combat_mode else search_obs_radius

        px = float(marine.position.x)
        py = float(marine.position.y)

        local = []
        for u in getattr(bot, "enemy_units", []):
            if getattr(u, "type_id", None) != UnitTypeId.ROACH:
                continue
            dx = float(u.position.x) - px
            dy = float(u.position.y) - py
            dist = float(np.sqrt(dx * dx + dy * dy) + 1e-6)
            if dist <= obs_radius:
                local.append(u)
        return local

    def _get_observed_enemies_for_unit(self, bot, marine) -> List[object]:
        now = int(bot.state.game_loop)
        cached_loop = int(getattr(bot, "_roach_obs_game_loop", -1))
        obs_by_tag = getattr(bot, "_roach_obs_enemies_by_tag", None)

        if (obs_by_tag is not None) and (cached_loop == now):
            return list(obs_by_tag.get(int(marine.tag), []))

        return self._fallback_local_observed_enemies(bot, marine)

    def _get_global_observed_focus(self, bot):
        """
        Global focus is computed ONLY from current observed union cache.
        No oracle access to full true enemy set.
        """
        focus_tag = getattr(bot, "_roach_obs_global_focus_tag", None)
        global_obs = getattr(bot, "_roach_obs_global_enemies", {}) or {}

        if focus_tag is None:
            return None
        return global_obs.get(int(focus_tag), None)


    # main execution
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

        m_cent = np.array([ms.center.x, ms.center.y], dtype=np.float32)
        rep = calculate_wall_repulsion(m_cent, W, H, map_margin)

        ally_positions = np.array(
            [[float(u.position.x), float(u.position.y)] for u in ms],
            dtype=np.float32,
        )

        global_focus_unit = self._get_global_observed_focus(bot)

        moved_count = 0
        total = int(ms.amount)

        for idx, m in enumerate(ms):
            tag = int(m.tag)

            a = int(marine_actions.get(tag, 0))
            a %= (self.N_MOVE * self.N_TGT)
            move_idx = a // self.N_TGT
            tgt_idx = a % self.N_TGT
            # print("move_idx: ", move_idx)
            # print("tgt_idx: ", tgt_idx)

            px = float(m.position.x)
            py = float(m.position.y)
            pos = np.array([px, py], dtype=np.float32)

            ally_towards = np.zeros(2, dtype=np.float32)
            ally_away = np.zeros(2, dtype=np.float32)
            if total > 1:
                others = np.delete(ally_positions, idx, axis=0)
                c = others.mean(axis=0)
                vec_a = c - pos
                ally_towards = safe_normalize(vec_a)
                ally_away = -ally_towards

            enemies = self._get_observed_enemies_for_unit(bot, m)
            enemy_visible = (len(enemies) > 0)

            # No currently observed enemy -> use virtual frame if available
            if not enemy_visible:
                virtual_frames = getattr(bot, "_combat_virtual_frame_by_tag", {}) or {}
                vf = virtual_frames.get(tag, None)

                if vf is not None and vf.get("point", None) is not None:
                    vpt = vf["point"]
                    vx = float(vpt.x) - px
                    vy = float(vpt.y) - py

                    vdir = safe_normalize(np.array([vx, vy], dtype=np.float32))
                    vaway = -vdir

                    strafe_left = np.array([-vdir[1], vdir[0]], dtype=np.float32)
                    strafe_right = np.array([vdir[1], -vdir[0]], dtype=np.float32)

                    move_vec = self._movement_vec(
                        move_idx=move_idx,
                        toward=vdir,
                        away=vaway,
                        left=strafe_left,
                        right=strafe_right,
                        ally_towards=ally_towards,
                        ally_away=ally_away,
                        rep=rep,
                        tag=tag,
                    )
                else:
                    move_vec = self._movement_vec(
                        move_idx=move_idx,
                        toward=np.zeros(2, np.float32),
                        away=np.zeros(2, np.float32),
                        left=np.zeros(2, np.float32),
                        right=np.zeros(2, np.float32),
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

                move_vec = move_vec + wall_repulsion_w * rep
                final_move = safe_normalize(move_vec)

                tgt_pt = self._calc_target(px, py, final_move, W, H, step_size, map_margin)
                if self._can_issue_unit(tag, bot, tgt_pt):
                    m.move(tgt_pt)
                    self._mark_issued_unit(tag, bot, tgt_pt)
                    moved_count += 1
                continue


            # Observed combat
            ex = np.array([float(u.position.x) for u in enemies], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in enemies], dtype=np.float32)

            dx = ex - px
            dy = ey - py
            dists = np.sqrt(dx * dx + dy * dy) + 1e-6

            j_near = int(np.argmin(dists))
            nearest_enemy = enemies[j_near]

            target_unit, target_dist = self._select_target(
                enemies=enemies,
                px=px,
                py=py,
                tgt_idx=tgt_idx,
                global_focus_unit=global_focus_unit,
            )

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

            retreat_like = (move_idx in (2, 7))
            can_attack = (target_unit is not None) and (target_dist <= attack_range) and (tgt_idx != 0)

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

            move_vec = move_vec + wall_repulsion_w * rep
            final_move = safe_normalize(move_vec)
            tgt_pt = self._calc_target(px, py, final_move, W, H, step_size, map_margin)

            if self._can_issue_unit(tag, bot, tgt_pt):
                m.move(tgt_pt)
                self._mark_issued_unit(tag, bot, tgt_pt)
                moved_count += 1

        return moved_count / float(total) if total > 0 else 0.0


    # target selection
    def _select_target(
        self,
        enemies,
        px: float,
        py: float,
        tgt_idx: int,
        global_focus_unit: Optional[object],
    ) -> Tuple[Optional[object], float]:
        """
        tgt_idx:
          0 -> NO_ATTACK
          1 -> nearest to self
          2 -> lowest hp (local observed only)
          3 -> global focus lowest hp (from observed union only)
        """
        if tgt_idx == 0 or not enemies:
            return None, 1e9

        if tgt_idx == 1: # nearest
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

        if tgt_idx == 2: # local lowest hp
            best_u = min(enemies, key=lambda u: float(getattr(u, "health", 1e9)))
            dx = float(best_u.position.x) - px
            dy = float(best_u.position.y) - py
            return best_u, float(np.sqrt(dx * dx + dy * dy) + 1e-6)

        if tgt_idx == 3: # global observed focus
            if global_focus_unit is not None:
                # only use global focus if marine currently observes it
                for u in enemies:
                    if int(u.tag) == int(global_focus_unit.tag):
                        dx = float(u.position.x) - px
                        dy = float(u.position.y) - py
                        return u, float(np.sqrt(dx * dx + dy * dy) + 1e-6)

            # fallback to local lowest hp if team-wide focus isn't locally visible
            best_u = min(enemies, key=lambda u: float(getattr(u, "health", 1e9)))
            dx = float(best_u.position.x) - px
            dy = float(best_u.position.y) - py
            return best_u, float(np.sqrt(dx * dx + dy * dy) + 1e-6)

        return None, 1e9

    # movement primitives
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
            return safe_normalize(rep)
        if move_idx == 6: # TOWARD ally-center
            return ally_towards
        if move_idx == 7: # AWAY ally-center
            return ally_away
        if move_idx == 8: # ORBIT
            return left if (tag % 2 == 0) else right
        return np.zeros(2, dtype=np.float32)


    # utility
    def _calc_target(self, px: float, py: float, move_vec: np.ndarray, W: float, H: float,
                     step_size: float, map_margin: float) -> Point2:
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