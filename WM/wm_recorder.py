from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import json
import numpy as np

from sc2.data import Alliance
from sc2.ids.unit_typeid import UnitTypeId

from WM.params import WMRecorderParams


# helpers
def ut_value(ut) -> int:
    """Safely get integer unit type id."""
    v = getattr(ut, "value", None)
    if v is not None:
        return int(v)
    # 
    if isinstance(ut, (int, np.integer)):
        return int(ut)
    raise TypeError(f"Unsupported UnitTypeId type: {type(ut)}")

def point_xy(p) -> Tuple[float, float]:
    return float(p.x), float(p.y)

def mirror_x_coord(x: float, map_w: float) -> float:
    return float(map_w) - float(x)

def clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)




@dataclass
class EpisodeMeta:
    env_name: str = ""
    episode_id: int = -1
    generation: int = -1
    map_w: float = 0.0
    map_h: float = 0.0
    mirror_x: bool = False
    canonical_side: str = "right"

    # fixed slots within episode
    marine_tags: List[int] = None # len<=9
    enemy_tags: List[int] = None # len<=10
    enemy_types_init: List[int] = None

    marine_slot_order: str = "y_desc"
    enemy_slot_order: str = "type_then_y_desc"
    
    
    


@dataclass
class FrameRecord:
    game_loop: int

    marine_xy: np.ndarray # (9,2)
    marine_hp: np.ndarray # (9,)
    marine_alive: np.ndarray # (9,)
    marine_action: np.ndarray # (9,)

    enemy_xy: np.ndarray # (10,2)
    enemy_hp: np.ndarray # (10,)
    enemy_alive: np.ndarray # (10,)
    enemy_type: np.ndarray # (10,)

    reward: np.ndarray
    done: np.ndarray
    
    tokens: np.ndarray # (N, D)


class WMRecorder:
    """
    WM training
    - Marine: 9 slots, y-desc
    - Enemy: 10 slots, (type_priority -> y-desc -> x-asc)
    - token fields:
        [type_code, side_code, alive, x_norm, y_norm, hp_norm, dx_norm, dy_norm, distx_norm, disty_norm]
    """

    def __init__(
        self,
        out_dir: str | Path,
        env_name: str,
        params: WMRecorderParams
    ):
        self.out_dir = Path(out_dir)
        self.env_name = str(env_name)
        self.p = params
        
        self.max_marines = int(params.max_marines)
        self.max_enemies = int(params.max_enemies)

        assert params.canonical_side in ("right", "left")
        self.canonical_side = params.canonical_side

        self.hp_marine_max = float(params.hp_marine_max)
        self.hp_zergling_max = float(params.hp_zergling_max)
        self.hp_baneling_max = float(params.hp_baneling_max)

        self.save_npz = bool(params.save_npz)
        self.debug = bool(params.debug)

        self.meta: Optional[EpisodeMeta] = None

        self._tag2slot_marine: Dict[int, int] = {}
        self._tag2slot_enemy: Dict[int, int] = {}

        self.frames: List[FrameRecord] = []

        self._episode_started = False
        self._last_saved_path: Optional[Path] = None

        # for dx/dy
        self._prev_marine_xy_slots = np.zeros((self.max_marines, 2), dtype=np.float32)
        self._prev_enemy_xy_slots = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self._prev_marine_alive = np.zeros((self.max_marines,), dtype=np.int8)
        self._prev_enemy_alive = np.zeros((self.max_enemies,), dtype=np.int8)
        self._has_prev = False
        
        
        self._enemy_slot_type_init = np.zeros((self.max_enemies,), dtype=np.int8)

    
    # Episode lifecycle
    
    def begin_episode(self, bot, episode_id: int, generation: int = -1) -> None:
        map_w = float(bot.game_info.map_size.x)
        map_h = float(bot.game_info.map_size.y)

        mirror = self._infer_mirror(bot, map_w)
        # print("mirror:", mirror)

        # Marine slots (y desc in canonical space)
        marines = self._get_controlled_marines(bot)
        m_infos = []
        for u in marines:
            tag = int(u.tag)
            x, y = point_xy(u.position)
            if mirror:
                x = mirror_x_coord(x, map_w)
            m_infos.append((tag, x, y))
        
        m_infos.sort(key=lambda t: (-t[2], t[1]))  # y desc, x asc
        
        marine_tags = [t[0] for t in m_infos][: self.max_marines]

        # Enemy slots (type_priority -> y desc -> x asc)
        enemies = self._get_enemy_combat_units(bot)
        e_infos = []
        for e in enemies:
            tag = int(e["tag"])
            x = float(e["x"])
            y = float(e["y"])
            if mirror:
                x = mirror_x_coord(x, map_w)
            
            tcode = float(e["type_code"]) # 1 zergling, 2 baneling
            e_infos.append((tag, tcode, x, y))
        
        # baneling
        def _prio(tc: float) -> int:
            return 0 if int(tc) == int(self.p.codes.TYPE_BANELING) else 1
        
        e_infos.sort(key=lambda t: (_prio(t[1]), -t[3], t[2]))
        
        chosen = e_infos[: self.max_enemies]# (tag, tcode, x, y)
        enemy_tags = [int(t[0]) for t in chosen]
        enemy_types_init = [int(t[1]) for t in chosen] 
        
        # recorder
        self._enemy_slot_type_init.fill(0)
        for j, tc in enumerate(enemy_types_init):
            self._enemy_slot_type_init[j] = np.int8(tc)
        

        self.meta = EpisodeMeta(
            env_name=self.env_name,
            episode_id=int(episode_id),
            generation=int(generation),
            map_w=map_w,
            map_h=map_h,
            mirror_x=bool(mirror),
            canonical_side=self.canonical_side,
            marine_tags=list(marine_tags),
            enemy_tags=list(enemy_tags),
            enemy_types_init=list(enemy_types_init),
            marine_slot_order="y_desc",
            enemy_slot_order="baneling_first_then_y_desc",
        )

        self._tag2slot_marine = {tag: i for i, tag in enumerate(marine_tags)}
        self._tag2slot_enemy = {tag: i for i, tag in enumerate(enemy_tags)}

        self.frames = []
        self._episode_started = True

        # reset dx/dy history
        self._prev_marine_xy_slots.fill(0.0)
        self._prev_enemy_xy_slots.fill(0.0)
        self._prev_marine_alive.fill(0)
        self._prev_enemy_alive.fill(0)
        self._has_prev = False

        if self.debug:
            side = "MIRROR" if mirror else "NO_MIRROR"
            print(f"[WMREC] begin_episode ep={episode_id} gen={generation} {side} "
                  f"marine_slots={len(marine_tags)} enemy_slots={len(enemy_tags)}", flush=True)

    
    def finalize_last_step(self, final_bonus: float, force_done: bool = True) -> None:
        if not self.frames:
            return
        fr = self.frames[-1]
        fr.reward = np.float32(float(fr.reward) + float(final_bonus))
        if force_done:
            fr.done = np.int8(1)
    
    
    def set_last_reward_done(self, step_reward: float, done: bool) -> None:
        if not self.frames:
            return
        self.frames[-1].reward = np.float32(step_reward)
        self.frames[-1].done = np.int8(done)
    
    def record_step(self, bot, marine_actions: Dict[int, int], step_reward: float, done: bool) -> None:
        if not self._episode_started or self.meta is None:
            return

        reward = np.float32(step_reward)
        done = np.int8(done)
        
        map_w = self.meta.map_w
        map_h = self.meta.map_h
        mirror = self.meta.mirror_x

        # collect marines by tag
        xy_m = np.zeros((self.max_marines, 2), dtype=np.float32)
        hp_m = np.zeros((self.max_marines,), dtype=np.float32)
        alive_m = np.zeros((self.max_marines,), dtype=np.int8)
        act_m = np.zeros((self.max_marines,), dtype=np.int16)

        marines = self._get_controlled_marines(bot)
        marine_by_tag = {int(u.tag): u for u in marines}

        for tag, slot in self._tag2slot_marine.items():
            u = marine_by_tag.get(tag, None)
            act_m[slot] = int(marine_actions.get(tag, 0))
            if u is None:
                alive_m[slot] = 0
                continue
            x, y = point_xy(u.position)
            if mirror:
                x = mirror_x_coord(x, map_w)
            xy_m[slot] = (x, y)
            hp_m[slot] = float(getattr(u, "health", 0.0) or 0.0)
            alive_m[slot] = 1

        # collect enemies by tag
        xy_e = np.zeros((self.max_enemies, 2), dtype=np.float32)
        hp_e = np.zeros((self.max_enemies,), dtype=np.float32)
        alive_e = np.zeros((self.max_enemies,), dtype=np.int8)
        typ_e = np.zeros((self.max_enemies,), dtype=np.int8) # 1/2

        enemies = self._get_enemy_combat_units(bot) # list of dicts
        enemy_by_tag = {int(e["tag"]): e for e in enemies}

        for tag, slot in self._tag2slot_enemy.items():
            
            typ_e[slot] = self._enemy_slot_type_init[slot]

            e = enemy_by_tag.get(tag, None)
            if e is None:
                alive_e[slot] = 0
                # xy/hp 
                continue

            x = float(e["x"]); y = float(e["y"])
            if mirror:
                x = mirror_x_coord(x, map_w)
            xy_e[slot] = (x, y)
            hp_e[slot] = float(e["hp"])
            alive_e[slot] = 1

            cur_type = int(e["type_code"])
            typ_e[slot] = cur_type

            
            if int(self._enemy_slot_type_init[slot]) == 0:
                self._enemy_slot_type_init[slot] = np.int8(cur_type)
                
                if self.meta is not None and self.meta.enemy_types_init is not None:
                    if slot < len(self.meta.enemy_types_init):
                        self.meta.enemy_types_init[slot] = int(cur_type)

        # dx/dy
        dx_m = np.zeros_like(xy_m, dtype=np.float32)
        dx_e = np.zeros_like(xy_e, dtype=np.float32)
        if self._has_prev:
            # only if alive in both frames; else 0
            mask_m = (alive_m == 1) & (self._prev_marine_alive == 1)
            mask_e = (alive_e == 1) & (self._prev_enemy_alive == 1)
            dx_m[mask_m] = xy_m[mask_m] - self._prev_marine_xy_slots[mask_m]
            dx_e[mask_e] = xy_e[mask_e] - self._prev_enemy_xy_slots[mask_e]

        tokens = self._build_tokens_all(
            xy_m=xy_m, 
            hp_m=hp_m, 
            alive_m=alive_m, 
            dxy_m=dx_m,
            xy_e=xy_e, 
            hp_e=hp_e, 
            alive_e=alive_e, 
            typ_e=typ_e, 
            dxy_e=dx_e,
            map_w=map_w, 
            map_h=map_h,
        )
        # print("tokens:", tokens)#(19, 10)

        fr = FrameRecord(
            game_loop=int(bot.state.game_loop),
            marine_xy=xy_m, 
            marine_hp=hp_m, 
            marine_alive=alive_m, 
            marine_action=act_m,
            enemy_xy=xy_e, 
            enemy_hp=hp_e, 
            enemy_alive=alive_e, 
            enemy_type=typ_e,
            
            reward=reward,
            done=done,
            
            tokens=tokens,
        )
        self.frames.append(fr)

        # update prev
        self._prev_marine_xy_slots[:] = xy_m
        self._prev_enemy_xy_slots[:] = xy_e
        self._prev_marine_alive[:] = alive_m
        self._prev_enemy_alive[:] = alive_e
        self._has_prev = True
        
        # if self.debug and len(self.frames) < 10:
        #     print("[WMREC] enemy typ_e:", typ_e.tolist(), "alive:", alive_e.tolist(), flush=True)


    def end_episode(self, reason: str, ep_return: float, steps: int, final_snapshot: Optional[dict] = None) -> None:
        if not self._episode_started or self.meta is None:
            return
        self._end_info = dict(reason=str(reason), ep_return=float(ep_return), steps=int(steps))
        if final_snapshot is not None:
            self._end_info["final_snapshot"] = final_snapshot
        self._episode_started = False
        if self.debug:
            print(f"[WMREC] end_episode reason={reason} return={ep_return:.3f} steps={steps}", flush=True)

    def save_episode(self) -> Optional[Path]:
        if self.meta is None or len(self.frames) == 0:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ep = self.meta.episode_id
        gen = self.meta.generation
        base = f"wm_ep{ep:05d}_gen{gen:03d}"
        npz_path = self.out_dir / f"{base}.npz"
        json_path = self.out_dir / f"{base}.json"

        loops = np.array([f.game_loop for f in self.frames], dtype=np.int32)

        marine_xy = np.stack([f.marine_xy for f in self.frames], axis=0)
        marine_hp = np.stack([f.marine_hp for f in self.frames], axis=0)
        marine_alive = np.stack([f.marine_alive for f in self.frames], axis=0)
        marine_action = np.stack([f.marine_action for f in self.frames], axis=0)

        enemy_xy = np.stack([f.enemy_xy for f in self.frames], axis=0)
        enemy_hp = np.stack([f.enemy_hp for f in self.frames], axis=0)
        enemy_alive = np.stack([f.enemy_alive for f in self.frames], axis=0)
        enemy_type = np.stack([f.enemy_type for f in self.frames], axis=0)

        rewards = np.array([float(f.reward) for f in self.frames], dtype=np.float32)
        dones = np.array([int(f.done) for f in self.frames], dtype=np.int8)
        
        
        tokens = np.stack([f.tokens for f in self.frames], axis=0)

        # print("np.savez_compressed")
        np.savez_compressed(
            npz_path,
            loops=loops,
            marine_xy=marine_xy, 
            marine_hp=marine_hp, 
            marine_alive=marine_alive, 
            marine_action=marine_action,
            enemy_xy=enemy_xy, 
            enemy_hp=enemy_hp, 
            enemy_alive=enemy_alive, 
            enemy_type=enemy_type,
            rewards=rewards, 
            dones=dones,
            tokens=tokens,
        )

        meta_dict = asdict(self.meta)
        meta_dict["num_frames"] = int(len(self.frames))
        meta_dict["token_shape"] = list(tokens.shape)
        end_info = getattr(self, "_end_info", None)
        if end_info is not None:
            meta_dict["end"] = end_info

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)

        self._last_saved_path = npz_path
        if self.debug:
            print(f"[WMREC] saved: {npz_path} (+ {json_path})", flush=True)
        return npz_path


    # Internal: unit queries
    def _get_controlled_marines(self, bot):
        ms = bot.units(UnitTypeId.MARINE)
        ctrl = getattr(bot, "controlled_marine_tags", None)
        if ctrl:
            ctrl_set = set(int(x) for x in ctrl)
            return ms.filter(lambda u: int(u.tag) in ctrl_set)
        return ms

    def _get_enemy_combat_units(self, bot) -> List[dict]:
        """
        raw_data get enemy zergling/baneling
        return list of dict: {tag,x,y,hp,type_code}
        """
        ling_id = ut_value(UnitTypeId.ZERGLING)
        bane_id = ut_value(UnitTypeId.BANELING)

        out = []
        try:
            raw = bot.state.observation.raw_data
            for u in raw.units:
                if getattr(u, "alliance", None) != Alliance.Enemy.value:
                    continue
                hp = float(getattr(u, "health", 0.0) or 0.0)
                if hp <= 0:
                    continue
                ut = int(getattr(u, "unit_type", -1))
                if ut == ling_id:
                    tcode = self.p.codes.TYPE_ZERGLING
                elif ut == bane_id:
                    tcode = self.p.codes.TYPE_BANELING
                else:
                    continue
                out.append(dict(
                    tag=int(u.tag),
                    x=float(u.pos.x),
                    y=float(u.pos.y),
                    hp=hp,
                    type_code=tcode,
                ))
        except Exception:
            # fallback: bot.enemy_units
            for e in bot.enemy_units:
                hp = float(getattr(e, "health", 0.0) or 0.0)
                if hp <= 0:
                    continue
                if e.type_id == UnitTypeId.ZERGLING:
                    tcode = self.p.codes.TYPE_ZERGLING
                elif e.type_id == UnitTypeId.BANELING:
                    tcode = self.p.codes.TYPE_BANELING
                else:
                    continue
                out.append(dict(
                    tag=int(e.tag),
                    x=float(e.position.x),
                    y=float(e.position.y),
                    hp=hp,
                    type_code=tcode,
                ))
        return out

    def _infer_mirror(self, bot, map_w: float) -> bool:
        ms = bot.units(UnitTypeId.MARINE)
        if not ms or ms.amount == 0:
            return False
        m_mean_x = float(np.mean([float(u.position.x) for u in ms]))

        # enemy mean x from raw if possible
        e_mean_x = None
        try:
            raw = bot.state.observation.raw_data
            xs = []
            for u in raw.units:
                if getattr(u, "alliance", None) != Alliance.Enemy.value:
                    continue
                hp = float(getattr(u, "health", 0.0) or 0.0)
                if hp <= 0:
                    continue
                xs.append(float(u.pos.x))
            if xs:
                e_mean_x = float(np.mean(xs))
        except Exception:
            pass
        if e_mean_x is None:
            e_mean_x = float(map_w) / 2.0

        marine_is_left = (m_mean_x < e_mean_x)
        if self.canonical_side == "right":
            return bool(marine_is_left)
        else:
            return bool(not marine_is_left)

    # Token builder
    def _hp_norm_by_type(self, type_code: float, hp: float) -> float:
        if int(type_code) == int(self.p.codes.TYPE_MARINE):
            return float(hp) / max(1e-6, self.hp_marine_max)
        if int(type_code) == int(self.p.codes.TYPE_ZERGLING):
            return float(hp) / max(1e-6, self.hp_zergling_max)
        if int(type_code) == int(self.p.codes.TYPE_BANELING):
            return float(hp) / max(1e-6, self.hp_baneling_max)
        return float(hp) / 100.0

    def _build_tokens_all(
        self,
        xy_m: np.ndarray, hp_m: np.ndarray, alive_m: np.ndarray, dxy_m: np.ndarray,
        xy_e: np.ndarray, hp_e: np.ndarray, alive_e: np.ndarray, typ_e: np.ndarray, dxy_e: np.ndarray,
        map_w: float, map_h: float,
    ) -> np.ndarray:
        """
        token = [type_code, side_code, alive, x_norm, y_norm, hp_norm, dx_norm, dy_norm, distx_norm, disty_norm]
        N = 9 + 10, D = 10
        """
        N = self.max_marines + self.max_enemies
        D = 10
        tok = np.zeros((N, D), dtype=np.float32)

        W = max(1e-6, float(map_w))
        H = max(1e-6, float(map_h))

        # marines first
        for i in range(self.max_marines):
            alive = float(alive_m[i])
            if alive <= 0.0:
                # dead token stays zeros except type/side/alive (可选)
                tok[i, 0] = self.p.codes.TYPE_MARINE
                tok[i, 1] = self.p.codes.SIDE_SELF
                tok[i, 2] = 0.0
                continue

            x = float(xy_m[i, 0]); y = float(xy_m[i, 1])
            dx = float(dxy_m[i, 0]); dy = float(dxy_m[i, 1])
            hp = float(hp_m[i])

            tok[i, 0] = self.p.codes.TYPE_MARINE
            tok[i, 1] = self.p.codes.SIDE_SELF
            tok[i, 2] = 1.0
            tok[i, 3] = x / W
            tok[i, 4] = y / H
            tok[i, 5] = self._hp_norm_by_type(self.p.codes.TYPE_MARINE, hp)
            tok[i, 6] = dx / W
            tok[i, 7] = dy / H

            distx = min(x, W - x) / W
            disty = min(y, H - y) / H
            tok[i, 8] = distx
            tok[i, 9] = disty

        # enemies next
        base = self.max_marines
        for j in range(self.max_enemies):
            idx = base + j
            alive = float(alive_e[j])
            tcode = float(typ_e[j]) if alive > 0 else float(typ_e[j]) # dead
            if int(tcode) == 0:
               
                pass

            tok[idx, 0] = tcode
            tok[idx, 1] = self.p.codes.SIDE_ENEMY
            tok[idx, 2] = alive

            if alive <= 0.0:
                continue

            x = float(xy_e[j, 0]); y = float(xy_e[j, 1])
            dx = float(dxy_e[j, 0]); dy = float(dxy_e[j, 1])
            hp = float(hp_e[j])

            tok[idx, 3] = x / W
            tok[idx, 4] = y / H
            tok[idx, 5] = self._hp_norm_by_type(tcode, hp)
            tok[idx, 6] = dx / W
            tok[idx, 7] = dy / H

            distx = min(x, W - x) / W
            disty = min(y, H - y) / H
            tok[idx, 8] = distx
            tok[idx, 9] = disty

        # clip to reasonable range
        tok[:, 3:6] = clip01(tok[:, 3:6]) # x,y,hp
        tok[:, 8:10] = clip01(tok[:, 8:10]) # distx, disty

        return tok
