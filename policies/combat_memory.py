# policies/combat_memory.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sc2.position import Point2

from tasks.spec import UnitObs
from utils import marine_units


class CombatMemoryMaskedPolicy:
    """
    Combat Memory

      - input = observed raw 18 dims + local combat memory 14 dims
      - visible enemies automatically refresh short-term threat/target memory
      - winning learner registers only gate persistence / clearing
      - action side can use exported virtual frames when current enemy observation is empty
    """

    name = "combat_memory"

    # raw observed features dim
    BASE_DIM = 18
    MEM_DIM = 14

    # timer / memory hyperparams
    MAX_AGE = 6.0
    MAX_TIMER = 8.0

    # memory layout indices
    I_PANIC = 0
    I_CYC = 1
    I_HAZ = 2

    I_THREAT_DX = 3
    I_THREAT_DY = 4
    I_THREAT_DIST = 5
    I_THREAT_AGE = 6
    I_THREAT_TYPE = 7

    I_TARGET_DX = 8
    I_TARGET_DY = 9
    I_TARGET_DIST = 10
    I_TARGET_AGE = 11

    I_RETREAT_TIMER = 12
    I_TARGET_TIMER = 13

    def __init__(self, env_n: int = 36, base_dim: int = 18):
        self.env_n = int(env_n)
        self.base_dim = int(base_dim)

        self.agent_action_n = int(env_n)
        self.state_dim = int(self.BASE_DIM + self.MEM_DIM)

        # trainer must be created with nRegisters >= 12
        self.n_tpg_registers = 12

        self._mem: Dict[int, np.ndarray] = {}
        self._prev_hp: Dict[int, float] = {}

        # exported each step for executor
        self._virtual_frames_by_tag: Dict[int, dict] = {}
        
        self._memory_trace_rows = []


    # episode / step lifecycle
    def reset_episode(self):
        self._mem.clear()
        self._prev_hp.clear()
        self._virtual_frames_by_tag.clear()

    def begin_step(self, bot=None, task=None):
        self._virtual_frames_by_tag.clear()

        # age timers / ages once per environment step
        for tag, mem in self._mem.items():
            # ages
            mem[self.I_THREAT_AGE] = min(self.MAX_AGE, mem[self.I_THREAT_AGE] + 1.0)
            mem[self.I_TARGET_AGE] = min(self.MAX_AGE, mem[self.I_TARGET_AGE] + 1.0)

            # timers
            mem[self.I_RETREAT_TIMER] = max(0.0, mem[self.I_RETREAT_TIMER] - 1.0)
            mem[self.I_TARGET_TIMER] = max(0.0, mem[self.I_TARGET_TIMER] - 1.0)

            # small decay on local traces
            mem[self.I_PANIC] *= 0.985
            mem[self.I_HAZ] *= 0.990

    def export_virtual_frames(self, bot):
        """
        Build short-lived virtual frame from local memory.
        Used by action executor when no observed enemies are visible.
        """
        frames: Dict[int, dict] = {}

        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)
        map_scale = max(W, H)

        for m in marine_units(bot):
            tag = int(m.tag)
            mem = self._mem.get(tag, None)
            if mem is None:
                continue

            px = float(m.position.x)
            py = float(m.position.y)

            use_source = None
            dx = dy = dist_norm = age = threat_type = 0.0

            # prefer threat memory while retreat timer is active
            # print("mem[self.I_RETREAT_TIMER]: ", mem[self.I_RETREAT_TIMER])
            if (
                mem[self.I_RETREAT_TIMER] > 0.0
                and mem[self.I_THREAT_AGE] < self.MAX_AGE
                and self._has_vec(mem[self.I_THREAT_DX], mem[self.I_THREAT_DY])
            ):
                use_source = "threat"
                dx = float(mem[self.I_THREAT_DX])
                dy = float(mem[self.I_THREAT_DY])
                dist_norm = float(mem[self.I_THREAT_DIST])
                age = float(mem[self.I_THREAT_AGE])
                threat_type = float(mem[self.I_THREAT_TYPE])

            elif (
                mem[self.I_TARGET_TIMER] > 0.0
                and mem[self.I_TARGET_AGE] < self.MAX_AGE
                and self._has_vec(mem[self.I_TARGET_DX], mem[self.I_TARGET_DY])
            ):
                use_source = "target"
                dx = float(mem[self.I_TARGET_DX])
                dy = float(mem[self.I_TARGET_DY])
                dist_norm = float(mem[self.I_TARGET_DIST])
                age = float(mem[self.I_TARGET_AGE])
                threat_type = 0.0

            if use_source is None:
                continue

            # reconstruct short-range virtual point
            dist_world = float(np.clip(dist_norm * map_scale, 2.0, 10.0))
            
            pt = Point2((
                px + dx * dist_world,
                py + dy * dist_world,
            ))
            
            # print("use_source:", use_source)
            # print("pt:", pt)

            frames[tag] = {
                "point": pt,
                "source": use_source,
                "age": age,
                "threat_type": threat_type,
            }

        self._virtual_frames_by_tag = frames
        return frames


    # main act
    def act(self, agent, obs: UnitObs, bot=None, marine=None, task=None) -> int:
        tag = int(obs.tag)

        mem = self._get_mem(tag)
        # print("mem:", mem)

        # 1) automatic local trace update from current observed state
        self._update_trace_scalars(tag, obs)

        # 2) automatic visible-memory refresh (threat + target)
        self._refresh_visible_memory(tag, obs)

        # 3) build state = observed raw 18 + memory features 14
        state = np.concatenate(
            [
                np.asarray(obs.base_feats, dtype=np.float32).reshape(-1)[: self.BASE_DIM],
                self._mem_to_features(self._get_mem(tag)),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        # 4) TPG acts
        out = agent.act(state.tolist())
        env_a = self._decode_action(out)

        # 5) winning learner registers gate persistence
        regs = self._read_selected_registers(agent)
        # print("regs:", regs)
        self._apply_register_control(tag, obs, env_a, regs)
                
        
        # add log
        self._log_trace(bot, obs, env_a, regs)

        return int(env_a)


    # memory core
    def _get_mem(self, tag: int) -> np.ndarray:
        tag = int(tag)
        mem = self._mem.get(tag)
        if mem is None:
            mem = np.zeros((self.MEM_DIM,), dtype=np.float32)

            # initialize ages as stale
            mem[self.I_THREAT_AGE] = self.MAX_AGE
            mem[self.I_TARGET_AGE] = self.MAX_AGE

            self._mem[tag] = mem
        return mem

    def _mem_to_features(self, mem: np.ndarray) -> np.ndarray:
        x = np.asarray(mem, dtype=np.float32).copy()

        # normalize ages / timers for TPG input
        x[self.I_THREAT_AGE] = np.clip(x[self.I_THREAT_AGE] / self.MAX_AGE, 0.0, 1.0)
        x[self.I_TARGET_AGE] = np.clip(x[self.I_TARGET_AGE] / self.MAX_AGE, 0.0, 1.0)
        x[self.I_RETREAT_TIMER] = np.clip(x[self.I_RETREAT_TIMER] / self.MAX_TIMER, 0.0, 1.0)
        x[self.I_TARGET_TIMER] = np.clip(x[self.I_TARGET_TIMER] / self.MAX_TIMER, 0.0, 1.0)
        return x

    def _update_trace_scalars(self, tag: int, obs: UnitObs):
        mem = self._get_mem(tag)

        hp = float(obs.hp)
        prev_hp = float(self._prev_hp.get(tag, hp))
        delta_hp = max(0.0, prev_hp - hp)
        self._prev_hp[tag] = hp

        damage_term = min(1.0, delta_hp / 10.0)
        threat_term = float(0.6 * obs.bane_threat + 0.3 * obs.ling_threat + 0.1 * obs.enemy_density)

        # panic
        panic = float(mem[self.I_PANIC])
        panic = 0.92 * panic + 0.60 * damage_term + 0.35 * threat_term
        # print("panic:", panic)
        mem[self.I_PANIC] = float(np.clip(panic, 0.0, 2.0))

        # cyc
        cyc = float(mem[self.I_CYC])
        cyc = 0.50 * cyc + 0.50 * float(obs.cd_norm)
        # print("cyc:", cyc)
        mem[self.I_CYC] = float(np.clip(cyc, 0.0, 1.0))

        # hazard
        haz = float(mem[self.I_HAZ])
        danger_now = (
            0.45 * float(obs.enemy_density)
            + 0.45 * float(obs.bane_threat)
            + 0.20 * float(obs.ling_threat)
            + 0.35 * damage_term
        )
        haz = 0.90 * haz + 0.10 * danger_now
        # print("haz:", haz)
        mem[self.I_HAZ] = float(np.clip(haz, 0.0, 2.0))

    def _refresh_visible_memory(self, tag: int, obs: UnitObs):
        mem = self._get_mem(tag)
        base = np.asarray(obs.base_feats, dtype=np.float32).reshape(-1)

        vis = float(base[17])
        if vis <= 0.5:
            return

        # threat memory: prefer bane if currently observed, else nearest observed enemy
        bane_dx = float(base[12])
        bane_dy = float(base[13])
        bane_dist = float(base[14])
        bane_threat = float(base[15])

        enemy_dx = float(base[8])
        enemy_dy = float(base[9])
        enemy_dist = float(base[10])

        if bane_threat > 1e-4 or bane_dist < 0.999:
            mem[self.I_THREAT_DX] = bane_dx
            mem[self.I_THREAT_DY] = bane_dy
            mem[self.I_THREAT_DIST] = bane_dist
            mem[self.I_THREAT_AGE] = 0.0
            mem[self.I_THREAT_TYPE] = 1.0   # baneling
        elif self._has_vec(enemy_dx, enemy_dy):
            mem[self.I_THREAT_DX] = enemy_dx
            mem[self.I_THREAT_DY] = enemy_dy
            mem[self.I_THREAT_DIST] = enemy_dist
            mem[self.I_THREAT_AGE] = 0.0
            mem[self.I_THREAT_TYPE] = 0.25  # generic enemy / ling-ish

        # target memory: use nearest observed enemy direction as short-lived target frame
        if self._has_vec(enemy_dx, enemy_dy):
            mem[self.I_TARGET_DX] = enemy_dx
            mem[self.I_TARGET_DY] = enemy_dy
            mem[self.I_TARGET_DIST] = enemy_dist
            mem[self.I_TARGET_AGE] = 0.0

    def _apply_register_control(self, tag: int, obs: UnitObs, env_a: int, regs: np.ndarray):
        mem = self._get_mem(tag)
        # print("tag:{} mem:{}".format(tag, mem))

        # print("env_a:", env_a)
        move_idx = int(env_a) // 4
        tgt_idx = int(env_a) % 4
        # print("move_idx:", move_idx)
        # print("tgt_idx:", tgt_idx)

        # learned persistence length
        persistence = float(np.clip((regs[9] + 1.0) / 2.0, 0.0, 1.0))
        timer_len = 2.0 + round(persistence * (self.MAX_TIMER - 2.0))

        # strong current danger
        visible_danger = float(
            max(
                obs.bane_threat,
                0.6 * obs.enemy_density + 0.5 * obs.ling_threat,
            )
        )

        # retreat gate: either TPG explicitly wants it, or current state clearly looks dangerous
        if (regs[1] > 0.0 and visible_danger > 0.05) or move_idx in (2, 7):
            mem[self.I_RETREAT_TIMER] = max(mem[self.I_RETREAT_TIMER], timer_len)
            # print("retreat")


        # target commit gate: if policy selected an attack-target head and register says "keep it"
        if tgt_idx != 0 and (regs[2] > 0.0 or move_idx in (1, 3, 4, 8, 0)):
            mem[self.I_TARGET_TIMER] = max(mem[self.I_TARGET_TIMER], timer_len)
            # print("keep target")

        # clear stale memory
        if regs[10] > 0.0:
            if mem[self.I_THREAT_AGE] >= self.MAX_AGE:
                mem[self.I_THREAT_DX:self.I_THREAT_TYPE + 1] = 0.0
                mem[self.I_THREAT_AGE] = self.MAX_AGE

            if mem[self.I_TARGET_AGE] >= self.MAX_AGE:
                mem[self.I_TARGET_DX:self.I_TARGET_AGE + 1] = 0.0
                mem[self.I_TARGET_AGE] = self.MAX_AGE
            # print("clear stale memory")

    # utils
    def _decode_action(self, out: Any) -> int:
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, (list, np.ndarray)):
            out = int(np.asarray(out).ravel()[0])
        return int(out) % int(self.env_n)

    def _read_selected_registers(self, agent) -> np.ndarray:
        act_vars = getattr(agent, "actVars", None) or {}
        # print("act_vars:", act_vars)
        regs = act_vars.get("last_selected_registers", None)
        # print("regs:", regs)

        if regs is None:
            return np.zeros((self.n_tpg_registers,), dtype=np.float32)

        regs = np.asarray(regs, dtype=np.float32).reshape(-1)
        if regs.shape[0] < self.n_tpg_registers:
            regs = np.pad(regs, (0, self.n_tpg_registers - regs.shape[0]), constant_values=0.0)
        elif regs.shape[0] > self.n_tpg_registers:
            regs = regs[: self.n_tpg_registers]
        return regs

    @staticmethod
    def _has_vec(dx: float, dy: float) -> bool:
        return (abs(float(dx)) + abs(float(dy))) > 1e-6
    
    
    # new add
    # add inside CombatMemoryMaskedPolicy
    def _virtual_frame_status_from_mem(self, mem: np.ndarray):
        """
        Infer whether the current memory is sufficient to export a virtual frame.
        This mirrors export_virtual_frames(), but only returns a boolean + source.
        """
        if (
            mem[self.I_RETREAT_TIMER] > 0.0
            and mem[self.I_THREAT_AGE] < self.MAX_AGE
            and self._has_vec(mem[self.I_THREAT_DX], mem[self.I_THREAT_DY])
        ):
            return 1, "threat"

        if (
            mem[self.I_TARGET_TIMER] > 0.0
            and mem[self.I_TARGET_AGE] < self.MAX_AGE
            and self._has_vec(mem[self.I_TARGET_DX], mem[self.I_TARGET_DY])
        ):
            return 1, "target"

        return 0, "none"

    def _log_trace(self, bot, obs: UnitObs, env_a: int, regs: np.ndarray):
        """
        Save one per-step, per-marine trace row onto the bot.
        """
        if bot is None:
            return

        if not hasattr(bot, "_memory_trace_rows"):
            bot._memory_trace_rows = []

        tag = int(obs.tag)
        mem = self._get_mem(tag).copy()
        base = np.asarray(obs.base_feats, dtype=np.float32).reshape(-1)

        visible = float(base[17]) if base.shape[0] > 17 else 0.0
        move_idx = int(env_a) // 4
        tgt_idx = int(env_a) % 4

        has_vf, vf_source = self._virtual_frame_status_from_mem(mem)
        used_vf = int((visible <= 0.5) and (has_vf == 1))

        row = {
            "episode": int(getattr(bot, "_cur_ep_no", -1)),
            "game_loop": int(bot.state.game_loop),
            "tag": tag,

            "env_a": int(env_a),
            "move_idx": move_idx,
            "tgt_idx": tgt_idx,

            "visible": float(visible),
            "enemy_density": float(getattr(obs, "enemy_density", 0.0)),
            "hp": float(getattr(obs, "hp", 0.0)),
            "cd_norm": float(getattr(obs, "cd_norm", 0.0)),
            "bane_threat": float(getattr(obs, "bane_threat", 0.0)),
            "ling_threat": float(getattr(obs, "ling_threat", 0.0)),

            "panic": float(mem[self.I_PANIC]),
            "cyc": float(mem[self.I_CYC]),
            "haz": float(mem[self.I_HAZ]),

            "threat_dx": float(mem[self.I_THREAT_DX]),
            "threat_dy": float(mem[self.I_THREAT_DY]),
            "threat_dist": float(mem[self.I_THREAT_DIST]),
            "threat_age": float(mem[self.I_THREAT_AGE]),
            "threat_type": float(mem[self.I_THREAT_TYPE]),

            "target_dx": float(mem[self.I_TARGET_DX]),
            "target_dy": float(mem[self.I_TARGET_DY]),
            "target_dist": float(mem[self.I_TARGET_DIST]),
            "target_age": float(mem[self.I_TARGET_AGE]),

            "retreat_timer": float(mem[self.I_RETREAT_TIMER]),
            "target_timer": float(mem[self.I_TARGET_TIMER]),

            "reg1_retreat_gate": float(regs[1]) if len(regs) > 1 else 0.0,
            "reg2_target_gate": float(regs[2]) if len(regs) > 2 else 0.0,
            "reg9_persistence": float(regs[9]) if len(regs) > 9 else 0.0,

            "has_virtual_frame": int(has_vf),
            "used_virtual_frame": int(used_vf),
            "virtual_frame_source": vf_source,
        }

        bot._memory_trace_rows.append(row)