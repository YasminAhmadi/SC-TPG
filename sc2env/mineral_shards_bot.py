from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Alliance
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2

from tpg.agent import Agent
from tpg.trainer import Trainer

from config.schema import TrainConfig
from utils import *


# from WM.wm_recorder import WMRecorder


class SwarmBot(BotAI):
    def __init__(self, trainer: Trainer, task, policy, cfg: TrainConfig):
        super().__init__()
        self.trainer: Trainer = trainer
        self.task = task
        self.policy = policy
        self.cfg = cfg
        self.env_key = self.task.name

        self.agents: List[Agent] = []
        self.cur_idx: int = 0

        self.agent_returns: List[List[float]] = []
        self.agent_episodes_done: List[int] = []

        # trainer save generation / resume，use pickle
        if not hasattr(self.trainer, "generation"):
            self.trainer.generation = 0
        if not hasattr(self.trainer, "resume"):
            self.trainer.resume = None
        self.generation = int(self.trainer.generation)

        self.ep_reward = 0.0
        self.ep_start_loop = 0
  
        
        self._last_move_ratio: float = 0.0
        
        self._enemy_alive_tags = set()
        self._enemy_seen_tags = set()
        
        # termination robust bookkeeping
        self._term_prev_m_all: Optional[int] = None
        self._term_prev_e_bio: Optional[int] = None
        self._zero_marine_streak = 0
        
        # constants. not for this mineral task
        self._INIT_E_BIO = int(getattr(self.task, "INIT_E_BIO", 10))
        self._INIT_M_ALL = int(getattr(self.task, "INIT_M_ALL", 9))
        
        # for this mineral task, False
        self._HAS_ENEMY_BIO = bool(getattr(self.task, "HAS_ENEMY_BIO", True))
        # enemy bio types: task-defined (MvZB: ling+bane; Roaches: roach; etc.)
        default_bio = {UnitTypeId.ZERGLING, UnitTypeId.BANELING}
        self._ENEMY_BIO_TYPES = set(getattr(self.task, "ENEMY_BIO_TYPES", default_bio))
        # print("self._ENEMY_BIO_TYPES:", self._ENEMY_BIO_TYPES)

        # MvZB: True; Beacon, mineral shards: False
        self._USE_ENEMY_STREAK_WIN = bool(getattr(self.task, "USE_ENEMY_STREAK_WIN", self._HAS_ENEMY_BIO))
        
        # find zerglings
        self._PARTIAL_OBS_ENEMY = bool(getattr(self.task, "PARTIAL_OBS_ENEMY", False))
        if self._PARTIAL_OBS_ENEMY:
            self._USE_ENEMY_STREAK_WIN = False
        
        # record
        self.survivors = 0
        self.kills = 0
        
        self._wm_pending = False # world model pending flag
        self._zero_enemy_streak = 0 
        self._rew_ov_final = None

        

        # controlled marine tag
        self.controlled_marine_tags: Optional[set[int]] = None
        
        self._need_clean_stray_marines_once: bool = False

        # Used to remember "the tags of all friendly marines on the map at this moment" before a reset.
        self._prev_all_marine_tags: Optional[set[int]] = None

        # chat reset
        self._waiting_reset = False
        self._reset_wait_left = 0
        self._reset_tries_left = 0
        self._prev_marine_centroid: Optional[Point2] = None
        self._prev_enemy_centroid: Optional[Point2] = None

    def _ensure_eval_buffers(self):
        n = len(self.agents)

        if n <= 0:
            self.agent_returns = []
            self.agent_episodes_done = []
            self.cur_idx = 0
            return

        if (not isinstance(self.agent_returns, list)) or (len(self.agent_returns) != n):
            self.agent_returns = [[] for _ in range(n)]

        if (not isinstance(self.agent_episodes_done, list)) or (len(self.agent_episodes_done) != n):
            self.agent_episodes_done = [0 for _ in range(n)]

        if self.cur_idx < 0:
            self.cur_idx = 0
        if self.cur_idx >= n:
            self.cur_idx = n - 1

    async def on_start(self):
        self.client.game_step = 2 # try 2/4/6/8
        # await announce(self, "boot")

        
        # record
        # self.wmrec = WMRecorder(
        #     out_dir=self.cfg.wm_dir,
        #     env_name=self.env_key,
        #     params=self.task.WM_PARAMS
        # )
        
        if not hasattr(self, "_wm_ep_id"):
            # print("not hasattr(self, _wm_ep_id)")
            self._wm_ep_id = 0
        
        st = getattr(self.trainer, "battle_state", None)
        # print("st:", st)
        if st is not None:
            # restore
            self.generation = st["generation"]
            self.agents = self.trainer.getAgents()
            self._ensure_eval_buffers()
            self.cur_idx = st["cur_idx"]
            self.agent_returns = st["agent_returns"]
            self.agent_episodes_done = st["agent_episodes_done"]

            print(f"[Resume Gen {self.generation:03d}] from agent {self.cur_idx + 1}/{len(self.agents)}")
            await self.begin_episode_via_chatreset()
        else:
            await self._start_new_generation(first=True)

    async def _start_new_generation(self, first: bool = False):
        self.agents = self.trainer.getAgents()

        # print("_start_new_generation")
        t = self.trainer
        # print(
        #     f"[DBG] gen_mut={t.mutateParams.get('generation')} "
        #     f"gen_attr={t.generation} "
        #     f"rootTeams={len(t.rootTeams)} "
        #     f"agents={len(t.getAgents())} "
        #     f"elites={len(getattr(t, 'elites', []))}",
        #     flush=True
        # )

        n = len(self.agents)
        gen_on_trainer = getattr(self.trainer, "generation", 0)
        # gen_on_trainer = int(self.trainer.mutateParams.get("generation", 0))
        # print("gen_on_trainer:", gen_on_trainer)

        if first:
            resume = getattr(self.trainer, "resume", None)
            if not isinstance(resume, dict):
                resume = None

            if (
                    resume is not None
                    and resume.get("generation") == gen_on_trainer
                    and resume.get("num_agents") == n
            ):
                # restore this gen's eval state
                self.generation = gen_on_trainer
                self.agent_returns = resume["agent_returns"]
                self.agent_episodes_done = resume["agent_episodes_done"]
                self.cur_idx = resume["cur_idx"]
                print(
                    f"[Resume Gen {self.generation:03d}] from agent {self.cur_idx + 1}/{n}"
                )
            else:
                # restart
                self.generation = gen_on_trainer
                self.agent_returns = [[] for _ in range(n)]
                self.agent_episodes_done = [0 for _ in range(n)]
                self.cur_idx = 0
                self.trainer.resume = None

                print(f"[Gen {self.generation:03d}] start; {n} agents")
        else:
            # evolve to next gen
            self.generation = gen_on_trainer

            self.trainer.resume = None

            self.agent_returns = [[] for _ in range(n)]
            self.agent_episodes_done = [0 for _ in range(n)]
            self.cur_idx = 0

            print(f"[Gen {self.generation:03d}] start; {n} agents")

        await self.begin_episode_via_chatreset()

    
    def _start_new_episode_counters(self):
        self._enemy_alive_tags = set()
        self._enemy_seen_tags = set()
        
        self._tags_relocked_once = False
        
        # record
        self._has_acted_this_episode = False
        self._wm_pending = False
        self._zero_enemy_streak = 0
        self._rew_ov_final = None
        
        self.ep_reward = 0.0
        self.ep_start_loop = self.state.game_loop
        # print("self.ep_start_loop:", self.ep_start_loop)
        
        #
        ms_all = self.units(UnitTypeId.MARINE)
        cur_tags = {int(u.tag) for u in ms_all}
        prev_tags = self._prev_all_marine_tags or set()


        new_tags = cur_tags - prev_tags
        # print("new_tags:", new_tags)

        if new_tags:
            # print("exist new_tags:", new_tags)
            self.controlled_marine_tags = set(new_tags)
        else:
            self.controlled_marine_tags = set(cur_tags)
        
        
        # WM record: episode start (AFTER reset + AFTER controlled tags locked)
        # self.wmrec.begin_episode(self, episode_id=self._wm_ep_id, generation=int(self.generation))
        self._wm_ep_id += 1
        
        self._prev_all_marine_tags = set(cur_tags)
        
        self._prev_enemy_hp = float(sum_enemy_hp(self))
        # print("self._prev_enemy_hp:", self._prev_enemy_hp)
        self._prev_marine_hp = sum_hp(marine_units(self))
        # print("self._prev_marine_hp:", self._prev_marine_hp)
        
        
        if self._prev_marine_hp <= 1e-6 and ms_all and ms_all.exists:
            self._prev_marine_hp = float(sum(float(getattr(u, "health", 0.0)) for u in ms_all))
        
        # print("[EP-START] marines(all)=", ms_all.amount if ms_all else 0,
        #   "controlled=", len(self.controlled_marine_tags))
        
        self._last_d_to_enemy = None
        self._prev_enemy_count = count_enemies(self)
        self._prev_marine_count = count_marines(self)
        self._last_move_ratio = 0.0

        # HP
        self._init_enemy_hp = self._prev_enemy_hp
        self._init_marine_hp = self._prev_marine_hp
        

        # reset 
        self.task.reset_episode(self)
        # reset policy
        if hasattr(self.policy, "reset_episode"):
            self.policy.reset_episode()
        
        # beacon
        self._ep_score_start = float(getattr(getattr(self.state, "score", None), "score", 0.0) or 0.0)
        self._ep_score_prev = self._ep_score_start

        
        # reset termination bookkeeping
        self._zero_marine_streak = 0
        self._term_prev_m_all = int(count_marines(self))
        # print("self._term_prev_m_all:", self._term_prev_m_all)
        self._term_prev_e_bio = int(count_enemy_bio(self))
        # print("self._term_prev_e_bio:", self._term_prev_e_bio)
        
        self.survivors = 0
        self.kills = 0


        self._last_frame_snap = None

        # Debug
        n_m = count_marines(self)
        n_e = count_enemies(self)
        # print(f"_start_new_episode_counters [EP-START] marines={n_m}, enemies={n_e}")


        try:
            raw = self.state.observation.raw_data
            # print("raw:", raw)
            try:
                marine_id = int(UnitTypeId.MARINE.value)
            except Exception:
                marine_id = int(UnitTypeId.MARINE)

            raw_marine_self = count_units_in_raw_data(raw, marine_id, Alliance.Self.value)
            # print("raw_marine_self:", raw_marine_self)    
        except Exception as e:
            print("[EP-START-RAW] raw-data unavailable:", e)


        self._need_clean_stray_marines_once = True

        
        self._latched_terminal_reason = None
        

        self._alive_marine_tags = set(self.controlled_marine_tags or [])
        
        # no enemy bio for this task
        if self._HAS_ENEMY_BIO:
            BIO = self._ENEMY_BIO_TYPES
            self._alive_enemy_tags = {int(u.tag) for u in self.enemy_units if u.type_id in BIO}
        else:
            self._alive_enemy_tags = set()
        
    
    async def on_unit_destroyed(self, unit_tag: int):
        t = int(unit_tag)

        # reset/episode
        if getattr(self, "_waiting_reset", False):
            return
        if not getattr(self, "_has_acted_this_episode", False):
            return

        sM = getattr(self, "_alive_marine_tags", None)
        sE = getattr(self, "_alive_enemy_tags", None)

        # all_dead
        if getattr(self, "_latched_terminal_reason", None) == "all_dead":
            return

        if sM is not None and t in sM:
            sM.remove(t)
            if len(sM) == 0:
                self._latched_terminal_reason = "all_dead"
            return 


        if sE is not None and t in sE:
            sE.remove(t)
            if len(sE) == 0:
                if (sM is None) or (len(sM) > 0):
                    self._latched_terminal_reason = "win"
                else:
                    self._latched_terminal_reason = "all_dead"
            return
    
    async def request_reset(self):
        
        # print(f"[RESET-SEND] loop={self.state.game_loop} tries_left={self._reset_tries_left} waiting={self._waiting_reset}", flush=True)

        try:
            await self.client.chat_send("reset", False)
            
            
        except Exception:
            self._waiting_reset = False

            self._start_new_episode_counters()
            print(f"[Gen {self.generation:03d}] episode begin (chat send failed)")
            return
        self._waiting_reset = True
        self._reset_wait_left = self.cfg.reset_wait_steps
    
    
    async def begin_episode_via_chatreset(self):
        # print("begin_episode_via_chatreset")
        
        
        # marine tag
        cur_ms_all = self.units(UnitTypeId.MARINE)
        self._prev_all_marine_tags = {int(u.tag) for u in cur_ms_all}

        self._prev_marine_centroid = get_centroid(marine_units(self))
        self._prev_enemy_centroid = get_enemy_centroid(self)
        self._reset_tries_left = self.cfg.reset_max_tries

        await self.request_reset()
    
    def _evolve_one_gen_single_task(self, scores):
        trainer = self.trainer

        teams = []
        for a in self.agents:
            if hasattr(a, "team"):
                teams.append(a.team)
            elif hasattr(a, "getTeam"):
                teams.append(a.getTeam())
            else:
                raise RuntimeError("Agent has no team attribute/method")

        assert len(teams) == len(scores), (len(teams), len(scores))

        for team, score in zip(teams, scores):
            team.outcomes[self.env_key] = float(score)

        g0 = self.trainer.generation
        self.trainer.evolve([self.env_key])
        g1 = self.trainer.generation
        # print(f"[DBG] evolve generation: {g0} -> {g1}", flush=True)

        # debug
        # print("_evolve_one_gen_single_task")
        t = self.trainer
        # print(
        #     f"[DBG] gen_mut={t.mutateParams.get('generation')} "
        #     f"gen_attr={t.generation} "
        #     f"rootTeams={len(t.rootTeams)} "
        #     f"agents={len(t.getAgents())} "
        #     f"elites={len(getattr(t, 'elites', []))}",
        #     flush=True
        # )

    async def _finish_episode_for_current_agent(self, reason: str, already_reset: bool = False, final_snapshot: Optional[dict] = None):

        self._ensure_eval_buffers()

        steps = self.state.game_loop - self.ep_start_loop
        # print("final steps:", steps)
        final_bonus = self.task.final_reward(self, reason, steps)
        # did not use final bonus for mineral task.
        # print("final_bonus:", final_bonus)
        
        record_reward = self.ep_reward
        self.ep_reward += final_bonus
        
        idx = self.cur_idx
        self.agent_returns[idx].append(self.ep_reward)
        self.agent_episodes_done[idx] += 1
        ep_no = self.agent_episodes_done[idx]

        print(f"[Gen {self.generation:03d} | Agent {idx + 1}/{len(self.agents)}] "
            f"ep#{ep_no} return={self.ep_reward:.3f} ({reason})")


        if ep_no < self.cfg.episodes_per_agent:
            # print("ep_no < EPISODES_PER_AGENT")
            self.update_resume_state()
            await self.begin_episode_via_chatreset()
            return

        # all done
        self.cur_idx += 1

        # all agents in this generation have finished their episodes: evolve and start next gen
        if self.cur_idx >= len(self.agents):
            fitness = [
                float(np.mean(rs)) if len(rs) > 0 else 0.0
                for rs in self.agent_returns
            ]
            fitness_arr = np.asarray(fitness, dtype=np.float32)

            best = float(fitness_arr.max())
            mean = float(fitness_arr.mean())
            std = float(fitness_arr.std())
            var = float(fitness_arr.var())

            best_idx = int(np.argmax(fitness_arr))
            best_agent = self.agents[best_idx]

            #To CSV
            log_fitness_csv(self.cfg, self.generation, best, mean, std, var)


            ckpt_path = self.cfg.ckpt_dir / f"agent_gen{self.generation:03d}.pkl"
            # if self.generation >= 1600:
            best_agent.saveToFile(str(ckpt_path))

            print(
                f"[Gen {self.generation:03d}] "
                f"best={best:.2f} | mean={mean:.2f} | std={std:.2f} | var={var:.2f} "
                f"-> saved {ckpt_path}"
            )

            # draw trace. old version
            # draw_trace(self.cfg, self, best_agent)
            
            ###debug
            # print("_finish_episode_for_current_agent")
            t = self.trainer
            # print(
            #     f"[DBG] gen_mut={t.mutateParams.get('generation')} "
            #     f"gen_attr={t.generation} "
            #     f"rootTeams={len(t.rootTeams)} "
            #     f"agents={len(t.getAgents())} "
            #     f"elites={len(getattr(t, 'elites', []))}",
            #     flush=True
            # )


            # evolve
            self._evolve_one_gen_single_task(fitness_arr.tolist())
            self.trainer.resume = None
            await self._start_new_generation(first=False)
            return

        
        
        self.update_resume_state()
        
        
        if already_reset:

            self._waiting_reset = False
            self._reset_wait_left = 0
            self._reset_tries_left = 0

            self._episode_pending_start = True
            self._need_clean_stray_marines_once = True
            self._stray_kill_wait = 0
            # print("[EVAL] next episode: AUTO reset -> pending_start")
            
            
            await self.begin_episode_via_chatreset()
            return

            
        # if self.wmrec:
        #         self.wmrec.finalize_last_step(final_bonus, force_done=True)
        #         self.wmrec.end_episode(reason, ep_return=record_reward, steps=steps, final_snapshot=final_snapshot)
        #         self.wmrec.save_episode()    
        
        # cleanup WM pending + final override
        self._wm_pending = False
        self._rew_ov_final = None    
            
        # print("[EVAL] next episode: switch agent -> start new episode")    
        await self.begin_episode_via_chatreset()
        return
        
        
    def _validate_tags(self):

        if getattr(self, "_waiting_reset", False):
            return

        N = 40
        if (self.state.game_loop - getattr(self, "ep_start_loop", 0)) > N:
            return


        if getattr(self, "_tags_relocked_once", False):
            return

        actual_tags = list(self_marine_tags_raw(self)) # raw: all self marines (no filter)
        actual_total = len(actual_tags)
        if actual_total == 0:
            return

        tags = getattr(self, "controlled_marine_tags", None)
        if not tags:
            return

        
        actual_set = set(actual_tags)
        inter = actual_set & set(tags)


        if len(inter) == 0:
            chosen = set(sorted(actual_tags)[-9:])

            print(f"[WARN] Tag mismatch: locked={len(tags)} actual={actual_total} -> relock {len(chosen)}", flush=True)
            self.controlled_marine_tags = chosen


            if hasattr(self.task, "fe"):
                self.task.fe.reset()

            self._tags_relocked_once = True
    
    # def _make_frame_snapshot(self, m_all: int, e_bio: int) -> dict:
    #     mhp = float(sum_marine_hp(self))
    #     ehp = float(sum_enemy_hp(self))

    #     # mean_d_norm
    #     mean_d_norm = 0.0
    #     e_cent = get_enemy_centroid(self)
    #     ms = marine_units(self)
    #     if e_cent is not None and ms.exists:
    #         dists = [u.position.distance_to(e_cent) for u in ms]
    #         mean_d = float(sum(dists) / max(1, len(dists)))
    #         max_d = max(float(self.game_info.map_size.x), float(self.game_info.map_size.y))
    #         mean_d_norm = float(mean_d / max_d)

    #     return dict(
    #         m_all=int(m_all),
    #         e_bio=int(e_bio),
    #         marine_hp=mhp,
    #         enemy_hp=ehp,
    #         alive_marines=int(m_all),
    #         mean_d_norm=float(mean_d_norm),
    #     )
    def _make_frame_snapshot(self, m_all: int, e_bio: int) -> dict:
        ms = marine_units(self)
        mhp = 0.0
        if ms and ms.exists:
            mhp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in ms))
        
        
        ehp = 0.0
        if self._HAS_ENEMY_BIO:
            BIO = self._ENEMY_BIO_TYPES
            es = [u for u in self.enemy_units if u.type_id in BIO]
            if es:
                ehp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in es))

        # mean_d_norm
        mean_d_norm = 0.0
        e_cent = get_enemy_centroid(self)
        ms = marine_units(self)
        if e_cent is not None and ms.exists:
            dists = [u.position.distance_to(e_cent) for u in ms]
            mean_d = float(sum(dists) / max(1, len(dists)))
            max_d = max(float(self.game_info.map_size.x), float(self.game_info.map_size.y))
            mean_d_norm = float(mean_d / max_d)

        # print("snapshot:")
        # print("m_all:", m_all)
        # print("e_bio:", e_bio)
        
        return dict(
            m_all=int(m_all),
            e_bio=int(e_bio),
            marine_hp=mhp,
            enemy_hp=ehp,
            alive_marines=int(m_all),
            mean_d_norm=float(mean_d_norm),
        )
    
    async def on_step(self, iteration: int):
        BIO = self._ENEMY_BIO_TYPES#{UnitTypeId.ZERGLING}  # FindZerglings 
        vis = {int(u.tag) for u in self.enemy_units if u.type_id in BIO}
        if vis:
            self._enemy_seen_tags |= vis
            self._enemy_alive_tags |= vis
        
        e_visible = len(vis)
        e_alive_est = len(self._enemy_alive_tags)
        # print("e_visible:", e_visible)
        # print("e_alive_est:", e_alive_est)
        
        # First frame: Clearing away the remaining "wild self-margins" from the previous round.
        if self._need_clean_stray_marines_once:
            await kill_stray_marines(self)
            self._need_clean_stray_marines_once = False

        # wait for reset to take effect
        if self._waiting_reset:
            if self._reset_wait_left > 0:
                self._reset_wait_left -= 1
                return
            m1 = get_centroid(marine_units(self))
            e1 = get_enemy_centroid(self)
            if reset_effective(self._prev_marine_centroid, self._prev_enemy_centroid, m1, e1, min_delta=self.cfg.reset_min_delta):
                self._waiting_reset = False
                # _ = self._wm.end_episode(force_done=True)
                self._start_new_episode_counters()
                # print(f"[Gen {self.generation:03d}] episode begin (chat reset OK)")
                return
            else:
                self._reset_tries_left -= 1
                if self._reset_tries_left > 0:
                    print("self.request_reset()")
                    await self.request_reset()
                    return
                self._waiting_reset = False
                # _ = self._wm.end_episode(force_done=True)
                self._start_new_episode_counters()
                print(f"[Gen {self.generation:03d}] episode begin (chat reset FAILED, fallback)")
                return

        if not self.agents or self.cur_idx >= len(self.agents):
            return

        self._validate_tags()
        
        
        # current observations
        cur_m_all = int(count_marines(self))#self.units(UnitTypeId.MARINE).amount#
        # print("cur_m_all:", cur_m_all)
        cur_e_bio = int(count_enemy_bio(self))
        # print("cur_e_bio:", cur_e_bio)

        loops_used = int(self.state.game_loop - self.ep_start_loop)
        done_timeout = (loops_used >= self.cfg.loop_timeout)
        # print("done_timeout:", done_timeout)
        
        # streak
        if cur_m_all == 0 and cur_e_bio > 0:
            self._zero_marine_streak += 1
        else:
            self._zero_marine_streak = 0
        
        if self._USE_ENEMY_STREAK_WIN:
            if cur_e_bio == 0 and cur_m_all > 0:
                self._zero_enemy_streak += 1
            else:
                self._zero_enemy_streak = 0
        else:
            # Beacon/mineral. 
            self._zero_enemy_streak = 0


        cur_score = float(getattr(getattr(self.state, "score", None), "score", 0.0) or 0.0)

        score_win = False
        if getattr(self.task, "WIN_BY_SCORE", False):
            prev = float(getattr(self, "_ep_score_prev", cur_score))
            delta = cur_score - prev
            if delta >= float(getattr(self.task, "SCORE_DELTA_WIN", 1.0)) - 1e-6:
                score_win = True
            self._ep_score_prev = cur_score

        # print("cur_score:", cur_score)
        # print("score_win:", score_win)
        
        # print("self._zero_marine_streak:", self._zero_marine_streak)
        done_all_dead = (self._zero_marine_streak >= 2)

        # task-defined win (preferred)
        check_win = False
        # if hasattr(self.task, "check_win"):
        #     try:
        #         check_win = bool(self.task.check_win(self))
        #     except Exception as e:
        #         print("[WARN] task.check_win failed:", e)
        #         check_win = False

        task_win = score_win or check_win
        # print("task_win:", task_win)
        
        done_win = (
            task_win
            or (getattr(self, "_latched_terminal_reason", None) == "win")
            or (self._USE_ENEMY_STREAK_WIN and self._zero_enemy_streak >= 2)
        )
        # print("done_win:", done_win)

        done_flag = done_all_dead or done_win or done_timeout
        # print("done_flag:", done_flag)

        if done_all_dead:
            term_reason = "all_dead"
        elif done_win:
            term_reason = "win"
        elif done_timeout:
            term_reason = f"timeout({loops_used})"
        else:
            term_reason = None
        # print("done_flag:", done_flag)
               
        
        # auto_reset_jump
        prev_m = self._term_prev_m_all
        # print("prev_m:", prev_m)
        prev_e = self._term_prev_e_bio
        # print("prev_e:", prev_e)

        auto_reset_jump = False
        if getattr(self.task, "AUTO_RESET_JUMP", False):
            if not bool(getattr(self.task, "PARTIAL_OBS_ENEMY", False)):
                # print("not PARTIAL_OBS_ENEMY")
                if (prev_e is not None) and (prev_e <= 1) and (cur_e_bio >= self._INIT_E_BIO - 1):
                    # print("judge")
                    auto_reset_jump = True
                    
            if (prev_m is not None) and (prev_m == 0) and (cur_m_all >= self._INIT_M_ALL - 1):
                auto_reset_jump = True
            if (prev_m is not None) and (prev_m < cur_m_all) and (cur_m_all >= self._INIT_M_ALL - 1):
                auto_reset_jump = True
            # print("auto_reset_jump:", auto_reset_jump)
            
        # print("auto_reset_jump:", auto_reset_jump)
        if auto_reset_jump:
            snap = self._last_frame_snap or dict(
                m_all=int(prev_m or 0),
                e_bio=int(prev_e or 0),
                marine_hp=0.0,
                enemy_hp=0.0,
                alive_marines=int(prev_m or 0),
                mean_d_norm=0.0,
            )

            reason = getattr(self, "_latched_terminal_reason", None)
            if reason is None:
                reason = "win" if (prev_e is not None and prev_e <= 1) else "all_dead"

            snap2 = dict(snap)
            if reason == "win":
                snap2["e_bio"] = 0
                snap2["enemy_hp"] = 0.0
            if reason == "all_dead":
                snap2["m_all"] = 0
                snap2["alive_marines"] = 0
                snap2["marine_hp"] = 0.0


            self._rew_ov_final = snap2


            # if self.wmrec and self._wm_pending:
            #     self.wmrec.set_last_reward_done(0.0, True)

            await self._finish_episode_for_current_agent(reason, already_reset=True, final_snapshot=snap2)
            return



        # print("task_win:", task_win)
        # print("")
        if score_win and getattr(self.task, "WIN_BY_SCORE", False):
            self._wm_pending = False
            await self._finish_episode_for_current_agent("win")
            return
        
        
        # print("self._wm_pending:", self._wm_pending)
        if self._wm_pending:
            step_reward = self.task.step_reward(self, self._last_move_ratio)
            # print("step_reward:", step_reward)
            self.ep_reward += step_reward
            # if self.wmrec:
            #     self.wmrec.set_last_reward_done(step_reward, done_flag)
            self._wm_pending = False

        if done_flag:
            if done_timeout or self._has_acted_this_episode:
                await self._finish_episode_for_current_agent(term_reason)
                return
            else:

                pass

        # actions
        agent = self.agents[self.cur_idx]
        ms = marine_units(self)

        state21_cache = {}
        if ms:
            for m in ms:
                obs = self.task.fe.get_features(self, m)
                # print("obs:", obs)
                state21_cache[m.tag] = obs

        marine_actions = {}
        if ms:
            for m in ms:
                obs = state21_cache[m.tag]
                # print("s21:", s21)
                out = self.policy.act(agent=agent, obs=obs, bot=self, marine=m, task=self.task)
                # print("out:", out)
                a = self.task.env_action_space.clip(out)
                # print("a:", a)
                marine_actions[m.tag] = a
        # print("marine_actions:", marine_actions)

        # if self.wmrec:
            # self.wmrec.record_step(self, marine_actions, 0.0, False)
            # self._wm_pending = True


        self._last_frame_snap = self._make_frame_snapshot(cur_m_all, cur_e_bio)

        # update prev
        self._term_prev_m_all = cur_m_all
        self._term_prev_e_bio = cur_e_bio

        # send actions
        self._last_move_ratio = await self.task.apply_actions(self, marine_actions)
        self._wm_pending = True
        self._has_acted_this_episode = True

       
        
    async def on_end(self, game_result):
        self.save_progress_to_trainer()
        episodes_info = {i: ep for i, ep in enumerate(self.agent_episodes_done)}
        print(
            f"[on_end] result={game_result}, gen={self.generation}, cur_idx={self.cur_idx}, "
            f"episodes_done={episodes_info}"
        )

    #  trainer save
    def save_progress_to_trainer(self):
        self.trainer.battle_state = dict(
            generation=self.generation,
            cur_idx=self.cur_idx,
            agent_returns=self.agent_returns,
            agent_episodes_done=self.agent_episodes_done,
        )

    def update_resume_state(self):
        self.trainer.resume = {
            "generation": self.generation,
            "num_agents": len(self.agents),
            "cur_idx": self.cur_idx,
            "agent_returns": self.agent_returns,
            "agent_episodes_done": self.agent_episodes_done,
        }