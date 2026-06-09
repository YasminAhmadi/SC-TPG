#sc2env/mvzb_bot.py
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

        # trainer save generation / resume. use pickle
        if not hasattr(self.trainer, "generation"):
            self.trainer.generation = 0
        if not hasattr(self.trainer, "resume"):
            self.trainer.resume = None
        self.generation = int(self.trainer.generation)

        self.ep_reward = 0.0
        self.ep_start_loop = 0
  
        # reward shaping history
        self._last_move_ratio: float = 0.0
        

        # termination robust bookkeeping
        self._term_prev_m_all: Optional[int] = None
        self._term_prev_e_bio: Optional[int] = None
        self._zero_marine_streak = 0
        
        # constants. 9 for marine, 10 for enemy bio (6+4)
        self._INIT_E_BIO = int(getattr(self.task, "INIT_E_BIO", 10))
        self._INIT_M_ALL = int(getattr(self.task, "INIT_M_ALL", 9))
        

        # enemy bio types: task-defined (MvZB: ling+bane; Roaches: roach; etc.)
        default_bio = {UnitTypeId.ZERGLING, UnitTypeId.BANELING}
        self._ENEMY_BIO_TYPES = set(getattr(self.task, "ENEMY_BIO_TYPES", default_bio))
        # print("self._ENEMY_BIO_TYPES:", self._ENEMY_BIO_TYPES)


        self._rew_ov_final = None #

        

        #  9 controlled marine. tag
        self.controlled_marine_tags: Optional[set[int]] = None
        
        # clean up stray marines after reset
        self._need_clean_stray_marines_once: bool = False

        # save all marine tags seen at the start of each episode, to help lock the controlled marines after reset
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

        # cur_idx check
        if self.cur_idx < 0:
            self.cur_idx = 0
        if self.cur_idx >= n:
            self.cur_idx = n - 1

    async def on_start(self):
        self.client.game_step = 2 # smaller for more frequent updates. 2/4/6/8...
        await self._start_new_generation(first=True)

        # st = getattr(self.trainer, "battle_state", None)
        # # print("st:", st)
        # if st is not None:
        #     self.generation = st["generation"]
        #     self.agents = self.trainer.getAgents()
        #     self._ensure_eval_buffers()
        #     self.cur_idx = st["cur_idx"]
        #     self.agent_returns = st["agent_returns"]
        #     self.agent_episodes_done = st["agent_episodes_done"]

        #     print(f"[Resume Gen {self.generation:03d}] from agent {self.cur_idx + 1}/{len(self.agents)}")
        #     await self.begin_episode_via_chatreset()
        # else:
        #     # no history to resume, start fresh
        #     await self._start_new_generation(first=True)

    async def _start_new_generation(self, first: bool = False):
        self.agents = self.trainer.getAgents()

        # print("_start_new_generation")
        t = self.trainer
        print(
            f"[DBG] gen_mut={t.mutateParams.get('generation')} "
            f"gen_attr={t.generation} "
            f"rootTeams={len(t.rootTeams)} "
            f"agents={len(t.getAgents())} "
            f"elites={len(getattr(t, 'elites', []))}",
            flush=True
        )

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
                # resume from saved state!!!
                self.generation = gen_on_trainer
                self.agent_returns = resume["agent_returns"]
                self.agent_episodes_done = resume["agent_episodes_done"]
                self.cur_idx = resume["cur_idx"]
                print(
                    f"[Resume Gen {self.generation:03d}] from agent {self.cur_idx + 1}/{n}"
                )
            else:
                # restart this generation
                self.generation = gen_on_trainer
                self.agent_returns = [[] for _ in range(n)]
                self.agent_episodes_done = [0 for _ in range(n)]
                self.cur_idx = 0
                self.trainer.resume = None

                print(f"[Gen {self.generation:03d}] start; {n} agents")
        else:
            # for next genertation
            self.generation = gen_on_trainer

            self.trainer.resume = None

            self.agent_returns = [[] for _ in range(n)]
            self.agent_episodes_done = [0 for _ in range(n)]
            self.cur_idx = 0

            print(f"[Gen {self.generation:03d}] start; {n} agents")

        await self.begin_episode_via_chatreset()

    
    def _start_new_episode_counters(self):

        self._enemy_seen_tags = set()
        
        self._tags_relocked_once = False
        
        # record
        self._has_acted_this_episode = False

        self._rew_ov_final = None
        
        self.ep_reward = 0.0
        self.ep_start_loop = self.state.game_loop
        # print("self.ep_start_loop:", self.ep_start_loop)
        
        # lock marine tags at the start of the episode
        ms_all = self.units(UnitTypeId.MARINE)
        cur_tags = {int(u.tag) for u in ms_all}
        prev_tags = self._prev_all_marine_tags or set()
        # print("prev_tags:", prev_tags)


        new_tags = cur_tags - prev_tags
        # print("new_tags:", new_tags)

        if new_tags:
            # print("exist new_tags:", new_tags)
            self.controlled_marine_tags = set(new_tags)
        else:
            self.controlled_marine_tags = set(cur_tags)
        
        
        # for next episode, save all marine tags seen at the start, to help lock the controlled marines after reset
        self._prev_all_marine_tags = set(cur_tags)
        

        self._prev_enemy_hp = float(sum_enemy_hp(self))
        # print("self._prev_enemy_hp:", self._prev_enemy_hp)
        self._prev_marine_hp = sum_hp(marine_units(self))
        # print("self._prev_marine_hp:", self._prev_marine_hp)
        
        # check again
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
        

        # reset eposide
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

        # snapshot clean
        self._last_frame_snap = None

        # Debug. check
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

        # for next episode
        self._need_clean_stray_marines_once = True


        
        # controlled marine tags
        self._alive_marine_tags = set(self.controlled_marine_tags or [])
        
        # enemy bio tags
        BIO = self._ENEMY_BIO_TYPES
        self._alive_enemy_tags = {int(u.tag) for u in self.enemy_units if u.type_id in BIO}

        
    

    
    async def request_reset(self):
        
        # print(f"[RESET-SEND] loop={self.state.game_loop} tries_left={self._reset_tries_left} waiting={self._waiting_reset}", flush=True)

        try:
            await self.client.chat_send("reset", False)
            
            
        except Exception:
            self._waiting_reset = False

            self._start_new_episode_counters()
            # print(f"[Gen {self.generation:03d}] episode begin (chat send failed)")
            return
        self._waiting_reset = True
        self._reset_wait_left = self.cfg.reset_wait_steps
    
    
    async def begin_episode_via_chatreset(self):
        # print("begin_episode_via_chatreset")
        
        
        # 0) record pre marine tages
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
        # print("final_bonus:", final_bonus)
        
        record_reward = self.ep_reward
        self.ep_reward += final_bonus
        
        idx = self.cur_idx
        self.agent_returns[idx].append(self.ep_reward)
        self.agent_episodes_done[idx] += 1
        ep_no = self.agent_episodes_done[idx]

        print(f"[Gen {self.generation:03d} | Agent {idx + 1}/{len(self.agents)}] "
            f"ep#{ep_no} return={self.ep_reward:.3f} ({reason})")


        # This agent hasn't completed the required EPISODES_PER_AGENT process yet.
        if ep_no < self.cfg.episodes_per_agent:
            # print("ep_no < EPISODES_PER_AGENT")
            self.update_resume_state()
            await self.begin_episode_via_chatreset()
            return

        # all finished
        self.cur_idx += 1

        # eval all agents in this generation
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

            # To CSV
            gen = int(getattr(self.trainer, "generation", self.generation))
            self.generation = gen
            
            
            log_fitness_csv(self.cfg, gen, best, mean, std, var)


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
            print(
                f"[DBG] gen_mut={t.mutateParams.get('generation')} "
                f"gen_attr={t.generation} "
                f"rootTeams={len(t.rootTeams)} "
                f"agents={len(t.getAgents())} "
                f"elites={len(getattr(t, 'elites', []))}",
                flush=True
            )


            # evolve next generation
            self._evolve_one_gen_single_task(fitness_arr.tolist())
            self.trainer.resume = None
            await self._start_new_generation(first=False)
            return

        
        
        self.update_resume_state()
        
        
        if already_reset:
            # The map has been automatically reset to the next game: do not send another "chat reset" message.
            self._waiting_reset = False
            self._reset_wait_left = 0
            self._reset_tries_left = 0

            self._episode_pending_start = True
            self._need_clean_stray_marines_once = True
            self._stray_kill_wait = 0
            # print("[EVAL] next episode: AUTO reset -> pending_start")
            
            
            await self.begin_episode_via_chatreset()
            return

               
        

        self._rew_ov_final = None    
            
        # print("[EVAL] next episode: switch agent -> start new episode")    
        await self.begin_episode_via_chatreset()
        return
        
        
    def _validate_tags(self):
        # check
        if getattr(self, "_waiting_reset", False):
            return

        # Triggering is only allowed within the first N frames of each round to avoid jitter throughout the round.
        N = 40
        if (self.state.game_loop - getattr(self, "ep_start_loop", 0)) > N:
            return

        # Avoid spamming and repeatedly changing tags
        if getattr(self, "_tags_relocked_once", False):
            return

        actual_tags = list(self_marine_tags_raw(self))  # raw: all self marines (no filter)
        actual_total = len(actual_tags)
        if actual_total == 0:
            return

        tags = getattr(self, "controlled_marine_tags", None)
        if not tags:
            # print("no controlled_marine_tags yet, skip tag validation")
            return

        # 
        actual_set = set(actual_tags)
        inter = actual_set & set(tags)

        
        if len(inter) == 0:
            
            chosen = set(sorted(actual_tags)[-9:])

            print(f"[WARN] Tag mismatch: locked={len(tags)} actual={actual_total} -> relock {len(chosen)}", flush=True)
            self.controlled_marine_tags = chosen
            
            if hasattr(self.task, "fe"):
                self.task.fe.reset()

            self._tags_relocked_once = True
    

    def _make_frame_snapshot(self, m_all: int, e_bio: int) -> dict:
        """Save a "final snapshot of the previous frame" for auto-reset frame skipping."""

        ms = marine_units(self)

        # self HP: sum of all marines HP 
        marine_hp = 0.0
        if ms and ms.exists:
            marine_hp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in ms))

        # Enemy HP: sum of all enemy bio HP
        enemy_hp = 0.0
        BIO = self._ENEMY_BIO_TYPES
        es_bio = [u for u in self.enemy_units if u.type_id in BIO]
        if es_bio:
            enemy_hp = float(sum(float(getattr(u, "health", 0.0) or 0.0) for u in es_bio))

        # mean_d_norm：Marine's average distance from the enemy's center (normalized)
        mean_d_norm = 0.0
        e_cent = get_enemy_centroid(self)
        if e_cent is not None and ms and ms.exists:
            dists = [u.position.distance_to(e_cent) for u in ms]
            mean_d = float(sum(dists) / max(1, len(dists)))
            max_d = max(float(self.game_info.map_size.x), float(self.game_info.map_size.y))
            mean_d_norm = float(mean_d / max_d)

        return {
            "m_all": int(m_all),
            "e_bio": int(e_bio),

            # final_reward / eval
            "alive_marines": int(m_all),
            "marine_hp": float(marine_hp),
            "enemy_hp": float(enemy_hp),
            "enemy_count": int(e_bio),
            "enemy_count_all": int(count_enemies(self)),#

            "mean_d_norm": float(mean_d_norm),
        }
    
    
    
    def _accumulate_step_reward(self) -> None:
        """
        Accumulate dense step reward for the transition caused by the previous action.
        """
        if not bool(getattr(self.cfg, "use_step_reward", False)):
            return

        # No previous action has been issued in this episode, so there is no transition to reward.
        if not bool(getattr(self, "_has_acted_this_episode", False)):
            return

        step_r = float(self.task.step_reward(self, self._last_move_ratio))
        self.ep_reward += step_r

        if bool(getattr(self.cfg, "debug_step_reward", False)):
            print(
                f"[STEP-REWARD] r={step_r:.3f} ep_reward={self.ep_reward:.3f}",
                flush=True,
            )
    
    
    
    async def on_step(self, iteration: int):
      
        # First frame: Clearing away the remaining from the previous round.
        if self._need_clean_stray_marines_once:
            await kill_stray_marines(self)
            self._need_clean_stray_marines_once = False

        # waiting for reset to take effect
        if self._waiting_reset:
            if self._reset_wait_left > 0:
                self._reset_wait_left -= 1
                return
            m1 = get_centroid(marine_units(self))
            e1 = get_enemy_centroid(self)
            if reset_effective(self._prev_marine_centroid, self._prev_enemy_centroid, m1, e1, min_delta=self.cfg.reset_min_delta):
                self._waiting_reset = False

                self._start_new_episode_counters()
                # print(f"[Gen {self.generation:03d}] episode begin (chat reset OK)")
                return
            else:
                self._reset_tries_left -= 1
                if self._reset_tries_left > 0:
                    # print("self.request_reset()")
                    await self.request_reset()
                    return
                self._waiting_reset = False

                self._start_new_episode_counters()
                # print(f"[Gen {self.generation:03d}] episode begin (chat reset FAILED, fallback)")
                return

        if not self.agents or self.cur_idx >= len(self.agents):
            return

        self._validate_tags()
        
        
        # A) First, read the count of the "current observation"
        cur_m_all = int(count_marines(self))#self.units(UnitTypeId.MARINE).amount#
        # print("cur_m_all:", cur_m_all)
        cur_e_bio = int(count_enemy_bio(self))
        # print("cur_e_bio:", cur_e_bio)

        loops_used = int(self.state.game_loop - self.ep_start_loop)
        done_timeout = (loops_used >= self.cfg.loop_timeout)
        # print("done_timeout:", done_timeout)
        

        reason = None
        if done_timeout:
            reason = f"timeout({loops_used})"

        # B) The `auto_reset_jump` function must be performed before reward settlement, otherwise the reward will jump randomly across resets.
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
                    
            # marine
            if (prev_m is not None) and (prev_m == 0) and (cur_m_all >= self._INIT_M_ALL - 1):
                auto_reset_jump = True
            if (prev_m is not None) and (prev_m < cur_m_all) and (cur_m_all >= self._INIT_M_ALL - 1):
                auto_reset_jump = True
            # print("auto_reset_jump:", auto_reset_jump)
            
        # print("auto_reset_jump:", auto_reset_jump)
        if auto_reset_jump:
            # Inferring the final outcome using the previous frame snapshot.
            snap = self._last_frame_snap or dict(
                m_all=int(prev_m or 0),
                e_bio=int(prev_e or 0),
                marine_hp=0.0,
                enemy_hp=0.0,
                alive_marines=int(prev_m or 0),
                enemy_count=int(prev_e or 0),
                enemy_count_all=int(prev_e or 0),
                mean_d_norm=0.0,
            )

            reason = "win" if (prev_e is not None and prev_e <= 1) else "all_dead"
            # print("reason:", reason)

            snap2 = dict(snap)
            if reason == "win":
                snap2["e_bio"] = 0
                snap2["enemy_hp"] = 0.0
                snap2["enemy_count"] = 0
                snap2["enemy_count_all"] = 0
            if reason == "all_dead":
                snap2["m_all"] = 0
                snap2["alive_marines"] = 0
                snap2["marine_hp"] = 0.0

            # Let final_reward use snap2, not the new world after reset.
            self._rew_ov_final = snap2


            await self._finish_episode_for_current_agent(reason, already_reset=True, final_snapshot=snap2)
            return

        # self._accumulate_step_reward()


        # if done_timeout:
        #     await self._finish_episode_for_current_agent(reason)
        #     return
        if done_timeout:
            self._rew_ov_final = self._make_frame_snapshot(cur_m_all, cur_e_bio)
            await self._finish_episode_for_current_agent(reason, final_snapshot=self._rew_ov_final)
            return
        
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
                # print("obs:", obs)
                out = self.policy.act(agent=agent, obs=obs, bot=self, marine=m, task=self.task)
                # print("out:", out)
                a = self.task.env_action_space.clip(out)
                # print("a:", a)
                marine_actions[m.tag] = a
        # print("marine_actions:", marine_actions)


        # save snapshot
        self._last_frame_snap = self._make_frame_snapshot(cur_m_all, cur_e_bio)

        # update prev
        self._term_prev_m_all = cur_m_all
        self._term_prev_e_bio = cur_e_bio

        # send actions to the environment
        self._last_move_ratio = await self.task.apply_actions(self, marine_actions)

        self._has_acted_this_episode = True

       
        
    async def on_end(self, game_result):
        self.save_progress_to_trainer()
        episodes_info = {i: ep for i, ep in enumerate(self.agent_episodes_done)}
        print(
            f"[on_end] result={game_result}, gen={self.generation}, cur_idx={self.cur_idx}, "
            f"episodes_done={episodes_info}"
        )

    # trainer save
    def save_progress_to_trainer(self):
        gen = int(getattr(self.trainer, "generation", self.generation))
        self.generation = gen
        self.trainer.battle_state = dict(
            generation=gen,
            cur_idx=self.cur_idx,
            agent_returns=self.agent_returns,
            agent_episodes_done=self.agent_episodes_done,
        )

    def update_resume_state(self):
        gen = int(getattr(self.trainer, "generation", self.generation))
        self.generation = gen
        self.trainer.resume = {
            "generation": gen,
            "num_agents": len(self.agents),
            "cur_idx": self.cur_idx,
            "agent_returns": self.agent_returns,
            "agent_episodes_done": self.agent_episodes_done,
        }