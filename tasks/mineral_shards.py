# tasks/mineral_shards.py
from __future__ import annotations

from features.mineral_shards import BaseFeatureExtractor
from rewards.mineral_shards import RewardCalculator
from actions.mineral_shards import ActionExecutor

from rewards.params import RewardParams
from actions.params import ActionParams
from policies.base import DiscreteActionSpace
from WM.params import WMRecorderParams


class MineralShardsTask:
    name = "MineralShards"
    map_name = "CollectMineralShards"

    ENV_N = 9*4
    env_action_space = DiscreteActionSpace(ENV_N)

    # No enemy for this task
    HAS_ENEMY_BIO = False
    USE_ENEMY_STREAK_WIN = False
    INIT_M_ALL = 2
    INIT_E_BIO = 0

    # Do not use score as the winning condition.
    WIN_BY_SCORE = False

    ACTION_PARAMS = ActionParams(
        order_cooldown=6,
        wall_repulsion=0.45,
        step_size=2.6,
        map_margin=2.0,
        attack_range_approx=2.5,
        cd_ready_thresh=0.2,
    )

    # RewardParams
    # R_KILL_ENEMY: This is considered as the "reward coefficient for each shard found".
    # R_DELDIST: Approaching the latest shard shaping
    # R_EDGE_PEN: Penalty of standing against the wall (to prevent spinning around against the wall)
    REWARD_PARAMS = RewardParams(
        R_DELDIST=0.35,
        R_DMG_DEALT=0.0,
        R_DMG_TAKEN=0.0,
        R_KILL_ENEMY=5.0, # 1/5/8..
        R_LOSS_MARINE=0.0,
        R_EDGE_PEN=0.06,
        R_TIMEOUT_BASE=0.0,
        R_ALIVE_BONUS=0.0,
        R_MOVE_NEAR_ENEMY=0.12, # Movement bonus when close to the objective and when there is "progress"
        MOVE_REWARD_DIST_THRESH=0.10,
    )

    WM_PARAMS = WMRecorderParams(
        max_marines=2,
        max_enemies=0,
        canonical_side="none",
        hp_marine_max=45.0,
        hp_zergling_max=35.0,
        hp_baneling_max=30.0,
        debug=False,
    )

    def __init__(self, cfg=None):
        self.cfg = cfg
        
        self.fe = BaseFeatureExtractor()
        self.rew = RewardCalculator(self.REWARD_PARAMS)
        self.exec = ActionExecutor(self.ACTION_PARAMS)

    def reset_episode(self, bot):
        self.fe.reset()
        self.exec.reset()
        self.rew.reset(bot)
        if hasattr(bot, "policy"):
            bot.policy.reset_episode()

    def featurize_unit(self, bot, marine):
        return self.fe.get_features(bot, marine)

    def featurize_for_trace(self, bot):
        return self.fe.featurize_for_trace(bot)

    def observe(self, bot):
        return self.fe.observe_swarm(bot)

    async def apply_actions(self, bot, actions):
        return await self.exec.apply_actions(bot, actions)

    def step_reward(self, bot, move_ratio):
        return self.rew.calculate_step_reward(bot, move_ratio)

    def final_reward(self, bot, reason, steps):
        return self.rew.calculate_final_reward(bot, reason, steps)

    # Don't check win in this task
    def check_win(self, bot) -> bool:
        return False
