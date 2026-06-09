# tasks/beacon.py
from __future__ import annotations

from features.beacon import BaseFeatureExtractor
from rewards.beacon import RewardCalculator
from actions.beacon import ActionExecutor

from rewards.params import RewardParams
from actions.params import ActionParams
from policies.base import DiscreteActionSpace
from WM.params import WMRecorderParams


class BeaconTask:
    name = "Beacon"
    ENV_N = 9*4 # action count
    HAS_ENEMY_BIO = False
    USE_ENEMY_STREAK_WIN = False
    INIT_M_ALL = 1
    INIT_E_BIO = 0
    WIN_RADIUS = 0.5
    WIN_BY_SCORE = True
    SCORE_DELTA_WIN = 1.0
    map_name = "MoveToBeacon"
    env_action_space = DiscreteActionSpace(ENV_N)

    ACTION_PARAMS = ActionParams(
        order_cooldown=6,
        wall_repulsion=0.45,
        step_size=2.5,
        map_margin=2.0,
        attack_range_approx=2.5, 
        cd_ready_thresh=0.2, 
    )

    # RewardParams
    REWARD_PARAMS = RewardParams(
        R_DELDIST=1.0, # close to beacon
        R_DMG_DEALT=0.0,
        R_DMG_TAKEN=0.0,
        R_KILL_ENEMY=0.0,
        R_LOSS_MARINE=0.0,
        R_EDGE_PEN=0.08,  # wall penalty
        R_TIMEOUT_BASE=30.0, # time penalty base
        R_ALIVE_BONUS=100.0, # win bonus
        R_MOVE_NEAR_ENEMY=0.15, 
        MOVE_REWARD_DIST_THRESH=0.08, # dist_norm
    )

    WM_PARAMS = WMRecorderParams(
        max_marines=1,
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
