# tasks/mvzb.py
from __future__ import annotations

import importlib
from sc2.ids.unit_typeid import UnitTypeId

from rewards.mvzb import RewardCalculator
from rewards.params import RewardParams
from actions.params import ActionParams
from policies.base import DiscreteActionSpace
from utils import *


class MvZBTask:
    """
    MvZB task with env-variant driven feature/action loading.

    Supported env_variant:
      "base"   -> features.mvzb / actions.mvzb
      "masked" -> features.mvzb_mask / actions.mvzb_mask
    """

    ENEMY_BIO_TYPES = {UnitTypeId.ZERGLING, UnitTypeId.BANELING}
    ENV_N = 9 * 4

    AUTO_RESET_JUMP = True
    HAS_ENEMY_BIO = True
    USE_ENEMY_STREAK_WIN = True
    INIT_M_ALL = 9
    INIT_E_BIO = 10

    # default; may be overridden by instance if needed
    map_name = "DefeatZerglingsAndBanelings2"
    env_action_space = DiscreteActionSpace(ENV_N)

    ACTION_PARAMS = ActionParams(
        order_cooldown=6,
        wall_repulsion=0.4,
        step_size=2.5,
        map_margin=2.0,
        attack_range_approx=5.5,
        cd_ready_thresh=0.2,
    )

    REWARD_PARAMS = RewardParams(
        R_DELDIST=0.15,
        R_DMG_DEALT=0.45,
        R_DMG_TAKEN=0.18,
        R_KILL_ENEMY=8.0,
        R_LOSS_MARINE=14.0,
        R_EDGE_PEN=0.05,
        R_TIMEOUT_BASE=35.0,
        R_ALIVE_BONUS=30.0,
        R_MOVE_NEAR_ENEMY=0.0,
        MOVE_REWARD_DIST_THRESH=0.35,
        R_FRAC_ENEMY=120.0,
        R_FRAC_OUR=110.0,
        R_DEAD_FINAL=8.0,
    )

    VARIANTS = {
        "base": {
            "feature_module": "features.mvzb",
            "action_module": "actions.mvzb",
            "name": "MvZB",
            "partial_obs_enemy": False,
        },
                
        "masked": {
            "feature_module": "features.mvzb_mask",
            "action_module": "actions.mvzb_mask",
            "name": "MvZB_masked",
            "partial_obs_enemy": True,
        },
    }

    def __init__(self, cfg=None, env_variant: str | None = None):
        self.cfg = cfg

        # The explicitly passed env_variant is preferred; otherwise, it is read from cfg; otherwise, the default base is used.
        variant = env_variant
        if variant is None and cfg is not None:
            variant = getattr(cfg, "env_variant", "base")
        self.env_variant = str(variant or "base").lower()

        if self.env_variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown MvZB env_variant: {self.env_variant}. "
                f"Supported: {list(self.VARIANTS.keys())}"
            )

        spec = self.VARIANTS[self.env_variant]

        # instance-level metadata
        self.name = spec["name"]
        self.PARTIAL_OBS_ENEMY = bool(spec["partial_obs_enemy"])

        # Dynamically import the corresponding version of features/actions
        fe_module = importlib.import_module(spec["feature_module"])
        act_module = importlib.import_module(spec["action_module"])

        FeatureExtractor = getattr(fe_module, "BaseFeatureExtractor")
        ActionExecutor = getattr(act_module, "ActionExecutor")

        self.fe = FeatureExtractor()
        self.rew = RewardCalculator(self.REWARD_PARAMS)
        self.exec = ActionExecutor(self.ACTION_PARAMS)

        # name
        self.ae = self.exec

    def reset_episode(self, bot):
        # feature reset
        if hasattr(self.fe, "reset"):
            try:
                self.fe.reset(bot)
            except TypeError:
                self.fe.reset()

        if hasattr(self.exec, "reset"):
            self.exec.reset()

        if hasattr(self.rew, "reset"):
            self.rew.reset(bot)

        if hasattr(bot, "policy") and hasattr(bot.policy, "reset_episode"):
            bot.policy.reset_episode()

    def featurize_unit(self, bot, marine):
        return self.fe.get_features(bot, marine)

    def featurize_for_trace(self, bot):
        return self.fe.featurize_for_trace(bot)

    def observe(self, bot):
        # check
        if not hasattr(self.fe, "observe_swarm"):
            raise AttributeError(
                f"{type(self.fe).__name__} has no method observe_swarm()."
            )
        return self.fe.observe_swarm(bot)

    async def apply_actions(self, bot, actions):
        return await self.exec.apply_actions(bot, actions)

    def step_reward(self, bot, move_ratio):
        return self.rew.calculate_step_reward(bot, move_ratio)

    def final_reward(self, bot, reason, steps):
        return self.rew.calculate_final_reward(bot, reason, steps)