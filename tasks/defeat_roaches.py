# tasks/defeat_roaches.py
from __future__ import annotations

import importlib

from sc2.ids.unit_typeid import UnitTypeId
from features.defeat_roaches import RoachFeatureExtractor
from rewards.defeat_roaches  import RewardCalculator
from actions.defeat_roaches import ActionExecutor  # ✅
from rewards.params import RewardParams
from actions.params import ActionParams
from policies.base import DiscreteActionSpace
# from WM.params import WMRecorderParams

class DefeatRoachesTask:
    # name = "DefeatRoaches2"
    ENEMY_BIO_TYPES = {UnitTypeId.ROACH}
    ENV_N = 9*4
    AUTO_RESET_JUMP = True
    HAS_ENEMY_BIO = True
    USE_ENEMY_STREAK_WIN = True
    PARTIAL_OBS_ENEMY = False

    INIT_M_ALL = 9
    INIT_E_BIO = 4


    map_name = "DefeatRoaches2"

    env_action_space = DiscreteActionSpace(ENV_N)

    ACTION_PARAMS = ActionParams(
        order_cooldown=6,
        wall_repulsion=0.4,
        step_size=2.5,
        map_margin=2.0,
        attack_range_approx=5.5, # 5/6
        cd_ready_thresh=0.2,
    )

    REWARD_PARAMS = RewardParams(
        R_DELDIST=0.12,
        R_DMG_DEALT=0.35, # The roach is more tough, appropriately increasing the cause damage signal.
        R_DMG_TAKEN=0.18,
        R_KILL_ENEMY=18.0, # 4 enemies
        R_LOSS_MARINE=18.0,
        R_EDGE_PEN=0.05,
        R_TIMEOUT_BASE=60.0,
        R_ALIVE_BONUS=20.0,
        R_MOVE_NEAR_ENEMY=0.00,
        MOVE_REWARD_DIST_THRESH=0.35,

    )

    # WM_PARAMS = WMRecorderParams(
    #     max_marines=9,
    #     max_enemies=4,
    #     canonical_side="right",
    #     hp_marine_max=45.0,

    #     debug=True,
    # )
    VARIANTS = {
        "base": {
            "feature_module": "features.defeat_roaches",
            "action_module": "actions.defeat_roaches",
            "name": "DefeatRoaches2",
            "partial_obs_enemy": False,
        },
        "masked": {
            "feature_module": "features.defeat_roaches_mask",
            "action_module": "actions.defeat_roaches_mask",
            "name": "DefeatRoaches2",
            "partial_obs_enemy": True,
        },
    }

    def __init__(self, cfg=None, env_variant: str | None = None):
        self.cfg = cfg

        variant = env_variant
        if variant is None and cfg is not None:
            variant = getattr(cfg, "env_variant", "base")
        self.env_variant = str(variant or "base").lower()

        if self.env_variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown defeat roach env_variant: {self.env_variant}. "
                f"Supported: {list(self.VARIANTS.keys())}"
            )

        spec = self.VARIANTS[self.env_variant]

        # instance-level metadata
        self.name = spec["name"]
        self.PARTIAL_OBS_ENEMY = bool(spec["partial_obs_enemy"])

        # feature / action
        fe_module = importlib.import_module(spec["feature_module"])
        # print("fe_module:", fe_module)
        act_module = importlib.import_module(spec["action_module"])

        FeatureExtractor = getattr(fe_module, "RoachFeatureExtractor")
        ActionExecutor = getattr(act_module, "ActionExecutor")

        self.fe = FeatureExtractor()
        self.rew = RewardCalculator(self.REWARD_PARAMS)
        self.exec = ActionExecutor(self.ACTION_PARAMS)

        self.ae = self.exec

    def reset_episode(self, bot):
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
        return self.fe.observe_swarm(bot)

    async def apply_actions(self, bot, actions):
        return await self.exec.apply_actions(bot, actions)

    def step_reward(self, bot, move_ratio):
        return self.rew.calculate_step_reward(bot, move_ratio)

    def final_reward(self, bot, reason, steps):
        return self.rew.calculate_final_reward(bot, reason, steps)
