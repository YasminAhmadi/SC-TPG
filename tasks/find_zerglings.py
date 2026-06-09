# tasks/find_zerglings.py
from __future__ import annotations

from sc2.ids.unit_typeid import UnitTypeId
from features.find_zerglings import BaseFeatureExtractor
from rewards.find_zerglings import RewardCalculator
from actions.find_zerglings import ActionExecutor

from rewards.params import RewardParams
from actions.params import ActionParams
from policies.base import DiscreteActionSpace
# from WM.params import WMRecorderParams
from utils import marine_units


class FindZerglingsTask:
    """
    FindAndDefeatZerglings:
    - 3 marines
    - enemies may be initially not visible (need exploration)
    """
    name = "FindZerglings2"
    ENEMY_BIO_TYPES = {UnitTypeId.ZERGLING}
    ENV_N = 9*4

    map_name = "FindAndDefeatZerglings2"

    # Important: do NOT use "enemy==0 streak => win" in exploration tasks
    HAS_ENEMY_BIO = True
    USE_ENEMY_STREAK_WIN = False
    PARTIAL_OBS_ENEMY = True

    AUTO_RESET_JUMP = False
    INIT_M_ALL = 3
    INIT_E_BIO = 0 # not used here

    env_action_space = DiscreteActionSpace(ENV_N)

    # action params (mostly same scale as your other tasks)
    ACTION_PARAMS = ActionParams(
        order_cooldown=6,
        wall_repulsion=0.40,
        step_size=2.5,
        map_margin=2.0,
        attack_range_approx=5.5,
        cd_ready_thresh=0.2,
    )

    # reward params (reuse RewardParams fields)
    REWARD_PARAMS = RewardParams(
        R_SURVIVE = 1.0,
        # Here we repurpose R_DELDIST as "exploration coverage progress weight"
        R_DELDIST=1.0,

        # Use score delta (kills) as main signal via R_KILL_ENEMY
        R_DMG_DEALT=0.0,
        R_DMG_TAKEN=0.10,

        R_KILL_ENEMY=10.0, # multiplied by score delta (often 1 per kill)
        R_LOSS_MARINE=15.0, # if marines can die, keep this

        R_EDGE_PEN=0.05,
        R_TIMEOUT_BASE=0, #30.0,
        R_ALIVE_BONUS=20.0,

        # For combat "distance progress" shaping only when enemy visible
        R_MOVE_NEAR_ENEMY=0.25,
        MOVE_REWARD_DIST_THRESH=0.30,
    )


    # win heuristic
    WIN_COVERAGE_FRAC = 0.985
    WIN_NO_ENEMY_STREAK = 6

    def __init__(self, cfg=None):
        self.cfg = cfg
        
        self.fe = BaseFeatureExtractor()
        self.rew = RewardCalculator(self.REWARD_PARAMS)
        self.exec = ActionExecutor(self.ACTION_PARAMS)

        self._win_streak = 0

    def reset_episode(self, bot):
        self._win_streak = 0
        self.fe.reset()
        self.exec.reset()
        self.rew.reset(bot)
        if hasattr(bot, "policy"):
            bot.policy.reset_episode()

    def check_win(self, bot) -> bool:
        """
        Exploration-safe win:
        - coverage high AND no enemy visible for several consecutive frames
        This avoids the classic bug: "enemy_units empty => win" while enemies are just hidden.
        """
        # Any visible enemy => not win
        if getattr(bot, "enemy_units", None) and bot.enemy_units.exists:
            self._win_streak = 0
            return False

        cov = float(getattr(bot, "_explore_seen_frac", 0.0) or 0.0)
        if cov >= float(self.WIN_COVERAGE_FRAC):
            self._win_streak += 1
        else:
            self._win_streak = 0

        return self._win_streak >= int(self.WIN_NO_ENEMY_STREAK)

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
