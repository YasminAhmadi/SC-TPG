# training/loop.py
from __future__ import annotations

from sc2 import maps
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer

from seed import set_global_seed

from training.checkpoint import load_or_create_trainer, save_trainer

def run_once(cfg, task, policy, make_bot):
    """
    cfg: TrainConfig
    task: TaskSpec
    policy: PolicySpec
    make_bot: (trainer, task, policy) -> BotAI
    """
    # whther to use seed, open when needed
    # set_global_seed(cfg.seed)
    
    # n_actions = policy.wrap_action_space(task.env_action_space).n
    n_actions = policy.agent_action_n
    # print("n_actions:", n_actions)
    trainer = load_or_create_trainer(cfg.trainer_ckpt, n_actions=n_actions)

    print(f"\n[MAIN] ===== launch SC2 | gen={getattr(trainer,'generation',0)} | task={task.name} | policy={policy.name} =====")
    try:
        result = run_game(
            maps.get(task.map_name),
            [
                Bot(Race.Terran, make_bot(trainer, task, policy, cfg)),
                Computer(Race.Zerg, Difficulty.VeryEasy),
            ],
            realtime=cfg.realtime,
            # save_replay_as=cfg.run_dir / f"replay{getattr(trainer,'generation',0)}.SC2Replay",#whether to save replay. Will not save for training
        )
        print("[MAIN] SC2 game ended with:", result)
    except Exception as e:
        print("[MAIN] SC2 game crashed with:", e)

    save_trainer(trainer, cfg.trainer_ckpt)
