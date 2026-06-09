# train.py
from __future__ import annotations
import os
import pickle
from pathlib import Path
import importlib

from config.schema import TrainConfig

from tasks.beacon import BeaconTask
from tasks.mineral_shards import MineralShardsTask
from tasks.find_zerglings import FindZerglingsTask
from tasks.defeat_roaches import DefeatRoachesTask
from tasks.mvzb import MvZBTask

from sc2env.mvzb_bot import SwarmBot
from policies.registry import make_policy
from training.loop import run_once

TASKS = {
    "beacon": BeaconTask,
    "mineral_shards": MineralShardsTask,
    "find_zerglings": FindZerglingsTask,
    "DefeatRoaches": DefeatRoachesTask,
    "mvzb": MvZBTask,
}




def make_bot(trainer, task, policy, cfg):
    bot_module_name = cfg.task_name.lower() 
    
    module_path = f"sc2env.{bot_module_name}_bot"
    
    try:
        bot_module = importlib.import_module(module_path)
        bot_class = getattr(bot_module, "SwarmBot")
        
        return bot_class(trainer=trainer, task=task, policy=policy, cfg=cfg)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find SwarmBot in {module_path}. Check if the file exists.")
        raise e



def _gen_from_trainer_ckpt(ckpt_path: Path) -> int | None:

    if not ckpt_path.exists():
        return None
    try:
        with ckpt_path.open("rb") as f:
            trainer = pickle.load(f)

        for attr in ("generation", "gen", "cur_gen", "current_gen", "n_generations", "epoch"):
            if hasattr(trainer, attr):
                v = getattr(trainer, attr)
                if isinstance(v, int):
                    return v
        return None
    except Exception:
        return None



def get_current_gen(cfg: TrainConfig) -> int:
    g = _gen_from_trainer_ckpt(cfg.trainer_ckpt)
    if g is not None:
        return g

    return 0

def main():
    task_name = os.getenv("TASK_NAME", "mvzb")
    policy_name = os.getenv("POLICY_NAME", "fixed")
    seed = int(os.getenv("SEED", "0"))

    cfg = TrainConfig(task_name=task_name, policy_name=policy_name, seed=seed)
    cfg.ensure_dirs()

    task = TASKS[cfg.task_name]()
    policy = make_policy(cfg.policy_name, env_n=task.env_action_space.n)


    while True:
        cur_gen = get_current_gen(cfg)
        if cur_gen >= cfg.gens:
            print(f"[DONE] reached target gens: cur_gen={cur_gen}, target={cfg.gens}")
            break

        print(f"[TRAIN] cur_gen={cur_gen} -> running one more...")
        run_once(cfg, task, policy, make_bot)

if __name__ == "__main__":
    main()
