# train_once.py
from __future__ import annotations
from seed import set_global_seed

from config.schema import TrainConfig
from tasks.beacon import BeaconTask
from tasks.mineral_shards import MineralShardsTask
from tasks.find_zerglings import FindZerglingsTask
from tasks.defeat_roaches import DefeatRoachesTask
from tasks.mvzb import MvZBTask


# from sc2env.beacon_bot import SwarmBot
# from sc2env.mineral_shards_bot import SwarmBot
# from sc2env.find_zerglings_bot import SwarmBot

# from sc2env.defeatroaches_bot import SwarmBot
from sc2env.mvzb_bot import SwarmBot



from policies.registry import make_policy

from training.loop import run_once

def make_bot(trainer, task, policy, cfg):
    return SwarmBot(trainer=trainer, task=task, policy=policy, cfg=cfg)

def main():
    # while True:
    # cfg = TrainConfig(task_name="beacon", env_variant="base", policy_name="fixed", seed=0)
    # cfg = TrainConfig(task_name="beacon", env_variant="base", policy_name="no", seed=0)
    
    # cfg = TrainConfig(task_name="mineral_shards", env_variant="base", policy_name="fixed", seed=0)
    # cfg = TrainConfig(task_name="mineral_shards", env_variant="base", policy_name="no", seed=0)
    
    # cfg = TrainConfig(task_name="find_zerglings", env_variant="base", policy_name="no_masked", seed=0)#FindAndDefeatZerglings_fog2
    # cfg = TrainConfig(task_name="find_zerglings", env_variant="base", policy_name="no", seed=0)
    
    # cfg = TrainConfig(task_name="roach", env_variant="base", policy_name="no", seed=0)
    # cfg = TrainConfig(task_name="roach", env_variant="base", policy_name="combat", seed=0)
    
    cfg = TrainConfig(task_name="mvzb", env_variant="base", policy_name="no", seed=0)
    # cfg = TrainConfig(task_name="mvzb", env_variant="base", policy_name="combat", seed=0)
    
    # cfg = TrainConfig(task_name="mvzb", env_variant="base", policy_name="fixed", seed=0)
    # cfg = TrainConfig(task_name="mvzb", env_variant="base", policy_name="random", seed=0)
   
    # whether to use seed, open when needed
    # set_global_seed(cfg.seed)
    cfg.ensure_dirs()
    
    
    # task = BeaconTask(cfg=cfg)
    # task = MineralShardsTask(cfg=cfg)
    # task = FindZerglingsTask(cfg=cfg)
    # task = DefeatRoachesTask(cfg=cfg)
    task = MvZBTask(cfg=cfg)
    
    
    policy = make_policy(cfg.policy_name, env_n=task.env_action_space.n)

    run_once(cfg, task, policy, make_bot)

if __name__ == "__main__":
    main()
