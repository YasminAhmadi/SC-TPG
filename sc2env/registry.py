# sc2enc/registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import importlib

from tasks.beacon import BeaconTask
from tasks.mineral_shards import MineralShardsTask
from tasks.find_zerglings import FindZerglingsTask
from tasks.defeat_roaches import DefeatRoachesTask
from tasks.mvzb import MvZBTask


@dataclass(frozen=True)
class EnvSpec:
    task_cls: type
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    bot_module: str = ""
    bot_class: str = "SwarmBot"

def get_env_spec(task_name: str, env_variant: str) -> EnvSpec:
    key = (str(task_name).lower(), str(env_variant).lower())
    if key not in ENV_REGISTRY:
        raise ValueError(f"Unknown env spec: task={task_name}, env_variant={env_variant}")
    return ENV_REGISTRY[key]


def make_task_from_cfg(cfg):
    spec = get_env_spec(cfg.task_name, cfg.env_variant)
    return spec.task_cls(cfg=cfg, **spec.task_kwargs)


def make_bot_factory_from_cfg(cfg):
    spec = get_env_spec(cfg.task_name, cfg.env_variant)
    module = importlib.import_module(spec.bot_module)
    bot_cls = getattr(module, spec.bot_class)

    def make_bot(trainer, task, policy, cfg):
        return bot_cls(trainer=trainer, task=task, policy=policy, cfg=cfg)

    return make_bot

ENV_REGISTRY: dict[tuple[str, str], EnvSpec] = {
    ("beacon", "base"): EnvSpec(
        task_cls=BeaconTask,
        bot_module="sc2env.beacon_bot",
    ),
    ("mineral_shards", "base"): EnvSpec(
        task_cls=MineralShardsTask,
        bot_module="sc2env.mineral_shards_bot",
    ),
    ("find_zerglings", "base"): EnvSpec(
        task_cls=FindZerglingsTask,
        bot_module="sc2env.find_zerglings_bot",
    ),
    # Defeat Roaches
    ("defeatroaches", "base"): EnvSpec(
        task_cls=DefeatRoachesTask,
        task_kwargs={"env_variant": "base"},
        bot_module="sc2env.defeatroaches_bot",
    ),
    ("defeatroaches", "masked"): EnvSpec(
        task_cls=DefeatRoachesTask,
        task_kwargs={"env_variant": "masked"},
        bot_module="sc2env.defeatroaches_mask_bot",
    ),

    # MvZB
    ("mvzb", "base"): EnvSpec(
        task_cls=MvZBTask,
        task_kwargs={"env_variant": "base"},
        bot_module="sc2env.mvzb_bot",
    ),
    ("mvzb", "masked"): EnvSpec(
        task_cls=MvZBTask,
        task_kwargs={"env_variant": "masked"},
        bot_module="sc2env.mvzb_mask_bot",
    ),
}