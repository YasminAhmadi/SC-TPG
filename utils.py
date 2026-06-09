# utils.py
import numpy as np
from typing import List, Optional, Iterable, Dict, Tuple, Any
from enum import Enum

from sc2.position import Point2
from sc2.data import Alliance
from sc2.ids.unit_typeid import UnitTypeId

from config.schema import *
from tpg.mu_try import render_trace_graph


def _utid(x) -> int:
    """ UnitTypeId / Enum / int -> int id, Compatible """
    try:
        # In python-sc2, UnitTypeId is mostly Enum, and .value is int.
        v = getattr(x, "value", None)
        if v is not None:
            return int(v)
    except Exception:
        pass
    # fallback：In some cases, x itself is an int.
    try:
        return int(x)
    except Exception:
        return 0

def _safe_unit_type_value(ut) -> int:
    """
    Compatible with different SC2 versions: UnitTypeId.MARINE.value / int(UnitTypeId.MARINE)
    """
    try:
        return int(getattr(ut, "value"))
    except Exception:
        try:
            return int(ut)
        except Exception:
            # error
            return int(getattr(ut, "name", "0") == "MARINE")

def point_xy(p: Point2) -> Tuple[float, float]:
    return float(p.x), float(p.y)


def mirror_x(x: float, map_w: float) -> float:
    # Mirror the map horizontally by its width W: x' = W - x
    return float(map_w) - float(x)


def clip_position(p: Point2, map_w: float, map_h: float, margin: float = 0.8) -> Point2:
    x = max(margin, min(map_w - margin, float(p.x)))
    y = max(margin, min(map_h - margin, float(p.y)))
    return Point2((x, y))


def safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm


def count_units_in_raw_data(raw_data, unit_type_id: int, alliance_id: int) -> int:
    count = 0
    for u in raw_data.units:
        u_alliance = getattr(u, "alliance", None)
        u_type = int(getattr(u, "unit_type", -1))
        u_health = float(getattr(u, "health", 0.0))
        if u_alliance == alliance_id and u_type == unit_type_id and u_health > 0:
            count += 1
    return count


def get_centroid(units) -> Optional[Point2]:
    if not units:
        return None
    return units.center


def marine_units(bot):
    """
    Prioritize returning controlled marines (controlled_marine_tags), but if tags temporarily mismatch (empty intersection), fallback to all marines to avoid instant all_dead.
    """
    ms = bot.units(UnitTypeId.MARINE)
    if not ms:
        return ms

    tags = getattr(bot, "controlled_marine_tags", None)
    if not tags:
        return ms

    try:
        tags_set = set(tags)
        filt = ms.filter(lambda u: int(u.tag) in tags_set)
        if filt.amount > 0:
            return filt
    except Exception:
        pass

    return ms


def self_marine_tags_raw(bot) -> set[int]:
    try:
        raw = bot.state.observation.raw_data
    except Exception:
        return {int(u.tag) for u in bot.units(UnitTypeId.MARINE)}

    marine_id = _utid(UnitTypeId.MARINE)

    tags = set()
    for u in raw.units:
        if getattr(u, "alliance", None) != Alliance.Self.value:
            continue
        if int(getattr(u, "unit_type", -1)) != marine_id:
            continue

        dt = getattr(u, "display_type", None)
        if dt is not None and int(dt) != 1:  # Visible only
            continue

        if float(getattr(u, "health", 0.0)) <= 0:
            continue

        tags.add(int(u.tag))
    return tags


def count_marines(bot) -> int:
    """
    Final snapshot override: If `alive_marines` is provided in `bot._rew_ov_final`, use it (for auto-reset frame skipping calculation).
    Otherwise: prioritize counting based on the intersection of controlled tags in `raw_data`; if that fails, fall back to `units.amount`.
    """
    snap = getattr(bot, "_rew_ov_final", None)
    if snap is not None and "alive_marines" in snap:
        return int(snap["alive_marines"])

    tags = getattr(bot, "controlled_marine_tags", None)
    if tags:
        raw_tags = self_marine_tags_raw(bot)
        inter = raw_tags & set(tags)
        if len(inter) > 0:
            return int(len(inter))

    try:
        return int(bot.units(UnitTypeId.MARINE).amount)
    except Exception:
        # 0
        return 0


def get_enemy_centroid(bot) -> Optional[Point2]:
    pts = []
    
    try:
        raw = bot.state.observation.raw_data
        for u in raw.units:
            if getattr(u, "alliance", None) == Alliance.Enemy.value:
                pts.append(Point2((u.pos.x, u.pos.y)))
    except Exception:
        for u in bot.enemy_units:
            pts.append(u.position)

    if not pts:
        return None
    
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    
    return Point2((sum(xs) / len(xs), sum(ys) / len(ys)))


def count_enemies(bot) -> int:
    try:
        raw = bot.state.observation.raw_data
        return sum(
            1 for u in raw.units
            if getattr(u, "alliance", None) == Alliance.Enemy.value and float(getattr(u, "health", 0.0)) > 0
        )
    except Exception:
        return sum(1 for u in bot.enemy_units if getattr(u, "health", 0.0) > 0)


# def count_enemy_bio(bot) -> int:
#     snap = getattr(bot, "_rew_ov_final", None)
#     if snap is not None and "e_bio" in snap:
#         return int(snap["e_bio"])

#     ling_id = _utid(UnitTypeId.ZERGLING)
#     bane_id = _utid(UnitTypeId.BANELING)

#     try:
#         raw = bot.state.observation.raw_data
#         n = 0
#         for u in raw.units:
#             if getattr(u, "alliance", None) != Alliance.Enemy.value:
#                 continue
#             if float(getattr(u, "health", 0.0)) <= 0:
#                 continue
#             ut = int(getattr(u, "unit_type", -1))
#             if ut == ling_id or ut == bane_id:
#                 n += 1
#         return int(n)
#     except Exception:
#         return sum(
#             1 for u in bot.enemy_units
#             if u.type_id in (UnitTypeId.ZERGLING, UnitTypeId.BANELING) and getattr(u, "health", 0.0) > 0
        # )

def count_enemy_bio(bot) -> int:
    """
    Count "enemy biological units" for termination/win logic.

    Priority:
    1) If bot has _rew_ov_final snapshot (auto-reset jump), use snap["e_bio"].
    2) Else count from raw_data (most robust to partial wrappers).
    3) Fallback to bot.enemy_units.

    Enemy bio types are task-defined:
      task.ENEMY_BIO_TYPES = {UnitTypeId.ZERGLING, UnitTypeId.BANELING}  (MvZB)
      task.ENEMY_BIO_TYPES = {UnitTypeId.ROACH}                          (DefeatRoaches)
    Backward compatible: default to (ZERGLING, BANELING) if not provided.
    """
    snap = getattr(bot, "_rew_ov_final", None)
    if snap is not None and "e_bio" in snap:
        return int(snap["e_bio"])

    #task-defined types
    types = None
    try:
        task = getattr(bot, "task", None)
        types = getattr(task, "ENEMY_BIO_TYPES", None) if task is not None else None
    except Exception:
        types = None

    if not types:
        types = (UnitTypeId.ZERGLING, UnitTypeId.BANELING)

    # map to unit_type int ids for raw_data matching
    type_ids = {int(_utid(t)) for t in types}

    #prefer raw_data
    try:
        raw = bot.state.observation.raw_data
        n = 0
        for u in raw.units:
            if getattr(u, "alliance", None) != Alliance.Enemy.value:
                continue
            if float(getattr(u, "health", 0.0)) <= 0:
                continue
            ut = int(getattr(u, "unit_type", -1))
            if ut in type_ids:
                n += 1
        return int(n)
    
    except Exception:
        # fallback to bot.enemy_units
        types_set = set(types)
        return int(sum(
            1 for u in bot.enemy_units
            if u.type_id in types_set and float(getattr(u, "health", 0.0)) > 0.0
        ))


def sum_hp_units(units: Iterable) -> float:
    total = 0.0
    for u in units:
        hp = getattr(u, "health", 0.0) or 0.0
        if hp > 0:
            total += float(hp)
    return float(total)


def sum_hp(units: Iterable) -> float:
    """sum_hp(Units)"""
    return sum_hp_units(units)


def sum_marine_hp(bot) -> float:
    snap = getattr(bot, "_rew_ov_final", None)
    if snap is not None and "marine_hp" in snap:
        return float(snap["marine_hp"])
    return float(sum_hp_units(marine_units(bot)))


def sum_enemy_hp_units(units) -> float:
    return float(sum(getattr(u, "health", 0.0) for u in units if (getattr(u, "health", 0.0) or 0.0) > 0))


def sum_enemy_hp(bot) -> float:
    snap = getattr(bot, "_rew_ov_final", None)
    if snap is not None and "enemy_hp" in snap:
        return float(snap["enemy_hp"])
    return float(sum_enemy_hp_units(bot.enemy_units))


def enemy_points(bot) -> List[Point2]:
    pts = []
    try:
        raw = bot.state.observation.raw_data
        for u in raw.units:
            if getattr(u, "alliance", None) == Alliance.Enemy.value:
                pts.append(Point2((u.pos.x, u.pos.y)))
    except Exception:
        for u in bot.enemy_units:
            pts.append(u.position)
    
    return pts


def count_units_by_type(units, types):
    return sum(1 for u in units if u.type_id in types)


def calculate_wall_repulsion(pos, w, h, margin):
    rep = np.array([0.0, 0.0], dtype=np.float32)
    x, y = pos[0], pos[1]
    if x < margin:
        rep[0] += (margin - x) / margin
    if x > w - margin:
        rep[0] -= (x - (w - margin)) / margin
    if y < margin:
        rep[1] += (margin - y) / margin
    if y > h - margin:
        rep[1] -= (y - (h - margin)) / margin
    return rep


async def clear_battlefield(bot):
    try:
        raw = bot.state.observation.raw_data
    except Exception as e:
        print("[WARN] clear_battlefield!!!: no raw_data:", e, flush=True)
        return

    ling_id = _utid(UnitTypeId.ZERGLING)
    bane_id = _utid(UnitTypeId.BANELING)

    tags = []
    for u in raw.units:
        if getattr(u, "alliance", None) != Alliance.Enemy.value:
            continue
        ut = int(getattr(u, "unit_type", -1))
        if ut in (ling_id, bane_id):
            tags.append(u.tag)

    if not tags:
        return

    try:
        await bot._client.debug_kill_unit(tags)
        print(f"[INFO] clear_battlefield: killed {len(tags)} enemy combat units", flush=True)
    except Exception as e:
        print("[WARN] clear_battlefield!!!: debug_kill_unit failed:", e, flush=True)


def reset_effective(m0: Optional[Point2], e0: Optional[Point2], m1: Optional[Point2], e1: Optional[Point2], min_delta: float) -> bool:
    if m0 is None and m1 is not None:
        return True
    if e0 is None and e1 is not None:
        return True
    if m0 is not None and m1 is None:
        return True
    if e0 is not None and e1 is None:
        return True

    moved_m = (m0 is not None and m1 is not None and m1.distance_to(m0) > min_delta)
    moved_e = (e0 is not None and e1 is not None and e1.distance_to(e0) > min_delta)
    return moved_m or moved_e


async def kill_stray_marines(bot):
    tags_keep = getattr(bot, "controlled_marine_tags", None)
    if not tags_keep:
        return

    try:
        raw = bot.state.observation.raw_data
    except Exception:
        return

    marine_id = _utid(UnitTypeId.MARINE)

    stray_tags = []
    keep = set(tags_keep)
    for u in raw.units:
        if getattr(u, "alliance", None) != Alliance.Self.value:
            continue
        if int(getattr(u, "unit_type", -1)) != marine_id:
            continue
        if int(u.tag) not in keep:
            stray_tags.append(u.tag)

    if stray_tags:
        try:
            await bot._client.debug_kill_unit(stray_tags)
            # print(f"[DBG-KILL-STRAY] killed {len(stray_tags)} stray self MARINE units", flush=True)
        except Exception as e:
            print("[WARN] kill_stray_marines failed!!!:", e, flush=True)


def log_fitness_csv(cfg: TrainConfig, gen: int, best: float, mean: float, std: float, var: float):
    import csv
    log_path = cfg.fitness_csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["generation", "best", "mean", "std", "var"])
        writer.writerow([gen, best, mean, std, var])


async def announce(bot, msg: str):
    try:
        await bot.client.chat_send(msg, False)
    except Exception:
        pass
    print(msg, flush=True)


def draw_trace(cfg: TrainConfig, bot: Any, best_agent: Any) -> None:
    """
    OLD VERSION
    bot: Only needs to provide `featurize_for_trace` for the task/policy (you can use `bot.features` now).
    best_agent: TPG agent.
    """
    try:

        if hasattr(bot, "task") and hasattr(bot.task, "fe") and hasattr(bot.task.fe, "featurize_for_trace"):
            state_vec = bot.task.fe.featurize_for_trace(bot).tolist()
        else:
            # fallback：for old version
            state_vec = bot.features.featurize_for_trace(bot).tolist()

        trace = {}
        _ = best_agent.act(state_vec, path_trace=trace)

        trace_path = cfg.trace_dir / f"trace_gen{bot.generation:03d}"
        render_trace_graph(trace, filename=str(trace_path))
        print(f"[Gen {bot.generation:03d}] trace saved -> {trace_path}")
    except Exception as e:
        print(f"[Gen {bot.generation:03d}] trace render failed: {e}", flush=True)
