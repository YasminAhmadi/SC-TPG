#features/mvzb.py
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from config.schema import *
from utils import marine_units, safe_normalize
from tasks.spec import UnitObs

class BaseFeatureExtractor:
    def __init__(self):
        # Local memory: tag -> [panic, attack_cycle, hazard]
        self.marine_mem = {}
        # Tag -> prev_hp
        self.marine_prev_hp = {}

    def reset(self):
        self.marine_mem.clear()
        self.marine_prev_hp.clear()
    
    
    def get_features(self, bot, marine) -> np.ndarray:
        """
        extract features + update Memory
        """
        W = float(bot.game_info.map_size.x)
        H = float(bot.game_info.map_size.y)

        px = float(marine.position.x)
        py = float(marine.position.y)

        # 1.self features
        x = px / W
        y = py / H
        hp = float(getattr(marine, "health", 45.0))
        hp_ratio = hp / 45.0
        
        cd = float(getattr(marine, "weapon_cooldown", 0.0))
        # print("cd:", cd)
        cd_norm = min(1.0, cd / 10.0)
        
        dmin = min(px, py, W - px, H - py) / max(W, H)

        # 2. ally relationship
        ms = marine_units(bot)
        ally_dir_x = 0.0
        ally_dir_y = 0.0
        ally_density = 0.0
        
        if ms.amount > 1:
            others = [u for u in ms if u.tag != marine.tag]
            if others:
                ox = np.array([float(u.position.x) for u in others], dtype=np.float32)
                oy = np.array([float(u.position.y) for u in others], dtype=np.float32)
                cx = float(ox.mean())
                cy = float(oy.mean())
                
                vec = np.array([cx - px, cy - py], dtype=np.float32)
                dir_vec = safe_normalize(vec)
                ally_dir_x = float(dir_vec[0])
                ally_dir_y = float(dir_vec[1])

                dists = np.sqrt((ox - px) ** 2 + (oy - py) ** 2)
                R_allies = 3.5
                cnt = float((dists < R_allies).sum())
                ally_density = max(0.0, min(1.0, cnt / 8.0))

        # 3. enemy relationship
        enemies = list(bot.enemy_units)
        # print("enemies:", enemies)
        enemy_dir_x = 0.0
        enemy_dir_y = 0.0
        enemy_dist_norm = 1.0
        enemy_density = 0.0

        bane_dir_x = 0.0
        bane_dir_y = 0.0
        bane_dist_norm = 1.0
        bane_threat = 0.0
        ling_threat = 0.0
        vis = 0.0

        if enemies:
            vis = 1.0
            ex = np.array([float(u.position.x) for u in enemies], dtype=np.float32)
            ey = np.array([float(u.position.y) for u in enemies], dtype=np.float32)
            dx = ex - px
            dy = ey - py
            dists = np.sqrt(dx * dx + dy * dy) + 1e-6

            # close enemy direction & distance
            j = int(np.argmin(dists))
            dir_vec = np.array([dx[j], dy[j]], dtype=np.float32)
            dir_normed = safe_normalize(dir_vec) # utils
            
            enemy_dir_x = float(dir_normed[0])
            enemy_dir_y = float(dir_normed[1])
            
            raw_dist = float(dists[j])
            enemy_dist_norm = float(raw_dist / max(W, H))

            # enemy density
            R_enemy = 4.0
            enemy_density = max(0.0, min(1.0, float((dists < R_enemy).sum()) / 10.0))

            # Baneling / Zergling threat
            bane_x = []
            bane_y = []
            bane_d = []
            ling_d = []
            for k, u in enumerate(enemies):
                if u.type_id == UnitTypeId.BANELING:
                    bane_x.append(ex[k])
                    bane_y.append(ey[k])
                    bane_d.append(dists[k])
                elif u.type_id == UnitTypeId.ZERGLING:
                    ling_d.append(dists[k])

            if bane_d:
                idx = int(np.argmin(np.array(bane_d)))
                bx = float(bane_x[idx])
                by = float(bane_y[idx])
                bvec = np.array([bx - px, by - py], dtype=np.float32)
                bvec = safe_normalize(bvec) # utils

                bane_dir_x = float(bvec[0])
                bane_dir_y = float(bvec[1])
                
                raw_dist = float(bane_d[idx])
                bane_dist_norm = float(raw_dist / max(W, H))

                R_bane = 3.0
                bcnt = float((np.array(bane_d) < R_bane).sum())
                bane_threat = max(0.0, min(1.0, bcnt / 4.0))

            if ling_d:
                R_ling = 3.0
                lcnt = float((np.array(ling_d) < R_ling).sum())
                ling_threat = max(0.0, min(1.0, lcnt / 6.0))


        #concatenate all features
        base_feats = np.array(
            [
                x, y, hp_ratio, cd_norm, dmin,
                ally_dir_x, ally_dir_y, ally_density,
                enemy_dir_x, enemy_dir_y, enemy_dist_norm, enemy_density,
                bane_dir_x, bane_dir_y, bane_dist_norm, bane_threat,
                ling_threat, vis,
            ],
            dtype=np.float32,
        )

        return UnitObs(
            tag=int(marine.tag),
            base_feats=base_feats,
            hp=float(hp),
            cd_norm=float(cd_norm),
            enemy_density=float(enemy_density),
            bane_threat=float(bane_threat),
            ling_threat=float(ling_threat),
        )

    def featurize_for_trace(self, bot) -> np.ndarray:
        STATE_DIM = 21
        """Trace graph. old version"""
        ms = marine_units(bot)
        if not ms:
            return np.zeros(STATE_DIM, dtype=np.float32)

        obs = self.get_features(bot, ms.first)

        
        if isinstance(obs, UnitObs):
            base = np.asarray(obs.base_feats, dtype=np.float32).reshape(-1)

            mem = np.asarray(self.marine_mem.get(obs.tag, [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)

            vec = np.concatenate([base, mem], axis=0)

            if vec.shape[0] < STATE_DIM:
                vec = np.concatenate([vec, np.zeros(STATE_DIM - vec.shape[0], dtype=np.float32)], axis=0)
            elif vec.shape[0] > STATE_DIM:
                vec = vec[:STATE_DIM]

            return vec.astype(np.float32)


        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != STATE_DIM:
            if arr.shape[0] < STATE_DIM:
                arr = np.concatenate([arr, np.zeros(STATE_DIM - arr.shape[0], dtype=np.float32)], axis=0)
            else:
                arr = arr[:STATE_DIM]
        return arr