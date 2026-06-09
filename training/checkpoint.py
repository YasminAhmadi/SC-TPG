# training/checkpoint.py
from __future__ import annotations
import pickle
from pathlib import Path
from tpg.trainer import Trainer

def load_or_create_trainer(ckpt_path: Path, n_actions: int) -> Trainer:
    if ckpt_path.exists():
        with ckpt_path.open("rb") as f:
            trainer: Trainer = pickle.load(f)

        # # for old version ckpt
        # if not hasattr(trainer, "generation"):
        #     trainer.generation = 0
        # if not hasattr(trainer, "resume"):
        #     trainer.resume = None

        if hasattr(trainer, "battle_state") and trainer.battle_state is not None:
            print("[CKPT] drop battle_state (crash-unsafe)", flush=True)
            trainer.battle_state = None

        resume = getattr(trainer, "resume", None)
        if not (isinstance(resume, dict) and int(resume.get("generation", -999)) == int(trainer.generation)):
            trainer.resume = None
        
        print(f"[CKPT] loaded trainer @ {ckpt_path}, generation={trainer.generation}")
        return trainer

    # discrete integer int(n_actions)
    # discrete action type + continuous real-valued vector list(range(n_actions))
    trainer = Trainer(actions=int(n_actions))#list(range(n_actions)) Doesn't matter for results.
    trainer.generation = 0
    trainer.resume = None
    print(f"[CKPT] created NEW trainer, generation=0 | n_actions={n_actions}")
    return trainer


def save_trainer(trainer: Trainer, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with ckpt_path.open("wb") as f:
        pickle.dump(trainer, f)
    print(f"[CKPT] saved trainer -> {ckpt_path} | generation={trainer.generation}")
