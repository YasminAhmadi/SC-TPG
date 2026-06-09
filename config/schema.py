# config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainConfig:
    # experiment identity
    task_name: str = "mvzb"
    env_variant: str = "base"
    policy_name: str = "fixed"
    seed: int = 0
    
    

    # training control
    gens: int = 2000 # Set the number of generations. try 1000/ 1500 / 2000 / 3000
    episodes_per_agent: int = 2
    realtime: bool = False # whether to run in realtime

    # SC2 loop timeout
    loop_timeout: int = int(55 * 22.4)
    
    # reward mode. whether to use the step reward. False for easy training. MineralShards is better with step reward.
    # If true, use a light terminal bonus
    use_step_reward: bool = False
    debug_step_reward: bool = False
    
    # freeze
    freeze_act: bool = False
    freeze_act_path: str = ""
    freeze_act_reload_each_episode: bool = True

    # reset / robustness
    reset_wait_steps: int = 10
    reset_max_tries: int = 8
    reset_min_delta: float = 1.0

    # run paths. save path
    runs_root: Path = Path("runs")
    run_tag: str = "" 


    def run_id(self) -> str:
        task = str(self.task_name).lower()
        variant = str(self.env_variant).lower()
        policy = str(self.policy_name).lower()
        tag = f"_{self.run_tag}" if self.run_tag else ""
        return f"{task}_{variant}_{policy}_s{self.seed:03d}{tag}"
    
    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id()

    @property
    def trainer_ckpt(self) -> Path:
        return self.run_dir / "trainer.pkl"

    @property
    def fitness_csv(self) -> Path:
        return self.run_dir / "fitness.csv"

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def trace_dir(self) -> Path:
        return self.run_dir / "traces"

    @property
    def wm_dir(self) -> Path:
        return self.run_dir / "wm_data"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.wm_dir.mkdir(parents=True, exist_ok=True)

