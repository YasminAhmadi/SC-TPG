# policies/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np

from tasks.spec import UnitObs


@dataclass(frozen=True)
class DiscreteActionSpace:
    n: int
    def clip(self, a: int) -> int:
        return int(a) % int(self.n)

@dataclass(frozen=True)
class CompositeActionSpace:
    """
    (mem_action, env_action)：
      combined = mem_action * env_n + env_action
    n = env_n * mem_n
    """
    env: DiscreteActionSpace
    mem: DiscreteActionSpace

    @property
    def n(self) -> int:
        return int(self.env.n) * int(self.mem.n)

    def clip(self, a: int) -> int:
        return int(a) % self.n

    def decode(self, a: int) -> tuple[int, int]:
        c = self.clip(a)
        env_a = c % int(self.env.n)
        mem_a = c // int(self.env.n)
        return int(env_a), int(mem_a)

    def encode(self, env_a: int, mem_a: int) -> int:
        return int(mem_a) * int(self.env.n) + (int(env_a) % int(self.env.n))


class MemoryPolicyBase:
    name: str = "base"

    def __init__(
        self,
        mem_dim: int,
        action_space, # DiscreteActionSpace or CompositeActionSpace
        base_dim: int = 18,
        mem_action_space: Optional[DiscreteActionSpace] = None,
    ):
        self.mem_dim = int(mem_dim)
        self.base_dim = int(base_dim)

        if isinstance(action_space, CompositeActionSpace):
            self.action_space = action_space.env
            self.mem_action_space = action_space.mem
        else:
            self.action_space = action_space
            self.mem_action_space = mem_action_space

        self._mem: Dict[int, np.ndarray] = {}

    def decode_env_move_tgt(self, env_a: int, n_tgt: int = 4) -> Tuple[int, int]:
        """
        env_a (0..env_n-1) -> factorized actions:
          env_a = move_idx * n_tgt + tgt_idx
        For ActionExecutor。
        """
        env_a = int(self.action_space.clip(env_a))
        move_idx = env_a // int(n_tgt)
        tgt_idx = env_a % int(n_tgt)
        return int(move_idx), int(tgt_idx)
    
    @property
    def state_dim(self) -> int:
        return self.base_dim + self.mem_dim
    
    @property
    def agent_action_n(self) -> int:
        if self.mem_action_space is None:
            return int(self.action_space.n)
        return int(self.action_space.n) * int(self.mem_action_space.n)
    
    def reset_episode(self) -> None:
        self._mem.clear()
        self._reset_extra()

    def _reset_extra(self) -> None:
        pass

    def get_mem(self, tag: int) -> np.ndarray:
        tag = int(tag)
        m = self._mem.get(tag)
        if m is None:
            m = np.zeros((self.mem_dim,), dtype=np.float32)
            self._mem[tag] = m
        return m

    def set_mem(self, tag: int, mem: np.ndarray) -> None:
        mem = np.asarray(mem, dtype=np.float32).reshape(-1)
        if mem.shape[0] != self.mem_dim:
            raise ValueError(f"mem dim mismatch!!!: got {mem.shape[0]} expected {self.mem_dim}")
        self._mem[int(tag)] = mem

    def featurize(self, obs: UnitObs) -> np.ndarray:
        base = np.asarray(obs.base_feats, dtype=np.float32).reshape(-1)[: self.base_dim]
        mem = self.get_mem(obs.tag)
        return np.concatenate([base, mem], axis=0)

    def decode_action(self, out: Any) -> Tuple[int, Optional[int]]:
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, (list, np.ndarray)):
            out = int(np.asarray(out).ravel()[0])
        out = int(out)

        env_n = int(self.action_space.n)

        if self.mem_action_space is None:
            # print("if self.mem_action_space is None")
            return self.action_space.clip(out), None

        mem_n = int(self.mem_action_space.n)
        total = env_n * mem_n
        c = out % total
        env_action = c % env_n
        mem_action = c // env_n
        # print("env_action:", env_action)
        # print("mem_action:", mem_action)
        return int(env_action), int(mem_action)

    def act(self, agent, obs: UnitObs, bot=None, marine=None, task=None) -> int:
        mem = self.get_mem(obs.tag) if self.mem_dim > 0 else None
        # print("mem:", mem)

        full_state = self.featurize(obs).astype(np.float32, copy=False)

        out = agent.act(full_state.tolist())
        env_a, mem_a = self.decode_action(out)

        # only for memory
        if self.mem_dim > 0:
            self.apply_memory_action(obs, mem_a)

        return env_a

    def apply_memory_action(self, obs: UnitObs, mem_action: Optional[int]) -> None:
        raise NotImplementedError
