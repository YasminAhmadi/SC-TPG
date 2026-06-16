# policies/registry.py
from __future__ import annotations

from typing import Any

def make_policy(name: str, **kwargs: Any):
    name = str(name).lower()
    
   
    if name in ("no", "nomemory"):
        from policies.no_memory import NoMemoryPolicy
        return NoMemoryPolicy(**kwargs)
    
    if name in ("combat"):
        from policies.combat_memory import CombatMemoryMaskedPolicy
        return CombatMemoryMaskedPolicy(**kwargs)
    
    if name in ("no_masked"):
        # from policies.no_memory_masked import NoMemoryPolicyMasked
        # return NoMemoryPolicyMasked(**kwargs)
        from policies.no_memory import NoMemoryPolicy
        return NoMemoryPolicy(**kwargs)

    if name in ("fixed", "fixed_memory"):
        from policies.fixed_memory import FixedMemoryPolicy
        return FixedMemoryPolicy(**kwargs)
    
    if name == "evolved":#in ("evolved", "evolved_memory"):
        from policies.evolved_memory import EvolvedMemoryPolicy
        return EvolvedMemoryPolicy(**kwargs)

    raise ValueError(f"Unknown policy: {name}")
