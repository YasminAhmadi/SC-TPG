# seed.py
from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
