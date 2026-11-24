from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    act: np.ndarray
    real_lengths: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs": self.obs,
            "act": self.act,
            "real_lengths": self.real_lengths,
        }
