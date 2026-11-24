# Inference-only ACT source retained in-tree.
from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar


class _DatasetStatsUnpickler(pickle.Unpickler):
    """Restricted loader for the NumPy arrays in official ACT statistics."""

    _ALLOWED_GLOBALS: ClassVar[set] = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"ACT dataset statistics contain a forbidden pickle global: {module}.{name}")
        return super().find_class(module, name)


def build_policy_class():
    """Create the ACT policy class after torch dependencies are available.

    Args:
        None.
    """
    import torch.nn as nn
    import torchvision.transforms as transforms

    from .modeling import build_ACT_model

    class ACTPolicy(nn.Module):
        def __init__(self, args: Any) -> None:
            super().__init__()
            self.model = build_ACT_model(args)

        def forward(self, qpos, image):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
            a_hat, _is_pad_hat = self.model(qpos, image)
            return a_hat

    return ACTPolicy


def load_dataset_stats(checkpoint_dir: Path) -> Mapping[str, Any] | None:
    """Load ACT dataset normalization statistics when staged with weights.

    Args:
        checkpoint_dir: Directory expected to contain dataset_stats.pkl.
    """
    stats_path = checkpoint_dir / "dataset_stats.pkl"
    if not stats_path.is_file():
        return None
    with stats_path.open("rb") as handle:
        payload = _DatasetStatsUnpickler(handle).load()
    if not isinstance(payload, Mapping):
        raise TypeError("ACT dataset_stats.pkl root must be a mapping")

    import numpy as np

    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"ACT dataset statistics are missing required keys: {missing}")
    validated: dict[str, np.ndarray] = {}
    for key in required:
        value = np.asarray(payload[key])
        if value.dtype.kind not in {"f", "i", "u"} or value.ndim != 1:
            raise TypeError(f"ACT statistic {key!r} must be a one-dimensional numeric array")
        if not np.isfinite(value).all():
            raise ValueError(f"ACT statistic {key!r} contains non-finite values")
        validated[key] = value.astype(np.float32, copy=False)
    return validated
