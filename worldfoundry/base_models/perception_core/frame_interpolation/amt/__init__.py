"""Canonical AMT frame-interpolation runtime."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.core.io.paths import package_data_path


def config_path(name: str = "AMT-S.yaml") -> Path:
    return package_data_path('models', 'runtime', 'configs', 'amt') / name


def checkpoint_path() -> Path:
    capability = BASE_MODEL_CAPABILITIES["vbench_metric_checkpoint_assets"]
    for asset in capability.assets:
        if asset.id == "vbench_amt_s_checkpoint":
            status = asset.check()
            return Path(status["matched_path"] or status["local_path"])
    raise RuntimeError("vbench_amt_s_checkpoint is not registered in BASE_MODEL_CAPABILITIES.")


__all__ = ["checkpoint_path", "config_path"]
