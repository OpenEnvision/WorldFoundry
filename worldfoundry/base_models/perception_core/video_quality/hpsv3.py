"""HPSv3 runtime and asset paths used by WBench quality metrics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

RUNTIME_ROOT = Path(__file__).resolve().parent / "hpsv3_runtime"


def _asset_path(asset_id: str) -> Path:
    for asset in BASE_MODEL_CAPABILITIES["wbench_hpsv3"].assets:
        if asset.id == asset_id:
            status = asset.check()
            return Path(status["matched_path"] or status["local_path"])
    raise RuntimeError(f"wbench_hpsv3 asset is not registered: {asset_id}")


def add_runtime_to_path() -> Path:
    """Expose the vendored `hpsv3` package for callers that import it directly."""
    runtime = RUNTIME_ROOT
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    return runtime


def checkpoint_path() -> Path:
    return _asset_path("wbench_hpsv3_checkpoint")


def config_path() -> Path:
    return _asset_path("wbench_hpsv3_config")


def qwen_model_dir() -> Path:
    return _asset_path("wbench_hpsv3_qwen2_vl_model")


def resolved_config_path() -> Path:
    """Return HPSv3 config, rewritten to local Qwen2-VL weights when available."""
    config = config_path()
    qwen_dir = qwen_model_dir()
    if not qwen_dir.is_dir():
        return config

    import yaml

    with config.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["model_name_or_path"] = str(qwen_dir)
    output = config.parent / "_HPSv3_7B_resolved.yaml"
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return output


def load_inferencer(*, device: str = "cuda"):
    add_runtime_to_path()

    import numpy as np
    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = list[np.ndarray]

    from hpsv3.inference import HPSv3RewardInferencer

    return HPSv3RewardInferencer(
        config_path=str(resolved_config_path()),
        checkpoint_path=str(checkpoint_path()),
        device=device,
    )


__all__ = [
    "RUNTIME_ROOT",
    "add_runtime_to_path",
    "checkpoint_path",
    "config_path",
    "load_inferencer",
    "qwen_model_dir",
    "resolved_config_path",
]
