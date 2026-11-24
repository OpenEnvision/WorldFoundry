"""Path helpers for the in-tree 4DWorldBench runtime."""

from __future__ import annotations

from pathlib import Path


def keye_model_path() -> str:
    """Resolve the Keye-VL model path or Hugging Face repo id used by judge dimensions."""
    from worldfoundry.base_models.llm_mllm_core.mllm.keye_vl import model_path

    return model_path()


def droid_checkpoint_path() -> str:
    """Resolve the DROID-SLAM checkpoint from WorldFoundry assets or explicit env."""
    import os

    explicit = os.environ.get("WORLDFOUNDRY_4DWORLDBENCH_DROID_CKPT") or os.environ.get("WORLDFOUNDRY_DROID_SLAM_CKPT")
    if explicit:
        return explicit
    try:
        from worldfoundry.base_models.three_dimensions.slam.droid_slam import checkpoint_path

        return str(checkpoint_path())
    except Exception as exc:
        ckpt_root = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
        if ckpt_root:
            return str(Path(ckpt_root) / "droid.pth")
        raise RuntimeError(
            "DROID-SLAM checkpoint is required; set WORLDFOUNDRY_4DWORLDBENCH_DROID_CKPT "
            "or register the droid_slam base-model asset."
        ) from exc


__all__ = ["droid_checkpoint_path", "keye_model_path"]
