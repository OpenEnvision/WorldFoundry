"""WRBench: out-of-the-box unified camera control for video-generation models.

Quickstart::

    import wrbench
    from wrbench.datasets import natural25_first_frame_path

    # Preset combination
    wrbench.compile_camera(model="wan22-fun-5b-cam", camera="yaw:left:60@40,yaw:right:60@41",
                            image=natural25_first_frame_path("bedroom_cat_bed_jump"), out="out.mp4")

    # Arbitrary angle, near-per-frame
    script = wrbench.presets.sweep("yaw", "left", 37, frames=49)
    wrbench.compile_camera(model="wan22-fun-5b-cam", camera=script, image=natural25_first_frame_path("bedroom_cat_bed_jump"), out="out.mp4")

With ``dry_run=True`` this compiles the model-native payload and writes auditable
sidecars without running any heavy model. Real generation is delegated to the
canonical WorldFoundry model catalog and in-tree pipeline runner.
"""

from __future__ import annotations

from wrbench import presets
from wrbench.actions import CameraScript, FrameAction, parse_camera_script
from wrbench.builder import build_camera_trajectory
from wrbench.registry import (
    active_model_keys,
    all_records,
    canonical_model_key,
    deferred_model_keys,
    model_record,
)
from wrbench.runner import compile_camera

__version__ = "0.1.0"


def list_models(include_deferred: bool = False) -> list[str]:
    """Canonical keys of supported models."""
    keys = active_model_keys()
    if include_deferred:
        keys = keys + deferred_model_keys()
    return keys


__all__ = [
    "__version__",
    "CameraScript",
    "FrameAction",
    "parse_camera_script",
    "build_camera_trajectory",
    "compile_camera",
    "presets",
    "list_models",
    "canonical_model_key",
    "model_record",
    "all_records",
    "active_model_keys",
    "deferred_model_keys",
]
