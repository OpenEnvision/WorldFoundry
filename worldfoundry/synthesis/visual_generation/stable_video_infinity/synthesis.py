"""WorldFoundry synthesis adapter for Stable Video Infinity."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class _StableVideoInfinityRuntime:
    """Delay torch and DiffSynth imports until checkpoint-backed execution."""

    def __new__(cls, *args: Any, **kwargs: Any):
        from .worldfoundry_runtime import StableVideoInfinityRuntime

        signature = inspect.signature(StableVideoInfinityRuntime)
        parameters = signature.parameters
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return StableVideoInfinityRuntime(*args, **kwargs)


class StableVideoInfinitySynthesis(RuntimeVideoSynthesis):
    """Image-to-long-video synthesis backed by the in-tree SVI runtime."""

    MODEL_NAME = "stable-video-infinity"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = _StableVideoInfinityRuntime
    PRIMARY_PATH_KEY = "svi_lora_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/stable_video_infinity/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME

    def _prediction_runtime_overrides(
        self,
        kwargs: Mapping[str, Any],
        *,
        fps: int | None,
    ) -> dict[str, Any]:
        aliases = {
            "steps": "num_inference_steps",
            "num_steps": "num_inference_steps",
            "guidance_scale": "cfg_scale_text",
            "cfg_scale": "cfg_scale_text",
            "seed": "base_seed",
            "shift": "sigma_shift",
            "time_shift": "sigma_shift",
        }
        direct = {
            "num_clips",
            "num_frames",
            "num_motion_frames",
            "num_inference_steps",
            "cfg_scale_text",
            "sigma_shift",
            "ref_pad_cfg",
            "ref_pad_num",
            "prompt_repeat_times",
            "use_first_prompt_only",
            "repeat_first_clip",
            "prompt_prefix",
            "base_seed",
            "seed_stride",
            "max_width",
            "height",
            "width",
            "tiled",
            "tile_size",
            "tile_stride",
            "negative_prompt",
        }
        overrides: dict[str, Any] = {}
        if fps is not None:
            overrides["fps"] = fps
        for key, value in kwargs.items():
            if value is None:
                continue
            canonical = aliases.get(key, key)
            if canonical in direct:
                overrides[canonical] = value
        return overrides


__all__ = ["StableVideoInfinitySynthesis"]
