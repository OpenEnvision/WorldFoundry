"""Inference-only adapters and SOL-Engine acceleration stack for official LTX-2.5 pipelines."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .acceleration import (
        LTX25AcceleratedExecutor,
        LTX25AccelerationConfig,
        LTX25CFGParallelRunner,
        LTX25FirstBlockCache,
        LTX25FirstBlockCacheConfig,
        LTX25ParallelConfig,
        LTX25UlyssesAttentionAdapter,
        accelerate_ltx25_transformer,
        accelerate_ltx25_video_vae,
        coerce_ltx25_acceleration_config,
        convert_ltx25_conv_video_decoder_state_dict,
        convert_ltx25_conv_video_encoder_state_dict,
        gather_ltx25_video_hidden,
        ltx25_sol_profile,
        shard_ltx25_video_args,
    )
    from .runtime import (
        CONV_VIDEO_VAE_RELPATH,
        DEV_TRANSFORMER_RELPATH,
        DIFF_VIDEO_VAE_RELPATH,
        LTX25_STAGE_1_SIGMAS,
        LTX25_STAGE_2_SIGMAS,
        LTX25DistilledRuntime,
        LTX25RuntimePlan,
        resample_video_for_ltx25,
    )

_ACCELERATION_EXPORTS = (
    "LTX25AcceleratedExecutor",
    "LTX25AccelerationConfig",
    "LTX25CFGParallelRunner",
    "LTX25FirstBlockCache",
    "LTX25FirstBlockCacheConfig",
    "LTX25ParallelConfig",
    "LTX25UlyssesAttentionAdapter",
    "accelerate_ltx25_transformer",
    "accelerate_ltx25_video_vae",
    "coerce_ltx25_acceleration_config",
    "convert_ltx25_conv_video_decoder_state_dict",
    "convert_ltx25_conv_video_encoder_state_dict",
    "gather_ltx25_video_hidden",
    "ltx25_sol_profile",
    "shard_ltx25_video_args",
)

_RUNTIME_EXPORTS = (
    "CONV_VIDEO_VAE_RELPATH",
    "DEV_TRANSFORMER_RELPATH",
    "DIFF_VIDEO_VAE_RELPATH",
    "LTX25_STAGE_1_SIGMAS",
    "LTX25_STAGE_2_SIGMAS",
    "LTX25DistilledRuntime",
    "LTX25RuntimePlan",
    "resample_video_for_ltx25",
)

_LAZY_EXPORTS: dict[str, str] = {
    **dict.fromkeys(_ACCELERATION_EXPORTS, f"{__name__}.acceleration"),
    **dict.fromkeys(_RUNTIME_EXPORTS, f"{__name__}.runtime"),
}

__all__ = [
    "CONV_VIDEO_VAE_RELPATH",
    "DEV_TRANSFORMER_RELPATH",
    "DIFF_VIDEO_VAE_RELPATH",
    "LTX25_STAGE_1_SIGMAS",
    "LTX25_STAGE_2_SIGMAS",
    "LTX25AcceleratedExecutor",
    "LTX25AccelerationConfig",
    "LTX25CFGParallelRunner",
    "LTX25DistilledRuntime",
    "LTX25FirstBlockCache",
    "LTX25FirstBlockCacheConfig",
    "LTX25ParallelConfig",
    "LTX25RuntimePlan",
    "LTX25UlyssesAttentionAdapter",
    "accelerate_ltx25_transformer",
    "accelerate_ltx25_video_vae",
    "coerce_ltx25_acceleration_config",
    "convert_ltx25_conv_video_decoder_state_dict",
    "convert_ltx25_conv_video_encoder_state_dict",
    "gather_ltx25_video_hidden",
    "ltx25_sol_profile",
    "resample_video_for_ltx25",
    "shard_ltx25_video_args",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
