"""Opt-in inference approximations and acceleration reporting.

- cache: cross-step residual/prediction reuse.
- token_pruning: token selection and reconstruction.
- quantization: low-precision Linear execution and FP8/NVFP4 packing.
- technology: inventory of active acceleration families.
- mha_schedule: projection fusion and precision schedules for resident attention.
- profiles / sol_engine: pure-Python acceleration profiles and 5-pillar stack integrator.

CUDA graph dispatch, frame prefetch, overlap and prewarm are owned by
core.execution. One-shot encoder placement is owned by core.vram.

Re-exports are resolved lazily via PEP 562 ``__getattr__`` so importing
pure-Python control-plane submodules (``profiles``, ``sol_engine``) never
triggers eager ``torch`` imports or CUDA probing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "AdaCacheResidualCache": "worldfoundry.core.acceleration.cache",
    "AdaptiveResidualCache": "worldfoundry.core.acceleration.cache",
    "BlockTaylorSeerCache": "worldfoundry.core.acceleration.cache",
    "CustomTaylorResidualCache": "worldfoundry.core.acceleration.cache",
    "FixedStepCache": "worldfoundry.core.acceleration.cache",
    "MagCacheResidualCache": "worldfoundry.core.acceleration.cache",
    "TaylorSeerResidualCache": "worldfoundry.core.acceleration.cache",
    "TeaCacheResidualCache": "worldfoundry.core.acceleration.cache",
    "Float8Linear": "worldfoundry.core.acceleration.quantization.linear",
    "WeightOnlyLinear": "worldfoundry.core.acceleration.quantization.linear",
    "quantization_runtime_report": "worldfoundry.core.acceleration.quantization.linear",
    "replace_linear_with_float8": "worldfoundry.core.acceleration.quantization.linear",
    "replace_linear_with_weight_only": "worldfoundry.core.acceleration.quantization.linear",
    "reset_quantization_runtime_window": "worldfoundry.core.acceleration.quantization.linear",
    "set_low_precision_enabled": "worldfoundry.core.acceleration.quantization.linear",
    "NVFP4Linear": "worldfoundry.core.acceleration.quantization.nvfp4",
    "dequantize_nvfp4": "worldfoundry.core.acceleration.quantization.nvfp4",
    "quantize_nvfp4": "worldfoundry.core.acceleration.quantization.nvfp4",
    "replace_linear_with_nvfp4": "worldfoundry.core.acceleration.quantization.nvfp4",
    "AccelerationTechnology": "worldfoundry.core.acceleration.technology",
    "acceleration_technology_report": "worldfoundry.core.acceleration.technology",
    "TokenPruner": "worldfoundry.core.acceleration.token_pruning",
    "TokenPruneState": "worldfoundry.core.acceleration.token_pruning",
    "prune_tokens": "worldfoundry.core.acceleration.token_pruning",
    "restore_tokens": "worldfoundry.core.acceleration.token_pruning",
    "select_token_indices": "worldfoundry.core.acceleration.token_pruning",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
