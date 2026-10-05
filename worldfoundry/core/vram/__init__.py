"""VRAM orchestration: wrapped-layer offload, balanced device maps, and delayed disk materialization.

The three strategies are split by residency cost instead of sharing one swap loop:

- ``layers`` / ``memory``: ``AutoWrapped*`` modules move onto the GPU when used
  and can unload afterward. ``enable_vram_management`` wraps recursively — good
  for single-GPU inference when peak VRAM is tight but layers can run serially.
  ``DynamicSwapInstaller`` hooks forward to swap whole modules or layers.
- ``device_map``: multi-GPU *static* placement by layer parameter count
  (``balanced_layer_devices``). This is not runtime ping-pong. It is orthogonal
  to TP/CP: a device map shards modules, parallel groups shard tensor dims.
- ``disk_map``: keep weights on disk / mmap and materialize on the first
  forward. Use when even CPU RAM cannot hold the checkpoint, or when opening
  safetensors should be deferred.
- ``layerwise_offload``: per-layer CPU-offload handles and a mutation scope so
  training / LoRA weight writes do not race with swap-in/out.
- ``initialization``: ``init_weights_on_device`` / ``skip_model_initialization``
  skip real init on meta/CPU during construction to cut load-time peaks.

Symbols load lazily. Callers should choose a parallel plan (TP/CP or not) before
picking a device map or offload — stacking both often copies weights twice.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy export table — importing this package must not create a CUDA context
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "AutoTorchModule": "worldfoundry.core.vram.layers",
    "AutoWrappedLinear": "worldfoundry.core.vram.layers",
    "AutoWrappedModule": "worldfoundry.core.vram.layers",
    "AutoWrappedNonRecurseModule": "worldfoundry.core.vram.layers",
    "DeviceMapHandle": "worldfoundry.core.vram.device_map",
    "DiskMap": "worldfoundry.core.vram.disk_map",
    "LayerwiseOffloadHandle": "worldfoundry.core.vram.layerwise_offload",
    "balanced_layer_devices": "worldfoundry.core.vram.device_map",
    "enable_balanced_device_map": "worldfoundry.core.vram.device_map",
    "DynamicSwapInstaller": "worldfoundry.core.vram.memory",
    "WanAutoCastLayerNorm": "worldfoundry.core.vram.layers",
    "cpu": "worldfoundry.core.vram.memory",
    "enable_layerwise_cpu_offload": "worldfoundry.core.vram.layerwise_offload",
    "enable_vram_management": "worldfoundry.core.vram.layers",
    "enable_vram_management_recursively": "worldfoundry.core.vram.layers",
    "fake_diffusers_current_device": "worldfoundry.core.vram.memory",
    "fill_vram_config": "worldfoundry.core.vram.layers",
    "get_cuda_free_memory_gb": "worldfoundry.core.vram.memory",
    "gpu": "worldfoundry.core.vram.memory",
    "visible_cuda_devices": "worldfoundry.core.vram.device_map",
    "gpu_complete_modules": "worldfoundry.core.vram.memory",
    "init_weights_on_device": "worldfoundry.core.vram.initialization",
    "layerwise_offload_mutation_scope": "worldfoundry.core.vram.layerwise_offload",
    "load_model_as_complete": "worldfoundry.core.vram.memory",
    "log_gpu_memory": "worldfoundry.core.vram.memory",
    "move_model_to_device_with_memory_preservation": "worldfoundry.core.vram.memory",
    "move_direct_tensors_to_device": "worldfoundry.core.vram.layers",
    "offload_model_from_device_for_memory_preservation": "worldfoundry.core.vram.memory",
    "patched_diffusers_current_device": "worldfoundry.core.vram.memory",
    "skip_model_initialization": "worldfoundry.core.vram.initialization",
    "unload_complete_models": "worldfoundry.core.vram.memory",
}


# ──────────────────────────────────────────────────────────────────────────
# Attribute resolve — cache on first success so later access skips import
# ──────────────────────────────────────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Import the owning submodule once and cache the symbol on this module.

    Failure: :exc:`AttributeError` when ``name`` is not in ``_EXPORT_MODULES``.
    Successful lookups write into ``globals()`` so later access skips
    ``import_module``.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose already-materialized globals plus every lazy ``__all__`` name."""
    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
