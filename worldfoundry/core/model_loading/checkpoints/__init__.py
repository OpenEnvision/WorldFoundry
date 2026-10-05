"""Checkpoint loading and state-dict remapping helpers.

Lazy exports cover single-file, sharded safetensors, and DCP paths. DCP
loads do not fail on missing keys unless the caller sets ``check_success``
or uses :func:`dcp_load_state_dict` shape validation. Shared cache writes
are rank-0 only and atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy facade — keep DCP / safetensors / torch off the import path
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "DefaultLoadPlanner": "worldfoundry.core.model_loading.checkpoints.dcp",
    "DistributedCheckpointer": "worldfoundry.core.model_loading.checkpoints.dcp",
    "ModelWrapper": "worldfoundry.core.model_loading.checkpoints.dcp",
    "assign_state_dict_strict": "worldfoundry.core.model_loading.checkpoints.assignment",
    "get_storage_reader": "worldfoundry.core.model_loading.checkpoints.load",
    "load_checkpoint": "worldfoundry.core.model_loading.checkpoints.load",
    "load_distributed_checkpoint": "worldfoundry.core.model_loading.checkpoints.load",
    "load_safetensors_into_model_streaming": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "load_sharded_safetensors_parallel_with_progress": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "load_single_checkpoint": "worldfoundry.core.model_loading.checkpoints.load",
    "load_tensor_state_dict": "worldfoundry.core.model_loading.checkpoints.safe_loading",
    "load_weights_only": "worldfoundry.core.model_loading.checkpoints.safe_loading",
    "remap_checkpoint_keys": "worldfoundry.core.model_loading.checkpoints.remap",
    "require_mapping": "worldfoundry.core.model_loading.checkpoints.safe_loading",
    "require_tensor": "worldfoundry.core.model_loading.checkpoints.safe_loading",
    "safetensor_checkpoint_files": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "select_profile_checkpoint": "worldfoundry.core.model_loading.checkpoints.selection",
    "selected_checkpoint_options": "worldfoundry.core.model_loading.checkpoints.selection",
    "submodule_state_dict": "worldfoundry.core.model_loading.checkpoints.remap",
    "unwrap_model": "worldfoundry.core.model_loading.checkpoints.sharded_safetensors",
    "tensor_state_dict": "worldfoundry.core.model_loading.checkpoints.safe_loading",
    "validate_state_dict_compatibility": "worldfoundry.core.model_loading.checkpoints.assignment",
    "dcp_load_state_dict": "worldfoundry.core.model_loading.checkpoints.dcp",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access and cache it."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports to completion tools without importing backends."""

    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
