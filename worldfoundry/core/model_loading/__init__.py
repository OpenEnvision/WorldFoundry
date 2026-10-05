"""Model construction, checkpoint loading, adapters and placement policy.

checkpoints owns format/URI loading, safe tensor dictionaries, key remapping,
streaming shards, compatibility checks and distributed checkpoint assignment.
The remaining modules construct models, merge adapters, configure placement and
apply load-time optimization policy. Public APIs resolve lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy export table — do not import LoRA / file / factory at package import
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "audit_wan_lora_targets": "worldfoundry.core.model_loading.peft",
    "merge_peft_adapter": "worldfoundry.core.model_loading.peft",
    "GeneralLoRALoader": "worldfoundry.core.model_loading.lora",
    "LightX2VLoRALoader": "worldfoundry.core.model_loading.lora",
    "AttentionBackend": "worldfoundry.core.model_loading.policy",
    "InferenceModel": "worldfoundry.core.model_loading.inference_model",
    "LoaderModelConfig": "worldfoundry.core.model_loading.config",
    "ModelConfig": "worldfoundry.core.model_loading.config",
    "OffloadMode": "worldfoundry.core.model_loading.policy",
    "OffloadPolicy": "worldfoundry.core.model_loading.policy",
    "QuantizationMode": "worldfoundry.core.model_loading.policy",
    "QuantizationPolicy": "worldfoundry.core.model_loading.policy",
    "RuntimePolicy": "worldfoundry.core.model_loading.policy",
    "count_parameters": "worldfoundry.core.model_loading.factory",
    "count_params": "worldfoundry.core.model_loading.factory",
    "convert_keys_dict_to_single_str": "worldfoundry.core.model_loading.checkpoints.file",
    "convert_state_dict_keys_to_single_str": "worldfoundry.core.model_loading.checkpoints.file",
    "convert_state_dict_to_keys_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "get_init_context": "worldfoundry.core.model_loading.model",
    "get_obj_from_str": "worldfoundry.core.model_loading.factory",
    "hash_model_file": "worldfoundry.core.model_loading.checkpoints.file",
    "hash_state_dict_keys": "worldfoundry.core.model_loading.checkpoints.file",
    "build_rename_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "load_keys_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "load_model": "worldfoundry.core.model_loading.model",
    "load_model_loader_registry": "worldfoundry.core.model_loading.registry_config",
    "load_model_with_disk_offload": "worldfoundry.core.model_loading.model",
    "merge_ordered_lora_": "worldfoundry.core.model_loading.lora",
    "merge_named_lora_": "worldfoundry.core.model_loading.lora",
    "merge_flattened_path_lora_": "worldfoundry.core.model_loading.lora",
    "merge_rank_scaled_lora_": "worldfoundry.core.model_loading.lora",
    "load_state_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "load_state_dict_non_strict": "worldfoundry.core.model_loading.checkpoints.state_dict",
    "load_state_dict_from_folder": "worldfoundry.core.model_loading.checkpoints.file",
    "load_state_dict_from_gguf": "worldfoundry.core.model_loading.checkpoints.file",
    "load_state_dict_from_safetensors_index": "worldfoundry.core.model_loading.checkpoints.file",
    "load_torch_checkpoint": "worldfoundry.core.model_loading.checkpoints.file",
    "load_torch_state_dict": "worldfoundry.core.model_loading.checkpoints.file",
    "instantiate_from_config": "worldfoundry.core.model_loading.factory",
    "non_strict_load_model": "worldfoundry.core.model_loading.checkpoints.state_dict",
    "search_for_embeddings": "worldfoundry.core.model_loading.checkpoints.file",
    "search_for_files": "worldfoundry.core.model_loading.checkpoints.file",
    "search_parameter": "worldfoundry.core.model_loading.checkpoints.file",
    "resolve_symbol": "worldfoundry.core.model_loading.factory",
    "split_state_dict_with_prefix": "worldfoundry.core.model_loading.checkpoints.file",
    "ModelLoaderRegistry": "worldfoundry.core.model_loading.registry_config",
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
