"""Canonical checkpoint descriptions and native component loading.

This package is the only supported way a diffusion recipe materializes weights
for a ``Denoiser`` / encoder / VAE.  A recipe declares a frozen
:class:`CheckpointSpec`; :class:`NativeCheckpointResolver` turns it into local
paths; :class:`NativeModuleLoader` applies :class:`~..optimizations.policy.RuntimePolicy`
and returns a ``torch.nn.Module``.  The runner never downloads or remaps
checkpoints itself — it only consumes the already-built protocol objects.

Wan helpers (``WanInferenceComponents`` and friends) are lazy-imported so a
generic ``from ...loaders import CheckpointSpec`` does not pull optional
transformer dependencies.
"""

from importlib import import_module

from .checkpoints import CheckpointSpec
from .materialize import MaterializedCheckpoint, NativeCheckpointResolver
from .metadata import checkpoint_json_config, safetensors_json_metadata
from .module import CheckpointConfigResolver, ModuleLoadSpec, NativeModuleLoader

_WAN_EXPORTS = {
    "WanConditioningComponents",
    "WanInferenceComponents",
    "load_wan_conditioning_components",
    "load_wan_inference_components",
    "load_wan_transformer_checkpoint",
    "load_wan_vae_checkpoint",
}


def __getattr__(name):
    if name not in _WAN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.wan_components"), name)
    globals()[name] = value
    return value

__all__ = [
    "CheckpointSpec",
    "CheckpointConfigResolver",
    "MaterializedCheckpoint",
    "ModuleLoadSpec",
    "NativeCheckpointResolver",
    "NativeModuleLoader",
    "checkpoint_json_config",
    "safetensors_json_metadata",
    "WanInferenceComponents",
    "WanConditioningComponents",
    "load_wan_conditioning_components",
    "load_wan_inference_components",
    "load_wan_transformer_checkpoint",
    "load_wan_vae_checkpoint",
]
