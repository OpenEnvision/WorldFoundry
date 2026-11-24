"""Structured, Python and declarative model configuration.

lazy_config owns deferred object graphs; cosmos_config and hydra support attrs
runtime configurations; model_config defines architecture values; factory builds
objects from ``target`` / ``class_path`` mappings. Python files, override
formatting, field validation and legacy Hydra helpers live here too.
Checkpoint placement ModelConfig is owned by core.model_loading.config.
"""

from .flags import FLAGS, INTERNAL, VALIDATION, VERBOSE
from .model_config import (
    ArchConfig,
    DiffusionModelConfig,
    DiTArchConfig,
    DiTConfig,
    ModelConfig,
    build_kwargs_from_config,
    require_config_value,
)


def __getattr__(name: str):
    # Shared dataclass helpers must work without Cosmos / Hydra dependencies.
    if name in {"CheckpointConfig", "Config", "EMAConfig", "ObjectStoreConfig", "make_freezable"}:
        from . import cosmos_config as module
    elif name in {"LazyCall", "LazyConfig", "LazyDict", "instantiate"}:
        from . import lazy_config as module
    elif name in {"get_obj_from_str", "instantiate_from_config", "resolve_symbol"}:
        from . import factory as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "FLAGS",
    "INTERNAL",
    "VALIDATION",
    "VERBOSE",
    "ArchConfig",
    "CheckpointConfig",
    "Config",
    "DiTArchConfig",
    "DiTConfig",
    "DiffusionModelConfig",
    "EMAConfig",
    "LazyCall",
    "LazyConfig",
    "LazyDict",
    "ModelConfig",
    "ObjectStoreConfig",
    "build_kwargs_from_config",
    "get_obj_from_str",
    "instantiate",
    "instantiate_from_config",
    "make_freezable",
    "require_config_value",
    "resolve_symbol",
]
