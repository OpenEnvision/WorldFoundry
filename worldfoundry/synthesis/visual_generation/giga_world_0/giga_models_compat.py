"""Selective importer for the GigaWorld-0 subset of ``giga_models``.

The upstream package imports every vision and VLA surface from its top-level
``__init__``.  That makes unrelated optional dependencies (for example
``calflops``) mandatory before the diffusion pipeline can be imported.  This
module builds the normal package hierarchy without executing those broad
initializers, then imports only the modules used by GigaWorld-0.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType
from typing import Iterator


@dataclass(frozen=True)
class GigaWorldComponents:
    """Classes needed to construct the upstream GigaWorld-0 pipeline."""

    pipeline_class: type
    scheduler_class: type
    transformer_class: type


def _package_module(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def giga_models_root() -> Path:
    """Resolve the installed package without importing its broad initializer."""

    try:
        root = Path(distribution("giga-models").locate_file("giga_models"))
    except PackageNotFoundError as exc:
        raise ImportError(
            "GigaWorld-0 requires the pinned open-gigaai/giga-models package"
        ) from exc
    if not (root / "models" / "diffusion" / "giga_world_0").is_dir():
        raise ImportError("the installed giga-models package has no GigaWorld-0 runtime")
    return root


def _offline_download(*_args, **_kwargs):
    raise RuntimeError(
        "WorldFoundry GigaWorld-0 is local-only; stage checkpoints under "
        "${WORLDFOUNDRY_CKPT_DIR} instead of downloading during inference"
    )


@contextmanager
def selective_giga_world_imports() -> Iterator[GigaWorldComponents]:
    """Import the diffusion runtime while isolating upstream optional extras."""

    root = giga_models_root()
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "giga_models" or name.startswith("giga_models.")
    }
    for name in tuple(previous):
        sys.modules.pop(name, None)

    try:
        _package_module("giga_models", root)
        _package_module("giga_models.models", root / "models")
        _package_module("giga_models.models.diffusion", root / "models" / "diffusion")
        _package_module("giga_models.pipelines", root / "pipelines")
        _package_module("giga_models.pipelines.diffusion", root / "pipelines" / "diffusion")
        _package_module("giga_models.exports", root / "exports")
        utils = _package_module("giga_models.utils", root / "utils")

        import_module("giga_models.acceleration")
        import_module("giga_models.exports.transformer_engine")
        schedulers = import_module("giga_models.schedulers")
        utils_impl = import_module("giga_models.utils.utils")
        utils.load_state_dict = utils_impl.load_state_dict  # type: ignore[attr-defined]
        utils.download_from_huggingface = _offline_download  # type: ignore[attr-defined]

        models = import_module("giga_models.models.diffusion.giga_world_0")
        pipelines = import_module("giga_models.pipelines.diffusion.giga_world_0")
        yield GigaWorldComponents(
            pipeline_class=pipelines.GigaWorld0Pipeline,
            scheduler_class=schedulers.EDMRESMultistepScheduler,
            transformer_class=models.GigaWorld0Transformer3DModel,
        )
    finally:
        for name in tuple(sys.modules):
            if name == "giga_models" or name.startswith("giga_models."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


__all__ = [
    "GigaWorldComponents",
    "giga_models_root",
    "selective_giga_world_imports",
]
