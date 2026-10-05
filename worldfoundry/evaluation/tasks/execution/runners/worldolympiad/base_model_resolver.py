"""Resolve the WorldOlympiad base models from ``worldfoundry.base_models``.

The vendored WorldOlympiad runtime under ``runtime/worldolympiad`` is
byte-identical to upstream and is adapted *only* through CLI arguments and
environment variables. It needs
four base models — QwenVL, Depth Anything 3, SAM3, and CLIP — which the runner
previously resolved ad-hoc from a caller-supplied ``--weights-dir`` and an
external ``--da3-src`` checkout.

This module resolves those models from the in-tree
``worldfoundry.base_models`` capability registry instead, so an operator no
longer needs a separate weights directory or an external Depth-Anything-3
source tree when the registered assets are staged. Every resolver returns
``None`` when its asset is not staged, so the runner falls back to the
existing ``--weights-dir`` / ``--da3-src`` flags unchanged.

Reused surfaces:

* ``worldfoundry.base_models.capabilities.BASE_MODEL_CAPABILITIES`` — registry.
* ``BaseModelAsset.check(env)`` — returns ``{"ready": bool, "matched_path": str|None}``,
  the first ready path among env vars, ``local_path`` and ``alternate_paths``.
* ``BaseModelCapability.owner_path()`` — absolute path of the vendored code package.
* ``expand_capability_path(value, env)`` — expands ``${WORLDFOUNDRY_*}`` tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.base_models.capabilities import (  # noqa: E402
    BASE_MODEL_CAPABILITIES,
    BaseModelCapability,
    expand_capability_path,
)

# Capability ids the WorldOlympiad runtime reuses.
QWEN3_VL_CAPABILITY = "qwen3_vl_8b_instruct"
DA3_CAPABILITY = "depth_anything_v3"
SAM3_CAPABILITY = "sam3"

# In-tree code paths not represented by a capability entry. CLIP code lives
# under the perception tree as the OpenAI-CLIP package, importable as ``clip``.
CLIP_CODE_REL = "worldfoundry/base_models/perception_core/general_perception/openai_clip_runtime"

# The vendored runtime imports ``depth_anything_3``; base_models vendors the
# same package under the name ``depth_anything_v3``. A symlink bridges the name
# so no external Depth-Anything-3 checkout is required.
DA3_IMPORT_NAME = "depth_anything_3"
DA3_PACKAGE_DIR_NAME = "depth_anything_v3"
SHIM_SUBDIR = "base_model_shims"


@dataclass(frozen=True)
class ResolvedBaseModels:
    """Where each WorldOlympiad base model was resolved from.

    ``None`` means the asset is not staged and the runner must fall back to its
    explicit ``--weights-dir`` / ``--da3-src`` flags.
    """

    qwenvl_dir: Path | None
    da3_weights_dir: Path | None
    da3_code_shim_dir: Path | None
    sam3_path: Path | None
    clip_code_dir: Path | None
    clip_cache_dir: Path

    def pythonpath_entries(self) -> list[Path]:
        """PYTHONPATH entries that make ``import clip`` / ``import depth_anything_3`` resolve into base_models."""
        entries: list[Path] = []
        if self.clip_code_dir is not None:
            entries.append(self.clip_code_dir)
        if self.da3_code_shim_dir is not None:
            entries.append(self.da3_code_shim_dir)
        return entries

    def env_overrides(self) -> dict[str, str]:
        """Env vars to export for the runtime subprocess when resolved from base_models."""
        overrides: dict[str, str] = {}
        if self.qwenvl_dir is not None:
            overrides["QWENVL_MODEL_PATH"] = str(self.qwenvl_dir)
        if self.sam3_path is not None:
            overrides["SAM3_MODEL"] = str(self.sam3_path)
        if self.clip_cache_dir is not None:
            overrides["CLIP_DOWNLOAD_ROOT"] = str(self.clip_cache_dir)
        return overrides

    def as_provenance(self) -> dict[str, Any]:
        return {
            "qwenvl_dir": _or_none(self.qwenvl_dir),
            "da3_weights_dir": _or_none(self.da3_weights_dir),
            "da3_code_shim_dir": _or_none(self.da3_code_shim_dir),
            "sam3_path": _or_none(self.sam3_path),
            "clip_code_dir": _or_none(self.clip_code_dir),
            "clip_cache_dir": str(self.clip_cache_dir),
        }


def _or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _resolve_asset(capability_id: str, env: Mapping[str, str] | None = None) -> Path | None:
    """Return the first ready asset path for a capability, or ``None`` when unstaged."""
    capability: BaseModelCapability | None = BASE_MODEL_CAPABILITIES.get(capability_id)
    if capability is None:
        return None
    for asset in capability.assets:
        status = asset.check(env)
        if status.get("ready") and status.get("matched_path"):
            return Path(status["matched_path"])
    return None


def resolve_qwenvl_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Resolve the Qwen3-VL-8B-Instruct judge model directory."""
    return _resolve_asset(QWEN3_VL_CAPABILITY, env)


def resolve_da3_weights_dir(env: Mapping[str, str] | None = None) -> Path | None:
    """Resolve the Depth Anything 3 model directory (``config.json`` + ``model.safetensors``)."""
    return _resolve_asset(DA3_CAPABILITY, env)


def resolve_sam3_path(env: Mapping[str, str] | None = None) -> Path | None:
    """Resolve the SAM3 checkpoint file.

    Best-effort: base_models ships the native SAM3 checkpoint, while the vendored
    runtime loads SAM3 via the ``ultralytics`` package. If the resolved ``.pt``
    is not ultralytics-loadable in a given environment, the operator overrides
    with ``--sam3-model`` / ``WORLDFOUNDRY_SAM3_CKPT``.
    """
    return _resolve_asset(SAM3_CAPABILITY, env)


def da3_code_dir() -> Path:
    """Absolute path of the vendored ``depth_anything_v3`` code package in base_models."""
    return BASE_MODEL_CAPABILITIES[DA3_CAPABILITY].owner_path()


def resolve_clip_code_dir() -> Path | None:
    """Return the base_models ``openai_clip_runtime`` dir so ``import clip`` resolves there."""
    path = expand_capability_path(CLIP_CODE_REL)
    return path if path.is_dir() else None


def resolve_clip_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve a managed CLIP weight cache for OpenAI-CLIP ``clip.load(download_root=...)``.

    OpenAI CLIP's ``load("ViT-B/32", download_root=...)`` fetches ``ViT-B-32.pt``
    into this directory on first use; it is not the HF ``clip-vit-base-patch32``
    dir registered under the ``clip_vit_b32`` capability, so weights are not
    shared across the two formats — only the CLIP *code* is reused from base_models.
    """
    environ = os.environ if env is None else dict(env)
    explicit = environ.get("CLIP_DOWNLOAD_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    return expand_capability_path("${WORLDFOUNDRY_CKPT_DIR}/clip", environ)


def build_da3_code_shim(shim_parent: Path) -> Path | None:
    """Symlink ``depth_anything_3 -> <base_models>/depth_anything_v3`` under ``shim_parent``.

    The vendored runtime imports ``depth_anything_3``; base_models vendors the
    same package as ``depth_anything_v3``. The symlink bridges the name so the
    runtime's ``from depth_anything_3.api import DepthAnything3`` (and every
    submodule it imports — ``specs``, ``utils.geometry``, ``utils.pose_align``,
    ``model.utils.gs_renderer``) resolves into base_models with no external
    checkout. Returns the shim directory (to put on PYTHONPATH) or ``None`` if
    the base_models code package is missing.
    """
    target = da3_code_dir()
    if not target.is_dir():
        return None
    shim_dir = Path(shim_parent).expanduser() / SHIM_SUBDIR
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / DA3_IMPORT_NAME
    if link.is_symlink() or link.exists():
        try:
            link.unlink()
        except OSError:
            return None
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return None
    return shim_dir


def resolve_all(
    *,
    shim_parent: Path,
    env: Mapping[str, str] | None = None,
) -> ResolvedBaseModels:
    """Resolve every WorldOlympiad base model from ``worldfoundry.base_models``.

    The DA3 code shim is built whenever the base_models code package exists —
    independent of whether DA3 weights are staged — so ``--dry-run`` and import
    checks work without weights.
    """
    return ResolvedBaseModels(
        qwenvl_dir=resolve_qwenvl_dir(env),
        da3_weights_dir=resolve_da3_weights_dir(env),
        da3_code_shim_dir=build_da3_code_shim(shim_parent),
        sam3_path=resolve_sam3_path(env),
        clip_code_dir=resolve_clip_code_dir(),
        clip_cache_dir=resolve_clip_cache_dir(env),
    )


__all__ = [
    "ResolvedBaseModels",
    "QWEN3_VL_CAPABILITY",
    "DA3_CAPABILITY",
    "SAM3_CAPABILITY",
    "DA3_IMPORT_NAME",
    "DA3_PACKAGE_DIR_NAME",
    "SHIM_SUBDIR",
    "resolve_qwenvl_dir",
    "resolve_da3_weights_dir",
    "resolve_sam3_path",
    "da3_code_dir",
    "resolve_clip_code_dir",
    "resolve_clip_cache_dir",
    "build_da3_code_shim",
    "resolve_all",
]
