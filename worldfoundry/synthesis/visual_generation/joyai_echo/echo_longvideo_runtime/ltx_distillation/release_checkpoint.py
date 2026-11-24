"""Validation and resolution for the three public Echo 1.5 checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_NAME = "checkpoint.json"
SCHEMA_VERSION = 1
FULL_DMD = "echo15_full_dmd"
FP8 = "echo15_fp8"
FP4 = "echo15_fp4"
RELEASE_CHECKPOINTS = {
    FULL_DMD: "bf16",
    FP8: "fp8",
    FP4: "fp4",
}


@dataclass(frozen=True)
class ReleaseCheckpoint:
    """Resolved files for one validated public release checkpoint."""

    root: Path
    name: str
    precision: str
    model_path: Path
    modelopt_path: Path | None = None


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid checkpoint manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint manifest must be a JSON object: {path}")
    return payload


def _resolve_member(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"checkpoint manifest field files.{label} must be a relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"checkpoint manifest field files.{label} must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"checkpoint manifest field files.{label} escapes its directory") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {resolved}")
    return resolved


def resolve_release_checkpoint(path: str | Path) -> ReleaseCheckpoint:
    """Resolve one of the three supported checkpoint directory formats.

    Runtime callers provide one directory only. The manifest selects the loader
    and any internal companion file, so precision and weight paths cannot drift.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"release checkpoint directory not found: {root}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}")
    name = manifest.get("name")
    if name not in RELEASE_CHECKPOINTS:
        supported = ", ".join(RELEASE_CHECKPOINTS)
        raise ValueError(f"unsupported release checkpoint {name!r}; choose one of: {supported}")
    expected_precision = RELEASE_CHECKPOINTS[name]
    if manifest.get("precision") != expected_precision:
        raise ValueError(
            f"checkpoint {name} must declare precision={expected_precision!r}, got {manifest.get('precision')!r}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("checkpoint manifest field files must be a JSON object")
    model_field = "components" if name == FP4 else "model"
    model_path = _resolve_member(root, files.get(model_field), label=model_field)
    if model_path.suffix != ".safetensors":
        raise ValueError(f"checkpoint {model_field} must use the safetensors format")

    modelopt_path = None
    if name == FP4:
        if "model" in files:
            raise ValueError("echo15_fp4 must be standalone: use files.components, not a BF16 files.model")
        modelopt_path = _resolve_member(root, files.get("modelopt"), label="modelopt")
        if modelopt_path.suffix != ".pt":
            raise ValueError("echo15_fp4 ModelOpt state must use the .pt format")
    elif "modelopt" in files:
        raise ValueError(f"checkpoint {name} must not contain a ModelOpt state")

    return ReleaseCheckpoint(
        root=root,
        name=name,
        precision=expected_precision,
        model_path=model_path,
        modelopt_path=modelopt_path,
    )
