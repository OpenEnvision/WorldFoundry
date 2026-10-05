# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from safetensors.torch import load_file

from worldfoundry.core.io.filesystem.file_utils import file_sha256 as sha256_file

from .config import RepEncoderConfig


CHECKPOINT_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "manifest.json"


def resolve_checkpoint(path: str | Path) -> tuple[Path, Path, Path]:
    source = Path(path).expanduser().resolve()
    root = source if source.is_dir() else source.parent
    model_path = root / CHECKPOINT_FILENAME if source.is_dir() else source
    config_path = root / CONFIG_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    for required in (model_path, config_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"RepEncoder checkpoint artifact is missing: {required}")
    return model_path, config_path, manifest_path


def read_checkpoint_metadata(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, RepEncoderConfig, dict[str, Any], str]:
    model_path, config_path, manifest_path = resolve_checkpoint(path)
    config_payload = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping) or manifest.get("format") != "worldcrafter_repencoder_v1":
        raise ValueError("unsupported RepEncoder manifest")
    actual_sha256 = sha256_file(model_path)
    declared_sha256 = str(manifest.get("model_sha256", ""))
    if declared_sha256 != actual_sha256:
        raise RuntimeError(
            f"RepEncoder manifest SHA256 mismatch: {declared_sha256} != {actual_sha256}"
        )
    if expected_sha256 is not None and str(expected_sha256) != actual_sha256:
        raise RuntimeError(
            f"RepEncoder checkpoint SHA256 mismatch: {expected_sha256} != {actual_sha256}"
        )
    return model_path, RepEncoderConfig.from_dict(config_payload), dict(manifest), actual_sha256


def load_checkpoint_state(
    model: nn.Module,
    model_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> None:
    state = load_file(str(model_path), device=str(torch.device(device)))
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(
            "RepEncoder checkpoint key mismatch: "
            f"missing={sorted(set(expected) - set(state))[:8]}, "
            f"unexpected={sorted(set(state) - set(expected))[:8]}"
        )
    dtype_mismatch = {
        name: (value.dtype, expected[name].dtype)
        for name, value in state.items()
        if value.dtype != expected[name].dtype
    }
    if dtype_mismatch:
        raise TypeError(
            "RepEncoder checkpoint violates mixed dtype ownership: "
            f"{list(dtype_mismatch.items())[:8]}"
        )
    nonfinite = [name for name, value in state.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise FloatingPointError(f"RepEncoder checkpoint contains non-finite tensors: {nonfinite[:8]}")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict RepEncoder checkpoint load failed: {incompatible}")


__all__ = [
    "CHECKPOINT_FILENAME",
    "CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "load_checkpoint_state",
    "read_checkpoint_metadata",
    "resolve_checkpoint",
    "sha256_file",
]
