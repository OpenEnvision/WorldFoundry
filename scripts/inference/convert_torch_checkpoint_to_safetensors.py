#!/usr/bin/env python3
"""Safely convert a tensor-only PyTorch checkpoint to safetensors."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file


def _torch_version() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse Torch version: {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def _unwrap_state_dict(value: object) -> Mapping[str, torch.Tensor]:
    while isinstance(value, Mapping) and len(value) == 1:
        key = next(iter(value))
        if key not in {"state_dict", "module", "model", "model_state"}:
            break
        value = value[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint did not contain a state dict: {type(value).__name__}")
    invalid = sorted(str(name) for name, tensor in value.items() if not isinstance(tensor, torch.Tensor))
    if invalid:
        raise TypeError(f"state dict contains non-tensor entries: {invalid[:20]}")
    return value  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    incomplete = output.with_name(output.name + ".incomplete")

    if _torch_version() < (2, 6):
        raise RuntimeError(
            "pickle checkpoint conversion requires Torch >=2.6; older weights_only loading "
            "is affected by CVE-2025-32434"
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if incomplete.exists():
        raise FileExistsError(incomplete)

    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state_dict = _unwrap_state_dict(checkpoint)
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(state_dict), incomplete, metadata={"format": "pt", "source": source.name})
    os.replace(incomplete, output)
    print(f"{output}\t{output.stat().st_size}\t{len(state_dict)} tensors")


if __name__ == "__main__":
    main()
