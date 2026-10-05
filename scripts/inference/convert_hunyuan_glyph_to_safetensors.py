#!/usr/bin/env python3
"""Safely convert the Glyph-SDXL ByT5 tensor state dict to safetensors."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from safetensors.torch import save_file


def _torch_version() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse Torch version: {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Glyph-SDXL byt5_model.pt with weights-only Torch >=2.6 loading."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if _torch_version() < (2, 6):
        raise RuntimeError(
            "Glyph pickle conversion requires Torch >=2.6 because older weights_only loading "
            "is affected by CVE-2025-32434"
        )

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source.suffix not in {".pt", ".pth", ".bin"} or not source.is_file():
        raise FileNotFoundError(f"missing Glyph pickle checkpoint: {source}")
    if output.suffix != ".safetensors":
        raise ValueError(f"output must use the .safetensors suffix: {output}")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"expected a non-empty tensor state dict, got {type(payload)!r}")
    invalid = [key for key, value in payload.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise TypeError(f"Glyph checkpoint contains non-tensor state entries: {invalid[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    save_file(dict(payload), temporary)
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
