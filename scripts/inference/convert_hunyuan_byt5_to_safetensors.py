#!/usr/bin/env python3
"""Convert the official Google ByT5 pickle checkpoint in a Torch >=2.6 process."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration


def _torch_version() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse Torch version: {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely convert google/byt5-small from pytorch_model.bin to model.safetensors. "
            "Run this helper in an isolated Torch >=2.6 environment."
        )
    )
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()

    if _torch_version() < (2, 6):
        raise RuntimeError(
            "ByT5 pickle conversion requires Torch >=2.6 because older weights_only loading "
            "is affected by CVE-2025-32434"
        )
    if not (model_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"missing ByT5 pickle checkpoint: {model_dir / 'pytorch_model.bin'}")

    model = T5ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=False,
    )
    model.save_pretrained(
        model_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    output = model_dir / "model.safetensors"
    if not output.is_file():
        raise RuntimeError(f"ByT5 conversion did not create {output}")
    print(output)


if __name__ == "__main__":
    main()
