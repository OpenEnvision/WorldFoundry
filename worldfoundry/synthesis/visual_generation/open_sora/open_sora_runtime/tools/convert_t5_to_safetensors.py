"""Convert the published DeepFloyd T5 .bin shards into a separate safe cache.

Run from the WorldFoundry repository root, for example:

    python worldfoundry/synthesis/visual_generation/open_sora/open_sora_runtime/tools/convert_t5_to_safetensors.py \
      --source ../ckpts/DeepFloyd--t5-v1_1-xxl \
      --output ../ckpts/DeepFloyd--t5-v1_1-xxl-safetensors

The source directory is never modified. Only convert checkpoints you trust.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be a separate directory outside the source checkpoint")

    index = json.loads((source / "pytorch_model.bin.index.json").read_text())
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Source checkpoint has an empty or invalid weight map")

    shard_names = sorted(set(weight_map.values()))
    if any(not name.startswith("pytorch_model-") or not name.endswith(".bin") for name in shard_names):
        raise ValueError(f"Unexpected source shard names: {shard_names}")
    converted_names = {name: name.replace("pytorch_model-", "model-", 1).replace(".bin", ".safetensors") for name in shard_names}
    output.mkdir(parents=True, exist_ok=True)
    for sidecar in ("config.json", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
        temporary = output / (sidecar + ".part")
        shutil.copy2(source / sidecar, temporary)
        os.replace(temporary, output / sidecar)

    for shard_name in shard_names:
        target = output / converted_names[shard_name]
        if target.exists():
            if target.is_symlink():
                raise ValueError(f"Converted shard must not be a symlink: {target}")
            with safe_open(target, framework="pt", device="cpu") as saved:
                existing_keys = set(saved.keys())
            expected_keys = {key for key, name in weight_map.items() if name == shard_name}
            if existing_keys != expected_keys:
                raise ValueError(f"Existing shard has unexpected keys: {target}")
            print(f"Verified existing {target}", flush=True)
            continue

        weights = torch.load(source / shard_name, map_location="cpu", weights_only=True, mmap=True)
        expected_keys = {key for key, name in weight_map.items() if name == shard_name}
        if not isinstance(weights, dict) or set(weights) != expected_keys:
            raise ValueError(f"Source shard keys disagree with index: {shard_name}")
        if not all(isinstance(value, torch.Tensor) for value in weights.values()):
            raise TypeError(f"Source shard contains non-tensor values: {shard_name}")
        temporary = target.with_name(target.name + ".part")
        try:
            save_file({key: value.detach().contiguous() for key, value in weights.items()}, temporary, metadata={"format": "pt"})
            with safe_open(temporary, framework="pt", device="cpu") as saved:
                if set(saved.keys()) != expected_keys:
                    raise ValueError(f"Converted shard keys disagree with index: {temporary}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Converted {target} ({len(expected_keys)} tensors)", flush=True)
        del weights

    converted_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": {key: converted_names[name] for key, name in weight_map.items()},
    }
    temporary_index = output / "model.safetensors.index.json.part"
    temporary_index.write_text(json.dumps(converted_index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_index, output / "model.safetensors.index.json")
    print(f"Ready: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trusted DeepFloyd .bin checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate derived safetensors directory")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
