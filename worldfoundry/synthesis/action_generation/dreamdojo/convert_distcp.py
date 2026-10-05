"""Convert an official DreamDojo distributed checkpoint for inference."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
from typing import Any


def convert_distcp(distcp_dir: Path, output_dir: Path) -> Path:
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    distcp_dir = distcp_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    target = output_dir / "model_ema_bf16.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target
    if not (distcp_dir / ".metadata").is_file():
        raise FileNotFoundError(f"DreamDojo DCP metadata not found: {distcp_dir / '.metadata'}")

    lock_path = output_dir / ".worldfoundry_distcp_conversion.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file():
            return target

        full_checkpoint = output_dir / ".worldfoundry_model.pt"
        temporary_target = output_dir / ".model_ema_bf16.pt.tmp"
        full_checkpoint.unlink(missing_ok=True)
        temporary_target.unlink(missing_ok=True)
        try:
            dcp_to_torch_save(distcp_dir, full_checkpoint)
            state_dict: dict[str, Any] = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
            ema_state: dict[str, Any] = {}
            for key, value in state_dict.items():
                if not key.startswith("net_ema."):
                    continue
                converted_key = key.replace("net_ema.", "net.", 1)
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    value = value.bfloat16()
                ema_state[converted_key] = value
            if not ema_state:
                raise ValueError(f"DreamDojo checkpoint has no net_ema weights: {distcp_dir}")
            torch.save(ema_state, temporary_target)
            os.replace(temporary_target, target)
        finally:
            full_checkpoint.unlink(missing_ok=True)
            temporary_target.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distcp_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(convert_distcp(args.distcp_dir, args.output_dir))


if __name__ == "__main__":
    main()
