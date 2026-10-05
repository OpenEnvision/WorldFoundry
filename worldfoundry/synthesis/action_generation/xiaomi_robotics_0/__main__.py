"""Command-line inference entrypoint for the in-tree Xiaomi-Robotics-0 runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from worldfoundry.core.io.formats.serialization import jsonable, read_json

from .runtime import XiaomiRobotics0RuntimeConfig, runtime_for


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worldfoundry.synthesis.action_generation.xiaomi_robotics_0",
        description="Run Xiaomi-Robotics-0 action-chunk inference using only the in-tree model code.",
    )
    parser.add_argument("--variant", default="libero", help="Official variant (default: libero).")
    parser.add_argument("--checkpoint", help="Local checkpoint directory or Hugging Face repo ID.")
    parser.add_argument("--revision", help="Checkpoint revision; official variants default to pinned SHAs.")
    parser.add_argument("--robot-type", help="Processor action-statistics key; inferred for fine-tuned variants.")
    parser.add_argument("--camera-keys", help="Comma-separated observation image keys; required for pretrain.")
    parser.add_argument("--view-labels", help="Comma-separated prompt view labels; required for pretrain.")
    parser.add_argument("--instruction", default="", help="Natural-language robot instruction.")
    parser.add_argument("--base-image", help="Base/ego image path.")
    parser.add_argument("--wrist-image", help="Left-wrist image path for two-view variants.")
    parser.add_argument("--state-json", help="JSON list or path to a JSON state-vector file.")
    parser.add_argument("--output", default="xiaomi_robotics_0_action_trace.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the resolved checkpoint/runtime contract without importing model dependencies.",
    )
    return parser


def _state(value: str | None) -> Any:
    if not value:
        raise ValueError("--state-json is required for inference")
    path = Path(value).expanduser()
    return read_json(path) if path.is_file() else json.loads(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = {
        "variant": args.variant,
        "checkpoint_path": args.checkpoint,
        "revision": args.revision,
        "robot_type": args.robot_type,
        "camera_keys": args.camera_keys,
        "view_labels": args.view_labels,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "num_steps": args.num_steps,
        "seed": args.seed,
    }
    options = {key: value for key, value in options.items() if value not in (None, "")}
    config = XiaomiRobotics0RuntimeConfig.from_options(options, device=args.device)
    if args.plan_only:
        print(json.dumps(jsonable(config), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.instruction:
        raise ValueError("--instruction is required for inference")
    images = [value for value in (args.base_image, args.wrist_image) if value]
    result = runtime_for(config).predict(
        instruction=args.instruction,
        image=images,
        observation={"state": _state(args.state_json)},
        output_path=args.output,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
