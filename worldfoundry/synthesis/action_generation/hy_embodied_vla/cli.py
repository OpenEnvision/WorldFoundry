"""Command-line entrypoint for Hy-VLA planning and local inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from worldfoundry.core.io.formats.serialization import write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worldfoundry.synthesis.action_generation.hy_embodied_vla",
        description=(
            "Inference-only Tencent Hy-Embodied-0.5-VLA runtime. Checkpoints are not "
            "bundled and implicit multi-gigabyte downloads are disabled."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    plan = subparsers.add_parser("plan", help="write a runtime plan without loading torch or weights")
    _add_runtime_arguments(plan)
    plan.add_argument("--output", type=Path, default=Path("hy_embodied_vla_plan.json"))

    infer = subparsers.add_parser("infer", help="run one checkpoint-backed action-chunk inference")
    _add_runtime_arguments(infer)
    image_group = infer.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", help="one RGB image replicated across the three official views")
    image_group.add_argument("--image-top", help="top/head RGB image")
    infer.add_argument("--image-left", help="left-hand RGB image")
    infer.add_argument("--image-right", help="right-hand RGB image")
    infer.add_argument("--prompt", required=True, help="robot instruction")
    infer.add_argument(
        "--state-json",
        required=True,
        help="JSON vector or path to a JSON file containing the state vector",
    )
    infer.add_argument("--output", type=Path, default=Path("hy_embodied_vla_action_trace.json"))
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    from .runtime import (
        DEFAULT_BLEND_MODE,
        DEFAULT_STATE_FORMAT,
        DEFAULT_TORCH_DTYPE,
        DEFAULT_VARIANT,
        OFFICIAL_REPOSITORIES,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="local checkpoint directory or repo ID (defaults to the selected official variant)",
    )
    parser.add_argument(
        "--variant", choices=tuple(OFFICIAL_REPOSITORIES), default=DEFAULT_VARIANT
    )
    parser.add_argument(
        "--revision",
        help="checkpoint revision (official repos default to the audited immutable revision)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default=DEFAULT_TORCH_DTYPE)
    parser.add_argument(
        "--state-format",
        choices=("normalized", "posrot20", "robotwin_wxyz"),
        default=DEFAULT_STATE_FORMAT,
    )
    parser.add_argument(
        "--blend-mode",
        choices=("auto", "rel_only", "abs_only", "rel_abs"),
        default=DEFAULT_BLEND_MODE,
    )
    parser.add_argument("--history-size", type=int)


def _runtime_config(args: argparse.Namespace):
    from .runtime import OFFICIAL_REPOSITORIES, HyEmbodiedVLARuntimeConfig

    return HyEmbodiedVLARuntimeConfig(
        checkpoint=args.checkpoint or OFFICIAL_REPOSITORIES[args.variant],
        variant=args.variant,
        revision=args.revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=True,
        allow_checkpoint_download=False,
        history_size=args.history_size,
        state_format=args.state_format,
        blend_mode=args.blend_mode,
    )


def _read_state(value: str) -> Any:
    source = Path(value).expanduser()
    text = source.read_text(encoding="utf-8") if source.is_file() else value
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise TypeError("--state-json must contain a JSON array")
    return parsed


def _plan(args: argparse.Namespace) -> int:
    from .runtime import build_plan_payload

    config = _runtime_config(args)
    payload = build_plan_payload(config=config, context={}, profile={})
    destination = args.output.expanduser().resolve()
    write_json(destination, payload)
    print(destination)
    return 0


def _infer(args: argparse.Namespace) -> int:
    from .runtime import HyEmbodiedVLARuntime

    images: Any
    if args.image:
        images = args.image
    else:
        images = {
            key: value
            for key, value in {
                "top_head": args.image_top,
                "hand_left": args.image_left,
                "hand_right": args.image_right,
            }.items()
            if value is not None
        }
    runtime = HyEmbodiedVLARuntime(_runtime_config(args))
    result = runtime.predict_action(
        instruction=args.prompt,
        images=images,
        state=_read_state(args.state_json),
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "plan":
        return _plan(args)
    if args.command == "infer":
        return _infer(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


__all__ = ["main"]
