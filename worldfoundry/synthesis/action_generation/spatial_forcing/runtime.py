"""Callable and command-line entrypoints for Spatial-Forcing inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .inference import (
    DEFAULT_CHECKPOINT,
    OFFICIAL_CHECKPOINTS,
    SpatialForcingRuntime,
    SpatialForcingRuntimeConfig,
)

_RUNTIME_CACHE: dict[tuple[Any, ...], SpatialForcingRuntime] = {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _runtime_for(config: SpatialForcingRuntimeConfig) -> SpatialForcingRuntime:
    key = (
        config.checkpoint_location,
        config.revision,
        config.device,
        config.torch_dtype,
        config.cache_dir,
        config.local_files_only,
        config.attn_implementation,
        config.unnorm_key,
        config.task_suite_name,
        config.num_images_in_input,
        config.center_crop,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = SpatialForcingRuntime(config)
        _RUNTIME_CACHE[key] = runtime
    return runtime


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one checkpoint-backed Spatial-Forcing LIBERO action prediction."""

    del action_context
    options = dict(runtime_options or {})
    variant = str(options.get("variant") or "")
    variant_spec = OFFICIAL_CHECKPOINTS.get(variant)
    location = str(
        checkpoint_path
        or options.get("checkpoint_ref")
        or options.get("repo_id")
        or (variant_spec.repo_id if variant_spec is not None else "")
        or DEFAULT_CHECKPOINT.repo_id
    )
    config = SpatialForcingRuntimeConfig(
        checkpoint_location=location,
        revision=str(options.get("revision") or "") or None,
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        cache_dir=str(options.get("cache_dir") or "") or None,
        local_files_only=_as_bool(options.get("local_files_only"), True),
        attn_implementation=str(options.get("attn_implementation") or "auto"),
        unnorm_key=str(options.get("unnorm_key") or "") or None,
        task_suite_name=str(options.get("task_suite_name") or "") or None,
        num_images_in_input=int(options.get("num_images_in_input") or 2),
        center_crop=_as_bool(options.get("center_crop"), True),
    )
    return _runtime_for(config).predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worldfoundry.synthesis.action_generation.spatial_forcing.runtime",
        description="Run one official Spatial-Forcing OpenVLA/LIBERO inference step.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.repo_id, help="Local checkpoint or HF repo ID")
    parser.add_argument("--revision", default=None, help="Optional HF revision; official repos are pinned by default")
    parser.add_argument("--instruction", required=True, help="Robot task instruction")
    parser.add_argument("--full-image", required=True, type=Path, help="Agent-view RGB image")
    parser.add_argument("--wrist-image", required=True, type=Path, help="Wrist-camera RGB image")
    parser.add_argument(
        "--state", required=True, nargs=8, type=float, metavar="FLOAT", help="Eight LIBERO state values"
    )
    parser.add_argument("--device", default="cuda", help="Inference device, for example cuda:0")
    parser.add_argument("--torch-dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="auto", help="auto, flash_attention_2, sdpa, or eager")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--task-suite-name", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper; model imports and checkpoint loading remain lazy until execution."""

    args = _parser().parse_args(argv)
    result = predict_action(
        instruction=args.instruction,
        image=args.full_image,
        observation={
            "full_image": args.full_image,
            "wrist_image": args.wrist_image,
            "state": args.state,
        },
        action_context=(),
        checkpoint_path=str(args.checkpoint),
        device=args.device,
        runtime_options={
            "revision": args.revision,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "unnorm_key": args.unnorm_key,
            "task_suite_name": args.task_suite_name,
            "local_files_only": True,
        },
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "predict_action"]
