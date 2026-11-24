"""Callable and command-line entrypoints for X-WAM inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .inference import (
    DEFAULT_VARIANT,
    OFFICIAL_BASE_MODEL,
    OFFICIAL_CHECKPOINT,
    OFFICIAL_VARIANTS,
    XWAMRuntime,
    XWAMRuntimeConfig,
)

_RUNTIME_CACHE: dict[tuple[Any, ...], XWAMRuntime] = {}


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


def _runtime_for(config: XWAMRuntimeConfig) -> XWAMRuntime:
    key = (
        config.checkpoint_location,
        config.revision,
        config.variant,
        config.base_model_location,
        config.base_revision,
        config.device,
        config.torch_dtype,
        config.cache_dir,
        config.local_files_only,
        config.denoise_steps,
        config.action_denoise_steps,
        config.cfg_scale,
        config.compile_model,
        config.generate_world,
        config.run_depth,
        config.world_video_path,
    )
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = XWAMRuntime(config)
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
    """Run one official X-WAM RoboCasa or RoboTwin inference step."""

    del action_context
    options = dict(runtime_options or {})
    location = str(
        checkpoint_path
        or options.get("checkpoint_ref")
        or options.get("checkpoint_repo_id")
        or options.get("repo_id")
        or OFFICIAL_CHECKPOINT.repo_id
    )
    config = XWAMRuntimeConfig(
        checkpoint_location=location,
        revision=str(options.get("revision") or "") or None,
        variant=str(options.get("variant") or DEFAULT_VARIANT),
        base_model_location=str(
            options.get("base_model_path")
            or options.get("base_model_ref")
            or options.get("base_model_repo_id")
            or OFFICIAL_BASE_MODEL.repo_id
        ),
        base_revision=str(options.get("base_revision") or "") or None,
        device=device,
        torch_dtype=str(options.get("torch_dtype") or "auto"),
        cache_dir=str(options.get("cache_dir") or "") or None,
        local_files_only=_as_bool(options.get("local_files_only"), True),
        denoise_steps=int(options.get("denoise_steps") or 50),
        action_denoise_steps=int(options.get("action_denoise_steps") or 10),
        cfg_scale=float(options.get("cfg_scale") or options.get("cfg") or 0.0),
        compile_model=_as_bool(options.get("compile_model"), False),
        generate_world=_as_bool(options.get("generate_world"), False),
        run_depth=_as_bool(options.get("run_depth"), False),
        world_video_path=str(options.get("world_video_path") or "") or None,
    )
    return _runtime_for(config).predict_action(
        instruction=instruction,
        image=image,
        observation=observation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worldfoundry.synthesis.action_generation.x_wam.runtime",
        description="Run one official X-WAM world-action inference step.",
    )
    parser.add_argument("--checkpoint", default=OFFICIAL_CHECKPOINT.repo_id, help="Local root or HF repo ID")
    parser.add_argument("--revision", default=None, help="Policy Hub revision; official release is pinned by default")
    parser.add_argument("--base-model", default=OFFICIAL_BASE_MODEL.repo_id, help="Wan2.2-TI2V-5B root or HF ID")
    parser.add_argument(
        "--base-revision", default=None, help="Wan base revision; official release is pinned by default"
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=tuple(name for name, spec in OFFICIAL_VARIANTS.items() if spec.deployable),
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", required=True, nargs=3, type=Path, metavar=("VIEW0", "VIEW1", "VIEW2"))
    parser.add_argument("--state", required=True, nargs="+", type=float, metavar="FLOAT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-rank", type=int, default=0)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--step-id", type=int, default=0)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--world-video", type=Path, default=None, help="Run all 50 steps and save future-view mosaic")
    parser.add_argument("--run-depth", action="store_true", help="Include the depth branch in the saved mosaic")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper; heavyweight model imports occur only after argument parsing."""

    args = _parser().parse_args(argv)
    spec = OFFICIAL_VARIANTS[args.variant]
    if len(args.state) != spec.state_dimension:
        raise SystemExit(f"{args.variant} requires {spec.state_dimension} --state values, got {len(args.state)}")
    observation: dict[str, Any] = {
        "state": args.state,
        "env_rank": args.env_rank,
        "rollout_id": args.rollout_id,
        "step_id": args.step_id,
    }
    if args.seed is not None:
        observation["seed"] = args.seed
    result = predict_action(
        instruction=args.instruction,
        image=args.image,
        observation=observation,
        action_context=(),
        checkpoint_path=str(args.checkpoint),
        device=args.device,
        runtime_options={
            "revision": args.revision,
            "variant": args.variant,
            "base_model_ref": args.base_model,
            "base_revision": args.base_revision,
            "torch_dtype": args.torch_dtype,
            "denoise_steps": args.denoise_steps,
            "action_denoise_steps": args.action_denoise_steps,
            "cfg_scale": args.cfg_scale,
            "compile_model": args.compile_model,
            "local_files_only": True,
            "generate_world": args.world_video is not None,
            "run_depth": args.run_depth,
            "world_video_path": args.world_video,
        },
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "predict_action"]
