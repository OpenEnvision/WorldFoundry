"""Compatibility launcher for official EgoWM 3-DoF and 25-DoF SVD inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--motion-bucket-id", type=int, default=180)
    parser.add_argument(
        "--variant", choices=("25dof", "3dof"), default="25dof",
        help="Checkpoint action/state contract (default: 25dof).",
    )
    parser.add_argument(
        "--conditions-path",
        type=Path,
        help="JSON with physical_initial_state (25 numbers) and physical_actions (num_frames x 25).",
    )
    parser.add_argument(
        "--actions-path",
        type=Path,
        help="3-DoF JSON with normalized_actions (num_frames x 3), paired with --input-image.",
    )
    parser.add_argument(
        "--smoke-synthetic",
        action="store_true",
        help="Use synthetic actions and a zero initial state for a load/generation smoke test only.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.0,
        help="First action component; for 3-DoF, normalized offset from zero displacement.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")
    if args.conditions_path and args.smoke_synthetic:
        raise ValueError("conditions-path and smoke-synthetic are mutually exclusive")
    if args.actions_path and args.conditions_path:
        raise ValueError("actions-path and 25-DoF conditions-path are mutually exclusive")
    if args.actions_path and args.smoke_synthetic:
        raise ValueError("actions-path and smoke-synthetic are mutually exclusive")
    if args.actions_path and args.action_scale != 0.0:
        raise ValueError("action-scale cannot be combined with actions-path")
    if args.conditions_path and args.action_scale != 0.0:
        raise ValueError("action-scale cannot be combined with 25-DoF conditions-path")
    if args.variant == "25dof" and not args.conditions_path and not args.smoke_synthetic:
        raise ValueError(
            "25-DoF EgoWM requires --conditions-path with paired physical initial state and actions; "
            "use --smoke-synthetic only for a non-semantic smoke test"
        )
    if args.variant == "25dof" and args.actions_path:
        raise ValueError("3-DoF actions-path cannot be used with a 25-DoF checkpoint")
    if args.variant == "3dof" and (args.conditions_path or args.smoke_synthetic):
        raise ValueError("3-DoF EgoWM uses --actions-path or --action-scale; 25-DoF conditions and smoke mode are unsupported")
    for label, path in (
        ("official source", args.source_dir),
        ("checkpoint", args.checkpoint_path),
        ("base model", args.base_model_dir),
        ("input image", args.input_image),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if args.conditions_path and not args.conditions_path.is_file():
        raise FileNotFoundError(f"EgoWM conditions file is missing: {args.conditions_path}")
    if args.actions_path:
        load_3dof_actions(args.actions_path, args.num_frames)


def _numbers(value: object, *, label: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain only numbers")
    numbers = [float(item) for item in value]
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"{label} must contain only finite numbers")
    return numbers


def _normalize(values: list[float], stats: dict[str, list[float]]) -> list[float]:
    # Official SVD_25dof_nav_1xval.py: (value - min) / (max - min) * 2 - 1.
    return [(value - low) / (high - low) * 2 - 1
            for value, low, high in zip(values, stats["min"], stats["max"], strict=True)]


def _load_25dof_stats() -> dict[str, dict[str, list[float]]]:
    path = Path(__file__).parent / "nav_25dof_stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EgoWM 25-DoF statistics must be a JSON object")
    result = {}
    for label in ("state_stats", "action_stats_6"):
        raw = payload.get(label)
        if not isinstance(raw, dict):
            raise ValueError(f"EgoWM statistics missing {label}")
        minimum = _numbers(raw.get("min"), label=f"{label}.min", count=25)
        maximum = _numbers(raw.get("max"), label=f"{label}.max", count=25)
        if any(high <= low for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError(f"EgoWM {label} must have max > min in every dimension")
        result[label] = {"min": minimum, "max": maximum}
    return result


def load_25dof_conditions(path: Path, num_frames: int) -> tuple[list[float], list[list[float]]]:
    """Normalize physical 25D EVE1x navigation conditions with official statistics.

    Actions are differences between states six source-video frames apart. The
    caller supplies those physical differences and the matching initial state.
    Neither array is clipped, matching SVD_25dof_nav_1xval.py at upstream
    revision 4a24b6fb917b8b86cea892312971f39691c532e8.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EgoWM conditions must be a JSON object")
    state = _numbers(payload.get("physical_initial_state"), label="physical_initial_state", count=25)
    raw_actions = payload.get("physical_actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != num_frames:
        raise ValueError(f"physical_actions must contain exactly {num_frames} 25D rows")
    actions = [_numbers(row, label=f"physical_actions[{index}]", count=25)
               for index, row in enumerate(raw_actions)]
    stats = _load_25dof_stats()
    return _normalize(state, stats["state_stats"]), [
        _normalize(row, stats["action_stats_6"]) for row in actions
    ]


def load_3dof_actions(path: Path, num_frames: int) -> list[list[float]]:
    """Read explicit per-frame actions in the official 3-DoF normalized space."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("EgoWM 3-DoF actions must be a JSON object")
    raw_actions = payload.get("normalized_actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != num_frames:
        raise ValueError(f"normalized_actions must contain exactly {num_frames} 3D rows")
    return [_numbers(row, label=f"normalized_actions[{index}]", count=3)
            for index, row in enumerate(raw_actions)]


def checkpoint_action_dims(state_dict: dict) -> int:
    """Identify the checkpoint from its action and state embedding signature."""
    embedding = state_dict.get("add_action_embedding.linear_1.weight")
    if embedding is None:
        raise ValueError("EgoWM checkpoint has no action embedding")
    width = embedding.shape[1]
    has_state = any(key.startswith("add_state_embedding.") for key in state_dict)
    if width == 96 and not has_state:
        return 3
    if width == 800 and has_state:
        return 25
    raise ValueError(f"Unsupported EgoWM checkpoint: action width {width}, state embedding present={has_state}")


def _validate_checkpoint_variant(action_dims: int, variant: str) -> None:
    if action_dims != int(variant.removesuffix("dof")):
        raise ValueError(
            f"EgoWM --variant {variant} conflicts with checkpoint action/state signature ({action_dims}dof)"
        )


def run(args: argparse.Namespace) -> Path:
    """Load the official action/state-conditioned SVD classes and generate a video."""
    _validate_args(args)
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    base_model_dir = args.base_model_dir.expanduser().resolve()
    input_image = args.input_image.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if args.variant == "25dof" and args.conditions_path:
        initial_values, action_values = load_25dof_conditions(args.conditions_path, args.num_frames)
    elif args.variant == "3dof" and args.actions_path:
        action_values = load_3dof_actions(args.actions_path, args.num_frames)

    source_text = str(source_dir)
    inserted_source = source_text not in sys.path
    if inserted_source:
        sys.path.insert(0, source_text)
    try:
        import torch
        from PIL import Image

        from models.svd_wrapper import (
            DebugActionUnetFwise2,
            DebugActionUnetFwise2state,
            DebugSVDActionPipeline,
            DebugSVDActionStateConstcfgPipeline,
        )

        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        state_dict = checkpoint.get("unet", checkpoint)
        action_dims = checkpoint_action_dims(state_dict)
        if action_dims == 3:
            unet_class = DebugActionUnetFwise2
            pipeline_class = DebugSVDActionPipeline
        else:
            unet_class = DebugActionUnetFwise2state
            pipeline_class = DebugSVDActionStateConstcfgPipeline
        _validate_checkpoint_variant(action_dims, args.variant)
        unet = unet_class.from_pretrained(
            str(base_model_dir),
            subfolder="unet",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        unet.load_state_dict(state_dict, strict=True)
        del state_dict, checkpoint

        pipeline = pipeline_class.from_pretrained(
            str(base_model_dir),
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline.to(args.device)
        pipeline.set_progress_bar_config(disable=False)

        image = Image.open(input_image).convert("RGB").resize((args.width, args.height))
        actions = torch.zeros((args.num_frames, action_dims), dtype=torch.float32)
        if action_dims == 3:
            if args.actions_path:
                actions = torch.tensor(action_values, dtype=torch.float32)
            else:
                # Official 3-DoF scripts normalize x from [-2.5, 5] to [-1, 1].
                # Zero displacement is therefore -1/3, not zero. Treat the UI
                # action scale as an offset from that stationary baseline.
                actions[:, 0] = -1.0 / 3.0 + float(args.action_scale)
        elif args.conditions_path:
            actions = torch.tensor(action_values, dtype=torch.float32)
            initial_state = torch.tensor(initial_values, dtype=torch.float32)
        elif args.smoke_synthetic:
            actions[:, 0] = float(args.action_scale)
            initial_state = torch.zeros(25, dtype=torch.float32)
        generator = torch.manual_seed(args.seed)

        inference_kwargs = dict(
            decode_chunk_size=min(8, args.num_frames),
            generator=generator,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=0.1,
            actions=actions,
            fps=args.fps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
        )
        if action_dims == 25:
            inference_kwargs["init_state"] = initial_state
        else:
            # Official 3-DoF inference uses constant CFG 2.0 on every frame.
            # Equal endpoints reproduce that with the shared action pipeline.
            inference_kwargs["min_guidance_scale"] = 2.0
            inference_kwargs["max_guidance_scale"] = 2.0
        frames = pipeline(image, **inference_kwargs).frames[0]

        from worldfoundry.core.media.codecs.video import save_video_h264

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video_h264([image, *frames], output_path, fps=args.fps)
    finally:
        if inserted_source:
            sys.path.remove(source_text)

    print(f"EgoWM output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
