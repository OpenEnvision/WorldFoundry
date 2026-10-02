"""Checkpoint-compatible Ctrl-World three-view rollout launcher."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


NUM_HISTORY = 6
NUM_FRAMES = 5
ACTION_DIM = 7
ACTION_DIRECTIONS = {
    "stationary": (0, 0.0),
    "forward": (0, 1.0),
    "backward": (0, -1.0),
    "left": (1, -1.0),
    "right": (1, 1.0),
    "up": (2, 1.0),
    "down": (2, -1.0),
    "open": (6, -1.0),
    "close": (6, 1.0),
}
OFFICIAL_KEYBOARD_ACTIONS = {
    "forward": "f",
    "backward": "b",
    "left": "l",
    "right": "r",
    "up": "u",
    "down": "d",
    "open": "o",
    "close": "c",
}
OFFICIAL_ACTION_STATS = Path(__file__).resolve().parent / "action_stats.json"


def _clear_conflicting_models_namespace(runtime_root: Path) -> None:
    package = sys.modules.get("models")
    package_file = getattr(package, "__file__", None) if package is not None else None
    if package_file is not None:
        try:
            belongs_to_runtime = Path(package_file).resolve().is_relative_to(runtime_root)
        except OSError:
            belongs_to_runtime = False
        if belongs_to_runtime:
            return
    for module_name in tuple(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            sys.modules.pop(module_name, None)


def _ensure_vendored_runtime_importable() -> None:
    runtime_root = (Path(__file__).resolve().parent / "ctrl_world_runtime").resolve()
    _clear_conflicting_models_namespace(runtime_root)
    runtime_entry = str(runtime_root)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_entry]
    sys.path.insert(0, runtime_entry)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--clip-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--input-views", type=Path, nargs=3, metavar=("VIEW_0", "VIEW_1", "VIEW_2"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-mode",
        choices=("explicit-three-view", "synthetic-three-view"),
        default="explicit-three-view",
    )
    parser.add_argument("--action-direction", choices=tuple(ACTION_DIRECTIONS), default="right")
    parser.add_argument(
        "--action-mode",
        choices=("absolute-pose", "synthetic-zero-smoke"),
        default="absolute-pose",
        help="Use an official physical Cartesian pose, or explicitly opt into the old zero-pose smoke input.",
    )
    parser.add_argument("--initial-pose", type=float, nargs=ACTION_DIM, metavar=("X", "Y", "Z", "RX", "RY", "RZ", "GRIPPER"))
    parser.add_argument("--action-distance", type=float, default=0.08, help="Physical XYZ displacement in metres, as in official key_board_control.")
    parser.add_argument("--action-scale", type=float, help="Normalized delta for synthetic-zero-smoke only (default 0.2).")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the official Ctrl-World sampler requires a CUDA device")
    if (args.height, args.width) != (192, 320):
        raise ValueError("the released Ctrl-World checkpoint requires 192x320 pixels per view")
    if args.num_frames != NUM_FRAMES:
        raise ValueError(f"the released Ctrl-World action adapter requires exactly {NUM_FRAMES} generated frames")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.action_mode == "absolute-pose":
        if args.initial_pose is None:
            raise ValueError("absolute-pose mode requires --initial-pose with seven physical Cartesian pose values")
        if args.action_scale is not None:
            raise ValueError("--action-scale is only valid for synthetic-zero-smoke mode")
        if not all(math.isfinite(value) for value in args.initial_pose):
            raise ValueError("initial_pose must contain seven finite values")
    elif args.initial_pose is not None:
        raise ValueError("--initial-pose is only valid for absolute-pose mode")
    if not math.isfinite(args.action_distance) or args.action_distance < 0:
        raise ValueError("action_distance must be a finite nonnegative physical displacement")
    if args.action_scale is not None and (not math.isfinite(args.action_scale) or not 0.0 <= args.action_scale <= 1.0):
        raise ValueError("action_scale must be between 0 and 1 in normalized action space")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    for label, path in (
        ("Ctrl-World checkpoint", args.checkpoint_path),
        ("Stable Video Diffusion base model", args.base_model_dir),
        ("CLIP text model", args.clip_model_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if args.input_mode == "explicit-three-view":
        if args.input_views is None:
            raise ValueError("explicit-three-view mode requires --input-views with exactly three images")
        for index, path in enumerate(args.input_views):
            if not path.is_file():
                raise FileNotFoundError(f"input view {index} is missing: {path}")
    elif args.input_image is None:
        raise ValueError("synthetic-three-view mode requires --input-image")
    elif not args.input_image.is_file():
        raise FileNotFoundError(f"input image is missing: {args.input_image}")


def _synthetic_three_views(path: Path, *, height: int, width: int):
    """Derive deterministic view proxies from one image for checkpoint smoke only."""
    import numpy as np
    from PIL import Image, ImageOps

    base = np.asarray(
        ImageOps.fit(Image.open(path).convert("RGB"), (width, height), method=Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    shift = max(width // 32, 1)
    left = np.empty_like(base)
    left[:, :-shift] = base[:, shift:]
    left[:, -shift:] = base[:, -1:]
    right = np.empty_like(base)
    right[:, shift:] = base[:, :-shift]
    right[:, :shift] = base[:, :1]
    return np.stack((left, base, right), axis=0)


def _explicit_three_views(paths: list[Path] | tuple[Path, ...], *, height: int, width: int):
    """Resize three synchronized camera images without synthesizing view content."""
    import numpy as np
    from PIL import Image, ImageOps

    if len(paths) != 3:
        raise ValueError(f"Ctrl-World requires exactly three input views, got {len(paths)}")
    return np.stack(
        [
            np.asarray(
                ImageOps.fit(
                    Image.open(path).convert("RGB"),
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                ),
                dtype=np.uint8,
            )
            for path in paths
        ],
        axis=0,
    )


def _normalized_action_rollout(direction: str, scale: float):
    """Legacy out-of-distribution zero-pose action for explicit smoke tests only."""
    import torch

    actions = torch.zeros((1, NUM_HISTORY + NUM_FRAMES, ACTION_DIM), dtype=torch.float32)
    dimension, sign = ACTION_DIRECTIONS[direction]
    if sign:
        actions[0, NUM_HISTORY:, dimension] = torch.linspace(0.0, float(sign * scale), NUM_FRAMES)
    return actions


def _absolute_pose_action_rollout(initial_pose: list[float] | tuple[float, ...], direction: str, distance: float):
    """Match official cold-start history, keyboard rollout, and DROID normalization."""
    import numpy as np
    import torch

    _ensure_vendored_runtime_importable()
    from models.utils import key_board_control

    pose = np.asarray(initial_pose, dtype=np.float64)
    if pose.shape != (ACTION_DIM,) or not np.isfinite(pose).all():
        raise ValueError("initial_pose must contain seven finite physical Cartesian pose values")
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("action_distance must be a finite nonnegative physical displacement")
    if direction not in ACTION_DIRECTIONS:
        raise ValueError(f"unknown action direction: {direction}")

    # The official helper also clips XYZ and contains task 1799's special up/down paths.
    keyboard_action = OFFICIAL_KEYBOARD_ACTIONS.get(direction, "r")
    future = key_board_control(
        pose[None, :], keyboard_action, distance=0.0 if direction == "stationary" else distance
    )
    physical = np.concatenate((np.repeat(pose[None, :], NUM_HISTORY, axis=0), future), axis=0)

    stats = json.loads(OFFICIAL_ACTION_STATS.read_text(encoding="utf-8"))
    lower = np.asarray(stats["state_01"], dtype=np.float64)
    upper = np.asarray(stats["state_99"], dtype=np.float64)
    if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,) or not np.all(upper > lower):
        raise RuntimeError(f"invalid Ctrl-World DROID action statistics: {OFFICIAL_ACTION_STATS}")
    normalized = np.clip(2 * (physical - lower) / (upper - lower + 1e-8) - 1, -1, 1)
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)


def _model_args(args: argparse.Namespace, dtype) -> SimpleNamespace:
    return SimpleNamespace(
        svd_model_path=str(args.base_model_dir.resolve()),
        clip_model_path=str(args.clip_model_dir.resolve()),
        dtype=dtype,
        action_dim=ACTION_DIM,
        num_history=NUM_HISTORY,
        num_frames=NUM_FRAMES,
        text_cond=True,
        frame_level_cond=True,
        his_cond_zero=False,
        motion_bucket_id=args.motion_bucket_id,
        fps=7,
        width=args.width,
        height=args.height,
    )


def run(args: argparse.Namespace) -> Path:
    """Generate a three-camera Ctrl-World rollout with the released checkpoint."""
    _validate_args(args)
    _ensure_vendored_runtime_importable()

    import numpy as np
    import torch

    from models.ctrl_world import CrtlWorld
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
    from models.utils import split_ctrl_world_latents
    from worldfoundry.core.media.codecs.video import save_video_h264
    from worldfoundry.core.model_loading import load_torch_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Ctrl-World inference cannot run")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading local Stable Video Diffusion and CLIP components...")
    model = CrtlWorld(_model_args(args, dtype))
    print("Loading Ctrl-World checkpoint...")
    state_dict = load_torch_checkpoint(
        args.checkpoint_path.resolve(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    model.to(device=device, dtype=dtype).eval()

    views = (
        _explicit_three_views(args.input_views, height=args.height, width=args.width)
        if args.input_mode == "explicit-three-view"
        else _synthetic_three_views(args.input_image, height=args.height, width=args.width)
    )
    view_tensor = torch.from_numpy(views).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    view_tensor = view_tensor / 127.5 - 1.0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        latent_dist = model.pipeline.vae.encode(view_tensor).latent_dist
        view_latents = latent_dist.sample(generator=generator) * model.pipeline.vae.config.scaling_factor
        current_latent = torch.cat([view_latents[index] for index in range(3)], dim=1).unsqueeze(0)
        expected_shape = (1, 4, 72, 40)
        if tuple(current_latent.shape) != expected_shape:
            raise RuntimeError(f"Ctrl-World three-view latent shape must be {expected_shape}, got {tuple(current_latent.shape)}")
        history = current_latent.unsqueeze(1).repeat(1, NUM_HISTORY, 1, 1, 1)
        if args.action_mode == "absolute-pose":
            actions = _absolute_pose_action_rollout(args.initial_pose, args.action_direction, args.action_distance)
        else:
            print("Using synthetic zero-pose actions for smoke testing only; motion direction is not physically validated.")
            actions = _normalized_action_rollout(args.action_direction, args.action_scale if args.action_scale is not None else 0.2)
        actions = actions.to(device=device, dtype=dtype)
        text_token = model.action_encoder(
            actions,
            [args.prompt],
            model.tokenizer,
            model.text_encoder,
            frame_level_cond=True,
        )
        _, predicted_latents = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current_latent,
            text=text_token,
            width=args.width,
            height=args.height * 3,
            num_frames=NUM_FRAMES,
            history=history,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=NUM_FRAMES,
            max_guidance_scale=args.guidance_scale,
            fps=7,
            motion_bucket_id=args.motion_bucket_id,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )

        split_latents = split_ctrl_world_latents(predicted_latents, rows=3, cols=1)
        decoded_views = []
        for view_latent in split_latents:
            decoded = model.pipeline.vae.decode(
                view_latent / model.pipeline.vae.config.scaling_factor,
                num_frames=NUM_FRAMES,
            ).sample
            decoded = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255.0).to(torch.uint8)
            decoded_views.append(decoded.permute(0, 2, 3, 1).cpu().numpy())

    video = np.concatenate(decoded_views, axis=2)
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_h264(video, output_path, fps=args.fps)
    print(f"Ctrl-World output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
