"""Public 10-second pure-UCPE image-to-video inference entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io import resolve_data_path

from .helpers.action_camera import DEFAULT_PITCH_LIMIT_DEG, DEFAULT_ROTATION_SPEED_DEG, DEFAULT_TRANSLATION_SPEED
from .helpers.action_condition import action_config, build_action_condition, build_action_trajectory
from .helpers.action_overlay import overlay_genie_on_video
from .ltx_core.components.guiders import MultiModalGuiderParams
from .ltx_core.model.video_vae.tiling import TilingConfig
from .ltx_core.model.video_vae.video_vae import get_video_chunks_number
from .ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from .ltx_pipelines.utils.args import ImageConditioningInput
from .ltx_pipelines.utils.media_io import encode_video

ROOT = Path(__file__).resolve().parent
MOGE_FOV_MODULE = "worldfoundry.synthesis.visual_generation.joyai_echo.echo_wm_runtime.helpers.moge_fov"
DEFAULT_CONFIG = Path(resolve_data_path("models/runtime/configs/joyai-echo-wm/inference_wm.yaml"))
NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "game UI, video game interface, HUD, heads-up display, menu, status bar, "
    "health bar, score, minimap, crosshair, reticle, buttons, icons, subtitles, "
    "captions, watermark, logo, text overlay, user interface"
)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _override(value, default):
    return default if value is None else value


def _auto_fov(image: Path, model: str, python_bin: str, width: int, height: int) -> float:
    command = package_module_command(
        MOGE_FOV_MODULE,
        ["--image", image, "--model", model, "--target-width", width, "--target-height", height],
        python_executable=python_bin,
    )
    raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    return float(json.loads(raw)["fov_x_deg"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, required=True, help="First-frame image; this entrypoint is I2V-only.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--action-str", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gemma-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/echo_wm.mp4"))
    parser.add_argument("--auto-fov", action="store_true")
    parser.add_argument("--moge-model", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--moge-python", default=sys.executable)
    parser.add_argument("--fov-deg", type=float, default=None)
    parser.add_argument("--translation-speed", type=float, default=None)
    parser.add_argument("--rotation-speed-deg", type=float, default=None)
    parser.add_argument("--pitch-limit-deg", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--video-cfg", type=float, default=None, help="Video CFG scale (default: 4.0).")
    parser.add_argument("--audio-cfg", type=float, default=None, help="Audio CFG scale (default: 2.0).")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt; overrides config.")
    parser.add_argument("--stg-scale", type=float, default=None)
    parser.add_argument("--stg-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--action-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a second MP4 with a Genie-style WASD/rotation HUD overlay "
        "(default: enabled; disable with --no-action-overlay).",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"I2V first-frame image not found: {args.image}")
    cfg = _load_config(args.config)
    video_cfg = cfg.get("video", {})
    model_cfg = cfg.get("model", {})
    action_cfg = cfg.get("action", {})
    checkpoint = args.checkpoint or ROOT / model_cfg.get("checkpoint", "checkpoints/echo-wm-base.safetensors")
    gemma_path = args.gemma_path or ROOT / model_cfg["gemma_path"]
    width = _override(args.width, video_cfg.get("width", 1280))
    height = _override(args.height, video_cfg.get("height", 704))
    num_frames = _override(args.num_frames, video_cfg.get("num_frames", 241))
    fps = _override(args.fps, video_cfg.get("fps", 24.0))
    steps = _override(args.steps, video_cfg.get("steps", 30))
    seed = args.seed if args.seed is not None else video_cfg.get("seed", 42)
    legacy_guidance = args.guidance_scale
    video_cfg_scale = _override(
        args.video_cfg,
        legacy_guidance if legacy_guidance is not None else video_cfg.get("video_cfg", 4.0),
    )
    audio_cfg_scale = _override(
        args.audio_cfg,
        legacy_guidance if legacy_guidance is not None else video_cfg.get("audio_cfg", 2.0),
    )
    negative_prompt = args.negative_prompt or cfg.get("negative_prompt", NEGATIVE_PROMPT)
    stg_scale = _override(args.stg_scale, video_cfg.get("stg_scale", 1.0))
    stg_blocks = _override(args.stg_blocks, video_cfg.get("stg_blocks", [29]))
    fov = _override(args.fov_deg, action_cfg.get("fov_deg", 70.0))
    if not args.prompt:
        raise ValueError("Provide --prompt using the six-field format in PROMPT_SKILL.md")
    prompt = args.prompt
    if args.auto_fov:
        fov = _auto_fov(args.image, args.moge_model, args.moge_python, width, height)

    action = build_action_condition(
        args.action_str,
        num_frames=num_frames,
        width=width,
        height=height,
        translation_speed=_override(
            args.translation_speed, action_cfg.get("translation_speed", DEFAULT_TRANSLATION_SPEED)
        ),
        rotation_speed_deg=_override(
            args.rotation_speed_deg, action_cfg.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG)
        ),
        pitch_limit_deg=_override(args.pitch_limit_deg, action_cfg.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG)),
        fov_deg=fov,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fps=fps,
    )
    trajectory = None
    if args.action_overlay:
        trajectory = build_action_trajectory(
            args.action_str,
            num_frames=num_frames,
            translation_speed=_override(
                args.translation_speed, action_cfg.get("translation_speed", DEFAULT_TRANSLATION_SPEED)
            ),
            rotation_speed_deg=_override(
                args.rotation_speed_deg, action_cfg.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG)
            ),
            pitch_limit_deg=_override(args.pitch_limit_deg, action_cfg.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG)),
            fps=fps,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(checkpoint),
        gemma_root=str(gemma_path),
        loras=(),
        device=device,
        action_config=action_config(width, height),
    )
    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=video_cfg_scale, stg_scale=stg_scale, stg_blocks=stg_blocks
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=audio_cfg_scale, stg_scale=stg_scale, stg_blocks=stg_blocks
        ),
        images=[ImageConditioningInput(str(args.image), 0, 1.0)],
        action_cond=action,
        video_tiling_config=TilingConfig.default(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=int(fps),
        audio=None if args.no_audio else audio,
        output_path=str(args.output),
        video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
    )
    overlay_output = None
    if trajectory is not None:
        overlay_output = args.output.with_name(f"{args.output.stem}_action{args.output.suffix}")
        overlay_genie_on_video(args.output, trajectory, output_path=overlay_output)
    metadata = {
        "prompt": prompt,
        "action": args.action_str,
        "fov_deg": fov,
        "seed": seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "action_overlay": bool(args.action_overlay),
        "overlay_output": overlay_output.name if overlay_output else None,
        "video_cfg": video_cfg_scale,
        "audio_cfg": audio_cfg_scale,
        "negative_prompt": negative_prompt,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    if overlay_output:
        print(f"Saved {overlay_output}")


if __name__ == "__main__":
    main()
