from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

RUNTIME_ROOT = Path(__file__).resolve().parent / "pusav1_runtime"
LIGHTX2V_DEFAULT_SUBDIR = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


def _parse_csv_ints(value: str, fallback: Iterable[int]) -> list[int]:
    if not value:
        return list(fallback)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str, fallback: Iterable[float]) -> list[float]:
    if not value:
        return list(fallback)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _required_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required Pusa V1 file not found: {path}")
    return str(path)


def _prepare_images(image_paths: list[str], width: int, height: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for raw_path in image_paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Pusa V1 conditioning image not found: {source}")
        img = Image.open(source).convert("RGB")
        original_w, original_h = img.size
        ratio = min(width / original_w, height / original_h)
        new_w = max(1, int(original_w * ratio))
        new_h = max(1, int(original_h * ratio))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
        images.append(canvas)
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pusa V1 Wan2.2 inference from the in-tree WorldFoundry runtime.")
    parser.add_argument("--mode", choices=("t2v", "i2v", "multi"), default="t2v")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--base-model-root", required=True)
    parser.add_argument("--high-model-dir", default="")
    parser.add_argument("--low-model-dir", default="")
    parser.add_argument("--lightx2v-root", default="")
    parser.add_argument("--high-lora-path", required=True)
    parser.add_argument("--low-lora-path", required=True)
    parser.add_argument("--high-lora-alpha", type=float, default=1.5)
    parser.add_argument("--low-lora-alpha", type=float, default=1.4)
    parser.add_argument("--image-path", action="append", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cond-position", default="")
    parser.add_argument("--noise-multipliers", default="")
    parser.add_argument("--switch-dit-boundary", type=float, default=0.875)
    parser.add_argument("--lightx2v", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--num-persistent-param-in-dit", type=float, default=6e9)
    return parser.parse_args()


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    if args.cfg_scale is None:
        args.cfg_scale = 1.0 if args.lightx2v else 3.0
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    high_lora = Path(args.high_lora_path).expanduser().resolve()
    low_lora = Path(args.low_lora_path).expanduser().resolve()
    _required_file(high_lora)
    _required_file(low_lora)
    args.high_lora_path = str(high_lora)
    args.low_lora_path = str(low_lora)

    from worldfoundry.core.io import save_video
    from worldfoundry.synthesis.visual_generation.pusa_vidgen.native_pipeline import load_pusa_pipeline

    base_dir = Path(args.base_model_root).expanduser().resolve()
    high_model_dir = Path(args.high_model_dir or base_dir / "high_noise_model").expanduser().resolve()
    low_model_dir = Path(args.low_model_dir or base_dir / "low_noise_model").expanduser().resolve()
    lightx2v_high_path = None
    lightx2v_low_path = None
    if args.lightx2v:
        if not args.lightx2v_root:
            raise ValueError("--lightx2v requires --lightx2v-root")
        lightx2v_dir = Path(args.lightx2v_root).expanduser().resolve() / LIGHTX2V_DEFAULT_SUBDIR
        lightx2v_high_path = _required_file(lightx2v_dir / "high_noise_model.safetensors")
        lightx2v_low_path = _required_file(lightx2v_dir / "low_noise_model.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pusa_pipeline(
        base_model_root=base_dir,
        high_model_dir=high_model_dir,
        low_model_dir=low_model_dir,
        high_lora_path=args.high_lora_path,
        low_lora_path=args.low_lora_path,
        high_lora_alpha=args.high_lora_alpha,
        low_lora_alpha=args.low_lora_alpha,
        lightx2v_high_path=lightx2v_high_path,
        lightx2v_low_path=lightx2v_low_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=int(args.num_persistent_param_in_dit))
    if args.mode == "t2v":
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            tiled=True,
            switch_DiT_boundary=args.switch_dit_boundary,
            cfg_scale=args.cfg_scale,
        )
    else:
        if not args.image_path:
            raise ValueError("Pusa V1 i2v/multi mode requires at least one --image-path.")
        images = _prepare_images(args.image_path, args.width, args.height)
        cond_positions = _parse_csv_ints(args.cond_position, range(len(images)))
        noise_multipliers = _parse_csv_floats(args.noise_multipliers, [0.2] * len(images))
        if len(images) != len(cond_positions) or len(images) != len(noise_multipliers):
            raise ValueError("--image-path, --cond-position, and --noise-multipliers counts must match.")
        multi_frame_images = {
            frame_idx: (image, noise)
            for frame_idx, image, noise in zip(cond_positions, images, noise_multipliers)
        }
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            multi_frame_images=multi_frame_images,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            tiled=True,
            switch_DiT_boundary=args.switch_dit_boundary,
            cfg_scale=args.cfg_scale,
        )

    save_video(video, str(output_path), fps=args.fps, quality=args.quality)
    metadata = {
        "ok": True,
        "status": "success",
        "runtime": "worldfoundry.synthesis.visual_generation.pusa_vidgen.pusa_v1_runner",
        "runtime_root": str(RUNTIME_ROOT),
        "source": "Pusa V1 Wan2.2",
        "checkpoint_root": str(checkpoint_root),
        "output_path": str(output_path),
        "mode": args.mode,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "lightx2v": bool(args.lightx2v),
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
