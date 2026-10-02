"""Compatibility launcher for TesserAct RGB-depth-normal video inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_vendored_runtime_importable() -> None:
    runtime_root = str(Path(__file__).resolve().parent / "tesseract_runtime")
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--depth-path", type=Path)
    parser.add_argument("--normal-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--geometry-mode", choices=("synthetic-gradient", "files"), default="synthetic-gradient")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--use-dynamic-cfg", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--memory-efficient", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the official TesserAct sampler requires a CUDA device")
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be positive multiples of 16")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        raise ValueError("num_frames must be at least 5 and follow the CogVideoX 4n+1 layout")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    for label, path in (
        ("TesserAct checkpoint", args.checkpoint_dir),
        ("CogVideoX base model", args.base_model_dir),
        ("input image", args.input_image),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if args.geometry_mode == "files" and (args.depth_path is None or args.normal_path is None):
        raise ValueError("geometry_mode=files requires both --depth-path and --normal-path")
    if (args.depth_path is None) != (args.normal_path is None):
        raise ValueError("--depth-path and --normal-path must be supplied together")
    for label, path in (("depth", args.depth_path), ("normal", args.normal_path)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} conditioning is missing: {path}")


def _load_rgb(path: Path, *, height: int, width: int):
    import numpy as np
    from PIL import Image

    from tesseract.utils import crop_and_resize_frames, read_video_first_frame

    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi"}:
        image = np.asarray(read_video_first_frame(str(path)), dtype=np.float32)
    else:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    # Official RGBDN inference resizes float32 RGB before scaling to [0, 1].
    # Quantizing the resized result to uint8 changes nearly every input pixel.
    image = crop_and_resize_frames([image], (height, width), interpolation="bilinear")[0]
    return np.ascontiguousarray(image, dtype=np.float32)


def _synthetic_gradient_geometry(rgb):
    """Build deterministic non-metric geometry strictly for a runtime smoke test."""
    import numpy as np

    rgb_float = rgb.astype(np.float32) / 255.0
    luminance = 0.2126 * rgb_float[..., 0] + 0.7152 * rgb_float[..., 1] + 0.0722 * rgb_float[..., 2]
    vertical = np.linspace(0.0, 1.0, rgb.shape[0], dtype=np.float32)[:, None]
    inverse_depth = np.clip(0.65 * luminance + 0.35 * (1.0 - vertical), 0.0, 1.0)
    grad_y, grad_x = np.gradient(inverse_depth)
    normal = np.stack((-2.0 * grad_x, -2.0 * grad_y, np.ones_like(inverse_depth)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    normal = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    return inverse_depth.astype(np.float32), normal.astype(np.float32)


def _file_geometry(depth_path: Path, normal_path: Path, *, height: int, width: int):
    import cv2
    import numpy as np

    from tesseract.utils import crop_and_resize_frames

    depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth conditioning must be a two-dimensional array, got {depth.shape}")
    # Match the released RGBDN SFT inference script's depth convention.
    depth = 1.0 - depth
    depth = crop_and_resize_frames([depth], (height, width), interpolation="bilinear")[0]
    if float(depth.min()) < 0.0:
        depth = (depth + 1.0) / 2.0
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

    normal_bgr = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
    if normal_bgr is None:
        raise ValueError(f"could not decode normal conditioning image: {normal_path}")
    normal = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
    normal = crop_and_resize_frames([normal], (height, width), interpolation="bilinear")[0]
    return depth, normal.astype(np.float32) / 255.0


def _conditioning_tensor(args: argparse.Namespace):
    import numpy as np
    import torch

    rgb = _load_rgb(args.input_image, height=args.height, width=args.width)
    if args.geometry_mode == "files":
        depth, normal = _file_geometry(
            args.depth_path,
            args.normal_path,
            height=args.height,
            width=args.width,
        )
    else:
        depth, normal = _synthetic_gradient_geometry(rgb)
    rgb_float = rgb.astype(np.float32) / 255.0
    rgb_depth_normal = np.concatenate((rgb_float, np.repeat(depth[..., None], 3, axis=-1), normal), axis=-1)
    return torch.from_numpy(rgb_depth_normal).permute(2, 0, 1).unsqueeze(0).contiguous()


def run(args: argparse.Namespace) -> Path:
    """Generate a TesserAct RGB/depth/normal rollout with the official model classes."""
    _validate_args(args)
    _ensure_vendored_runtime_importable()

    import numpy as np
    import torch
    from diffusers import CogVideoXDPMScheduler

    from tesseract.modules.tesseract_model import TesserActDepthNormal
    from tesseract.modules.tesseract_pipeline import TesserActImageToDepthNormalVideoPipeline
    from worldfoundry.core.media.codecs.video import save_video_h264

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TesserAct inference cannot run")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    conditioning = _conditioning_tensor(args)
    print("Loading TesserAct RGBDN transformer...")
    transformer = TesserActDepthNormal.from_pretrained_modify(
        str(args.checkpoint_dir.resolve()),
        torch_dtype=dtype,
    ).eval()
    transformer.config.in_channels += 16 * 4
    if hasattr(transformer.patch_embed, "pos_embedding"):
        del transformer.patch_embed.pos_embedding
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False

    print("Loading local CogVideoX tokenizer, text encoder, VAE, and scheduler...")
    pipe = TesserActImageToDepthNormalVideoPipeline.from_pretrained(
        str(args.base_model_dir.resolve()),
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if args.memory_efficient:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        result = pipe(
            image=conditioning,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            generator=generator,
        )
    frames = result.frames[0]
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_h264(frames, output_path, fps=args.fps)
    print(f"TesserAct output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
