"""Checkpoint-compatible Genie Envisioner three-view rollout launcher."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path


NUM_VIEWS = 3


def _clear_conflicting_namespace(name: str, runtime_root: Path) -> None:
    package = sys.modules.get(name)
    package_file = getattr(package, "__file__", None) if package is not None else None
    if package_file is not None:
        try:
            belongs_to_runtime = Path(package_file).resolve().is_relative_to(runtime_root)
        except OSError:
            belongs_to_runtime = False
        if belongs_to_runtime:
            return
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(f"{name}."):
            sys.modules.pop(module_name, None)


def _ensure_vendored_runtime_importable() -> None:
    runtime_root = (Path(__file__).resolve().parent / "genie_envisioner_runtime").resolve()
    for namespace in ("models", "utils"):
        _clear_conflicting_namespace(namespace, runtime_root)
    runtime_entry = str(runtime_root)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_entry]
    sys.path.insert(0, runtime_entry)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
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
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n-previous", type=int, default=4)
    parser.add_argument("--latent-chunk", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--decode-timestep", type=float, default=0.03)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the official Genie Envisioner sampler requires a CUDA device")
    if (args.height, args.width) != (192, 256):
        raise ValueError("the released GE-base checkpoint requires 192x256 pixels per view")
    if args.n_previous != 4:
        raise ValueError("the released GE-base configuration requires exactly four history frames")
    if args.latent_chunk < 1:
        raise ValueError("latent_chunk must be at least 1")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.guidance_scale < 1.0:
        raise ValueError("guidance_scale must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    for label, path in (
        ("Genie Envisioner checkpoint", args.checkpoint_path),
        ("LTX-Video base model", args.base_model_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if args.input_mode == "explicit-three-view":
        if args.input_views is None:
            raise ValueError("explicit-three-view mode requires --input-views with three images or history directories")
        _validate_view_paths(args.input_views, n_previous=args.n_previous)
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

    if len(paths) != NUM_VIEWS:
        raise ValueError(f"Genie Envisioner requires exactly {NUM_VIEWS} input views, got {len(paths)}")
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


def _validate_view_paths(paths: list[Path] | tuple[Path, ...], *, n_previous: int) -> bool:
    """Return whether the views are numbered history directories, rejecting mixed inputs."""
    if len(paths) != NUM_VIEWS:
        raise ValueError(f"Genie Envisioner requires exactly {NUM_VIEWS} views, got {len(paths)}")
    if all(path.is_file() for path in paths):
        return False
    if not all(path.is_dir() for path in paths):
        raise ValueError("input views must all be images or all be history directories")
    expected = {f"{index}.png" for index in range(n_previous)}
    for view in paths:
        actual = {path.name for path in view.glob("*.png")}
        if actual != expected:
            raise ValueError(
                f"history directory {view} needs exactly {sorted(expected)}; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        for name in expected:
            if not (view / name).is_file() or (view / name).stat().st_size == 0:
                raise ValueError(f"history image is missing or empty: {view / name}")
    return True


def _explicit_three_view_history(paths: list[Path] | tuple[Path, ...], *, n_previous: int, height: int, width: int):
    """Match official ``load_images``: sorted 0..3 PNGs and OpenCV linear resize."""
    import cv2
    import numpy as np

    views = []
    for view in paths:
        frames = []
        original_shape = None
        for index in range(n_previous):
            path = view / f"{index}.png"
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.ndim != 3 or bgr.shape[0] < 1 or bgr.shape[1] < 1:
                raise ValueError(f"cannot decode history image: {path}")
            if original_shape is None:
                original_shape = bgr.shape[:2]
            elif bgr.shape[:2] != original_shape:
                raise ValueError(f"history image size changed within {view}: {path}")
            rgb = bgr[:, :, ::-1]
            resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(np.transpose(resized.astype(np.float32) / 255.0 * 2.0 - 1.0, (2, 0, 1)))
        views.append(np.stack(frames, axis=1))
    return np.stack(views, axis=0)


def _transformer_config() -> dict[str, object]:
    """Return the released GE-base LTX transformer architecture."""
    return {
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 64,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 2048,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1.0e-6,
        "num_attention_heads": 32,
        "num_layers": 28,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
        "action_expert": False,
        "action_in_channels": 14,
        "action_num_attention_heads": 16,
        "action_attention_head_dim": 32,
    }


def run(args: argparse.Namespace) -> Path:
    """Generate a three-view GE-base rollout with released local checkpoints."""
    _validate_args(args)
    _ensure_vendored_runtime_importable()

    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from einops import rearrange
    from safetensors.torch import load_file
    from transformers import T5EncoderModel, T5Tokenizer

    from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
    from models.pipeline.custom_pipeline import CustomPipeline
    from worldfoundry.core.media.codecs.video import save_video_h264

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Genie Envisioner inference cannot run")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.mixed_precision
    ]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_model_dir = args.base_model_dir.expanduser().resolve()
    print("Loading local LTX tokenizer and text encoder...")
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_dir, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_dir,
        subfolder="text_encoder",
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()

    print("Loading local LTX VAE...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        base_model_dir,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=dtype).eval()
    vae.enable_slicing()
    vae.enable_tiling()

    print("Loading Genie Envisioner transformer checkpoint...")
    transformer = LTXVideoTransformer3DModel(**_transformer_config())
    state_dict = load_file(str(args.checkpoint_path.expanduser().resolve()), device="cpu")
    transformer.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    transformer = transformer.to(device=device, dtype=dtype).eval()

    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = CustomPipeline(scheduler, vae, text_encoder, tokenizer, transformer)

    history_dirs = args.input_mode == "explicit-three-view" and _validate_view_paths(
        args.input_views, n_previous=args.n_previous
    )
    if history_dirs:
        views = _explicit_three_view_history(
            args.input_views, n_previous=args.n_previous, height=args.height, width=args.width
        )
        # Upstream passes float32 normalized observations to the sampler.
        history = torch.from_numpy(views.copy()).to(device=device)
    else:
        views = (
            _explicit_three_views(args.input_views, height=args.height, width=args.width)
            if args.input_mode == "explicit-three-view"
            else _synthetic_three_views(args.input_image, height=args.height, width=args.width)
        )
        view_tensor = torch.from_numpy(views.copy()).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
        view_tensor = view_tensor / 127.5 - 1.0
        history = view_tensor.unsqueeze(2).repeat(1, 1, args.n_previous, 1, 1)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        predictions = pipe.infer(
            image=history,
            n_prev=args.n_previous,
            prompt=[args.prompt],
            negative_prompt="",
            height=args.height,
            width=args.width,
            chunk=args.latent_chunk,
            frame_rate=30,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            decode_timestep=args.decode_timestep,
            decode_noise_scale=args.decode_noise_scale,
            return_dict=False,
            return_init=True,
            n_view=NUM_VIEWS,
            return_action=False,
            return_video=True,
            noise_seed=args.seed,
            pixel_wise_timestep=True,
            n_chunk=1,
        )[0]

    generated = predictions["video"].clamp(-1, 1)
    video = ((generated + 1.0) * 127.5).to(torch.uint8)
    video = rearrange(video, "v c t h w -> t h (v w) c", v=NUM_VIEWS).cpu().numpy()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_h264(video, output_path, fps=args.fps)
    print(f"Genie Envisioner output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
