"""Checkpoint-compatible GigaWorld-0 GR1 image-to-video launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NEGATIVE_PROMPT = (
    "static, motion blur, low resolution, grainy, pixelated, poorly lit, "
    "choppy motion, artifacts, unnatural transitions, visual noise, flickering"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-model-dir", type=Path, required=True)
    parser.add_argument("--text-encoder-model-dir", type=Path, required=True)
    parser.add_argument("--vae-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("natten", "torch"), default="natten")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the official GigaWorld-0 sampler requires a CUDA device")
    if args.height < 64 or args.width < 64 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be at least 64 and divisible by 16")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        raise ValueError("num_frames must be at least 5 and satisfy (num_frames - 1) % 4 == 0")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.guidance_scale < 1.0:
        raise ValueError("guidance_scale must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    if args.max_text_length < 8 or args.max_text_length > 512:
        raise ValueError("max_text_length must be between 8 and 512")
    required = (
        ("GigaWorld-0 transformer", args.transformer_model_dir / "diffusion_pytorch_model.safetensors"),
        ("GigaWorld-0 transformer config", args.transformer_model_dir / "config.json"),
        ("safe T5 encoder", args.text_encoder_model_dir / "model.safetensors"),
        ("T5 encoder config", args.text_encoder_model_dir / "config.json"),
        ("T5 tokenizer", args.text_encoder_model_dir / "spiece.model"),
        ("Wan VAE", args.vae_model_dir / "diffusion_pytorch_model.safetensors"),
        ("Wan VAE config", args.vae_model_dir / "config.json"),
        ("input image", args.input_image),
    )
    for label, path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")


def _transformer_config(path: Path, *, attention_backend: str) -> dict[str, object]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("_class_name") != "GigaWorld0Transformer3DModel":
        raise ValueError("transformer config does not describe GigaWorld0Transformer3DModel")
    if config.get("in_channels") != 17 or config.get("out_channels") != 16:
        raise ValueError("the GR1 image-conditioned checkpoint must use 17 input and 16 output channels")
    if attention_backend == "torch":
        config["natten_parameters"] = None
    return config


def _local_text_encoder(model_dir: Path, *, device, dtype, max_length: int):
    import torch
    from diffusers.utils.accelerate_utils import apply_forward_hook
    from transformers import T5EncoderModel, T5TokenizerFast

    class LocalT5TextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_length = max_length
            self.tokenizer = T5TokenizerFast.from_pretrained(
                model_dir, local_files_only=True
            )
            self.text_encoder = T5EncoderModel.from_pretrained(
                model_dir,
                dtype=dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            ).to(device=device, dtype=dtype).eval()

        @property
        def device(self):
            return self.text_encoder.device

        @property
        def dtype(self):
            return self.text_encoder.dtype

        @apply_forward_hook
        @torch.inference_mode()
        def encode_prompts(self, prompts):
            if isinstance(prompts, str):
                prompts = [prompts]
            if not prompts:
                raise ValueError("the prompt list is empty")
            encoded = self.tokenizer.batch_encode_plus(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_length=True,
            )
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)
            hidden = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
            return hidden

    return LocalT5TextEncoder()


def _condition_image(path: Path, *, height: int, width: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image_width, image_height = image.size
    # Match the official scripts/inference.py resize followed by center crop.
    # ImageOps.fit uses Lanczos by default and changes the released GR1 inputs.
    if height / image_height < width / image_width:
        resized_width = width
        resized_height = round(width / image_width * image_height)
    else:
        resized_height = height
        resized_width = round(height / image_height * image_width)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    ).crop((left, top, left + width, top + height))


def run(args: argparse.Namespace) -> Path:
    """Generate a short image-conditioned rollout with released local weights."""

    _validate_args(args)

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan

    from worldfoundry.core.io.video import save_video_h264
    from worldfoundry.synthesis.visual_generation.giga_world_0.giga_models_compat import (
        selective_giga_world_imports,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GigaWorld-0 inference cannot run")
    if args.attention_backend == "natten":
        try:
            import natten
        except ImportError as exc:
            raise RuntimeError(
                "the official GigaWorld-0 attention path requires a NATTEN wheel matching Torch and CUDA"
            ) from exc
        if not hasattr(natten.functional, "neighborhood_attention_generic"):
            raise RuntimeError(
                "GigaWorld-0 requires NATTEN >= 0.20 with stride-aware "
                "neighborhood_attention_generic; install the source-compatible release for this Torch runtime"
            )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.mixed_precision
    ]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    transformer_config = _transformer_config(
        args.transformer_model_dir,
        attention_backend=args.attention_backend,
    )
    with selective_giga_world_imports() as components:
        print("Loading safe local T5-11B encoder...")
        text_encoder = _local_text_encoder(
            args.text_encoder_model_dir,
            device=device,
            dtype=dtype,
            max_length=args.max_text_length,
        )
        print("Loading local Wan Diffusers VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            args.vae_model_dir,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device=device, dtype=dtype).eval()
        vae.enable_slicing()
        vae.enable_tiling()

        print(f"Loading GigaWorld-0 GR1 transformer ({args.attention_backend} attention)...")
        transformer_load_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if args.attention_backend == "torch":
            transformer_load_kwargs["natten_parameters"] = None
        transformer = components.transformer_class.from_pretrained(
            args.transformer_model_dir,
            **transformer_load_kwargs,
        ).to(device=device, dtype=dtype).eval()
        scheduler = components.scheduler_class(
            prediction_type="rf",
            solver_order=1,
            final_sigmas_type="sigma_min",
            sigma_data=1.0,
        )
        pipe = components.pipeline_class(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        ).to(device)
        image = _condition_image(args.input_image, height=args.height, width=args.width)
        with torch.inference_mode():
            generated = pipe(
                prompt=args.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                seed=args.seed,
                output_type="pil",
            )[0]

    frames = np.stack(
        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in generated],
        axis=0,
    )
    if frames.shape != (args.num_frames, args.height, args.width, 3):
        raise RuntimeError(
            "unexpected GigaWorld-0 video shape: "
            f"expected {(args.num_frames, args.height, args.width, 3)}, got {frames.shape}"
        )
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_h264(frames, output_path, fps=args.fps)
    print(f"GigaWorld-0 output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
