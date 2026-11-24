"""Command-line entry point for MoVerse panorama inference."""

import argparse
from pathlib import Path

from .generate import generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perspective image to 360-degree panorama")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--back_prompt", default="")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument(
        "--basemodel_name_or_path",
        default="checkpoints/FLUX.1-Fill-dev",
    )
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--save_intermediate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panorama = generate(
        image=str(image_path),
        prompt=args.prompt,
        back_prompt=args.back_prompt,
        ckpt_dir=args.ckpt_dir,
        basemodel_name_or_path=args.basemodel_name_or_path,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        save_intermediate_dir=str(output_dir) if args.save_intermediate else None,
    )
    panorama.save(output_dir / f"{image_path.stem}_pano.png")


if __name__ == "__main__":
    main()
