#!/usr/bin/env python3
"""Estimate a conditioning image's horizontal FOV with MoGe-2 (prints JSON).

The action model consumes camera intrinsics (``K_pix``) alongside the pose
trajectory; ``action_string_camera.default_k_pix`` otherwise guesses a fixed 70°
horizontal FOV. Feeding the *image's own* FOV makes the UCPE Plücker embedding
consistent with the first frame, so a given camera translation produces the pixel
flow the model expects instead of a systematically wide/narrow one.

Run as a subprocess so the ~1.4 GB ViT-L never shares
the generator's VRAM:

    python -m worldfoundry.synthesis.visual_generation.joyai_echo.echo_wm_runtime.helpers.moge_fov \
        --image frame.png --target-width 1280 --target-height 704

Output (stdout, one JSON object):
    {"fov_x_deg": 63.2, "fov_x_raw_deg": 71.5, "crop_factor": 0.83, ...}

``fov_x_raw_deg`` is MoGe's estimate for the image as given. ``fov_x_deg`` is the
value to actually use: the trainer's ``_encode_conditioning_image`` fits the image
to the target resolution by **resize-to-cover + center-crop**, so when the input
aspect is wider than the target, the sides are cropped away and the effective
horizontal FOV shrinks by ``a_target / a_input``. Inputs taller than the target
keep their full horizontal FOV (only the top/bottom are cropped).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


def effective_fov_x(fov_x_raw_deg: float, in_w: int, in_h: int, target_w: int, target_h: int) -> tuple[float, float]:
    """Horizontal FOV after resize-to-cover + center-crop to the target size.

    Returns ``(fov_x_eff_deg, crop_factor)`` where ``crop_factor`` is the retained
    fraction of the original horizontal extent (1.0 = nothing cropped away).
    """
    a_in = in_w / in_h
    a_target = target_w / target_h
    crop_factor = min(1.0, a_target / a_in)  # >1 would mean *adding* FOV, impossible
    fov_eff = 2.0 * math.degrees(math.atan(crop_factor * math.tan(math.radians(fov_x_raw_deg) / 2.0)))
    return fov_eff, crop_factor


def estimate(
    image_path: Path, model_name: str, device: str, target_w: int, target_h: int, resolution_level: int = 9
) -> dict:
    from moge.model.v2 import MoGeModel

    image = Image.open(image_path).convert("RGB")
    in_w, in_h = image.size
    dev = torch.device(device)
    model = MoGeModel.from_pretrained(model_name).to(dev).eval()

    tensor = torch.tensor(np.asarray(image) / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
    with torch.inference_mode():
        out = model.infer(tensor, resolution_level=resolution_level)

    # MoGe returns NORMALIZED intrinsics (fx = focal / width, principal point 0.5).
    k = out["intrinsics"].float().cpu().numpy()
    fx_n, fy_n = float(k[0, 0]), float(k[1, 1])
    fov_x_raw = 2.0 * math.degrees(math.atan(0.5 / fx_n))
    fov_y_raw = 2.0 * math.degrees(math.atan(0.5 / fy_n))
    fov_x_eff, crop_factor = effective_fov_x(fov_x_raw, in_w, in_h, target_w, target_h)

    return {
        "fov_x_deg": round(fov_x_eff, 3),
        "fov_x_raw_deg": round(fov_x_raw, 3),
        "fov_y_raw_deg": round(fov_y_raw, 3),
        "crop_factor": round(crop_factor, 4),
        "fx_normalized": round(fx_n, 6),
        "input_width": in_w,
        "input_height": in_h,
        "target_width": target_w,
        "target_height": target_h,
        # Focal the trainer will derive from fov_x_deg at the target resolution.
        "fx_pixels_at_target": round((target_w / 2.0) / math.tan(math.radians(fov_x_eff) / 2.0), 2),
        "model": model_name,
        "device": str(dev),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local model.pt")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--target-width", type=int, default=1280)
    p.add_argument("--target-height", type=int, default=704)
    p.add_argument(
        "--resolution-level", type=int, default=9, help="MoGe inference resolution level (higher = finer, slower)."
    )
    args = p.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    result = estimate(args.image, args.model, args.device, args.target_width, args.target_height, args.resolution_level)
    # JSON on stdout only; progress/warnings from torch go to stderr.
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
