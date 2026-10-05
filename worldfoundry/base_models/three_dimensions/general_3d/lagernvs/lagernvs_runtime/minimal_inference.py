# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal end-to-end inference script for LagerNVS.

This script demonstrates the full pipeline:
  1. Load input images
  2. Create a target camera trajectory (using VGGT for pose estimation)
  3. Download and load the LagerNVS checkpoint from HuggingFace
  4. Render novel views
  5. Save output as an MP4 video

Prerequisites:
  - GPU with CUDA support (bfloat16 on Ampere+ GPUs, float16 otherwise)
  - HuggingFace token with access to the gated model repo.
    Set via: export HF_TOKEN=hf_your_token_here
    See the LagerNVS supported-model page for gated Hugging Face access.
  - Internet access for downloading VGGT (~4GB) and the LagerNVS checkpoint.
    On Meta devvms, prefix the command with `with-proxy`.

Usage:
  python minimal_inference.py --images path/to/img1.png path/to/img2.png
  python minimal_inference.py --images images/input_000000.png images/input_000001.png

This script uses the general model (facebook/lagernvs_general_512) which supports
inference without known source camera poses. For posed-only models (Re10k, DL3DV),
use run_eval.py with ground truth camera poses instead.
"""

import argparse
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
for _path in (
    _THIS_FILE.parents[6],
    _THIS_FILE.parents[3] / "point_clouds" / "vggt",
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import torch
from huggingface_hub import hf_hub_download
from inference_utils import create_target_camera_path, load_and_preprocess_images_compat, render_chunked, save_video
from models.encoder_decoder import EncDec_VitB8


def _resolve_checkpoint(model_repo):
    """Helper function to resolve checkpoint.

    Args:
        model_repo: The model repo.
    """
    path = Path(model_repo).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        ckpt = path / "model.pt"
        if ckpt.is_file():
            return str(ckpt)
        raise FileNotFoundError(f"LagerNVS checkpoint directory is missing model.pt: {path}")
    return hf_hub_download(model_repo, filename="model.pt")


def main():
    """Main."""
    parser = argparse.ArgumentParser(description="LagerNVS minimal inference")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to 1 or more input images",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=100,
        help="Number of frames to render (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path (default: output_video.mp4)",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="facebook/lagernvs_general_512",
        help="HuggingFace repo ID for the checkpoint",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="bidirectional_cross_attention",
        choices=["bidirectional_cross_attention", "full_attention"],
        help=(
            "Attention type for the renderer. "
            "Use 'full_attention' for Re10k model, "
            "'bidirectional_cross_attention' for General/DL3DV models."
        ),
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target size in pixels (default: 512)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="resize",
        choices=["resize", "square_crop"],
        help=(
            "Image preprocessing mode. "
            "'resize' preserves aspect ratio with longer side = target_size (General model). "
            "'square_crop' center-crops to square then resizes to target_size (256 models)."
        ),
    )
    args = parser.parse_args()

    model_repo_path = Path(args.model_repo).expanduser()
    assert model_repo_path.exists() or args.model_repo == "facebook/lagernvs_general_512", (
        f"Only the general model (facebook/lagernvs_general_512) is supported "
        f"for inference without known camera poses. Got: {args.model_repo}. "
        f"Posed-only models (Re10k, DL3DV) are intended only for benchmarking. "
        f"Use them in run_eval.py with ground truth camera poses."
    )

    # -------------------------------------------------------------------------
    # 1. Device and dtype setup
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 requires Ampere+ GPUs (Compute Capability 8.0+), fall back to float16
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"Device: {device}, dtype: {dtype}")

    # -------------------------------------------------------------------------
    # 2. Load and preprocess input images
    # -------------------------------------------------------------------------
    # load_and_preprocess_images preprocesses input images.
    # "resize" mode: longer side = target_size, aspect ratio preserved (General 512 model).
    # "square_crop" mode: center-crop to square, resize to target_size x target_size (256 models).
    # Returns tensor of shape (num_views, 3, H, W).
    image_names = args.images
    num_cond_views = len(image_names)

    images = load_and_preprocess_images_compat(
        image_names, mode=args.mode, target_size=args.target_size, patch_size=8
    )
    # Add batch dimension: (num_views, 3, H, W) -> (1, num_views, 3, H, W)
    images = images.to(device).unsqueeze(0)
    image_size_hw = (images.shape[-2], images.shape[-1])
    print(f"Loaded {num_cond_views} images, shape: {images.shape}")

    # -------------------------------------------------------------------------
    # 3. Create target camera trajectory
    # -------------------------------------------------------------------------
    # create_target_camera_path uses VGGT (downloaded automatically, ~4GB) to
    # estimate approximate input camera poses, then interpolates a smooth
    # B-spline camera path through them (multi-view) or creates a forward
    # dolly motion (single-view).
    #
    # Returns:
    #   rays:       (1, num_cond_views + video_length, 6, H, W) Plucker ray coords
    #               Conditioning views get zero rays (model doesn't use input poses).
    #   cam_tokens: (1, num_cond_views + video_length, 11) camera tokens encoding
    #               scene scale normalization info.
    print("Creating target camera path (downloads VGGT on first run)...")
    rays, cam_tokens = create_target_camera_path(
        image_names,
        args.video_length,
        num_cond_views,
        image_size_hw,
        device,
        dtype,
        mode=args.mode,
    )
    print(f"Rays shape: {rays.shape}, cam_tokens shape: {cam_tokens.shape}")

    # -------------------------------------------------------------------------
    # 4. Load the LagerNVS model
    # -------------------------------------------------------------------------
    # EncDec_VitB8 = EncoderDecoder with ViT-B/8 config:
    #   - Encoder: VGGT-based feature extractor (pretrained_vggt=False here
    #     because the full model checkpoint already includes trained encoder weights)
    #   - Decoder: 12-layer transformer renderer, patch_size=8, hidden_size=768
    #
    # attention_to_features_type controls how the renderer attends to encoder
    # features:
    #   "bidirectional_cross_attention" — General and DL3DV models
    #   "full_attention"                — Re10k model
    print(f"Loading model from {args.model_repo}...")
    model = EncDec_VitB8(
        pretrained_vggt=False,
        attention_to_features_type=args.attention_type,
    )

    # Download checkpoint from gated HuggingFace repo (requires HF_TOKEN)
    ckpt_path = _resolve_checkpoint(args.model_repo)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # -------------------------------------------------------------------------
    # 5. Render novel views
    # -------------------------------------------------------------------------
    # render_chunked processes target views in chunks of 16 to manage GPU memory.
    # It internally uses torch.amp.autocast with bfloat16.
    #
    # Input tuple: (cond_images, rays, cam_tokens)
    #   cond_images: (B, num_cond_views, 3, H, W)
    #   rays:        (B, num_cond_views + video_length, 6, H, W)
    #   cam_tokens:  (B, num_cond_views + video_length, 11)
    #
    # Output: (B, video_length, 3, H, W) — rendered RGB frames
    print(f"Rendering {args.video_length} frames...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            video_out = render_chunked(
                model,
                (images, rays, cam_tokens),
                num_cond_views=num_cond_views,
            )
    print(f"Output video shape: {video_out.shape}")

    # -------------------------------------------------------------------------
    # 6. Save output video
    # -------------------------------------------------------------------------
    save_video(video_out[0], args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
