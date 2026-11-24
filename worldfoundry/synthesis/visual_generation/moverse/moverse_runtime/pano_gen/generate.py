"""Python API for 360° panorama generation.

Usage (programmatic):
    from pano_gen.generate import generate

    pano = generate(
        image="input.jpg",
        prompt="a bright breakfast alcove, curved booth seating...",
        back_prompt="with additional alcove seating and a small kitchen behind",
        ckpt_dir="checkpoints/gimbal360",
        basemodel_name_or_path="checkpoints/FLUX.1-Fill-dev",
    )
    pano.save("panorama.png")

Usage (CLI):
    python pano_gen/inference.py --image input.jpg --prompt "..." --ckpt_dir /path/to/ckpts
"""

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from worldfoundry.base_models.three_dimensions.general_3d.geocalib.autolevel import (
    FlowEstimator,
    preprocess_image,
)

from .camera_utils import Perspective
from .fluxfill_pipeline import FluxFillPipeline
from .prompt_builder import build_pano_prompt


class PerspectiveToERP:
    """Converts a perspective image to equirectangular conditioning image + mask."""

    def __init__(self, pano_height: int, pano_width: int, ckpt_path: str):
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(ckpt_path)
        self.to_tensor = T.ToTensor()

    def _load_model(self, ckpt_path: str) -> FlowEstimator:
        """Load AutoLevel FlowEstimator from checkpoint."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        rigid_filter_cfg = state.get("rigid_filter_cfg", {})
        model = FlowEstimator(rigid_filter_cfg=rigid_filter_cfg)
        model.backbone.load_state_dict(state["backbone"], strict=True)
        model.ll_enc.load_state_dict(state["ll_enc"], strict=True)
        model.decoder.load_state_dict(state["decoder"], strict=True)
        model = model.to(self.device).eval()
        return model

    @torch.no_grad()
    def estimate_camera(self, img: Image.Image) -> dict:
        """Estimate camera parameters (vFOV, pitch, roll) from a PIL image."""
        img_t = self.to_tensor(img)
        crop_proc, _ = preprocess_image(img_t, short_side=320, divisor=32)
        inp = crop_proc.unsqueeze(0).to(self.device)

        out = self.model(inp)
        params = out["params"][0]
        fov_deg, pitch_deg, roll_deg = params[0].item(), params[1].item(), params[2].item()

        return {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "fov": fov_deg,
        }

    def convert(self, img: Image.Image) -> tuple:
        """Convert a perspective PIL image to ERP conditioning + mask.

        Args:
            img: PIL.Image (RGB).

        Returns:
            conditioning_image (PIL.Image, RGB): gray background with projected perspective content.
            mask_image (PIL.Image, L): white = inpaint region, black = known region.
            cam_param (dict): estimated camera parameters.
        """
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        height_orig, width_orig = img_bgr.shape[:2]
        cam_param = self.estimate_camera(img)
        vertical_fov = cam_param["fov"]
        roll = cam_param["roll"]
        pitch = cam_param["pitch"]

        aspect_ratio = width_orig / height_orig
        resized_height = int((vertical_fov / 180) * self.pano_height)
        resized_width = int(resized_height * aspect_ratio)
        img_bgr = cv2.resize(img_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        projector = Perspective(img_bgr, vertical_fov, 0, pitch, roll, vfov=True)
        erp_raw, erp_mask = projector.GetEquirec(self.pano_height, self.pano_width)

        erp_mask = cv2.erode(erp_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5)

        conditioning = np.full((self.pano_height, self.pano_width, 3), 127, dtype=np.uint8)
        conditioning[erp_mask == 1] = erp_raw[erp_mask == 1].astype(np.uint8)

        inpaint_mask = (1 - erp_mask.astype(np.uint8)) * 255

        conditioning_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        return Image.fromarray(conditioning_rgb), Image.fromarray(inpaint_mask[:, :, 0]), cam_param


# ---------------------------------------------------------------------------
# Model caches (avoid reloading on repeated calls)
# ---------------------------------------------------------------------------
_erp_converter_cache: dict = {}
_pipeline_cache: dict = {}


def _get_erp_converter(pano_height: int, pano_width: int, ckpt_path: str) -> PerspectiveToERP:
    key = (pano_height, pano_width, ckpt_path)
    if key not in _erp_converter_cache:
        _erp_converter_cache[key] = PerspectiveToERP(pano_height, pano_width, ckpt_path)
    return _erp_converter_cache[key]


def _get_pipeline(basemodel_name_or_path: str, lora_path: str, dtype: torch.dtype) -> FluxFillPipeline:
    key = (basemodel_name_or_path, lora_path, dtype)
    if key not in _pipeline_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = FluxFillPipeline.from_pretrained(basemodel_name_or_path, torch_dtype=dtype).to(device)
        # In offline mode (HF_HUB_OFFLINE=1), diffusers requires explicit weight_name.
        # lora_path is a full file path, so split into directory + filename.
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path)
            lora_filename = os.path.basename(lora_path)
            pipeline.load_lora_weights(lora_dir, weight_name=lora_filename)
        else:
            pipeline.load_lora_weights(lora_path)
        pipeline.fuse_lora(lora_scale=1.0)
        _pipeline_cache[key] = pipeline
    return _pipeline_cache[key]


def generate(
    image,
    prompt: str,
    back_prompt: str = "",
    ckpt_dir: str = "checkpoints/gimbal360",
    basemodel_name_or_path: str = "checkpoints/FLUX.1-Fill-dev",
    resolution: int = 960,
    guidance_scale: float = 30.0,
    num_inference_steps: int = 30,
    seed: int = 42,
    mixed_precision: str = "bf16",
    save_intermediate_dir: str | None = None,
) -> Image.Image:
    """Generate a 360° panorama from a perspective image.

    Pipeline:
        1. AutoLevel estimates camera params (vFOV, pitch, roll) from the input image.
        2. Perspective → ERP projection creates a conditioning image + inpaint mask.
        3. FluxFill (FLUX.1-Fill-dev + LoRA) completes the panorama.

    Args:
        image: Path to image file (str) or PIL.Image.
        prompt: Text prompt for panorama generation (required).
        back_prompt: Optional description of unseen 270° content.
        ckpt_dir: Directory containing ``autolevel.pth`` and ``pytorch_lora_weights.safetensors``.
        basemodel_name_or_path: Path or HF repo for FLUX.1-Fill-dev base model.
        resolution: Panorama height in pixels (width = 2 × height).
        guidance_scale: FluxFill guidance scale.
        num_inference_steps: Number of denoising steps.
        seed: Random seed for reproducibility.
        mixed_precision: "fp16", "bf16", or "fp32".
        save_intermediate_dir: If set, save ERP conditioning image and mask to this directory.

    Returns:
        PIL.Image (RGB): The generated 360° panorama (resolution × 2×resolution).
    """
    from pathlib import Path

    ckpt_dir_path = Path(ckpt_dir)
    autolevel_ckpt = str(ckpt_dir_path / "autolevel.pth")
    lora_path = str(ckpt_dir_path / "pytorch_lora_weights.safetensors")

    # Load input image
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise ValueError("image must be a path (str) or PIL.Image")

    # Dtype
    dtype = torch.float32
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16

    pano_width = resolution * 2
    pano_height = resolution

    # --- Step 1: Perspective → ERP conditioning + mask ---
    erp_converter = _get_erp_converter(pano_height, pano_width, autolevel_ckpt)
    conditioning_image, mask_image, cam_param = erp_converter.convert(img)
    conditioning_image = conditioning_image.resize((pano_width, pano_height), Image.BILINEAR)
    mask_image = mask_image.resize((pano_width, pano_height), Image.NEAREST)

    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)
        conditioning_image.save(os.path.join(save_intermediate_dir, "cond.png"))
        mask_image.save(os.path.join(save_intermediate_dir, "mask.png"))

    # --- Step 2: Build prompt ---
    prompt_text = build_pano_prompt(prompt, back_prompt)

    # --- Step 3: FluxFill generation ---
    pipeline = _get_pipeline(basemodel_name_or_path, lora_path, dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)

    with torch.no_grad():
        result = pipeline(
            prompt=prompt_text,
            image=conditioning_image,
            mask_image=mask_image,
            height=pano_height,
            width=pano_width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

    return result
