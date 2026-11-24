"""Module for the CropResize operator implementation."""

from pathlib import Path

import torch
import torchvision.transforms as transforms  # for process_perception()
from PIL import Image

from .base_operator import BaseOperator


class CropResize:
    """
    Custom transform to resize and crop images to a target size while preserving aspect ratio.

    Resizes the image to ensure it covers the target dimensions, then center-crops to the exact size.
    Useful for preparing consistent input dimensions for video generation models.
    """

    def __init__(self, size=(704, 1216)):
        """
        Args:
            size (tuple): Target dimensions (height, width) for the output image
        """
        self.target_h, self.target_w = size

    def __call__(self, img):
        """
        Apply the transform to an image.

        Args:
            img (PIL.Image): Input image to transform

        Returns:
            PIL.Image: Resized and cropped image with target dimensions
        """
        # Get original image dimensions
        w, h = img.size

        # Calculate scaling factor to ensure image covers target size
        scale = max(
            self.target_w / w,  # Scale needed to cover target width
            self.target_h / h,  # Scale needed to cover target height
        )

        # Resize image while preserving aspect ratio
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)

        # Center-crop to exact target dimensions
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)


OPERATION_2_ACTION_DICT = {
    "forward": "w",
    "left": "a",
    "right": "d",
    "backward": "s",
    "camera_l": "left_rot",
    "camera_r": "right_rot",
    "camera_up": "up_rot",
    "camera_down": "down_rot",
}


def _to_pil_image(image):
    """Gracefully loads, converts, and unifies arbitrary inputs into RGB PIL Images."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(Path(image).expanduser()).convert("RGB")
    return image


class HunyuanGameCraftOperator(BaseOperator):
    """Operator interface for Hunyuan Game Craft multi-modal environments.

    Translates discrete game commands ("forward", "camera_l") into action vectors
    and handles unified pre-processing of visual state images to match the GameCraft model's
    strict VAE dimensionality.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        """Initialize the operator with specific configurations."""
        if operation_types is None:
            operation_types = ["action_instruction"]
        if interaction_template is None:
            interaction_template = list(OPERATION_2_ACTION_DICT)
            interaction_template.extend(
                action for action in OPERATION_2_ACTION_DICT.values() if action not in interaction_template
            )
        super().__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

    def check_interaction(self, interaction):
        """Verifies that a provided action command maps correctly to the game environment."""
        if interaction not in self.interaction_template:
            raise ValueError(f"{interaction} not in template")
        return True

    def get_interaction(self, interaction):
        """Process and append the interaction to the current sequence."""
        if not isinstance(interaction, list):
            interaction = [interaction]
        for act in interaction:
            self.check_interaction(act)
        self.current_interaction.append(interaction)

    def process_perception(self, image, output_H, output_W, process_model):
        """Transforms raw image inputs into properly sized and normalized PyTorch tensors.

        Uses center-cropping to ensure that GameCraft models receive exact (output_H, output_W)
        shapes, normalizing raw 0-255 pixels into the [-1.0, 1.0] range expected by VAEs.
        """
        device = process_model.device
        dtype = process_model.weight_dtype

        # Define image transformation pipeline for input reference images
        closest_size = (output_H, output_W)
        ref_image_transform = transforms.Compose(
            [
                CropResize(closest_size),
                transforms.CenterCrop(closest_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
            ]
        )

        raw_ref_images = [_to_pil_image(image)]

        # Apply transformations and prepare tensor for model input
        ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in raw_ref_images]
        ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(device)

        # Encode reference images to latent space using VAE
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            if process_model.args.cpu_offload:
                # Move VAE components to GPU temporarily for encoding
                process_model.vae.quant_conv.to("cuda")
                process_model.vae.encoder.to("cuda")

            # Enable tiling for VAE to handle large images efficiently
            process_model.pipeline.vae.enable_tiling()

            # Encode image to latents and scale by VAE's scaling factor
            raw_last_latents = (
                process_model.vae.encode(ref_images_pixel_values).latent_dist.sample().to(dtype=dtype)
            )  # Shape: (B, C, F, H, W)
            raw_last_latents.mul_(process_model.vae.config.scaling_factor)
            raw_ref_latents = raw_last_latents.clone()

            # Clean up
            process_model.pipeline.vae.disable_tiling()
            if process_model.args.cpu_offload:
                # Move VAE components back to CPU after encoding
                process_model.vae.quant_conv.to("cpu")
                process_model.vae.encoder.to("cpu")

        visual_context = {
            "ref_images": raw_ref_images,
            "last_latents": raw_last_latents,
            "ref_latents": raw_ref_latents,
        }

        return visual_context

    def process_interaction(self):
        """Compiles the sequence of raw environmental actions into model-friendly action prompts.

        Resolves string mappings like "forward" -> "w" and appends the resolved sequence
        to the interaction history for stateful generation queries.
        """
        if len(self.current_interaction) == 0:
            raise ValueError("No interaction to process")
        now_interaction = self.current_interaction[-1]
        self.interaction_history.append(now_interaction)
        return [OPERATION_2_ACTION_DICT.get(act, act) for act in now_interaction]
