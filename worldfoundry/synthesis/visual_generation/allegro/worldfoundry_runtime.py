"""
This module provides an interface for generating videos using the Allegro Text-to-Image-to-Video (TI2V) model.
It includes functionality for preprocessing conditioning images and a class to manage the Allegro model
lifecycle and video generation process.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from worldfoundry.core.utils.python.import_guard import third_party_lazy_import_guard

from .allegro_runtime import load_allegro_components


def preprocess_images(first_frame, last_frame, height, width, device, dtype, components):
    """
    Convert Allegro conditioning image paths into normalized tensors.

    This function loads one or two conditioning image paths, applies necessary
    transformations (resizing, normalization), and converts them into PyTorch
    tensors suitable for the Allegro TI2V pipeline.

    Args:
        first_frame: Required first conditioning frame path.
        last_frame: Optional last conditioning frame path.
        height: Target frame height.
        width: Target frame width.
        device: Torch device for output tensors.
        dtype: Torch dtype for output tensors.
        components: In-tree Allegro component bundle containing transformation classes.

    Returns:
        A dictionary containing:
            - "conditional_images": A list of processed PyTorch tensors representing the
              conditioning images.
            - "conditional_images_indices": A list of integers indicating the positions
              of the conditional images (0 for first, -1 for last).

    Raises:
        ValueError: If `first_frame` is not provided or if more than two conditioning
                    images are implied.
    """
    import numpy as np
    import torch
    from einops import rearrange
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import Lambda

    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    transform = transforms.Compose(
        [
            components.to_tensor_video_cls(),
            components.center_crop_resize_video_cls((height, width)),
            norm_fun,
        ]
    )
    images = []
    if first_frame is not None and len(first_frame.strip()) != 0:
        images.append(first_frame)
    else:
        raise ValueError("Allegro TI2V requires a first conditioning frame.")
    if last_frame is not None and len(last_frame.strip()) != 0:
        images.append(last_frame)

    # Determine indices for the conditional images based on how many are provided.
    if len(images) == 1:
        conditional_images_indices = [0]
    elif len(images) == 2:
        conditional_images_indices = [0, -1]
    else:
        raise ValueError("Allegro TI2V supports one or two conditioning images.")

    # Load images, convert to RGB, and then to PyTorch tensors.
    conditional_images = [Image.open(image).convert("RGB") for image in images]
    conditional_images = [
        torch.from_numpy(np.copy(np.array(image))) for image in conditional_images
    ]
    # Rearrange dimensions from HWC to CHW and add a batch dimension.
    conditional_images = [
        rearrange(image, "h w c -> c h w").unsqueeze(0)
        for image in conditional_images
    ]
    # Apply the defined transformations and move to the target device/dtype.
    conditional_images = [
        transform(image).to(device=device, dtype=dtype)
        for image in conditional_images
    ]

    return {
        "conditional_images": conditional_images,
        "conditional_images_indices": conditional_images_indices,
    }


class Allegro:
    """
    Manages the Allegro Text-to-Image-to-Video (TI2V) model lifecycle and video generation.

    This class handles the initialization of all necessary Allegro components,
    loading pretrained weights, and providing an interface for generating videos
    from a text prompt and an initial conditioning image.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        guidance_scale: float = 8,
        num_sampling_steps: int = 100,
        seed: int = 123,
        generation_type: Literal["i2v", "t2v"] = "i2v",
        device: str = "cuda",
        num_frames: int = 88,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int | None = None,
    ):
        """
        Build an in-tree Allegro TI2V runtime.

        Initializes the Allegro model by loading its components (VAE, text encoder,
        tokenizer, scheduler, transformer) from the specified path and setting
        up generation parameters.

        Args:
            model_name: Registry name for the video model.
            model_path: External checkpoint asset directory where model weights are stored.
            guidance_scale: Classifier-free guidance scale. Higher values encourage
                            output to be more aligned with the prompt.
            num_sampling_steps: Diffusion sampling step count. More steps generally
                                 lead to higher quality but take longer.
            seed: CUDA generator seed used during sampling for reproducibility.
            generation_type: Supported generation mode, expected to be "i2v" (Image-to-Video).
            device: Explicit execution device assigned by the Workspace.
            num_frames: Native checkpoint length, fixed at 88 frames.
            height: Native checkpoint height, fixed at 720 pixels.
            width: Native checkpoint width, fixed at 1280 pixels.
            num_inference_steps: Standard alias overriding num_sampling_steps.

        Raises:
            ValueError: If `generation_type` is not "i2v".
        """
        self.model_name = model_name
        self.generation_type = generation_type
        if self.generation_type != "i2v":
            raise ValueError("Allegro runtime currently supports only i2v generation.")
        self.model_path = str(Path(model_path))
        self.allegro_ti2v_pipeline = None
        self.num_sampling_steps = int(num_sampling_steps if num_inference_steps is None else num_inference_steps)
        self.num_frames = int(num_frames)
        self.height, self.width = int(height), int(width)
        if self.num_frames < 1 or self.num_sampling_steps < 1:
            raise ValueError("Allegro frame and inference step counts must be positive.")
        if (self.num_frames, self.height, self.width) != (88, 720, 1280):
            raise ValueError("Allegro TI2V supports 88 frames at 720x1280; other generation shapes are unsupported.")
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.device_spec = str(device)
        self.components = None

        # Diffusers imports ``AutoImageProcessor`` while constructing its
        # single-file loader mixins.  Resolve all cold dependency exports under
        # the same process-wide guard used by Studio pipeline imports.  The
        # guard ends before any checkpoint I/O so model loading on independent
        # GPUs remains parallel.
        with third_party_lazy_import_guard():
            import transformers

            getattr(transformers, "AutoImageProcessor")
            components = load_allegro_components()
            import torch
            from diffusers.schedulers import EulerAncestralDiscreteScheduler
            from transformers import T5EncoderModel, T5Tokenizer

        self.device = torch.device(self.device_spec)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        model_path = Path(self.model_path)
        # Keep the selected CUDA context active for upstream helpers that use
        # generic ``cuda`` while moving every owned component explicitly.
        with device_context:
            vae = components.autoencoder_cls.from_pretrained(
                model_path / "vae",
                torch_dtype=torch.float32,
            ).to(self.device)
            vae.eval()

            text_encoder = T5EncoderModel.from_pretrained(
                model_path / "text_encoder",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            text_encoder.eval()

            tokenizer = T5Tokenizer.from_pretrained(model_path / "tokenizer")
            scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path / "scheduler")

            transformer = components.transformer_cls.from_pretrained(
                model_path / "transformer",
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            transformer.eval()

            self.allegro_ti2v_pipeline = components.pipeline_cls(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                transformer=transformer,
                device=self.device,
            ).to(self.device)

        self.negative_prompt = (
            "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, normal "
            "quality, jpeg artifacts, signature, watermark, username, blurry."
        )
        self.positive_prompt = (
            "(masterpiece), (best quality), (ultra-detailed), (unwatermarked), \n"
            "{} \n"
            "emotional, harmonious, vignette, 4k epic detailed, shot on kodak, "
            "35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
        )
        self.components = components

    def generate_video(self, prompt: str, image_path: str | None):
        """
        Generate a video from a text prompt and an initial conditioning frame.

        This method preprocesses the input image and prompt, then calls the
        initialized Allegro TI2V pipeline to generate a video.

        Args:
            prompt: Text prompt used for generation, describing the desired video content.
            image_path: Required path to the first conditioning frame for TI2V generation.

        Returns:
            A PyTorch tensor representing the generated video, with pixel values
            normalized to [0, 1] and dimensions permuted to (T, C, H, W).

        Raises:
            ValueError: If the `prompt` is empty.
        """
        import torch

        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            pre_results = preprocess_images(
                image_path,
                "",  # Allegro TI2V currently uses only one conditioning image.
                height=self.height,
                width=self.width,
                device=self.device,
                dtype=torch.bfloat16,
                components=self.components,
            )

            prompt = str(prompt or "").lower().strip()
            if not prompt:
                raise ValueError("Allegro TI2V requires a non-empty prompt.")
            prompt = self.positive_prompt.format(prompt)

            out_video = self.allegro_ti2v_pipeline(
                prompt,
                negative_prompt=self.negative_prompt,
                conditional_images=pre_results["conditional_images"],
                conditional_images_indices=pre_results["conditional_images_indices"],
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_sampling_steps,
                guidance_scale=self.guidance_scale,
                max_sequence_length=512,
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
            ).video[0]

        # Normalize video pixel values from [0, 255] to [0, 1].
        out_video = out_video / 255.0
        # Permute dimensions from (T, H, W, C) to (T, C, H, W) for common video tensor format.
        return out_video.permute(0, 3, 1, 2)
