"""
This module provides a runtime interface for the EasyAnimate image-to-video (I2V) model,
allowing for video generation based on text prompts and an initial image.

It handles model loading, configuration resolution, and the video generation pipeline,
including various options for memory optimization and text/image encoding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.evaluation.utils import worldfoundry_data_path

from .easyanimate_runtime import load_easyanimate_components


def default_model_path() -> str:
    """
    Return the default EasyAnimate I2V checkpoint asset path.

    Args:
        None: The path is derived from WORLDFOUNDRY_CKPT_DIR without importing the
        evaluation package.
    """
    # Resolve the root directory for checkpoints, defaulting to 'checkpoints' if the env var is not set.
    ckpt_root = Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", "checkpoints"))
    # Construct the full path to the specific EasyAnimate model.
    return str(
        ckpt_root.expanduser()
        / "hfd"
        / "alibaba-pai--EasyAnimateV5.1-7b-zh-InP"
    )


DEFAULT_EASYANIMATE_CONFIG_ROOT = worldfoundry_data_path(
    "models",
    "runtime", "configs",
    "easyanimate",
)
DEFAULT_EASYANIMATE_CONFIG = (
    "runtime/configs/easyanimate/easyanimate_video_v5.1_magvit_qwen.yaml"
)


def resolve_config_path(config_path: str | Path) -> str:
    """
    Resolve EasyAnimate config paths against data/models/runtime/configs.

    This function takes a config path and resolves it to an absolute path,
    handling cases where the path is relative to the `worldfoundry` data directory
    or refers to a file within the default EasyAnimate config root.

    Args:
        config_path: Absolute path, data/models runtime config path, or config file name.

    Returns:
        The resolved absolute path to the configuration file as a string.
    """
    path = Path(config_path)
    # If the path is already absolute, return it directly.
    if path.is_absolute():
        return str(path)

    # If the path starts with 'runtime_configs', it's relative to the 'models' directory within worldfoundry data.
    if path.parts and path.parts[:2] == ("runtime", "configs"):
        return str(worldfoundry_data_path("models", *path.parts))
    # If it's a single file name, assume it's in the default EasyAnimate config root.
    if len(path.parts) == 1:
        return str(DEFAULT_EASYANIMATE_CONFIG_ROOT / path)

    # For any other relative path, resolve it against the current working directory.
    return str(path.resolve())


class EasyAnimate:
    """
    A class to manage and run the EasyAnimate image-to-video generation pipeline.

    This class initializes the necessary components (transformer, VAE, tokenizers,
    text encoders, scheduler) from a specified model path and configuration,
    and provides a method to generate videos from text prompts and an input image.
    """

    def __init__(
        self,
        model_name: str,
        sample_size: list[int],
        video_length: int,
        fps: int,
        generation_type: Literal["t2v", "i2v"],
        guidance_scale: float = 6.0,
        seed: int = 43,
        num_inference_steps: int = 50,
        negative_prompt: str = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code.",
        config_path: str = DEFAULT_EASYANIMATE_CONFIG,
        model_path: str | None = None,
        sampler_name: str = "Flow",
        GPU_memory_mode: str = "model_cpu_offload",
        height: int | None = None,
        width: int | None = None,
        device: str = "cuda",
    ):
        """
        Build an in-tree EasyAnimate I2V runtime.

        Args:
            model_name: Registry name for the video model.
            sample_size: Output frame size as [height, width].
            video_length: Number of generated frames requested.
            fps: Output video frame rate metadata.
            generation_type: Supported generation mode, expected to be "i2v".
            guidance_scale: Classifier-free guidance scale.
            seed: CUDA generator seed used during sampling.
            num_inference_steps: Diffusion sampling step count.
            negative_prompt: Negative prompt passed to the EasyAnimate pipeline.
            config_path: In-tree EasyAnimate YAML config path under data/models/runtime/configs.
            model_path: External checkpoint asset directory. If None, uses default_model_path().
            sampler_name: Scheduler name used by EasyAnimate.
            GPU_memory_mode: EasyAnimate offload and qfloat8 mode selector.
            height: Optional per-call frame height override.
            width: Optional per-call frame width override.
            device: Execution device for offload hooks and sampling.
        """
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
            value = os.environ.get(name)
            if value is not None and value.strip().lower() not in {"1", "true", "yes", "on"}:
                raise ValueError(
                    f"EasyAnimate inference requires offline model loading; {name}={value!r} is not allowed."
                )
            os.environ[name] = "1"
        self.model_name = model_name
        # Apply height/width overrides to sample_size if provided.
        if height is not None and width is not None:
            sample_size = [int(height), int(width)]
        self.sample_size = sample_size
        self.video_length = int(video_length)
        if self.video_length < 1 or int(num_inference_steps) < 1:
            raise ValueError("EasyAnimate frame and inference step counts must be positive.")
        self.fps = fps
        self.generation_type = generation_type
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.negative_prompt = negative_prompt
        self.config_path = resolve_config_path(config_path)
        # Resolve the model path, using the default if not explicitly provided.
        requested_model_path = str(model_path or default_model_path())
        local_model_path = Path(requested_model_path).expanduser()
        if not local_model_path.is_dir():
            local_model_path = resolve_local_hf_model_path(requested_model_path)
        self.model_path = str(local_model_path.resolve())
        self.vae = None
        self.pipeline = None

        # Load dynamic components from the EasyAnimate runtime package.
        components = load_easyanimate_components()
        import torch

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            FlowMatchEulerDiscreteScheduler,
            PNDMScheduler,
        )
        from omegaconf import OmegaConf
        from transformers import (
            BertModel,
            BertTokenizer,
            CLIPImageProcessor,
            CLIPVisionModelWithProjection,
            Qwen2Tokenizer,
            Qwen2VLForConditionalGeneration,
            T5EncoderModel,
            T5Tokenizer,
        )

        # Extract specific components loaded from the EasyAnimate runtime.
        name_to_autoencoder_magvit = components.name_to_autoencoder_magvit
        name_to_transformer3d = components.name_to_transformer3d
        EasyAnimateInpaintPipeline = components.inpaint_pipeline_cls
        convert_weight_dtype_wrapper = components.convert_weight_dtype_wrapper
        self.get_image_to_video_latent = components.get_image_to_video_latent
        weight_dtype = torch.bfloat16

        # Load the model configuration from the resolved path.
        config = OmegaConf.load(self.config_path)
        model_path = self.model_path

        # Determine which Transformer3DModel to use based on the config.
        Choosen_Transformer3DModel = name_to_transformer3d[
            config["transformer_additional_kwargs"].get(
                "transformer_type", "Transformer3DModel"
            )
        ]

        # Prepare transformer additional arguments, including upcast_attention for float16.
        transformer_additional_kwargs = OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        )
        if weight_dtype == torch.float16:
            transformer_additional_kwargs["upcast_attention"] = True

        # Load the transformer model, applying qfloat8 dtype if specified for memory optimization.
        transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            model_path,
            subfolder="transformer",
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=torch.float8_e4m3fn
            if GPU_memory_mode == "model_cpu_offload_and_qfloat8"
            else weight_dtype,
            low_cpu_mem_usage=True,
        )

        # Determine and load the VAE model based on the config.
        Choosen_AutoencoderKL = name_to_autoencoder_magvit[
            config["vae_kwargs"].get("vae_type", "AutoencoderKL")
        ]
        vae = Choosen_AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            vae_additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)

        # Apply VAE upcasting if AutoencoderKLMagvit and float16 are used.
        if (
            config["vae_kwargs"].get("vae_type", "AutoencoderKL")
            == "AutoencoderKLMagvit"
            and weight_dtype == torch.float16
        ):
            vae.upcast_vae = True

        # Check configuration for multi-text encoders and LLM replacement.
        multi_text_encoder = config["text_encoder_kwargs"].get("enable_multi_text_encoder", False)
        replace_t5_to_llm = config["text_encoder_kwargs"].get("replace_t5_to_llm", False)

        # Load tokenizers based on multi-text encoder and LLM replacement settings.
        if multi_text_encoder:
            tokenizer = BertTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            if replace_t5_to_llm:
                tokenizer_2 = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_path, "tokenizer_2")
                )
            else:
                tokenizer_2 = T5Tokenizer.from_pretrained(
                    model_path, subfolder="tokenizer_2"
                )
        else:
            if replace_t5_to_llm:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_path, "tokenizer")
                )
            else:
                tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
            tokenizer_2 = None

        # Load text encoders based on multi-text encoder and LLM replacement settings.
        if multi_text_encoder:
            text_encoder = BertModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=weight_dtype
            )
            if replace_t5_to_llm:
                text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_path, "text_encoder_2"),
                    torch_dtype=weight_dtype,
                )
            else:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    model_path, subfolder="text_encoder_2", torch_dtype=weight_dtype
                )
        else:
            if replace_t5_to_llm:
                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_path, "text_encoder"),
                    torch_dtype=weight_dtype,
                )
            else:
                text_encoder = T5EncoderModel.from_pretrained(
                    model_path, subfolder="text_encoder", torch_dtype=weight_dtype
                )
            text_encoder_2 = None

        # Load CLIP image encoder and processor if enabled and required by the transformer.
        if transformer.config.in_channels != vae.config.latent_channels and config[
            "transformer_additional_kwargs"
        ].get("enable_clip_in_inpaint", True):
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_path, subfolder="image_encoder"
            ).to("cuda", weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_path, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None

        # Select the appropriate scheduler based on the provided sampler name.
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM": DDIMScheduler,
            "Flow": FlowMatchEulerDiscreteScheduler,
        }[sampler_name]

        # Load the scheduler and instantiate the EasyAnimate inpainting pipeline.
        scheduler = Choosen_Scheduler.from_pretrained(model_path, subfolder="scheduler")
        pipeline = EasyAnimateInpaintPipeline(
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
            clip_image_processor=clip_image_processor,
        )
        # Apply GPU memory optimization strategies based on the chosen mode.
        if GPU_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=self.device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            pipeline.enable_model_cpu_offload(device=self.device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
        else:  # Default to "model_cpu_offload" if not specified or unrecognized.
            pipeline.enable_model_cpu_offload(device=self.device)

        self.vae = vae
        self.pipeline = pipeline

    def generate_video(self, prompt: str, image_path: str | None):
        """
        Generate a video from a prompt and conditioning image.

        Args:
            prompt: Text prompt used for generation.
            image_path: Required input image path for image-to-video generation.

        Returns:
            A tensor or array representing the generated video frames.
        """
        import torch

        video_length = self.video_length

        validation_image_start = image_path
        validation_image_end = None

        # Adjust video_length based on VAE mini_batch_encoder for caching efficiency.
        if self.vae.cache_mag_vae:
            video_length = (
                int(
                    (video_length - 1 + self.vae.mini_batch_encoder - 1)
                    // self.vae.mini_batch_encoder
                    * self.vae.mini_batch_encoder
                )
                + 1
                if video_length != 1
                else 1
            )
        else:
            video_length = (
                int(
                    (video_length + self.vae.mini_batch_encoder - 1)
                    // self.vae.mini_batch_encoder
                    * self.vae.mini_batch_encoder
                )
                if video_length != 1
                else 1
            )
        # Prepare the input video latent, mask, and CLIP image features from the input image.
        input_video, input_video_mask, clip_image = self.get_image_to_video_latent(
            validation_image_start,
            validation_image_end,
            video_length=video_length,
            sample_size=self.sample_size,
        )

        # Execute the pipeline without gradient computation for inference.
        with torch.no_grad():
            output = self.pipeline(
                prompt,
                video_length=video_length,
                negative_prompt=self.negative_prompt,
                height=self.sample_size[0],
                width=self.sample_size[1],
                generator=torch.Generator(device=self.device).manual_seed(self.seed),
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
            )

        # Extract and normalize the generated frames from the pipeline output.
        sample = self._extract_pipeline_frames(output)
        return self._normalize_pipeline_frames(sample)[: self.video_length]

    @staticmethod
    def _extract_pipeline_frames(output: Any) -> Any:
        """
        Return the frame payload from EasyAnimate pipeline output variants.

        This static method attempts to find the video frames within various
        possible output structures from the EasyAnimate pipeline.
        It checks for attributes like 'videos', 'frames', or 'sample',
        and dictionary keys with the same names.

        Args:
            output: The raw output object or dictionary from the EasyAnimate pipeline.

        Returns:
            The extracted video frames, or the original output if no specific frame
            payload could be identified.
        """
        # Check for frame data in common attribute names.
        for attr in ("videos", "frames", "sample"):
            if hasattr(output, attr):
                value = getattr(output, attr)
                if value is not None:
                    return value
        # Check for frame data in common dictionary keys.
        if isinstance(output, dict):
            for key in ("videos", "frames", "sample"):
                value = output.get(key)
                if value is not None:
                    return value
        return output

    @staticmethod
    def _normalize_pipeline_frames(sample: Any) -> Any:
        """
        Normalize EasyAnimate output to a frame-first tensor/array (T, C, H, W).

        This static method handles various output formats from the pipeline (e.g.,
        torch tensors, numpy arrays, or wrapped lists/tuples) and ensures the
        final video frames are in a consistent (Frames, Channels, Height, Width)
        order, potentially removing batch dimensions if present.

        Args:
            sample: The raw frames output from the pipeline or extraction.

        Returns:
            The normalized video frames as a tensor or numpy array.
        """
        import numpy as np
        import torch

        # Define expected channel counts for video frames.
        channel_counts = {1, 3, 4}

        if torch.is_tensor(sample):
            # If 5D (Batch, Frames, Channels, Height, Width), remove the batch dimension.
            if sample.ndim == 5:
                sample = sample[0]
            # If 4D and channels are in the first dimension (C, F, H, W or C, H, W, F),
            # permute to (F, C, H, W) for consistency.
            if sample.ndim == 4 and sample.shape[0] in channel_counts:
                return sample.permute(1, 0, 2, 3)  # Assuming (C, F, H, W) -> (F, C, H, W) or (C, H, W, F) -> (F, C, H, W)
            return sample

        if isinstance(sample, np.ndarray):
            # If 5D (Batch, Frames, Channels, Height, Width), remove the batch dimension.
            if sample.ndim == 5:
                sample = sample[0]
            # If 4D and channels are in the first dimension (C, F, H, W or C, H, W, F),
            # permute to (F, C, H, W) for consistency.
            if sample.ndim == 4 and sample.shape[0] in channel_counts:
                return np.transpose(sample, (1, 0, 2, 3))  # Assuming (C, F, H, W) -> (F, C, H, W) or (C, H, W, F) -> (F, C, H, W)
            return sample

        # If the sample is a list or tuple containing a single item, unwrap and normalize recursively.
        if isinstance(sample, (list, tuple)) and len(sample) == 1:
            return EasyAnimate._normalize_pipeline_frames(sample[0])
        return sample
