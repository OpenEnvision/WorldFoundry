import contextlib
import os
import time
import types
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Default inference paths. Keep empty for the open-source config; pass paths via
# CLI flags or the run_inference_openloop*.sh environment variables.
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = ""
DEFAULT_CHECKPOINT = ""
DEFAULT_NORM_STATS = ""
DEFAULT_DEVICE = "cuda:0"

import copy  # noqa: E402 - Follows module-level setup, as upstream.
import html  # noqa: E402 - Follows module-level setup, as upstream.
import json  # noqa: E402 - Follows module-level setup, as upstream.
from collections.abc import Callable  # noqa: E402 - Follows module-level setup, as upstream.
from typing import Any  # noqa: E402 - Follows module-level setup, as upstream.

import numpy as np  # noqa: E402 - Follows module-level setup, as upstream.
import PIL  # noqa: E402 - Follows module-level setup, as upstream.
import regex as re  # noqa: E402 - Follows module-level setup, as upstream.
import torch  # noqa: E402 - Follows module-level setup, as upstream.
import torch.nn.functional as torch_F  # noqa: E402 - Follows module-level setup, as upstream.
from diffusers.callbacks import (  # noqa: E402 - Follows module-level setup, as upstream.
    MultiPipelineCallbacks,
    PipelineCallback,
)
from diffusers.image_processor import PipelineImageInput  # noqa: E402 - Follows module-level setup, as upstream.
from diffusers.loaders import WanLoraLoaderMixin  # noqa: E402 - Follows module-level setup, as upstream.
from diffusers.models import AutoencoderKLWan  # noqa: E402 - Follows module-level setup, as upstream.
from diffusers.pipelines.pipeline_utils import (  # noqa: E402 - Follows module-level setup, as upstream.
    DiffusionPipeline,
)
from diffusers.schedulers import (  # noqa: E402 - Follows module-level setup, as upstream.
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import (  # noqa: E402 - Follows module-level setup, as upstream.
    is_ftfy_available,
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402 - Follows module-level setup, as upstream.
from diffusers.video_processor import VideoProcessor  # noqa: E402 - Follows module-level setup, as upstream.
from PIL import Image  # noqa: E402 - Follows module-level setup, as upstream.
from transformers import (  # noqa: E402 - Follows module-level setup, as upstream.
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    UMT5EncoderModel,
)

from .transformer import CasualWorldActionTransformer  # noqa: E402 - Follows module-level setup, as upstream.
from .transformer_mot import CasualWorldActionTransformer_MoT  # noqa: E402 - Follows module-level setup, as upstream.

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _combine_classifier_free_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply the standard CFG equation to action-flow predictions."""

    return unconditional + guidance_scale * (conditional - unconditional)


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WAPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs: ClassVar[list] = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components: ClassVar[list] = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: CasualWorldActionTransformer | CasualWorldActionTransformer_MoT = None,
        transformer_2: CasualWorldActionTransformer | CasualWorldActionTransformer_MoT = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor
        self.action_scheduler = copy.deepcopy(scheduler)

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        action_chunk: torch.Tensor | None = None,
        action_dim: int | None = 14,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            action_shape = (batch_size, action_chunk, action_dim)
            action = randn_tensor(action_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if self.config.expand_timesteps:
            video_condition = image

        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask, action

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        action_chunk: int,
        state: torch.Tensor | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006 - Keeps the upstream default object and signature.
        max_sequence_length: int = 512,
        action_dim: int = 32,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.action_scheduler.set_timesteps(num_inference_steps, device=device)
        action_timesteps = self.action_scheduler.timesteps
        assert torch.all(timesteps == action_timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )
        state = state.unsqueeze(0).to(device=device, dtype=self.dtype)
        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
            action_chunk,
            action_dim=action_dim,
        )
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask, action = latents_outputs
        else:
            latents, condition = latents_outputs

        # 6. Denoising loop
        action = action.to(dtype=transformer_dtype, device=device)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        if not self.config.expand_timesteps:
            raise NotImplementedError("The action-only fast path expects expand_timesteps=True.")
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        ref_latents_for_action = condition[:, :, :1].to(transformer_dtype)
        empty_noisy_latents_for_action = condition[:, :, 1:1].to(transformer_dtype)
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        num_clean_latent_tokens = frame_per_tokens
        action_timestep_template = torch.zeros(
            1,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=ref_latents_for_action.device,
            dtype=ref_latents_for_action.dtype,
        )
        action_noise_t_index = num_state_tokens + num_clean_latent_tokens

        cache_models = [self.transformer]
        if getattr(self, "transformer_2", None) is not None:
            cache_models.append(self.transformer_2)
        old_action_only_prefix_cache_enabled = [
            getattr(model, "_enable_action_only_prefix_cache", False) for model in cache_models
        ]
        for model in cache_models:
            model._enable_action_only_prefix_cache = True
            if hasattr(model, "reset_action_only_prefix_cache"):
                model.reset_action_only_prefix_cache()

        try:
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    if boundary_timestep is None or t >= boundary_timestep:
                        # wan2.1 or high-noise stage in wan2.2
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                    else:
                        # low-noise stage in wan2.2
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2

                    timestep = action_timestep_template.clone()
                    timestep[:, action_noise_t_index:] = t.to(timestep.dtype)

                    skip_cache_context = (
                        getattr(current_model, "_enable_action_only_prefix_cache", False)
                        and getattr(self, "_skip_action_only_cache_context", True)
                        and not self.do_classifier_free_guidance
                    )
                    cache_context = (
                        contextlib.nullcontext() if skip_cache_context else current_model.cache_context("cond")
                    )
                    with cache_context:
                        if getattr(self, "_torch_compile_mark_step", False):
                            mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                            if mark_step is not None:
                                mark_step()

                        action_pred = current_model(
                            ref_latents=ref_latents_for_action,
                            noisy_latents=empty_noisy_latents_for_action,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                            action=action,
                            state=state,
                            action_only=True,
                        )

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            action_uncond = current_model(
                                ref_latents=ref_latents_for_action,
                                noisy_latents=empty_noisy_latents_for_action,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                return_dict=False,
                                action=action,
                                state=state,
                                action_only=True,
                            )
                        action_pred = _combine_classifier_free_guidance(
                            action_pred,
                            action_uncond,
                            current_guidance_scale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    action = self.action_scheduler.step(action_pred, t, action, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()
        finally:
            for model, old_enabled in zip(cache_models, old_action_only_prefix_cache_enabled, strict=False):
                model._enable_action_only_prefix_cache = old_enabled
                if hasattr(model, "reset_action_only_prefix_cache"):
                    model.reset_action_only_prefix_cache()

        if not return_dict:
            return action


from torchvision.transforms import InterpolationMode  # noqa: E402 - Follows the logging setup.
from torchvision.transforms import functional as F  # noqa: E402 - Follows the logging setup.


def process_images(input_images, dst_width, dst_height):
    height = input_images.height
    width = input_images.width
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))  # noqa: RUF046 - Keeps the explicit int conversion for non-builtin operands.
    input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
    # center crop
    x1 = (new_width - dst_width) // 2
    y1 = (new_height - dst_height) // 2
    input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
    return input_images


def get_ref_image_3views(images, dst_size, layout="tshape"):
    dst_width, dst_height = dst_size
    img_front, img_left, img_right = images

    if layout == "tshape":
        top_h = dst_height // 2
        bottom_h = dst_height - top_h
        left_w = dst_width // 2
        right_w = dst_width - left_w

        cam_high = process_images(img_front, dst_width=dst_width, dst_height=top_h)
        cam_left = process_images(img_left, dst_width=left_w, dst_height=bottom_h)
        cam_right = process_images(img_right, dst_width=right_w, dst_height=bottom_h)
        out = Image.new("RGB", (dst_width, dst_height))
        out.paste(cam_high, (0, 0))
        out.paste(cam_left, (0, top_h))
        out.paste(cam_right, (left_w, top_h))
    elif layout in {"horizontal"}:
        target_h = int(img_front.height)
        target_w = int(img_front.width)
        img_front_r = F.resize(img_front, (target_h, target_w), InterpolationMode.BILINEAR)
        img_left_r = F.resize(img_left, (target_h, target_w), InterpolationMode.BILINEAR)
        img_right_r = F.resize(img_right, (target_h, target_w), InterpolationMode.BILINEAR)

        out = Image.new("RGB", (target_w * 3, target_h))
        out.paste(img_front_r, (0, 0))
        out.paste(img_left_r, (target_w, 0))
        out.paste(img_right_r, (target_w * 2, 0))
        out = F.resize(out, (dst_height, dst_width), InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return out


def _require_nonempty_path(value: str | None, name: str) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be set")
    return str(value)


def _local_path(value: str | os.PathLike[str] | None, name: str, *, directory: bool) -> str:
    path = Path(_require_nonempty_path(None if value is None else str(value), name)).expanduser().resolve()
    predicate = path.is_dir if directory else path.is_file
    if not predicate():
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{name} must reference an existing local {kind}: {path}")
    return str(path)


def get_task_dir_from_checkpoint(checkpoint: str) -> str:
    """Resolve experiment task dir from checkpoint path under .../<task>/models/..."""
    checkpoint_path = os.path.abspath(checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    parts = checkpoint_path.split(os.sep)
    if "models" in parts:
        models_idx = parts.index("models")
        return os.sep.join(parts[:models_idx])

    return os.path.abspath(os.path.join(checkpoint_path, os.pardir, os.pardir))


def get_step_subdir_from_checkpoint(checkpoint: str) -> str:
    """Extract step subdir name like step_50000 from checkpoint path."""
    checkpoint_path = os.path.abspath(checkpoint)
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    for part in reversed(checkpoint_path.split(os.sep)):
        match = re.match(r"checkpoint_.*_step_(\d+)$", part)
        if match:
            return f"step_{match.group(1)}"
        match = re.match(r"step_(\d+)$", part)
        if match:
            return f"step_{match.group(1)}"

    return os.path.basename(checkpoint_path)


def get_output_dirs_from_checkpoint(checkpoint: str) -> tuple[str, str]:
    task_dir = get_task_dir_from_checkpoint(checkpoint)
    step_subdir = get_step_subdir_from_checkpoint(checkpoint)
    save_dir = os.path.join(task_dir, "open_loopresults", step_subdir)
    ref_image_save_dir = os.path.join(task_dir, "visualization", step_subdir)
    return save_dir, ref_image_save_dir


def load_fixed_prompt_embedding(t5_path: str, device: str) -> torch.Tensor:
    t5 = torch.load(t5_path, map_location="cpu", weights_only=True)
    if isinstance(t5, dict):
        t5 = t5.get("t5_embedding", next(iter(t5.values()), None))
    if not isinstance(t5, torch.Tensor):
        raise TypeError(f"T5 embedding file must contain a tensor: {t5_path}")
    t5 = t5.float()

    if t5.ndim == 2:
        t5 = t5[:64]
        if t5.shape[0] < 64:
            t5 = torch_F.pad(t5, (0, 0, 0, 64 - t5.shape[0]), value=0.0)
        t5 = t5.unsqueeze(0)
    elif t5.ndim == 3:
        t5 = t5[:, :64]
        if t5.shape[1] < 64:
            t5 = torch_F.pad(t5, (0, 0, 0, 64 - t5.shape[1]), value=0.0)
    else:
        raise ValueError(f"Unsupported T5 embedding shape from {t5_path}: {tuple(t5.shape)}")

    return t5.to(device)


def resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    normalized = str(dtype_name).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model dtype: {dtype_name}")
    return aliases[normalized]


def compile_policy_action_blocks(policy, mode="reduce-overhead", fullgraph=False, scope="action-blocks"):
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile")

    compiled = []
    for name in ("transformer", "transformer_2"):
        module = getattr(policy, name, None)
        if module is None:
            continue
        if scope == "action-stack":
            fn = getattr(module, "forward_action_stack_with_prefix_cache", None)
            if fn is None:
                continue
            module._compiled_forward_action_stack_with_prefix_cache = torch.compile(fn, mode=mode, fullgraph=fullgraph)
            compiled.append(f"{name}.forward_action_stack_with_prefix_cache")
            continue

        blocks = getattr(module, "blocks", None)
        if blocks is None:
            continue
        compiled_count = 0
        for block in blocks:
            fn = getattr(block, "forward_action_with_prefix_cache", None)
            if fn is None:
                continue
            block.forward_action_with_prefix_cache = torch.compile(fn, mode=mode, fullgraph=fullgraph)
            compiled_count += 1
        if compiled_count:
            compiled.append(f"{name}.blocks.forward_action_with_prefix_cache[{compiled_count}]")
    policy._torch_compile_mark_step = bool(compiled)
    return compiled


def get_policy(
    checkpoint=DEFAULT_CHECKPOINT,
    base_model=DEFAULT_BASE_MODEL,
    norm_stats=DEFAULT_NORM_STATS,
    data_paths=None,
    data_idx=1,
    device=DEFAULT_DEVICE,
    fixed_t5_path=None,
    model_dtype="bf16",
    compile_transformer=True,
    compile_mode="reduce-overhead",
    compile_fullgraph=False,
    compile_scope="action-blocks",
    seed=None,
    enable_model_cpu_offload=False,
):
    checkpoint = _local_path(checkpoint, "checkpoint", directory=True)
    base_model = _local_path(base_model, "base_model", directory=True)
    norm_stats = _local_path(norm_stats, "norm_stats", directory=False)
    data_paths = [str(Path(path).expanduser().resolve()) for path in (data_paths or ())]
    if fixed_t5_path is not None:
        fixed_t5_path = _local_path(fixed_t5_path, "fixed_t5_path", directory=False)

    if device.startswith("cuda"):
        torch.cuda.set_device(device)

    torch_dtype = resolve_torch_dtype(model_dtype)
    print(f"Loading base model from: {base_model}")
    print(f"Loading checkpoint from: {checkpoint}")
    print(f"Loading model dtype: {torch_dtype}")
    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=torch_dtype,
        local_files_only=True,
        use_safetensors=True,
    )
    transformer = CasualWorldActionTransformer_MoT.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        local_files_only=True,
        use_safetensors=True,
    )
    transformer.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe = WAPipeline.from_pretrained(
        base_model,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if enable_model_cpu_offload:
        if not str(device).startswith("cuda"):
            raise ValueError("model CPU offload requires a CUDA policy device")
        gpu_id = torch.device(device).index
        pipe.enable_model_cpu_offload(gpu_id=torch.cuda.current_device() if gpu_id is None else gpu_id)
    else:
        pipe.to(device=device, dtype=torch_dtype)
    pipe._model_dtype_name = str(torch_dtype)
    if compile_transformer:
        print(
            f"Compiling action-only MoT blocks with torch.compile(mode={compile_mode!r}, "
            f"scope={compile_scope!r}, fullgraph={compile_fullgraph})"
        )
        compiled_modules = compile_policy_action_blocks(
            pipe, mode=compile_mode, fullgraph=compile_fullgraph, scope=compile_scope
        )
        print(f"Compiled modules: {compiled_modules}")

    print(f"Loading norm stats from: {norm_stats}")
    with open(norm_stats) as f:
        stats_dict = json.load(f)

    dst_size = (320, 384)
    action_chunk = 48
    guidance_scale = 0.0
    num_inference_steps = 10

    state_min = torch.tensor(stats_dict["norm_stats"]["observation.state"]["q01"])[..., :14].to(device=device)
    state_max = torch.tensor(stats_dict["norm_stats"]["observation.state"]["q99"])[..., :14].to(device=device)

    delta_min = torch.tensor(stats_dict["norm_stats"]["action"]["q01"][:14])[..., :14].to(device=device)
    delta_max = torch.tensor(stats_dict["norm_stats"]["action"]["q99"][:14])[..., :14].to(device=device)
    eps = 1e-8
    state_range = (state_max - state_min).clamp_min(eps)
    delta_range = (delta_max - delta_min).clamp_min(eps)

    inference_generator = None
    if seed is not None:
        inference_generator = torch.Generator(device=device).manual_seed(int(seed))

    if fixed_t5_path is None and data_paths:
        fixed_t5_path = os.path.join(data_paths[0], "t5_embedding", f"episode_{int(data_idx):06d}.pt")

    fixed_prompt_embedding = None
    if fixed_t5_path:
        if not os.path.isfile(fixed_t5_path):
            print(f"Fixed T5 embedding not found, requests must provide prompt_embedding or prompt: {fixed_t5_path}")
        else:
            fixed_prompt_embedding = load_fixed_prompt_embedding(fixed_t5_path, device)
            print(f"Loaded fixed T5 embedding from: {fixed_t5_path}, shape={tuple(fixed_prompt_embedding.shape)}")

    def inference(self, data):
        images = {
            "observation.images.cam_high": data["observation.images.cam_high"],  # 3 H W, tensor float64
            "observation.images.cam_left_wrist": data["observation.images.cam_left_wrist"],  # 3 H W, tensor float64
            "observation.images.cam_right_wrist": data["observation.images.cam_right_wrist"],  # 3 H W, tensor float64
        }

        state = data["observation.state"].to(device)

        pil_images = [
            PIL.Image.fromarray(
                (images["observation.images.cam_high"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ),
            PIL.Image.fromarray(
                (images["observation.images.cam_left_wrist"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ),
            PIL.Image.fromarray(
                (images["observation.images.cam_right_wrist"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ),
        ]
        ref_image = get_ref_image_3views(pil_images, dst_size)

        state = state[..., :14]

        quiet = bool(data.get("_quiet", False))
        norm_state = ((state - state_min) / state_range) * 2 - 1

        norm_state = norm_state.to(device)
        if norm_state.ndim == 1:
            norm_state = norm_state.unsqueeze(0)
        norm_state = torch_F.pad(norm_state, (0, 32 - norm_state.shape[-1]), value=0.0)
        prompt_embedding = data.get("prompt_embedding", None)
        prompt = data.get("prompt", None)
        if prompt_embedding is None and prompt is None:
            prompt_embedding = fixed_prompt_embedding
        if prompt_embedding is None and prompt is None:
            raise ValueError(
                "Request did not include prompt_embedding or prompt, and no fixed T5 embedding was loaded."
            )
        if prompt_embedding is not None:
            prompt_embedding = prompt_embedding.to(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        model_start_time = time.perf_counter()
        pred_action = pipe(
            height=dst_size[1],
            width=dst_size[0],
            action_chunk=action_chunk,
            state=norm_state,
            num_frames=5,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=ref_image,
            return_dict=False,
            prompt_embeds=prompt_embedding,
            prompt=prompt,
            action_dim=32,
            generator=inference_generator,
        )
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
        model_end_time = time.perf_counter()
        self._last_model_inference_s = model_end_time - model_start_time
        if not quiet:
            print(f"Model inference time: {self._last_model_inference_s:.6f} seconds", flush=True)

        pred_action = pred_action[..., :14]
        pred_action = ((pred_action + 1) / 2) * delta_range + delta_min
        pred_action = pred_action.cpu().numpy()
        mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
        pred_action = pred_action[0] + state.repeat(action_chunk, 1).cpu().numpy() * mask
        return pred_action

    pipe.inference = types.MethodType(inference, pipe)
    return pipe
