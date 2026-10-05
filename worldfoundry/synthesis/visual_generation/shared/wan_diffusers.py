"""Shared helpers for Diffusers-style Wan inference pipelines."""

from __future__ import annotations

import inspect
from typing import List, Optional, Union

import torch
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from torch.nn import functional as F


logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Configure an inference scheduler and return its effective timesteps."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    parameters = set(inspect.signature(scheduler.set_timesteps).parameters)
    if timesteps is not None:
        if "timesteps" not in parameters:
            raise ValueError(
                f"{scheduler.__class__} does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        num_inference_steps = len(scheduler.timesteps)
    elif sigmas is not None:
        if "sigmas" not in parameters:
            raise ValueError(
                f"{scheduler.__class__} does not support custom sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        num_inference_steps = len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


def resize_mask(
    mask: torch.Tensor,
    latent: torch.Tensor,
    process_first_frame_only: bool = True,
) -> torch.Tensor:
    """Resize a video mask to the temporal and spatial latent dimensions."""
    target_size = list(latent.shape[2:])
    if not process_first_frame_only:
        return F.interpolate(
            mask,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

    first_target = target_size.copy()
    first_target[0] = 1
    first = F.interpolate(
        mask[:, :, :1],
        size=first_target,
        mode="trilinear",
        align_corners=False,
    )
    if target_size[0] == 1:
        return first
    remaining_target = target_size.copy()
    remaining_target[0] -= 1
    remaining = F.interpolate(
        mask[:, :, 1:],
        size=remaining_target,
        mode="trilinear",
        align_corners=False,
    )
    return torch.cat((first, remaining), dim=2)


class WanDiffusersInferenceMixin:
    """Stateless inference operations shared by resident Wan pipelines."""

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).input_ids
        if (
            untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
            and not torch.equal(text_input_ids, untruncated_ids)
        ):
            removed = self.tokenizer.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logger.warning(
                "Prompt text after token %d was truncated: %s",
                max_sequence_length,
                removed,
            )

        sequence_lengths = attention_mask.gt(0).sum(dim=1).long()
        embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )[0].to(dtype=dtype, device=device)
        _, sequence_length, _ = embeddings.shape
        embeddings = embeddings.repeat(1, num_videos_per_prompt, 1).view(
            batch_size * num_videos_per_prompt,
            sequence_length,
            -1,
        )
        return [value[:length] for value, length in zip(embeddings, sequence_lengths)]

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
            if isinstance(negative_prompt, str):
                negative_prompt = batch_size * [negative_prompt]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` and `prompt` must have the same type."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    "`negative_prompt` and `prompt` must have the same batch size."
                )
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                "The generator list length must equal the effective batch size."
            )
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )
        if latents is None:
            latents = randn_tensor(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device)
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_vae_mode(self, value, *, device, dtype):
        if value is None:
            return None
        value = value.to(device=device, dtype=dtype)
        def encode_mode(item):
            encoded = self.vae.encode(item)
            posterior = getattr(encoded, "latent_dist", None)
            if posterior is None:
                # Diffusers VAEs can also be called with return_dict=False.
                if not isinstance(encoded, (tuple, list)) or not encoded:
                    raise TypeError(
                        f"Unsupported VAE encode output: {type(encoded).__name__}"
                    )
                posterior = encoded[0]
            return posterior.mode()

        return torch.cat(
            [encode_mode(item) for item in value.split(1)],
            dim=0,
        )

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
        noise_aug_strength,
    ):
        return (
            self._encode_vae_mode(mask, device=device, dtype=self.vae.dtype),
            self._encode_vae_mode(
                masked_image,
                device=device,
                dtype=self.vae.dtype,
            ),
        )

    def prepare_control_latents(
        self,
        control,
        control_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        return (
            self._encode_vae_mode(control, device=device, dtype=dtype),
            self._encode_vae_mode(control_image, device=device, dtype=dtype),
        )

    def prepare_extra_step_kwargs(self, generator, eta):
        parameters = set(inspect.signature(self.scheduler.step).parameters)
        kwargs = {}
        if "eta" in parameters:
            kwargs["eta"] = eta
        if "generator" in parameters:
            kwargs["generator"] = generator
        return kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 or width % 8:
            raise ValueError("`height` and `width` must be divisible by 8.")
        invalid_callbacks = [
            key
            for key in callback_on_step_end_tensor_inputs or ()
            if key not in self._callback_tensor_inputs
        ]
        if invalid_callbacks:
            raise ValueError(
                f"Unsupported callback tensor inputs: {invalid_callbacks}."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError("`prompt` must be a string or list.")
        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "`negative_prompt_embeds` cannot be combined with `prompt`."
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Pass either `negative_prompt` or `negative_prompt_embeds`, not both."
            )
        if (
            prompt_embeds is not None
            and negative_prompt_embeds is not None
            and prompt_embeds.shape != negative_prompt_embeds.shape
        ):
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape."
            )


__all__ = [
    "WanDiffusersInferenceMixin",
    "resize_mask",
    "retrieve_timesteps",
]
