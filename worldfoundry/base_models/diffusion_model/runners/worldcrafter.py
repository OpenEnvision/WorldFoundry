# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
import html

from itertools import accumulate

from typing import Any, Callable

import numpy as np

import regex as re

import torch

from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.image_processor import PipelineImageInput

from diffusers.loaders import HeliosLoraLoaderMixin as _BaseLoraLoaderMixin

from diffusers.models import AutoencoderKLWan

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.utils import (
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

from diffusers.utils.torch_utils import randn_tensor

from diffusers.video_processor import VideoProcessor

from ..models.networks.worldcrafter.ucpe.bridge import build_ucpe_attention_kwargs_for_chunk

from .helios_output import HeliosPipelineOutput as WorldCrafterPipelineOutput

from ..schedulers.worldcrafter import WorldCrafterScheduler

from ..models.networks.worldcrafter.transformer import WorldCrafterTransformer3DModel


from .helios import HeliosPipeline, XLA_AVAILABLE, optimized_scale, calculate_shift, EXAMPLE_DOC_STRING
logger = logging.get_logger(__name__)
if XLA_AVAILABLE:
    from .helios import xm

def _render_repencoder_memory_latents(
    *,
    memory_provider: Any,
    generated_latents: torch.Tensor,
    camera_trajectory: dict[str, Any],
    chunk_index: int,
    num_latent_frames_per_chunk: int,
    vae_scale_factor_temporal: int,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    if chunk_index <= 0:
        raise ValueError("RepEncoder rendering is only valid after chunk 0")
    if generated_latents.ndim != 5 or generated_latents.shape[2] == 0:
        raise RuntimeError(
            "RepEncoder requires a non-empty [B,C,T,H,W] bank of already generated WorldCrafter latents"
        )
    render_memory = getattr(memory_provider, "render_memory", None)
    if not callable(render_memory):
        raise TypeError(
            "memory_provider must expose a callable render_memory(...) method"
        )

    recent_latents = generated_latents[:, :, -1:, :, :]
    try:
        memory_latents = render_memory(
            generated_latents=generated_latents,
            recent_latents=recent_latents,
            camera_trajectory=camera_trajectory,
            chunk_index=chunk_index,
            num_latent_frames_per_chunk=num_latent_frames_per_chunk,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            generator=generator,
        )
    except Exception as exc:
        raise RuntimeError(
            f"RepEncoder memory rendering failed for chunk_index={chunk_index}"
        ) from exc

    expected_shape = (
        generated_latents.shape[0],
        generated_latents.shape[1],
        4,
        generated_latents.shape[3],
        generated_latents.shape[4],
    )
    if not isinstance(memory_latents, torch.Tensor):
        raise TypeError(
            "memory_provider.render_memory(...) must return a torch.Tensor, "
            f"got {type(memory_latents)!r}"
        )
    if tuple(memory_latents.shape) != expected_shape:
        raise ValueError(
            "RepEncoder memory must have shape [B,C,4,H,W]; "
            f"expected {expected_shape}, got {tuple(memory_latents.shape)}"
        )
    if memory_latents.device != generated_latents.device:
        raise ValueError(
            "RepEncoder memory must stay on the WorldCrafter latent device; "
            f"expected {generated_latents.device}, got {memory_latents.device}"
        )
    if not memory_latents.is_floating_point():
        raise TypeError(
            f"RepEncoder memory must be floating point, got {memory_latents.dtype}"
        )
    if not torch.isfinite(memory_latents).all():
        raise FloatingPointError(
            f"RepEncoder memory contains non-finite values at chunk_index={chunk_index}"
        )
    return memory_latents


class WorldCrafterPipeline(HeliosPipeline):
    """Camera-queryable memory rollout over the shared Helios components."""
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 132,
        num_inference_steps: int = 50,
        sigmas: list[float] = None,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: (
            Callable[[int, int], None]
            | PipelineCallback
            | MultiPipelineCallbacks
            | None
        ) = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        callback_on_chunk_end: Callable[[int, torch.Tensor], None] | None = None,
        callback_on_chunk_state: Callable[[int, dict[str, Any]], None] | None = None,
        resume_state: dict[str, Any] | None = None,
        stop_after_chunk: int | None = None,
        max_sequence_length: int = 512,
        # ------------ I2V ------------
        image: PipelineImageInput | None = None,
        image_latents: torch.Tensor | None = None,
        fake_image_latents: torch.Tensor | None = None,
        add_noise_to_image_latents: bool = True,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        # ------------ V2V ------------
        video: PipelineImageInput | None = None,
        video_latents: torch.Tensor | None = None,
        add_noise_to_video_latents: bool = True,
        video_noise_sigma_min: float = 0.111,
        video_noise_sigma_max: float = 0.135,
        # ------------ Interactive ------------
        use_interpolate_prompt: bool = False,
        interpolate_time_list: list = [7, 7, 7],
        interpolation_steps: int = 3,
        # ------------ Stage 1 ------------
        memory_size: int = 4,
        history_sizes: list = [2, 1],
        num_latent_frames_per_chunk: int = 9,
        keep_first_frame: bool = True,
        is_skip_first_chunk: bool = False,
        # ------------ Camera control ------------
        camera_trajectory: dict[str, Any] | None = None,
        # ------------ RepEncoder 3D memory ------------
        memory_provider: Any | None = None,
        # ------------ Stage 2 ------------
        is_enable_stage2: bool = False,
        pyramid_num_stages: int = 3,
        pyramid_num_inference_steps_list: list = [10, 10, 10],
        # ------------ CFG Zero ------------
        use_zero_init: bool | None = True,
        zero_steps: int | None = 1,
        # ------------ DMD ------------
        is_amplify_first_chunk: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` <= `1`).
            height (`int`, defaults to `384`):
                The height in pixels of the generated image.
            width (`int`, defaults to `640`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `132`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                Video output format: `"np"`, `"pt"`, or `"pil"`; `"latent"` returns latent tensors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WorldCrafterPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WorldCrafterPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WorldCrafterPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the only element contains the generated video batch. No safety-classification flags are returned.
        """

        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")
        use_fast = bool(self.config.is_distilled)
        if use_fast:
            if not is_enable_stage2 or guidance_scale != 1.0:
                raise ValueError("Fast inference requires pyramid sampling and CFG=1")
            if pyramid_num_inference_steps_list is not None:
                raise ValueError("Fast steps are owned by the checkpoint DMD contract")
        elif camera_trajectory is not None and is_enable_stage2:
            raise ValueError("Base camera inference requires stage1 sampling")
        if memory_size != 4:
            raise ValueError(
                f"RepEncoder memory contract requires memory_size=4, got {memory_size}"
            )
        if num_latent_frames_per_chunk != 9:
            raise ValueError(
                "RepEncoder target slots [2,4,6,8] require num_latent_frames_per_chunk=9, "
                f"got {num_latent_frames_per_chunk}"
            )

        requested_window_num_frames = (
            num_latent_frames_per_chunk - 1
        ) * self.vae_scale_factor_temporal + 1
        requested_num_chunks = max(
            1,
            (max(num_frames, 1) + requested_window_num_frames - 1)
            // requested_window_num_frames,
        )
        if use_interpolate_prompt:
            requested_num_chunks = max(requested_num_chunks, sum(interpolate_time_list))
        if requested_num_chunks > 1:
            if camera_trajectory is None:
                raise ValueError(
                    "Multi-chunk RepEncoder inference requires a global metric camera trajectory"
                )
            if memory_provider is None:
                raise ValueError("Multi-chunk inference requires memory_provider")

        history_sizes = sorted(history_sizes, reverse=True)  # From big to small
        assert (
            memory_size <= num_latent_frames_per_chunk
        ), f"memory_size={memory_size} must be <= num_latent_frames_per_chunk={num_latent_frames_per_chunk}"

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            image,
            video,
            use_interpolate_prompt,
            num_videos_per_prompt,
            interpolate_time_list,
            interpolation_steps,
            guidance_scale,
        )

        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        vae_dtype = self.vae.dtype

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, self.vae.dtype)

        # 2. Define call parameters
        if use_interpolate_prompt or (prompt is not None and isinstance(prompt, str)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        if use_interpolate_prompt:
            interpolate_interval_idx = None
            interpolate_embeds = None
            interpolate_cumulative_list = list(accumulate(interpolate_time_list))

        all_prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            if use_interpolate_prompt:
                negative_prompt_embeds = negative_prompt_embeds[0].unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare image or video
        if image is not None:
            image = self.video_processor.preprocess(image, height=height, width=width)
            image_latents, fake_image_latents = self.prepare_image_latents(
                image,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=image_latents,
                fake_latents=fake_image_latents,
            )

        if image_latents is not None and add_noise_to_image_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator)
                * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma
                * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )
            fake_image_noise_sigma = (
                torch.rand(1, device=device, generator=generator)
                * (video_noise_sigma_max - video_noise_sigma_min)
                + video_noise_sigma_min
            )
            fake_image_latents = (
                fake_image_noise_sigma
                * randn_tensor(
                    fake_image_latents.shape, generator=generator, device=device
                )
                + (1 - fake_image_noise_sigma) * fake_image_latents
            )

        if video is not None:
            video = self.video_processor.preprocess_video(
                video, height=height, width=width
            )
            image_latents, video_latents = self.prepare_video_latents(
                video,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=video_latents,
            )

        if video_latents is not None and add_noise_to_video_latents:
            image_noise_sigma = (
                torch.rand(1, device=device, generator=generator)
                * (image_noise_sigma_max - image_noise_sigma_min)
                + image_noise_sigma_min
            )
            image_latents = (
                image_noise_sigma
                * randn_tensor(image_latents.shape, generator=generator, device=device)
                + (1 - image_noise_sigma) * image_latents
            )

            noisy_latents_chunks = []
            num_latent_chunks = video_latents.shape[2] // num_latent_frames_per_chunk
            for i in range(num_latent_chunks):
                chunk_start = i * num_latent_frames_per_chunk
                chunk_end = chunk_start + num_latent_frames_per_chunk
                latent_chunk = video_latents[:, :, chunk_start:chunk_end, :, :]

                chunk_frames = latent_chunk.shape[2]
                frame_sigmas = (
                    torch.rand(chunk_frames, device=device, generator=generator)
                    * (video_noise_sigma_max - video_noise_sigma_min)
                    + video_noise_sigma_min
                )
                frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)

                noisy_chunk = (
                    frame_sigmas
                    * randn_tensor(
                        latent_chunk.shape, generator=generator, device=device
                    )
                    + (1 - frame_sigmas) * latent_chunk
                )
                noisy_latents_chunks.append(noisy_chunk)
            video_latents = torch.cat(noisy_latents_chunks, dim=2)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        window_num_frames = (
            num_latent_frames_per_chunk - 1
        ) * self.vae_scale_factor_temporal + 1
        num_latent_chunk = max(
            1, (num_frames + window_num_frames - 1) // window_num_frames
        )
        history_video = None
        total_generated_latent_frames = 0

        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )
        if fake_image_latents is not None:
            history_latents = torch.cat([history_latents, fake_image_latents], dim=2)
            total_generated_latent_frames += 1
        if video_latents is not None:
            history_frames = history_latents.shape[2]
            video_frames = video_latents.shape[2]
            if video_frames < history_frames:
                keep_frames = history_frames - video_frames
                history_latents = torch.cat(
                    [history_latents[:, :, :keep_frames, :, :], video_latents], dim=2
                )
            else:
                history_latents = video_latents
            total_generated_latent_frames += video_latents.shape[2]

        generated_memory_latents = history_latents[:, :, :0, :, :]

        start_chunk = 0
        if resume_state is not None:
            if resume_state.get("format") != "worldcrafter_chunk_state_v1":
                raise ValueError("unsupported WorldCrafter resume-state format")
            start_chunk = int(resume_state["next_chunk_index"])
            if start_chunk <= 0 or start_chunk >= num_latent_chunk:
                raise ValueError(
                    f"resume next_chunk_index must be in [1, {num_latent_chunk - 1}], got {start_chunk}"
                )
            generated_memory_latents = resume_state["generated_memory_latents"].to(
                device=device, dtype=torch.float32
            )
            expected_generated = start_chunk * num_latent_frames_per_chunk
            if tuple(generated_memory_latents.shape) != (
                batch_size,
                num_channels_latents,
                expected_generated,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ):
                raise ValueError(
                    "resume generated_memory_latents shape does not match next_chunk_index: "
                    f"{tuple(generated_memory_latents.shape)}"
                )
            history_latents = resume_state["history_latents"].to(
                device=device, dtype=torch.float32
            )
            expected_history = (
                batch_size,
                num_channels_latents,
                sum(history_sizes),
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            if tuple(history_latents.shape) != expected_history:
                raise ValueError(
                    f"resume history_latents must have shape {expected_history}, "
                    f"got {tuple(history_latents.shape)}"
                )
            saved_image_latents = resume_state.get("image_latents")
            if saved_image_latents is None:
                if keep_first_frame:
                    raise ValueError("resume state is missing fixed image_latents")
                image_latents = None
            else:
                image_latents = saved_image_latents.to(
                    device=device, dtype=torch.float32
                )
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    "resumable WorldCrafter inference requires one torch.Generator"
                )
            generator.set_state(resume_state["generator_state"].cpu())
            total_generated_latent_frames = expected_generated

        final_chunk_index = num_latent_chunk - 1
        if stop_after_chunk is not None:
            final_chunk_index = int(stop_after_chunk)
            if final_chunk_index < start_chunk or final_chunk_index >= num_latent_chunk:
                raise ValueError("stop_after_chunk is outside this inference interval")

        # 6. Denoising loop
        if use_interpolate_prompt:
            if num_latent_chunk < max(interpolate_cumulative_list):
                num_latent_chunk = sum(interpolate_cumulative_list)
                print(f"Update num_latent_chunk to: {num_latent_chunk}")

        if not is_enable_stage2:
            patch_size = self.transformer.config.patch_size
            image_seq_len = (
                num_latent_frames_per_chunk
                * (height // self.vae_scale_factor_spatial)
                * (width // self.vae_scale_factor_spatial)
                // (patch_size[0] * patch_size[1] * patch_size[2])
            )
            sigmas = (
                np.linspace(0.999, 0.0, num_inference_steps + 1)[:-1]
                if sigmas is None
                else sigmas
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

        for k in range(start_chunk, num_latent_chunk):
            if use_interpolate_prompt:
                assert num_latent_chunk >= max(interpolate_cumulative_list)

                current_interval_idx = 0
                for idx, cumulative_val in enumerate(interpolate_cumulative_list):
                    if k < cumulative_val:
                        current_interval_idx = idx
                        break

                if current_interval_idx == 0:
                    prompt_embeds = all_prompt_embeds[0].unsqueeze(0)
                else:
                    interval_start = interpolate_cumulative_list[
                        current_interval_idx - 1
                    ]
                    position_in_interval = k - interval_start

                    if position_in_interval < interpolation_steps:
                        if (
                            interpolate_embeds is None
                            or interpolate_interval_idx != current_interval_idx
                        ):
                            interpolate_embeds = self.interpolate_prompt_embeds(
                                prompt_embeds_1=all_prompt_embeds[
                                    current_interval_idx - 1
                                ].unsqueeze(0),
                                prompt_embeds_2=all_prompt_embeds[
                                    current_interval_idx
                                ].unsqueeze(0),
                                interpolation_steps=interpolation_steps,
                            )
                            interpolate_interval_idx = current_interval_idx

                        prompt_embeds = interpolate_embeds[position_in_interval]
                    else:
                        prompt_embeds = all_prompt_embeds[
                            current_interval_idx
                        ].unsqueeze(0)
            else:
                prompt_embeds = all_prompt_embeds

            is_first_chunk = k == 0
            is_second_chunk = k == 1
            if is_first_chunk:
                first_memory_latents = generated_memory_latents.new_zeros(
                    batch_size,
                    num_channels_latents,
                    memory_size,
                    height // self.vae_scale_factor_spatial,
                    width // self.vae_scale_factor_spatial,
                )
            else:
                first_memory_latents = _render_repencoder_memory_latents(
                    memory_provider=memory_provider,
                    generated_latents=generated_memory_latents,
                    camera_trajectory=camera_trajectory,
                    chunk_index=k,
                    num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                    vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                    generator=generator,
                )
            if keep_first_frame:
                if is_first_chunk:
                    history_sizes_first_chunk = [1] + history_sizes.copy()
                    history_latents_first_chunk = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_chunk),
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                        device=device,
                        dtype=torch.float32,
                    )
                    if fake_image_latents is not None:
                        history_latents_first_chunk = torch.cat(
                            [history_latents_first_chunk, fake_image_latents], dim=2
                        )
                    if video_latents is not None:
                        history_frames = history_latents_first_chunk.shape[2]
                        video_frames = video_latents.shape[2]
                        if video_frames < history_frames:
                            keep_frames = history_frames - video_frames
                            history_latents_first_chunk = torch.cat(
                                [
                                    history_latents_first_chunk[
                                        :, :, :keep_frames, :, :
                                    ],
                                    video_latents,
                                ],
                                dim=2,
                            )
                        else:
                            history_latents_first_chunk = video_latents

                    indices = torch.arange(
                        0,
                        sum(
                            [
                                1,
                                memory_size,
                                *history_sizes,
                                num_latent_frames_per_chunk,
                            ]
                        ),
                    )
                    (
                        indices_prefix,
                        indices_latents_memory,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split(
                        [1, memory_size, *history_sizes, num_latent_frames_per_chunk],
                        dim=0,
                    )
                    indices_latents_history_short = torch.cat(
                        [indices_prefix, indices_latents_history_1x], dim=0
                    )

                    latents_memory = first_memory_latents
                    latents_prefix, latents_history_mid, latents_history_1x = (
                        history_latents_first_chunk[
                            :, :, -sum(history_sizes_first_chunk) :
                        ].split(history_sizes_first_chunk, dim=2)
                    )
                    if image_latents is not None:
                        latents_prefix = image_latents
                    latents_history_short = torch.cat(
                        [latents_prefix, latents_history_1x], dim=2
                    )
                else:
                    indices = torch.arange(
                        0,
                        sum(
                            [
                                1,
                                memory_size,
                                *history_sizes,
                                num_latent_frames_per_chunk,
                            ]
                        ),
                    )
                    (
                        indices_prefix,
                        indices_latents_memory,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split(
                        [1, memory_size, *history_sizes, num_latent_frames_per_chunk],
                        dim=0,
                    )
                    indices_latents_history_short = torch.cat(
                        [indices_prefix, indices_latents_history_1x], dim=0
                    )

                    latents_prefix = image_latents
                    latents_memory = first_memory_latents
                    latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat(
                        [latents_prefix, latents_history_1x], dim=2
                    )
            else:
                indices = torch.arange(
                    0, sum([memory_size, *history_sizes, num_latent_frames_per_chunk])
                )
                (
                    indices_latents_memory,
                    indices_latents_history_mid,
                    indices_latents_history_short,
                    indices_hidden_states,
                ) = indices.split(
                    [memory_size, *history_sizes, num_latent_frames_per_chunk], dim=0
                )
                latents_memory = first_memory_latents
                latents_history_mid, latents_history_short = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)

            indices_hidden_states = indices_hidden_states.unsqueeze(0)
            indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
            indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
            indices_latents_memory = indices_latents_memory.unsqueeze(0)

            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            if not is_enable_stage2:
                self.scheduler.set_timesteps(
                    num_inference_steps, device=device, sigmas=sigmas, mu=mu
                )
                timesteps = self.scheduler.timesteps
                num_warmup_steps = (
                    len(timesteps) - num_inference_steps * self.scheduler.order
                )
                self._num_timesteps = len(timesteps)
            else:
                if use_fast:
                    from ..models.networks.worldcrafter.fast.contract import resolve_dmd_inference_trace

                    num_inference_steps = resolve_dmd_inference_trace(
                        self.dmd_timestep_contract,
                        latent_shape=latents.shape[1:],
                        history_tensors=(
                            latents_history_short,
                            latents_history_mid,
                            latents_memory,
                        ),
                        num_stages=pyramid_num_stages,
                    ).num_steps
                else:
                    num_inference_steps = sum(pyramid_num_inference_steps_list)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                current_attention_kwargs = attention_kwargs
                if camera_trajectory is not None and not use_fast:
                    current_attention_kwargs = dict(attention_kwargs or {})
                    ucpe_attention_kwargs = build_ucpe_attention_kwargs_for_chunk(
                        transformer=self.transformer,
                        camera_trajectory=camera_trajectory,
                        height=height,
                        width=width,
                        num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                        chunk_index=k,
                        vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                    )
                    if ucpe_attention_kwargs is None:
                        raise ValueError(
                            f"UCPE camera control could not be built for latent chunk {k}; "
                            "check pose length and camera adapter patching"
                        )
                    current_attention_kwargs.update(ucpe_attention_kwargs)
                if is_enable_stage2:
                    from .worldcrafter_fast import sample_fast

                    # Upsample block noise uses a separate default generator.
                    # Forwarding the trajectory generator here changes its RNG
                    # consumption and the generated video.
                    latents = sample_fast(
                        self,
                        latents=latents,
                        pyramid_num_stages=pyramid_num_stages,
                        pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                        prompt_embeds=prompt_embeds,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_memory,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_memory,
                        attention_kwargs=current_attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        camera_trajectory=camera_trajectory,
                        num_latent_frames_per_chunk=num_latent_frames_per_chunk,
                        chunk_index=k,
                        camera_restart_each_chunk=False,
                        ucpe_pixel_center=True,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )
                else:
                    latents = self.stage1_sample(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_memory,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_memory,
                        attention_kwargs=current_attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        generator=generator,
                        num_warmup_steps=num_warmup_steps,
                        # ------------ CFG Zero ------------
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        # ------------ Callback ------------
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )

                if keep_first_frame and (
                    (is_first_chunk and image_latents is None)
                    or (is_skip_first_chunk and is_second_chunk)
                ):
                    image_latents = latents[:, :, 0:1, :, :]

                generated_memory_latents = torch.cat(
                    [generated_memory_latents, latents], dim=2
                )

                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([history_latents, latents], dim=2)
                real_history_latents = history_latents[
                    :, :, -total_generated_latent_frames:
                ]
                current_latents = (
                    real_history_latents[:, :, -num_latent_frames_per_chunk:].to(
                        vae_dtype
                    )
                    / latents_std
                    + latents_mean
                )
                current_video = self.vae.decode(current_latents, return_dict=False)[0]

                if callback_on_chunk_end is not None:
                    callback_on_chunk_end(k, current_video)

                if callback_on_chunk_state is not None:
                    if not isinstance(generator, torch.Generator):
                        raise TypeError(
                            "resumable WorldCrafter inference requires one torch.Generator"
                        )
                    callback_on_chunk_state(
                        k,
                        {
                            "format": "worldcrafter_chunk_state_v1",
                            "completed_chunk_index": int(k),
                            "next_chunk_index": int(k + 1),
                            "generated_memory_latents": generated_memory_latents.detach().cpu(),
                            "history_latents": history_latents[
                                :, :, -sum(history_sizes) :
                            ]
                            .detach()
                            .cpu(),
                            "image_latents": (
                                image_latents.detach().cpu()
                                if image_latents is not None
                                else None
                            ),
                            "generator_state": generator.get_state().cpu(),
                        },
                    )

                if history_video is None:
                    history_video = current_video
                else:
                    history_video = torch.cat([history_video, current_video], dim=2)
                if k == final_chunk_index:
                    break

        self._current_timestep = None

        if output_type != "latent":
            if not use_fast:
                # Preserve the existing base output contract. Fast decodes each
                # complete 33-frame chunk independently; applying the latent
                # length rule again would drop valid RGB frames (330 -> 329).
                generated_frames = history_video.size(2)
                generated_frames = (
                    (generated_frames - 1)
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
                history_video = history_video[:, :, :generated_frames]
            video = self.video_processor.postprocess_video(
                history_video, output_type=output_type
            )
        else:
            video = real_history_latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WorldCrafterPipelineOutput(frames=video)
