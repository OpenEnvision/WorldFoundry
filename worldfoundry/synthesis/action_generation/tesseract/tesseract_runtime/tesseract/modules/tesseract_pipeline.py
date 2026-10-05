import gc
import math
import torch
from typing import Optional, Union
from torch import randn as randn_tensor
import torch
import numpy as np
from transformers import T5Tokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
    CogVideoXDDIMScheduler,
)

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import is_peft_version, USE_PEFT_BACKEND, replace_example_docstring
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps, retrieve_latents

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False


class TesserActImageToDepthVideoPipeline(CogVideoXImageToVideoPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        image = image.unsqueeze(2)  # [B, C, F, H, W]

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            B, C, F, H, W = image.shape
            image = image.reshape(B * 2, C // 2, F, H, W)
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

        image_latents = torch.cat(image_latents, dim=0).to(dtype)
        B, C, F, H, W = image_latents.shape
        image_latents = image_latents.reshape(B // 2, C * 2, F, H, W)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        if not self.vae.config.invert_scale_latents:
            image_latents = self.vae_scaling_factor_image * image_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            image_latents = 1 / self.vae_scaling_factor_image * image_latents

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Select the first frame along the second dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        B, C, F, H, W = latents.shape
        latents = latents.reshape(B * 2, C // 2, F, H, W)
        frames = self.vae.decode(latents).sample
        B, C, F, H, W = frames.shape
        image_rgb = frames[0]  # [C, F, H, W]
        image_depth = frames[1]
        frames = torch.cat([image_rgb, image_depth], dim=-1).unsqueeze(0)
        return frames

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name=None, **kwargs):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer` and
        `self.text_encoder`. All kwargs are forwarded to `self.lora_state_dict`. See
        [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] for more details on how the state
        dict is loaded into `self.transformer`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
        base_state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items() if "lora" not in k}

        self.load_lora_into_transformer(
            lora_state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        self.transformer.load_state_dict(base_state_dict, strict=False)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=prompt_embeds.dtype)

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                # ==== depth here! ====
                _, _, C, _, _ = latent_model_input.shape
                latent_model_input_rgb = latent_model_input[:, :, : C // 2].clone()
                latent_model_input_depth = latent_model_input[:, :, C // 2 :].clone()
                latent_image_input_rgb = latent_image_input[:, :, : C // 2].clone()
                latent_image_input_depth = latent_image_input[:, :, C // 2 :].clone()
                latent_model_input = torch.cat(
                    [
                        latent_model_input_rgb,
                        latent_image_input_rgb,
                        latent_model_input_depth,
                        latent_image_input_depth,
                    ],
                    dim=2,
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


class TesserActImageToDepthNormalVideoPipeline(CogVideoXImageToVideoPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        image = image.unsqueeze(2)  # [B, C, F, H, W]

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            B, C, F, H, W = image.shape
            image = image.reshape(B * 3, C // 3, F, H, W)
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

        image_latents = torch.cat(image_latents, dim=0).to(dtype)
        B, C, F, H, W = image_latents.shape
        image_latents = image_latents.reshape(B // 3, C * 3, F, H, W)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        if not self.vae.config.invert_scale_latents:
            image_latents = self.vae_scaling_factor_image * image_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            image_latents = 1 / self.vae_scaling_factor_image * image_latents

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Select the first frame along the second dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents, image_latents

    # Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline.decode_latents
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        B, C, F, H, W = latents.shape
        latents = latents.reshape(B * 3, C // 3, F, H, W)
        # frames = self.vae.decode(latents).sample
        frames = []
        for i in range(3):
            frame = self.vae.decode(latents[[i]]).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        B, C, F, H, W = frames.shape
        image_rgb = frames[0]  # [C, F, H, W]
        image_depth = frames[1]
        normal_map = frames[2]
        frames = torch.cat([image_rgb, image_depth, normal_map], dim=-1).unsqueeze(0)
        return frames

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name=None, **kwargs):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer` and
        `self.text_encoder`. All kwargs are forwarded to `self.lora_state_dict`. See
        [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] for more details on how the state
        dict is loaded into `self.transformer`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
        base_state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items() if "lora" not in k}

        self.load_lora_into_transformer(
            lora_state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        self.transformer.load_state_dict(base_state_dict, strict=False)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        image_guidance_scale: float = 1,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_image_guidance = image_guidance_scale > 1.0

        # 3. Encode input prompt
        orig_prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        prompt_embeds = orig_prompt_embeds
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, orig_prompt_embeds], dim=0)
        if do_image_guidance:
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=prompt_embeds.dtype)

        latent_channels = self.transformer.config.in_channels // 2
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                if do_image_guidance:
                    latent_model_input = torch.cat([latent_model_input, latents], dim=0)
                    latent_image_input_drop = image_latents.clone()
                    latent_image_input_drop[:, 0] = 0
                    latent_image_input = torch.cat([latent_image_input, latent_image_input_drop], dim=0)

                # ==== depth normal here! ====
                _, _, C, _, _ = latent_model_input.shape
                latent_image_rgb = latent_image_input[:, :, : C // 3].clone()
                latent_image_depth = latent_image_input[:, :, C // 3 : C * 2 // 3].clone()
                latent_image_normal = latent_image_input[:, :, C * 2 // 3 :].clone()
                latent_model_input_rgb = latent_model_input[:, :, : C // 3].clone()
                latent_model_input_depth = latent_model_input[:, :, C // 3 : C * 2 // 3].clone()
                latent_model_input_normal = latent_model_input[:, :, C * 2 // 3 :].clone()
                latent_model_input = torch.cat(
                    [
                        latent_model_input_rgb,
                        latent_image_rgb,
                        latent_model_input_depth,
                        latent_image_depth,
                        latent_model_input_normal,
                        latent_image_normal,
                    ],
                    dim=2,
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                    image_guidance_scale = 1 + image_guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance and do_image_guidance:
                    noise_pred_image, noise_pred_text, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif do_image_guidance:
                    noise_pred_text, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_image + image_guidance_scale * (noise_pred_text - noise_pred_image)
                if do_image_guidance or do_classifier_free_guidance:
                    # no guidance for the first frame
                    noise_pred[:, 0] = noise_pred_text[:, 0]

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Free up memory
            del latent_model_input, latent_image_input, noise_pred
            gc.collect()
            torch.cuda.empty_cache()
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            W = video.shape[-2]
            # normalize normal map
            normal_map = (video[:, :, :, W // 3 * 2 :] - 0.5) * 2
            normal_map = (normal_map / (np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-8)) * 0.5 + 0.5
            video[:, :, :, W // 3 * 2 :] = normal_map
            # postprocess depth map
            depth_map = video[:, :, :, W // 3 : W // 3 * 2]
            depth_map = depth_map.mean(axis=-1, keepdims=True)
            video[:, :, :, W // 3 : W // 3 * 2] = depth_map
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
