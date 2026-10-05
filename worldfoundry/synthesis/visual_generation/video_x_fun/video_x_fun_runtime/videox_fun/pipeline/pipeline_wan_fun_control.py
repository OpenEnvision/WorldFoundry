import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, replace_example_docstring
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import T5Tokenizer

from worldfoundry.synthesis.visual_generation.shared.wan_diffusers import (
    WanDiffusersInferenceMixin,
    resize_mask,
    retrieve_timesteps,
)
from transformers import AutoTokenizer

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.video_x_fun import AutoencoderKLWan
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.video_x_fun import CLIPModel
from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.dreamx_world.text_encoder import WanT5EncoderModel
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.video_x_fun.transformer import WanTransformer3DModel
from worldfoundry.base_models.diffusion_model.schedulers.flow_dpm import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
)
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import (
    FlowUniPCMultistepScheduler,
)

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""
@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanFunControlPipeline(WanDiffusersInferenceMixin, DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, clip_image_encoder=clip_image_encoder, scheduler=scheduler
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        control_video: Union[torch.FloatTensor] = None,
        control_camera_video: Union[torch.FloatTensor] = None,
        start_image: Union[torch.FloatTensor] = None,
        ref_image: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Image = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
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
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        if comfyui_progressbar:
            pbar.update(1)

        # Prepare mask latent variables
        if control_camera_video is not None:
            control_latents = None
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        elif control_video is not None:
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"), height=height, width=width)
            control_video = control_video.to(dtype=torch.float32)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
            control_camera_latents = None
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_camera_latents = None

        if start_image is not None:
            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width)
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)

            start_image_latentes = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            start_image_latentes_conv_in = torch.zeros_like(latents)
            if latents.size()[2] != 1:
                start_image_latentes_conv_in[:, :, :1] = start_image_latentes
        else:
            start_image_latentes_conv_in = torch.zeros_like(latents)

        # Prepare clip latent variables
        if clip_image is not None:
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        else:
            clip_image = Image.new("RGB", (512, 512), color=(0, 0, 0))
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
            clip_context = torch.zeros_like(clip_context)

        if self.transformer.config.get("add_ref_conv", False):
            if ref_image is not None:
                video_length = ref_image.shape[2]
                ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height, width=width)
                ref_image = ref_image.to(dtype=torch.float32)
                ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)

                ref_image_latentes = self.prepare_control_latents(
                    None,
                    ref_image,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance
                )[1]
                ref_image_latentes = ref_image_latentes[:, :, 0]
            else:
                ref_image_latentes = torch.zeros_like(latents)[:, :, 0]
        else:
            if ref_image is not None:
                raise ValueError("The add_ref_conv is False, but ref_image is not None")
            else:
                ref_image_latentes = None

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spatial_compression_ratio, height // self.vae.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare mask latent variables
                if control_camera_video is not None:
                    control_latents_input = None
                    control_camera_latents_input = (
                        torch.cat([control_camera_latents] * 2) if do_classifier_free_guidance else control_camera_latents
                    ).to(device, weight_dtype)
                else:
                    control_latents_input = (
                        torch.cat([control_video_latents] * 2) if do_classifier_free_guidance else control_video_latents
                    ).to(device, weight_dtype)
                    control_camera_latents_input = None

                start_image_latentes_conv_in_input = (
                    torch.cat([start_image_latentes_conv_in] * 2) if do_classifier_free_guidance else start_image_latentes_conv_in
                ).to(device, weight_dtype)
                control_latents_input = start_image_latentes_conv_in_input if control_latents_input is None else \
                    torch.cat([control_latents_input, start_image_latentes_conv_in_input], dim = 1)

                clip_context_input = (
                    torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                )

                if ref_image_latentes is not None:
                    full_ref = (
                        torch.cat([ref_image_latentes] * 2) if do_classifier_free_guidance else ref_image_latentes
                    ).to(device, weight_dtype)
                else:
                    full_ref = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=control_latents_input,
                        y_camera=control_camera_latents_input,
                        full_ref=full_ref,
                        clip_fea=clip_context_input,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

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
                if comfyui_progressbar:
                    pbar.update(1)

        if output_type == "pil":
            video = self.decode_latents(latents)
            video = torch.from_numpy(video)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return video

        return WanPipelineOutput(videos=video)
