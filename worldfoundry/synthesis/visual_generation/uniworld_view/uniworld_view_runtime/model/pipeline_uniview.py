# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# update to diffusers 0.33.1 to support Wan video generation
import html
from collections.abc import Callable
from typing import Any, ClassVar

import PIL.Image
import regex as re
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

from .uniview_transformer import WanVACETransformer3DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import PIL.Image
        >>> from diffusers import AutoencoderKLWan, WanVACEPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        >>> from diffusers.utils import export_to_video, load_image
        def prepare_video_and_mask(first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int):
            first_img = first_img.resize((width, height))
            last_img = last_img.resize((width, height))
            frames = []
            frames.append(first_img)
            # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
            # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
            # match the original code.
            frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
            frames.append(last_img)
            mask_black = PIL.Image.new("L", (width, height), 0)
            mask_white = PIL.Image.new("L", (width, height), 255)
            mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
            return frames, mask

        >>> # Available checkpoints: Wan-AI/Wan2.1-VACE-1.3B-diffusers, Wan-AI/Wan2.1-VACE-14B-diffusers
        >>> model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        >>> first_frame = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
        ... )
        >>> last_frame = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png>>> "
        ... )

        >>> height = 512
        >>> width = 512
        >>> num_frames = 81
        >>> video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

        >>> output = pipe(
        ...     video=video,
        ...     mask=mask,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames=num_frames,
        ...     num_inference_steps=30,
        ...     guidance_scale=5.0,
        ...     generator=torch.Generator().manual_seed(42),
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


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


class WanVACEPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for controllable generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs: ClassVar[list[str]] = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanVACETransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
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
        # torch.Size([1, 512]), torch.Size([1, 512])
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
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
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
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        video=None,
        mask=None,
        ref_video=None,
        reference_images=None,
    ):
        base = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        if height % base != 0 or width % base != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {base} but are {height} and {width}.")

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

        if ref_video is not None:
            if not isinstance(ref_video, torch.Tensor):
                raise TypeError(f"`ref_video` must be a torch.Tensor, but got {type(ref_video)}")
            if ref_video.ndim != 5:
                raise ValueError(f"`ref_video` must have shape [B, C, T, H, W], but got {tuple(ref_video.shape)}")
            if ref_video.shape[1] != 3:
                raise ValueError(f"`ref_video` must have 3 channels (RGB), but got {ref_video.shape[1]}")
            if ref_video.shape[2] != 10:
                raise ValueError(
                    f"`ref_video` must have exactly 10 frames for UniView stage2, but got {ref_video.shape[2]}"
                )

        # if video is not None:
        #     if mask is not None:
        #         if len(video) != len(mask):
        #             raise ValueError(
        #                 f"Length of `video` {len(video)} and `mask` {len(mask)} do not match. Please make sure that"
        #                 " they have the same length."
        #             )
        #     if reference_images is not None:
        #         is_pil_image = isinstance(reference_images, PIL.Image.Image)
        #         is_list_of_pil_images = isinstance(reference_images, list) and all(
        #             isinstance(ref_img, PIL.Image.Image) for ref_img in reference_images
        #         )
        #         is_list_of_list_of_pil_images = isinstance(reference_images, list) and all(
        #             isinstance(ref_img, list) and all(isinstance(ref_img_, PIL.Image.Image) for ref_img_ in ref_img)
        #             for ref_img in reference_images
        #         )
        #         if not (is_pil_image or is_list_of_pil_images or is_list_of_list_of_pil_images):
        #             raise ValueError(
        #                 "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
        #                 "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
        #             )
        #         if is_list_of_list_of_pil_images and len(reference_images) != 1:
        #             raise ValueError(
        #                 "The pipeline only supports generating one video at a time at the moment. When passing a list "
        #                 "of list of reference images, where the outer list corresponds to the batch size and the inner "
        #                 "list corresponds to list of conditioning images per video, please make sure to only pass "
        #                 "one inner list of reference images (i.e., `[[<image1>, <image2>, ...]]`"
        #             )
        # elif mask is not None:
        #     raise ValueError("`mask` can only be passed if `video` is passed as well.")

    def preprocess_conditions(
        self,
        video: list[PipelineImageInput] | None = None,
        mask: list[PipelineImageInput] | None = None,
        reference_images: PIL.Image.Image | list[PIL.Image.Image] | list[list[PIL.Image.Image]] | None = None,
        batch_size: int = 1,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        # if video is not None:
        #     base = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        #     video_height, video_width = self.video_processor.get_default_height_width(video[0])

        #     if video_height * video_width > height * width:
        #         scale = min(width / video_width, height / video_height)
        #         video_height, video_width = int(video_height * scale), int(video_width * scale)

        #     if video_height % base != 0 or video_width % base != 0:
        #         logger.warning(
        #             f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
        #         )
        #         video_height = (video_height // base) * base
        #         video_width = (video_width // base) * base

        #     assert video_height * video_width <= height * width

        #     video = self.video_processor.preprocess_video(video, video_height, video_width)
        #     image_size = (video_height, video_width)  # Use the height/width of video (with possible rescaling)
        # else:
        #     video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
        #     image_size = (height, width)  # Use the height/width provider by user

        # if mask is not None:
        #     mask = self.video_processor.preprocess_video(mask, image_size[0], image_size[1])
        #     mask = torch.clamp((mask + 1) / 2, min=0, max=1)
        # else:
        #     mask = torch.ones_like(video)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        # Make a list of list of images where the outer list corresponds to video batch size and the inner list
        # corresponds to list of conditioning images per video
        if reference_images is None or isinstance(reference_images, PIL.Image.Image):
            reference_images = [[reference_images] for _ in range(video.shape[0])]
        elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), PIL.Image.Image):
            reference_images = [reference_images]
        elif (
            isinstance(reference_images, (list, tuple))
            and isinstance(next(iter(reference_images)), list)
            and isinstance(next(iter(reference_images[0])), PIL.Image.Image)
        ):
            reference_images = reference_images
        else:
            raise ValueError(
                "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
                "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
            )

        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

        ref_images_lengths = [len(reference_images_batch) for reference_images_batch in reference_images]
        if any(length != ref_images_lengths[0] for length in ref_images_lengths):
            raise ValueError(
                f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
                "may be added in the future."
            )

        reference_images_preprocessed = []
        for _i, reference_images_batch in enumerate(reference_images):
            preprocessed_images = []
            for _j, image in enumerate(reference_images_batch):
                if image is None:
                    continue
                image = self.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(image_size[0] / img_height, image_size[1] / img_width)  # noqa: F821 - Upstream leaves image_size unbound; UniView passes no reference images.
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image, size=(new_height, new_width), mode="bilinear", align_corners=False
                ).squeeze(0)  # [C, H, W]
                top = (image_size[0] - new_height) // 2  # noqa: F821 - Upstream leaves image_size unbound; UniView passes no reference images.
                left = (image_size[1] - new_width) // 2  # noqa: F821 - Upstream leaves image_size unbound; UniView passes no reference images.
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)  # noqa: F821 - Upstream leaves image_size unbound; UniView passes no reference images.
                canvas[:, top : top + new_height, left : left + new_width] = resized_image
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        return video, mask, reference_images_preprocessed

    def prepare_ref_latents(
        self,
        ref_video: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        if isinstance(generator, list):
            raise ValueError("Passing a list of generators is not yet supported.")

        if ref_video.shape[0] != 1:
            raise ValueError("Generating with more than one video is not yet supported.")

        vae_dtype = self.vae.dtype
        ref_video = ref_video.to(dtype=vae_dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        ref_latents = retrieve_latents(self.vae.encode(ref_video), generator, sample_mode="argmax")
        ref_latents = ((ref_latents.float() - latents_mean) * latents_std).to(vae_dtype)
        return ref_latents

    def prepare_video_latents_noref(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        if isinstance(generator, list):
            # TODO: support this
            raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

        # video.shape == torch.Size([1, 3, 81, 480, 832])
        if video.shape[0] != 1:
            # TODO: support this
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )

        vae_dtype = self.vae.dtype
        video = video.to(dtype=vae_dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        mask = mask.to(dtype=vae_dtype)
        mask = torch.where(mask > 0.5, 1.0, 0.0)
        inactive = video * (1 - mask)  # [1, 3, 81, 480, 832]
        reactive = video * mask  # [1, 3, 81, 480, 832]
        inactive = retrieve_latents(
            self.vae.encode(inactive), generator, sample_mode="argmax"
        )  # torch.Size([1, 16, 21, 60, 104])
        reactive = retrieve_latents(
            self.vae.encode(reactive), generator, sample_mode="argmax"
        )  # torch.Size([1, 16, 21, 60, 104])
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)  # torch.Size([1, 16, 21, 60, 104])

        return latents  # 在没有reference的情况下等于latents

    def prepare_masks_noref(
        self,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape[0] != 1:
            # TODO: support this
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )

        transformer_patch_size = self.transformer.config.patch_size[1]  # transformer_patch_size = 2

        mask_list = []
        for mask_ in mask:
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
            new_height = height // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            new_width = width // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
            ).squeeze(0)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        video: list[PipelineImageInput] | None = None,
        mask: list[PipelineImageInput] | None = None,
        ref_video: torch.Tensor | None = None,
        reference_images: list[PipelineImageInput] | None = None,
        conditioning_scale: float | list[float] | torch.Tensor = 1.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],  # noqa: B006 - Keeps the upstream default object and signature.
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
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
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Simplification of implementation for now
        if not isinstance(prompt, str):
            raise ValueError("Passing a list of prompts is not yet supported. This may be supported in the future.")
        if num_videos_per_prompt != 1:
            raise ValueError(
                "Generating multiple videos per prompt is not yet supported. This may be supported in the future."
            )

        # 1. Check inputs. Raise error if not correct
        # 这里定义的video和mask的类型是List[PipelineImageInput]，即List[PIL.Image.Image]
        ## 直接修改video和mask的类型为torch.Tensor, 不检查video和mask
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            video,
            mask,
            ref_video,
            reference_images,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
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

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(self.transformer.config.vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(device=device, dtype=transformer_dtype)

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

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        ## 直接传入处理好的render和mask,不处理视频和mask, 只生成空reference_images
        # # 5. Prepare latent variables
        video, mask, reference_images = self.preprocess_conditions(
            video,
            mask,
            reference_images,
            batch_size,
            height,
            width,
            num_frames,
            torch.float32,
            device,
        )
        # 处理后的video: torch.Size([1, 3, 81, 512, 512]), b c f h w, [-1,1]
        # 处理后的mask: torch.Size([1, 1, 81, 512, 512]), b 1 f h w, [0,1], 0代表保留， 1代表需要修复的洞

        if ref_video is not None:
            ref_video = ref_video.to(dtype=torch.float32, device=device)
            ref_latents = self.prepare_ref_latents(ref_video, generator, device).to(transformer_dtype)
        else:
            ref_latents = None

        num_reference_images = len(reference_images[0])
        ## with reference images
        # conditioning_latents = self.prepare_video_latents(video, mask, reference_images, generator, device)
        # mask = self.prepare_masks(mask, reference_images, generator)

        # without reference images
        conditioning_latents = self.prepare_video_latents_noref(video, mask, generator, device)  # 1, 32, 21, 60, 104
        mask = self.prepare_masks_noref(mask)  # 1, 64, 21, 60, 104

        conditioning_latents = torch.cat([conditioning_latents, mask], dim=1)  # 1, 96, 21, 60, 104
        conditioning_latents = conditioning_latents.to(transformer_dtype)  # vae dtype to transformer dtype

        num_channels_latents = self.transformer.config.in_channels  # 16
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames + num_reference_images * self.vae_scale_factor_temporal,
            torch.float32,
            device,
            generator,
            latents,
        )  # torch.Size([1, 16, 21, 60, 104])

        if conditioning_latents.shape[2] != latents.shape[2]:
            logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames to be generated. Generation quality may be affected."
            )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                #
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # torch.Size([1, 16, 21, 60, 104]), noise
                    timestep=timestep,  # torch.Size([1])
                    encoder_hidden_states=prompt_embeds,  # torch.Size([1, 512, 4096])
                    ref_latents=ref_latents,
                    control_hidden_states=conditioning_latents,  # torch.Size([1, 96, 21, 60, 104]), vace branch input
                    control_hidden_states_scale=conditioning_scale,  # tensor([1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0', dtype=torch.bfloat16), shape与vace layer数量相同为8
                    attention_kwargs=attention_kwargs,  # None
                    return_dict=False,
                )[0]  # torch.Size([1, 16, 21, 60, 104])

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        ref_latents=ref_latents,
                        control_hidden_states=conditioning_latents,
                        control_hidden_states_scale=conditioning_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

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

        self._current_timestep = None

        if output_type != "latent":
            latents = latents[:, :, num_reference_images:]
            latents = latents.to(vae_dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
