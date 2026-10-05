"""HunyuanVideo / 1.5 :class:`~...contracts.ConditionEncoder` adapters.

Original recipes use Llama/LLaVA + CLIP.  HunyuanVideo 1.5 adds ByT5
glyph tokens and optional SigLIP vision.  ``encode(DiffusionRequest)``
returns ``prompt_embeds`` / pooled / glyph tensors in
:class:`~...contracts.Conditioning`.  Prompt templates may contain
Chinese; they are not rewritten here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import NativeCheckpointResolver
from .constants import PROMPT_TEMPLATE
from .h15_text import PROMPT_TEMPLATE as H15_PROMPT_TEMPLATE
from .h15_text import TextEncoder as H15TextEncoder
from .h15_text.byT5 import load_glyph_byT5_v2
from .h15_text.byT5.format_prompt import MultilingualPromptFormat
from .h15_vision import VisionEncoder
from .i2v import TextEncoder as I2VTextEncoder
from .original import TextEncoder


def _precision(dtype: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, "bf16")


def _resource_directory(context: ComponentBuildContext, name: str, relative: str) -> Path:
    materialized = NativeCheckpointResolver().materialize(context.require_checkpoint(name))
    return materialized.directory(relative)


def _request_image(request: DiffusionRequest) -> object:
    for key in ("image", "images", "reference_image", "first_frame"):
        value = request.inputs.get(key)
        if value is not None:
            return value
    raise ValueError("HunyuanVideo I2V requires request.inputs['image']")


class HunyuanVideoPromptConditioner:
    """Llama/LLaVA plus CLIP conditioning for original HunyuanVideo."""

    def __init__(
        self,
        primary,
        clip: TextEncoder,
        *,
        image_to_video: bool = False,
        release_after_encode: bool = False,
    ) -> None:
        self.primary = primary
        self.clip = clip
        self.image_to_video = bool(image_to_video)
        self.release_after_encode = bool(release_after_encode)

    def _release_models(self) -> None:
        """Release one-shot text encoders before the memory-heavy denoising stage."""

        for encoder in (self.primary, self.clip):
            model = getattr(encoder, "model", None)
            if model is not None:
                encoder.model = None
                del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_branch(
        self,
        prompts: Sequence[str],
        *,
        request: DiffusionRequest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        tokens = self.primary.text2tokens(list(prompts), data_type="video")
        kwargs = {"data_type": "video", "device": device}
        if self.image_to_video:
            from worldfoundry.core import load_pil_image

            image = load_pil_image(_request_image(request))
            kwargs["semantic_images"] = [image] * len(prompts)
        primary = self.primary.encode(tokens, **kwargs)

        clip_tokens = self.clip.text2tokens(list(prompts), data_type="video")
        clip = self.clip.encode(clip_tokens, data_type="video", device=device)
        return {
            "text_states": primary.hidden_state.to(device=device, dtype=dtype),
            "text_mask": primary.attention_mask.to(device=device),
            "text_states_2": clip.hidden_state.to(device=device, dtype=dtype),
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return Llama/CLIP ``text_states`` tensors for both CFG branches."""
        shared = dict(request.inputs)
        shared["embedded_guidance_scale"] = float(
            request.inputs.get("embedded_guidance_scale", request.sampling.guidance_scale)
        )
        positive = self._encode_branch(request.prompts, request=request, device=device, dtype=dtype)
        if self.release_after_encode:
            self._release_models()
        return Conditioning(positive=positive, shared=shared)


def build_hunyuan_video_prompt_conditioner(context: ComponentBuildContext) -> HunyuanVideoPromptConditioner:
    image_to_video = bool(context.component_options.get("image_to_video", False))
    precision = _precision(context.policy.dtype)
    if image_to_video:
        primary_path = _resource_directory(context, "primary", "text_encoder_i2v")
        primary = I2VTextEncoder(
            text_encoder_type="llm-i2v",
            tokenizer_type="llm-i2v",
            max_length=359,
            text_encoder_precision=precision,
            text_encoder_path=str(primary_path),
            tokenizer_path=str(primary_path),
            use_attention_mask=True,
            i2v_mode=True,
            prompt_template=PROMPT_TEMPLATE["dit-llm-encode-i2v"],
            prompt_template_video=PROMPT_TEMPLATE["dit-llm-encode-video-i2v"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=True,
            device=context.policy.device,
            image_embed_interleave=4,
        )
    else:
        primary_path = _resource_directory(context, "primary", "text_encoder")
        primary = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            max_length=351,
            text_encoder_precision=precision,
            text_encoder_path=str(primary_path),
            tokenizer_path=str(primary_path),
            use_attention_mask=True,
            prompt_template=PROMPT_TEMPLATE["dit-llm-encode"],
            prompt_template_video=PROMPT_TEMPLATE["dit-llm-encode-video"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=True,
            device=context.policy.device,
        )
    clip_path = _resource_directory(context, "clip", "text_encoder_2")
    clip = TextEncoder(
        text_encoder_type="clipL",
        tokenizer_type="clipL",
        max_length=77,
        text_encoder_precision=precision,
        text_encoder_path=str(clip_path),
        tokenizer_path=str(clip_path),
        use_attention_mask=False,
        reproduce=True,
        device=context.policy.device,
    )
    return HunyuanVideoPromptConditioner(
        primary,
        clip,
        image_to_video=image_to_video,
        release_after_encode=bool(context.component_options.get("release_after_encode", False)),
    )


class HunyuanVideo15PromptConditioner:
    """Qwen2.5-VL, glyph ByT5, and optional SigLIP semantic conditioning."""

    _GLYPH_PATTERN = re.compile(r'"(.*?)"|“(.*?)”')

    def __init__(
        self,
        text_encoder: H15TextEncoder,
        *,
        byt5_tokenizer,
        byt5_model,
        byt5_max_length: int,
        prompt_format: MultilingualPromptFormat,
        vision_encoder: VisionEncoder | None = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.byt5_tokenizer = byt5_tokenizer
        self.byt5_model = byt5_model
        self.byt5_max_length = int(byt5_max_length)
        self.prompt_format = prompt_format
        self.vision_encoder = vision_encoder

    @torch.no_grad()
    def _byt5(self, prompts: Sequence[str], *, device: torch.device, dtype: torch.dtype):
        states = []
        masks = []
        for prompt in prompts:
            matches = self._GLYPH_PATTERN.findall(prompt)
            texts = list(dict.fromkeys(left or right for left, right in matches))
            if not texts:
                states.append(torch.zeros(1, self.byt5_max_length, 1472, device=device, dtype=dtype))
                masks.append(torch.zeros(1, self.byt5_max_length, device=device, dtype=torch.int64))
                continue
            formatted = self.prompt_format.format_prompt(
                texts,
                [{"color": None, "font-family": None} for _ in texts],
            )
            tokens = self.byt5_tokenizer(
                formatted,
                padding="max_length",
                max_length=self.byt5_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            mask = tokens.attention_mask.to(device)
            output = self.byt5_model(tokens.input_ids.to(device), attention_mask=mask.float())[0]
            states.append(output.to(dtype=dtype))
            masks.append(mask)
        return torch.cat(states), torch.cat(masks)

    @torch.no_grad()
    def _branch(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        tokens = self.text_encoder.text2tokens(list(prompts), data_type="video", max_length=256)
        output = self.text_encoder.encode(tokens, data_type="video", device=device)
        byt5_states, byt5_mask = self._byt5(prompts, device=device, dtype=dtype)
        return {
            "text_states": output.hidden_state.to(device=device, dtype=dtype),
            "text_mask": output.attention_mask.to(device=device),
            "text_states_2": None,
            "byt5_text_states": byt5_states,
            "byt5_text_mask": byt5_mask,
        }

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return HunyuanVideo 1.5 LLM + ByT5 (and optional SigLIP) tensors."""
        values = self._branch(request.prompts, device=device, dtype=dtype)
        shared = dict(request.inputs)
        shared["embedded_guidance_scale"] = float(
            request.inputs.get("embedded_guidance_scale", request.sampling.guidance_scale)
        )
        if self.vision_encoder is not None:
            from worldfoundry.core import load_pil_image

            image = np.asarray(load_pil_image(_request_image(request)))
            shared["vision_states"] = self.vision_encoder.encode_images(image).last_hidden_state.to(
                device=device, dtype=dtype
            )
        return Conditioning(positive=values, shared=shared)


def build_hunyuan_video15_prompt_conditioner(
    context: ComponentBuildContext,
) -> HunyuanVideo15PromptConditioner:
    root = NativeCheckpointResolver().materialize(context.require_checkpoint("resources")).root
    precision = _precision(context.policy.dtype)
    llm_path = root / "text_encoder" / "llm"
    text_encoder = H15TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        max_length=256,
        text_encoder_precision=precision,
        text_encoder_path=str(llm_path),
        tokenizer_path=str(llm_path),
        use_attention_mask=True,
        prompt_template=H15_PROMPT_TEMPLATE["li-dit-encode-image-json"],
        prompt_template_video=H15_PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=True,
        device=context.policy.device,
    )

    glyph_root = root / "text_encoder" / "Glyph-SDXL-v2"
    byt5 = load_glyph_byT5_v2(
        {
            "byT5_google_path": str(root / "text_encoder" / "byt5-small"),
            "byT5_ckpt_path": str(glyph_root / "checkpoints" / "byt5_model.pt"),
            "multilingual_prompt_format_color_path": str(glyph_root / "assets" / "color_idx.json"),
            "multilingual_prompt_format_font_path": str(
                glyph_root / "assets" / "multilingual_10-lang_idx.json"
            ),
            "byt5_max_length": int(context.component_options.get("byt5_max_length", 256)),
        },
        device=context.policy.device,
    )
    prompt_format = MultilingualPromptFormat(
        font_path=str(glyph_root / "assets" / "multilingual_10-lang_idx.json"),
        color_path=str(glyph_root / "assets" / "color_idx.json"),
    )
    vision = None
    if bool(context.component_options.get("image_to_video", False)):
        vision_root = NativeCheckpointResolver().materialize(
            context.require_checkpoint("vision")
        ).root
        vision = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision=precision,
            vision_encoder_path=str(vision_root),
            processor_path=str(vision_root),
            device=context.policy.device,
        )
    return HunyuanVideo15PromptConditioner(
        text_encoder,
        byt5_tokenizer=byt5["byt5_tokenizer"],
        byt5_model=byt5["byt5_model"],
        byt5_max_length=byt5["byt5_max_length"],
        prompt_format=prompt_format,
        vision_encoder=vision,
    )


__all__ = [
    "HunyuanVideo15PromptConditioner",
    "HunyuanVideoPromptConditioner",
    "build_hunyuan_video15_prompt_conditioner",
    "build_hunyuan_video_prompt_conditioner",
]
