"""Wan UMT5 / CLIP :class:`~...contracts.ConditionEncoder` adapters.

:class:`WanTextConditioner.encode` tokenizes ``DiffusionRequest.prompts``
and returns ``Conditioning(positive={\"context\": ...}, negative=...)``.
:class:`WanImageTextConditioner` adds CLIP visual tokens to ``shared``.
Factories load UMT5 + tokenizer (and optional CLIP) through the native
module loader.  Chinese / multilingual prompt strings are passed through
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ....components import ComponentBuildContext
from ....contracts import Conditioning, DiffusionRequest
from ....loaders import ModuleLoadSpec, NativeCheckpointResolver, NativeModuleLoader
from .clip import VisionTransformer
from .model import HuggingfaceTokenizer, T5LayerNorm, T5RelativeEmbedding, WanTextEncoder


class WanTextConditioner:
    """Tokenize prompts and produce padded UMT5 conditioning tensors."""

    def __init__(
        self,
        text_encoder: WanTextEncoder,
        tokenizer: HuggingfaceTokenizer,
        *,
        passthrough_inputs: bool = False,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.passthrough_inputs = bool(passthrough_inputs)

    @torch.no_grad()
    def _encode(
        self,
        prompts: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids, mask = self.tokenizer(
            list(prompts),
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(device=device)
        mask = mask.to(device=device)
        embeddings = self.text_encoder(ids, mask).to(dtype=dtype)
        sequence_lengths = mask.gt(0).sum(dim=1).tolist()
        for index, length in enumerate(sequence_lengths):
            embeddings[index, int(length) :] = 0
        return embeddings

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Return UMT5 ``context`` for positive (and CFG-negative) prompts."""
        positive = self._encode(request.prompts, device=device, dtype=dtype)
        negative: dict[str, torch.Tensor] = {}
        if request.sampling.guidance_scale != 1.0:
            prompts = request.negative_prompts or (("",) * request.batch_size)
            negative["context"] = self._encode(prompts, device=device, dtype=dtype)
        shared = dict(request.inputs) if self.passthrough_inputs else {}
        return Conditioning(
            positive={"context": positive},
            negative=negative,
            shared=shared,
        )


class WanImageTextConditioner(WanTextConditioner):
    """Add Wan's CLIP visual tokens to the shared native text conditioner."""

    def __init__(self, *args, image_encoder: VisionTransformer, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = image_encoder

    @staticmethod
    def _reference(request: DiffusionRequest) -> object:
        value = request.inputs.get("images", request.inputs.get("image"))
        if value is None:
            raise ValueError("Wan image-to-video conditioning requires request.inputs['images']")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise ValueError("Wan image-to-video conditioning requires a non-empty image sequence")
            return value[0]
        return value

    @torch.no_grad()
    def _encode_image(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from worldfoundry.core import load_pil_image

        image = load_pil_image(self._reference(request), first_sequence_item=False)
        array = np.asarray(image, dtype=np.float32)
        pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        pixels = pixels.div_(127.5).sub_(1.0).unsqueeze(0)
        pixels = pixels.repeat(request.batch_size, 1, 1, 1).to(device=device, dtype=dtype)
        pixels = F.interpolate(
            pixels,
            size=(self.image_encoder.image_size, self.image_encoder.image_size),
            mode="bicubic",
            align_corners=False,
        )
        mean = pixels.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        std = pixels.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        pixels = (pixels.mul(0.5).add(0.5) - mean) / std
        return self.image_encoder(pixels, use_31_block=True).to(dtype=dtype)

    def encode(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Conditioning:
        """Add CLIP ``clip_feature`` to the shared map of the UMT5 conditioner."""
        conditioning = super().encode(request, device=device, dtype=dtype)
        shared = dict(conditioning.shared)
        shared["clip_feature"] = self._encode_image(request, device=device, dtype=dtype)
        return Conditioning(
            positive=conditioning.positive,
            negative=conditioning.negative,
            shared=shared,
        )


class WanUMT5PromptEncoder:
    """Standalone native UMT5 encoder for non-recipe consumers.

    Pipelines should normally bind :class:`WanTextConditioner` through a
    recipe.  This adapter exists for representation or preprocessing code that
    needs prompt embeddings outside a diffusion sampling run.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        tokenizer_path: str,
        text_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> None:
        from worldfoundry.core.model_loading import load_model

        self.text_length = int(text_length)
        self.dtype = dtype
        self.device = torch.device(device)
        self.model = load_model(
            WanTextEncoder,
            checkpoint_path,
            torch_dtype=dtype,
            device=self.device,
        ).eval().requires_grad_(False)
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=self.text_length,
            clean="whitespace",
            local_files_only=True,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompts: str | Sequence[str],
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        target = self.device if device is None else torch.device(device)
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(target)
        mask = mask.to(target)
        embeddings = self.model(ids, mask).to(dtype=self.dtype)
        lengths = mask.gt(0).sum(dim=1).tolist()
        for index, length in enumerate(lengths):
            embeddings[index, int(length) :] = 0
        return embeddings


def convert_diffusers_umt5_encoder_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Map a Transformers UMT5 encoder checkpoint onto the native Wan encoder."""

    converted: dict[str, object] = {}
    direct = {
        "shared.weight": "token_embedding.weight",
        "encoder.final_layer_norm.weight": "norm.weight",
    }
    suffixes = {
        "layer.0.SelfAttention.q.weight": "attn.q.weight",
        "layer.0.SelfAttention.k.weight": "attn.k.weight",
        "layer.0.SelfAttention.v.weight": "attn.v.weight",
        "layer.0.SelfAttention.o.weight": "attn.o.weight",
        "layer.0.SelfAttention.relative_attention_bias.weight": "pos_embedding.embedding.weight",
        "layer.0.layer_norm.weight": "norm1.weight",
        "layer.1.DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
        "layer.1.DenseReluDense.wi_1.weight": "ffn.fc1.weight",
        "layer.1.DenseReluDense.wo.weight": "ffn.fc2.weight",
        "layer.1.layer_norm.weight": "norm2.weight",
    }
    for source, value in state_dict.items():
        target = direct.get(source)
        if target is None and source.startswith("encoder.block."):
            remainder = source.removeprefix("encoder.block.")
            block, separator, suffix = remainder.partition(".")
            mapped_suffix = suffixes.get(suffix)
            if separator and block.isdigit() and mapped_suffix is not None:
                target = f"blocks.{block}.{mapped_suffix}"
        if target is None:
            raise KeyError(f"unsupported Diffusers UMT5 encoder parameter: {source}")
        if target in converted:
            raise KeyError(f"UMT5 conversion produced duplicate parameter: {target}")
        converted[target] = value
    return converted


def convert_wan_clip_vision_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, object]:
    """Keep only the visual tower from Wan's combined CLIP checkpoint."""

    converted = {
        key.removeprefix("visual."): value
        for key, value in state_dict.items()
        if key.startswith("visual.")
    }
    if not converted:
        raise KeyError("Wan CLIP checkpoint does not contain a visual tower")
    return converted


def _build_wan_text_conditioner(
    context: ComponentBuildContext,
    *,
    state_dict_converter=None,
) -> WanTextConditioner:
    """Build Wan's tokenizer and UMT5 encoder from named checkpoint bindings."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    text_encoder = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanTextEncoder,
            state_dict_converter=state_dict_converter,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            layer_container="blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(text_encoder, WanTextEncoder):
        raise TypeError(f"expected WanTextEncoder, got {type(text_encoder).__name__}")

    tokenizer_checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("tokenizer"))
    tokenizer_subdir = str(context.component_options.get("tokenizer_subdir", "google/umt5-xxl"))
    tokenizer_path = tokenizer_checkpoint.directory(tokenizer_subdir)
    tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=int(context.component_options.get("text_length", 512)),
        clean=str(context.component_options.get("clean", "whitespace")),
        local_files_only=True,
    )
    return WanTextConditioner(
        text_encoder,
        tokenizer,
        passthrough_inputs=bool(context.component_options.get("passthrough_inputs", False)),
    )


def build_wan_text_conditioner(context: ComponentBuildContext) -> WanTextConditioner:
    """Build the original Wan UMT5 checkpoint layout."""

    return _build_wan_text_conditioner(context)


def build_diffusers_wan_text_conditioner(context: ComponentBuildContext) -> WanTextConditioner:
    """Build a Diffusers/Transformers-layout UMT5 checkpoint on the native encoder."""

    return _build_wan_text_conditioner(
        context,
        state_dict_converter=convert_diffusers_umt5_encoder_state_dict,
    )


def build_wan_image_text_conditioner(context: ComponentBuildContext) -> WanImageTextConditioner:
    """Build Wan's native UMT5 plus XLM-RoBERTa CLIP image conditioner."""

    from worldfoundry.core.vram import (
        AutoWrappedLinear,
        AutoWrappedModule,
        move_direct_tensors_to_device,
    )

    text = _build_wan_text_conditioner(context)
    image_encoder = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=VisionTransformer,
            config={
                "image_size": 224,
                "patch_size": 14,
                "dim": 1280,
                "mlp_ratio": 4,
                "out_dim": 1024,
                "num_heads": 16,
                "num_layers": 32,
                "pool_type": "token",
                "activation": "gelu",
                "attn_dropout": 0.0,
                "proj_dropout": 0.0,
                "embedding_dropout": 0.0,
            },
            state_dict_converter=convert_wan_clip_vision_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
            },
            layer_container="transformer",
        ),
        context.require_checkpoint("image_weights"),
        context.policy,
    )
    if not isinstance(image_encoder, VisionTransformer):
        raise TypeError(f"expected VisionTransformer, got {type(image_encoder).__name__}")
    move_direct_tensors_to_device(
        image_encoder,
        device=context.policy.device,
        dtype=context.policy.dtype,
    )
    return WanImageTextConditioner(
        text.text_encoder,
        text.tokenizer,
        passthrough_inputs=text.passthrough_inputs,
        image_encoder=image_encoder,
    )


__all__ = [
    "WanTextConditioner",
    "WanImageTextConditioner",
    "WanUMT5PromptEncoder",
    "build_diffusers_wan_text_conditioner",
    "build_wan_text_conditioner",
    "build_wan_image_text_conditioner",
    "convert_diffusers_umt5_encoder_state_dict",
    "convert_wan_clip_vision_state_dict",
]
