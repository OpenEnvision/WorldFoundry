# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors and The MiniMax and HuggingFace Teams.
# Adapted from HyperFlow 1dd2f342 and Diffusers v0.40.0 MiniMax-H3 inference.
"""Single-request HyperFlow orchestration over canonical MiniMax-H3 components."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_hyperflow import (
    load_hyperflow_audio_vae,
    load_hyperflow_video_vae,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.klvae import (
    DiagonalGaussianDistribution,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.hyperflow import (
    HyperFlowMetadata,
    load_hyperflow_transformer,
)
from worldfoundry.base_models.diffusion_model.models.encoders.minimax_h3_qwen3vl import MiniMaxH3Qwen3VLEncoder
from worldfoundry.base_models.diffusion_model.schedulers.hyperflow import HyperFlowSchedule

from .conditioning import prepare_presentation
from .media import PreparedMedia, ReferenceMedia
from .sampling import denoise, prepare_latents

_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
_LOG = logging.getLogger(__name__)


class HyperFlowComponent(nn.Module):
    """Expose canonical components to the release's automatic CPU-offload utility.

    Dispatching codec methods through forward ensures the existing offload hook
    runs for both encode and decode. Checkpoints load before this orchestration adapter.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_memory_footprint(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (*self.parameters(), *self.buffers()))

    def forward(self, *args: Any, method: str = "forward", **kwargs: Any) -> Any:
        return getattr(self.model, method)(*args, **kwargs)


def _place_components(
    components: dict[str, HyperFlowComponent], *, device: torch.device, offload: bool, margin: str
) -> Any:
    if not offload:
        for component in components.values():
            component.to(device)
        return None
    from diffusers import ComponentsManager

    manager = ComponentsManager()
    for name, component in components.items():
        manager.add(name, component)
    # Register everything first: add() otherwise re-arms the default 3 GB margin.
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=margin)
    return manager


@dataclass
class HyperFlowComponents:
    transformer: HyperFlowComponent
    vae: HyperFlowComponent
    audio_vae: HyperFlowComponent
    text_encoder: HyperFlowComponent | None
    metadata: HyperFlowMetadata
    video_config: dict[str, Any]
    audio_config: dict[str, Any]
    manager: Any
    conditioner_manager: Any


def load_components(
    model_path: Path,
    weights_path: Path,
    *,
    workflow: str,
    device: torch.device,
    attention_backend: str,
    sol_attn: bool,
    offload: bool,
    memory_reserve_margin: str,
    rank: int,
    degree: int,
) -> HyperFlowComponents:
    """Load only the selected H3 partition, strictly and entirely from local files."""
    encoder = None
    if rank == 0:
        encoder = HyperFlowComponent(
            MiniMaxH3Qwen3VLEncoder.from_pretrained(
                str(model_path / "text_encoder"),
                torch_dtype=torch.bfloat16,
                strict=True,
                preserve_rotary_precision=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
    transformer, metadata = load_hyperflow_transformer(
        model_path / ("transformer_ref" if workflow == "ref2va" else "transformer"),
        weights_path,
        attention_backend=attention_backend,
        sol_attn=sol_attn,
        degree=degree,
        rank=rank,
    )
    models = {
        "transformer": HyperFlowComponent(transformer),
        "vae": HyperFlowComponent(load_hyperflow_video_vae(model_path / "vae")),
        "audio_vae": HyperFlowComponent(load_hyperflow_audio_vae(model_path / "audio_vae")),
    }
    conditioner_manager = None
    if encoder is not None:
        if degree == 1:
            models["text_encoder"] = encoder
        else:
            conditioner_manager = _place_components(
                {"text_encoder": encoder}, device=device, offload=offload, margin="3GB"
            )
    manager = _place_components(models, device=device, offload=offload, margin=memory_reserve_margin)
    return HyperFlowComponents(
        models["transformer"],
        models["vae"],
        models["audio_vae"],
        encoder,
        metadata,
        json.loads((model_path / "vae" / "config.json").read_text()),
        json.loads((model_path / "audio_vae" / "config.json").read_text()),
        manager,
        conditioner_manager,
    )


def encode_references(
    components: HyperFlowComponents, references: list[ReferenceMedia], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sample each visual posterior under its own seed 42; take soundtrack means."""
    video_mean = torch.tensor(components.video_config["latents_mean"]).view(1, -1, 1, 1, 1)
    video_std = torch.tensor(components.video_config["latents_std"]).view(1, -1, 1, 1, 1)
    audio_mean = torch.tensor(components.audio_config["latents_mean"]).view(1, 1, -1)
    audio_std = torch.tensor(components.audio_config["latents_std"]).view(1, 1, -1)
    pixel_mean = torch.tensor(_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video_latents, audio_latents = [], []
    for reference in references:
        if reference.kind in ("image", "video"):
            if reference.kind == "image":
                pixels = torch.from_numpy(np.array(reference.image)).permute(2, 0, 1)[None, :, None]
                method = "_adaptive_encode"
            else:
                count = max(1, (len(reference.frames) - 5) // 17) * 17 + 5
                pixels = torch.from_numpy(reference.frames[:count].copy()).permute(3, 0, 1, 2)[None]
                method = "encode_temporal"
            pixels = (pixels.to(device, torch.float32).div(255.0) - pixel_mean) / pixel_std
            moments = components.vae(pixels, method=method)
            posterior = DiagonalGaussianDistribution(moments)
            latent = posterior.sample(generator=torch.Generator().manual_seed(42)).to(torch.float16).float().cpu()
            video_latents.append((latent - video_mean) / video_std)
        if reference.audio is not None:
            latent = components.audio_vae(reference.audio.to(device)[:, None], method="encode")
            latent = latent.float().cpu().transpose(1, 2)
            audio_latents.append(((latent - audio_mean) / audio_std).reshape(-1, audio_mean.shape[-1]))
    return video_latents, audio_latents


def decode_latents(
    components: HyperFlowComponents, video_latents: torch.Tensor, audio_latents: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode FP32 weights under the released FP16 video / FP32 audio recipe."""
    mean = torch.tensor(components.video_config["latents_mean"], device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(components.video_config["latents_std"], device=device).view(1, -1, 1, 1, 1)
    video_latents = video_latents * std + mean
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        video = components.vae(video_latents, method="decode_temporal")
    pixel_mean = torch.tensor(_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    video = (video[0].permute(1, 2, 3, 0).cpu() * 255).round().to(torch.uint8)
    mean = torch.tensor(components.audio_config["latents_mean"], device=device).view(1, -1, 1)
    std = torch.tensor(components.audio_config["latents_std"], device=device).view(1, -1, 1)
    audio = components.audio_vae(audio_latents * std + mean, method="decode")
    return video, audio.float()[:, 0].cpu()


@torch.no_grad()
def run_inference(
    model_path: Path,
    weights_path: Path,
    media: PreparedMedia | None,
    *,
    workflow: str,
    prompt: str,
    seed: int,
    device: torch.device,
    attention_backend: str,
    sol_attn: bool = False,
    offload: bool = True,
    memory_reserve_margin: str = "24GB",
    rank: int = 0,
    degree: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode on rank zero; each rank runs canonical codecs and Ulysses denoising."""
    presentation = processor = None
    if rank == 0:
        from transformers import AutoProcessor, AutoTokenizer

        assert media is not None
        tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_path / "processor", local_files_only=True)
        presentation = prepare_presentation(tokenizer, processor, prompt, media.references)
    components = load_components(
        model_path,
        weights_path,
        workflow=workflow,
        device=device,
        attention_backend=attention_backend,
        sol_attn=sol_attn,
        offload=offload,
        memory_reserve_margin=memory_reserve_margin,
        rank=rank,
        degree=degree,
    )
    payload: list[Any] = [None]
    if rank == 0:
        assert presentation is not None and components.text_encoder is not None
        ids, text_tags, vision = presentation
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        arguments = {
            name: value.to(device, torch.bfloat16) if name.startswith("pixel_") else value.to(device)
            for name, value in vision.items()
        }
        prompt_embeds = components.text_encoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=torch.tensor(processor.create_mm_token_type_ids([ids]), dtype=torch.long, device=device),
            **arguments,
        ).to(torch.bfloat16)
        payload[0] = (media, prompt_embeds.cpu(), text_tags)
        if degree > 1:
            components.text_encoder = components.conditioner_manager = None
            gc.collect()
            torch.cuda.empty_cache()
    if degree > 1:
        # Nonzero ranks warm the DiT while rank zero encodes, as in the release.
        components.transformer.to(device)
        dist.broadcast_object_list(payload, src=0)
    media, prompt_embeds, text_tags = payload[0]
    prompt_embeds = prompt_embeds.to(device)
    video_conditions, audio_conditions = encode_references(components, media.references, device)
    latents = prepare_latents(
        media,
        text_tags,
        video_conditions,
        audio_conditions,
        generator=torch.Generator().manual_seed(seed),
        device=device,
        video_channels=int(components.video_config["latent_channels"]),
        audio_channels=int(components.audio_config["latent_channels"]),
    )
    video_shift = float(json.loads((model_path / "scheduler" / "scheduler_config.json").read_text())["shift"])
    audio_shift = float(json.loads((model_path / "audio_scheduler" / "scheduler_config.json").read_text())["shift"])
    if (video_shift, audio_shift) != (components.metadata.video_shift, components.metadata.audio_shift):
        _LOG.warning("HyperFlow checkpoint shifts differ from the adapter's trained video/audio shifts.")
    schedule = HyperFlowSchedule(components.metadata.sigmas, video_shift=video_shift, audio_shift=audio_shift)
    video, audio = denoise(components.transformer, latents, prompt_embeds, schedule)
    return decode_latents(components, video, audio, device)
