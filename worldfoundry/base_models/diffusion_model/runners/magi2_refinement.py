# SPDX-FileCopyrightText: Copyright (c) 2025 SandAI
# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Released MAGI-2 video-only refinement with zero-SNR renoising and UniPC.

Adapted from SandAI MAGI-2-preview at f68a0f9bbccbea56e0177a9bd912abc3b18ffe61,
inference/pipeline/inference_engine.py.
The refiner doubles latent temporal resolution and leaves preview audio intact.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..models.encoders.magi2_refiner import Magi2RefinerDataProxy
from ..schedulers.magi2 import FlowUniPCMultistepScheduler


def refiner_signal_scale(*, device: torch.device, noise_index: int = 220) -> torch.Tensor:
    """Return the released zero-terminal-SNR DDPM signal scale at index 220."""
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000, dtype=torch.float64).square()
    signal = (1 - betas).cumprod(0).float().sqrt()
    first, last = signal[0].clone(), signal[-1].clone()
    signal = (signal - last) * first / (first - last)
    return signal.flip(0)[noise_index].to(device)


@torch.inference_mode()
def refine_video_latent(
    transformer: nn.Module,
    latent: torch.Tensor,
    context: torch.Tensor,
    null_context: torch.Tensor,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Refine one latent using the released shift-5, guidance-2 video preset."""
    latent = F.interpolate(
        latent, size=(latent.shape[2] * 2 - 1, height // 16, width // 16), mode="trilinear", align_corners=True
    )
    signal = refiner_signal_scale(device=latent.device)
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    latent = latent * signal + noise * (1 - signal.square()).sqrt()
    proxy = Magi2RefinerDataProxy(tuple(latent.shape[2:]), device=latent.device)
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(num_inference_steps, device=latent.device, shift=5.0)
    for timestep in scheduler.timesteps:
        positive = proxy.unpack_output(transformer(**proxy.build_model_inputs(latent, context)))
        negative = proxy.unpack_output(transformer(**proxy.build_model_inputs(latent, null_context)))
        velocity = negative + 2.0 * (positive - negative)
        latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
    return latent
