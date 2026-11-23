# SPDX-License-Identifier: Apache-2.0
# Adapted from DiffSynth-Studio / Self-Forcing (Apache-2.0) and
# apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Copyright (C) 2026 Apple Inc.).
"""Flow-matching scheduler and released variant recipes for FastFrames."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "FASTFRAMES_CONTEXT_MODES",
    "FASTFRAMES_NOISE_MODES",
    "FASTFRAMES_VARIANTS",
    "FF_L",
    "FF_S",
    "FastFramesFlowMatchScheduler",
    "FastFramesVariantConfig",
    "resolve_fastframes_variant",
]

FASTFRAMES_CONTEXT_MODES: tuple[str, ...] = ("refiner", "dit")
FASTFRAMES_NOISE_MODES: tuple[str, ...] = ("sequential", "pregenerated", "reseeded")


class FastFramesFlowMatchScheduler:
    """Rectified-flow schedule ``x_t = (1 - sigma) * x_0 + sigma * noise``.

    Uses a shifted sigma grid on ``num_train_timesteps=1000`` with float64
    flow-to-``x_0`` and ``x_0``-to-flow conversions matching upstream FastFrames.
    """

    def __init__(
        self,
        num_inference_steps: int = 1000,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigmas: torch.Tensor
        self.timesteps: torch.Tensor
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int = 100) -> None:
        """Compute shifted sigma and timestep grids for ``num_inference_steps`` steps."""
        sigmas = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1)[:-1]
        self.sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    @staticmethod
    def _index(timestep: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        """Advance one rectified-flow Euler step from ``timestep`` toward the next sigma."""
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = self._index(timestep, self.timesteps)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or bool((timestep_id + 1 >= len(self.timesteps)).any()):
            next_sigma: float | torch.Tensor = 0.0
        else:
            next_sigma = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        return sample + model_output * (next_sigma - sigma)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate ``[N, C, H, W]`` clean samples with ``noise`` at ``[N]`` timesteps."""
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        sigma = self.sigmas[self._index(timestep, self.timesteps)].reshape(-1, 1, 1, 1)
        return ((1.0 - sigma) * original_samples + sigma * noise).type_as(noise)

    def flow_to_x0(
        self,
        flow: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``x_0 = x_t - sigma_t * flow`` in float64 precision."""
        dtype = flow.dtype
        flow64, xt64, sigmas64, timesteps64 = (
            tensor.double().to(flow.device) for tensor in (flow, xt, self.sigmas, self.timesteps)
        )
        t64 = timestep.double().to(flow.device)
        sigma = sigmas64[self._index(t64, timesteps64)].reshape(-1, 1, 1, 1)
        return (xt64 - sigma * flow64).to(dtype)

    def x0_to_flow(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``flow = (x_t - x_0) / sigma_t`` in float64 precision."""
        dtype = x0.dtype
        x0_64, xt64, sigmas64, timesteps64 = (
            tensor.double().to(x0.device) for tensor in (x0, xt, self.sigmas, self.timesteps)
        )
        t64 = timestep.double().to(x0.device)
        sigma = sigmas64[self._index(t64, timesteps64)].reshape(-1, 1, 1, 1)
        return ((xt64 - x0_64) / sigma).to(dtype)


@dataclass(frozen=True)
class FastFramesVariantConfig:
    """Numerical recipe and checkpoint layout of a released FastFrames variant."""

    name: str
    variant_id: str
    wan_model: str
    first_frame_timesteps: tuple[float, ...]
    timesteps: tuple[float, ...]
    refine_timestep: float
    noise: str
    autocast: bool
    fast_ops: bool
    decoder_dtype: torch.dtype
    contiguous_stem: bool
    allow_tf32: bool
    num_frames: int = 21
    latent_size: tuple[int, int] = (60, 104)
    num_pixel_frames: int = 81
    fps: int = 16
    wan_dir: str = "Wan2.1-T2V-1.3B"
    dit_file: str = "causal-forcing++/framewise-2step.pt"
    refiner_file: str = "fastframes-s/kv_refiner.safetensors"
    decoder_file: str = "fastframes-s/pixel_decoder.safetensors"
    hf_repo_id: str = "apple/FastFrames-S"
    dit_repo_id: str = "zhuhz22/Causal-Forcing"
    wan_repo_id: str = "Wan-AI/Wan2.1-T2V-1.3B"


FF_S = FastFramesVariantConfig(
    name="FF-S",
    variant_id="fastframes-s",
    wan_model="Wan2.1-T2V-1.3B",
    first_frame_timesteps=(1000.0, 937.5, 833.3333129882812, 625.0),
    timesteps=(1000.0, 833.3333129882812),
    refine_timestep=833.3333129882812,
    noise="sequential",
    autocast=False,
    fast_ops=False,
    decoder_dtype=torch.float32,
    contiguous_stem=True,
    allow_tf32=False,
    wan_dir="Wan2.1-T2V-1.3B",
    dit_file="causal-forcing++/framewise-2step.pt",
    refiner_file="fastframes-s/kv_refiner.safetensors",
    decoder_file="fastframes-s/pixel_decoder.safetensors",
    hf_repo_id="apple/FastFrames-S",
    dit_repo_id="zhuhz22/Causal-Forcing",
    wan_repo_id="Wan-AI/Wan2.1-T2V-1.3B",
)

FF_L = FastFramesVariantConfig(
    name="FF-L",
    variant_id="fastframes-l",
    wan_model="Wan2.1-T2V-14B",
    first_frame_timesteps=(1000.0, 937.0, 833.0, 625.0),
    timesteps=(1000.0, 833.0),
    refine_timestep=833.0,
    noise="pregenerated",
    autocast=True,
    fast_ops=True,
    decoder_dtype=torch.bfloat16,
    contiguous_stem=False,
    allow_tf32=True,
    wan_dir="Wan2.1-T2V-14B",
    dit_file="fastframes-l/dit.safetensors",
    refiner_file="fastframes-l/kv_refiner.safetensors",
    decoder_file="fastframes-l/pixel_decoder.safetensors",
    hf_repo_id="apple/FastFrames-L",
    dit_repo_id="apple/FastFrames-L",
    wan_repo_id="Wan-AI/Wan2.1-T2V-14B",
)

FASTFRAMES_VARIANTS: dict[str, FastFramesVariantConfig] = {
    "fastframes-s": FF_S,
    "ff-s": FF_S,
    "fastframes-l": FF_L,
    "ff-l": FF_L,
}

_VARIANT_ALIASES: dict[str, FastFramesVariantConfig] = {
    "fastframes": FF_S,
    "fastframes-s": FF_S,
    "fastframes_s": FF_S,
    "fastframes-1.3b": FF_S,
    "ff-s": FF_S,
    "ff_s": FF_S,
    "ffs": FF_S,
    "fastframes-l": FF_L,
    "fastframes_l": FF_L,
    "fastframes-14b": FF_L,
    "ff-l": FF_L,
    "ff_l": FF_L,
    "ffl": FF_L,
}


def resolve_fastframes_variant(variant: str | FastFramesVariantConfig | None = None) -> FastFramesVariantConfig:
    """Resolve a FastFrames variant alias or config object to ``FF_S`` or ``FF_L``."""
    if isinstance(variant, FastFramesVariantConfig):
        return variant
    if variant is None:
        return FF_S
    normalized = str(variant).strip().lower()
    if normalized not in _VARIANT_ALIASES:
        supported = ("fastframes-s", "ff-s", "fastframes-l", "ff-l")
        raise ValueError(f"Unsupported FastFrames variant {variant!r}; expected one of {supported}")
    return _VARIANT_ALIASES[normalized]
