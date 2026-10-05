"""Gamma-World native :class:`~...contracts.Denoiser` factories.

Re-exports causal, few-step causal, and bidirectional adapters from
:mod:`.component`.  Each factory returns an object whose
``__call__(DenoiserInput) -> DenoiserOutput`` maps Wan-style
``context`` plus multi-view latents onto the Gamma-World DiT.
"""

from .component import (
    GammaWorldDenoiser,
    build_gamma_world_bidirectional_denoiser,
    build_gamma_world_causal_denoiser,
    build_gamma_world_causal_few_step_denoiser,
)

__all__ = [
    "GammaWorldDenoiser",
    "build_gamma_world_bidirectional_denoiser",
    "build_gamma_world_causal_denoiser",
    "build_gamma_world_causal_few_step_denoiser",
]
