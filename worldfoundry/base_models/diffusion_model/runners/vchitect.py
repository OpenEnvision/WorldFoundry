"""Vchitect-2's released step-dependent classifier-free guidance."""

from __future__ import annotations

import math

from torch import Tensor

from ..contracts import DenoiserOutput
from ..extensions import DiffusionRunContext
from .base import NativeDiffusionRunner


def vchitect_guidance_scale(base_scale: float, timestep: float, num_inference_steps: int) -> float:
    """Return the cosine CFG scale used by the released Vchitect-2 pipeline."""
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    phase = ((num_inference_steps - timestep) / num_inference_steps) ** 5.0
    return 1.0 + base_scale * (1.0 - math.cos(math.pi * phase)) / 2.0


class VchitectGuidanceRunner(NativeDiffusionRunner):
    """Keep the shared loop while reproducing Vchitect's varying CFG scale."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.cfg_parallel_degree != 1 or self.cfg_gate_step != 1.0:
            raise ValueError("Vchitect guidance requires local, uncached CFG branches")

    def predict(self, context: DiffusionRunContext, model_latents: Tensor) -> DenoiserOutput:
        if not context.conditioning.negative:
            return self._call_denoiser(context, latents=model_latents, branch="positive")
        if context.step is None:
            raise RuntimeError("Vchitect guidance requested before scheduler step selection")
        scale = vchitect_guidance_scale(
            context.request.sampling.guidance_scale,
            float(context.step.timestep.item()),
            context.request.sampling.num_inference_steps,
        )
        negative = self._call_denoiser(context, latents=model_latents, branch="negative")
        positive = self._call_denoiser(context, latents=model_latents, branch="positive")
        self._cfg_gate_negative_branch_calls += 1
        self._cfg_gate_positive_branch_calls += 1
        return self._guided_output(positive=positive, negative=negative, scale=scale)


__all__ = ["VchitectGuidanceRunner", "vchitect_guidance_scale"]
