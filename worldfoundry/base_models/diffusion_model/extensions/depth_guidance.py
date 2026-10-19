# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Decoded-depth guidance through the native denoiser/scheduler/codec roles.

The late-step gradient normalization follows WonderWorld's guided Marigold
algorithm (Yu et al., arXiv:2406.09394). No upstream runtime is imported.
"""

from __future__ import annotations

from typing import Protocol, cast

import torch

from ..contracts import DenoiserInput, DenoiserOutput, DiffusionRequest
from .base import DiffusionExtension, DiffusionRunContext


class DepthGuidanceDecoder(Protocol):
    """Differentiable scene-depth decoding at the released guidance precision."""

    def decode_for_guidance(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        """Decode a scheduler probe without final-output precision conversion."""
        ...


class DecodedDepthGuidance(DiffusionExtension):
    """Align generated depth to known geometry during the final Euler steps.

    Request inputs: target_depth and align_mask are BCHW; farther_mask adds a
    squared hinge penalty behind an occluder. All graph references are local
    to the current evaluation. Checkpoint parameters remain frozen.
    """

    extension_id = "decoded-depth-guidance"

    def before_denoiser(self, context: DiffusionRunContext, model_input: DenoiserInput) -> DenoiserInput:
        """Retain the current evaluation only, in the request-owned context."""
        context.state[self.extension_id] = model_input
        return model_input

    def after_denoiser(self, context: DiffusionRunContext, model_output: DenoiserOutput) -> DenoiserOutput:
        """Re-evaluate with gradients only when the request has depth anchors."""
        model_input = context.state.pop(self.extension_id)
        target = context.request.inputs.get("target_depth")
        mask = context.request.inputs.get("align_mask")
        requested_steps = context.request.inputs.get("guidance_steps", 8)
        if not isinstance(requested_steps, int) or requested_steps < 0:
            raise ValueError("depth guidance_steps must be a nonnegative integer")
        guidance_steps = min(requested_steps, context.request.sampling.num_inference_steps)
        if not isinstance(model_input, DenoiserInput) or context.step is None:
            raise RuntimeError("depth guidance requires a native denoiser evaluation")
        if (
            not isinstance(target, torch.Tensor)
            or not isinstance(mask, torch.Tensor)
            or not mask.any()
            or guidance_steps == 0
        ):
            return model_output
        offset = model_input.step_index - (model_input.total_steps - guidance_steps)
        if offset < 0:
            return model_output
        with torch.enable_grad():
            latent = model_input.latents.detach().requires_grad_(True)
            prediction = context.components.denoiser(model_input.with_updates(latents=latent)).sample
            proposed = context.components.scheduler.step(prediction, context.step, latent, generator=context.generator)
            decoder = cast(DepthGuidanceDecoder, context.components.decoder)
            depth = decoder.decode_for_guidance(proposed, context.request)
            loss = (depth.float() - target.detach().to(depth.device)).square()[mask.to(depth.device).bool()].mean()
            farther = context.request.inputs.get("farther_mask")
            if isinstance(farther, torch.Tensor) and farther.any():
                hinge = (
                    (target.detach().to(depth.device) - depth.float())
                    .clamp_min(0)
                    .square()[farther.to(depth.device).bool()]
                )
                active = hinge > 0
                if active.any():
                    loss = loss + hinge[active].mean()
            gradient = torch.autograd.grad(loss * 3e9, latent)[0]
        magnitude = gradient.square().sum().sqrt().detach()
        if not torch.isfinite(magnitude):
            raise RuntimeError("depth guidance produced a nonfinite gradient")
        if magnitude == 0:
            return model_output
        ratios = torch.cat((torch.linspace(2, 0.1, guidance_steps - 1), torch.tensor([0.1])))
        prediction_magnitude = prediction.detach().square().sum().sqrt()
        correction = gradient / magnitude * prediction_magnitude * ratios[offset]
        return DenoiserOutput(sample=(prediction.detach() + correction).detach())

    def on_run_error(self, context: DiffusionRunContext, error: BaseException) -> None:
        """Release a saved evaluation after cancellation or component failure."""
        del error
        context.state.pop(self.extension_id, None)
