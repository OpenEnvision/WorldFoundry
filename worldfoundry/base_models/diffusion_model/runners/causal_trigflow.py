# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-owned clean-prefill, causal few-step sampling, and window decode."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from worldfoundry.core.attention.cache.kvcache import CompactingKVCache

from ..contracts import Conditioning, DenoiserInput, DiffusionOutput, DiffusionRequest, LatentInitialization
from .base import RunnerComponents

if TYPE_CHECKING:
    from .strategies import ExecutionBuildContext


class CausalTrigFlowRunner:
    """Sample one re-encoded window with request-local bounded clean KV history.

    Denoisers consume TrigFlow time and return x0. Frame actions align with the
    transition ending at that frame, including clean observation prefill.
    """

    def __init__(
        self,
        *,
        model_id: str,
        components: RunnerComponents,
        device: torch.device,
        dtype: torch.dtype,
        actions_per_frame: int = 4,
    ) -> None:
        self.model_id, self.components = model_id, components
        self.device, self.dtype = torch.device(device), dtype
        self.actions_per_frame = actions_per_frame

    @torch.inference_mode()
    def run(self, request: DiffusionRequest) -> DiffusionOutput:
        """Prefill observed latents, generate future latents, decode, and release KV."""
        if request.batch_size != 1 or request.num_frames != 13:
            raise ValueError("causal-trigflow requires a single 13-frame observation window")
        schedule = self.components.scheduler.schedule(request.sampling, device=self.device, dtype=self.dtype)
        generator = torch.Generator(device=self.device).manual_seed(request.sampling.seed)
        encoded = request.inputs.get("text_conditioning")
        if isinstance(encoded, Conditioning):
            conditioning = {**encoded.shared, **encoded.positive, "actions": request.inputs["actions"]}
        else:
            encoded = self.components.conditioner.encode(request, device=self.device, dtype=self.dtype)
            conditioning = {**encoded.shared, **encoded.positive}
        initial = self.components.latent_initializer.initialize(
            request,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        if not isinstance(initial, LatentInitialization):
            raise TypeError("causal-trigflow requires encoded observation initialization")
        conditioning.update(initial.conditioning)
        observations, actions, mask = (
            conditioning.get(name) for name in ("condition_latents", "actions", "condition_mask")
        )
        if not all(isinstance(value, torch.Tensor) for value in (observations, actions, mask)):
            raise TypeError("causal-trigflow requires encoded frames, actions, and a condition mask")
        observations, actions, mask = (cast(torch.Tensor, value) for value in (observations, actions, mask))
        start = int(cast(int, conditioning["start_index"]))
        noise = initial.latents
        if not 1 <= start < noise.shape[2] or actions.shape[1] != (noise.shape[2] - 1) * self.actions_per_frame:
            raise ValueError("causal-trigflow actions must cover every latent transition in the window")
        caches: dict[int, CompactingKVCache] = {}
        output = noise.clone()
        output[:, :, :start] = observations[:, :, :start]
        try:
            for frame_index in range(noise.shape[2]):
                action = (
                    None
                    if frame_index == 0
                    else actions[:, (frame_index - 1) * self.actions_per_frame : frame_index * self.actions_per_frame]
                )
                values = {
                    **conditioning,
                    "frame_index": frame_index,
                    "action": action,
                    "condition_mask": mask[:, :, frame_index : frame_index + 1],
                    "kv_cache": caches,
                }
                if frame_index >= start:
                    original_noise = noise[:, :, frame_index : frame_index + 1]
                    frame = original_noise
                    for step in schedule:
                        prediction = self.components.denoiser(
                            DenoiserInput(
                                latents=self.components.scheduler.scale_model_input(frame, step),
                                timestep=step.timestep,
                                next_timestep=step.next_timestep,
                                conditioning=values,
                                step_index=step.index,
                                total_steps=len(schedule),
                            )
                        ).sample
                        frame = self.components.scheduler.step(prediction, step, original_noise, generator=generator)
                    output[:, :, frame_index : frame_index + 1] = frame
                if frame_index < noise.shape[2] - 1:
                    self.components.denoiser(
                        DenoiserInput(
                            latents=output[:, :, frame_index : frame_index + 1],
                            timestep=noise.new_zeros(()),
                            next_timestep=noise.new_zeros(()),
                            conditioning={**values, "store_kv": True},
                            step_index=0,
                            total_steps=1,
                        )
                    )
            sample = self.components.decoder.decode(output, request)
            return DiffusionOutput(
                sample=sample,
                latents=output,
                metadata={
                    **request.metadata,
                    "start_index": start,
                    "sampling_steps": len(schedule),
                    "sampling": "causal-trigflow",
                    "guidance": "distilled",
                },
            )
        finally:
            for cache in caches.values():
                cache.reset()
            caches.clear()


def build_causal_trigflow_strategy(context: ExecutionBuildContext) -> CausalTrigFlowRunner:
    """Bind canonical roles to the causal clean-prediction sampling strategy."""
    from .strategies import build_standard_strategy

    if context.extensions:
        raise ValueError("causal-trigflow does not support diffusion extensions")
    bound = build_standard_strategy(context)
    return CausalTrigFlowRunner(
        model_id=bound.model_id,
        components=bound.components,
        device=bound.device,
        dtype=bound.dtype,
    )
