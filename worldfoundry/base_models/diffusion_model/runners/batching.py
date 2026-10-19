# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose independent requests around the canonical sampling loop.

Only denoiser evaluations are merged. Encoding, random draws, solver histories
and decoding keep each original request's boundaries. A batch owns one denoiser
cache lifecycle and fails as a unit; it is never retried as singleton requests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor

from ..contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionOutput,
    DiffusionRequest,
    EncodedLatentInitializer,
    LatentInitialization,
    SamplingConfig,
    SchedulerStep,
)
from ..extensions import DiffusionExtension, DiffusionRunContext
from .base import NativeDiffusionRunner, RunnerComponents, Wan22DualExpertGuidanceRunner


def _merge_conditions(values: Sequence[Mapping[str, object]], sizes: tuple[int, ...]) -> dict[str, object]:
    """Concatenate batch-leading tensors; retain identical scalar conditions."""
    first = values[0]
    if any(value.keys() != first.keys() for value in values[1:]):
        raise ValueError("request batch conditioning keys must match")
    merged: dict[str, object] = {}
    for key, value in first.items():
        items = [item[key] for item in values]
        if isinstance(value, Tensor):
            tensors: list[Tensor] = []
            for item, size in zip(items, sizes):
                if not isinstance(item, Tensor) or item.ndim == 0 or item.shape[0] != size:
                    raise ValueError(f"request batch condition {key!r} must have a request batch dimension")
                if item.shape[1:] != value.shape[1:] or item.dtype != value.dtype or item.device != value.device:
                    raise ValueError(f"request batch condition {key!r} must have matching shape, dtype and device")
                tensors.append(item)
            merged[key] = torch.cat(tensors, dim=0)
        elif type(value) in (bool, int, float, str, type(None)) and all(
            type(item) is type(value) and item == value for item in items
        ):
            merged[key] = value
        else:
            raise ValueError(f"request batch condition {key!r} must be a batch-leading tensor or matching scalar")
    return merged


def validate_requests(requests: Sequence[DiffusionRequest], *, allow_images: bool = False) -> None:
    """Reject incompatible geometry/sampling/input policy before model execution."""
    if not requests or any(not isinstance(request, DiffusionRequest) for request in requests):
        raise ValueError("run_batch requires one or more DiffusionRequest values")
    first = requests[0]
    media_keys = {"image", "images"} if allow_images else set()
    first_policy = {key: value for key, value in first.inputs.items() if key not in media_keys}
    for request in requests:
        policy = {key: value for key, value in request.inputs.items() if key not in media_keys}
        unsupported = set(policy) - {
            "fps",
            "return_latent",
            "high_noise_guidance_scale",
            "low_noise_guidance_scale",
        }
        if unsupported or any(type(value) not in (bool, int, float) for value in policy.values()):
            raise ValueError("request batching supports scalar inputs and native image conditioning only")
        if any(type(value) not in (bool, int, float) for value in request.sampling.scheduler_options.values()):
            raise ValueError("request batching requires scalar scheduler options")
        if (
            (request.height, request.width, request.num_frames) != (first.height, first.width, first.num_frames)
            or replace(request.sampling, seed=first.sampling.seed) != first.sampling
            or policy != first_policy
        ):
            raise ValueError("request batch geometry, sampling options and input policy must match (seeds may differ)")


class _BatchReceipt(DiffusionExtension):
    extension_id = "request-batching"

    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.sizes = sizes
        self.batch_size = sum(sizes)
        self.request_id: str | None = None
        self.calls = 0

    def on_run_start(self, context: DiffusionRunContext) -> None:
        self.request_id = context.request_id

    def before_denoiser(self, context: DiffusionRunContext, model_input: DenoiserInput) -> DenoiserInput:
        if model_input.latents.shape[0] != self.batch_size:
            raise ValueError("batched denoiser input must have the merged batch dimension")
        return model_input.with_updates(request_batch_sizes=self.sizes)

    def after_denoiser(self, context: DiffusionRunContext, model_output: DenoiserOutput) -> DenoiserOutput:
        if model_output.sample.shape[0] != self.batch_size:
            raise ValueError("batched denoiser must preserve the merged batch dimension")
        self.calls += 1
        return model_output


class _RequestBatch:
    """Request-local adapters for the non-denoiser roles of one merged run."""

    def __init__(self, runner: NativeDiffusionRunner, requests: tuple[DiffusionRequest, ...]) -> None:
        self.runner = runner
        self.requests = requests
        self.sizes = tuple(request.batch_size for request in requests)
        self.generators = tuple(runner._generator(request.sampling.seed) for request in requests)
        self.schedules: tuple[tuple[SchedulerStep, ...], ...] = ()
        self.artifacts: tuple[Mapping[str, object], ...] = ()

    def encode(self, request: DiffusionRequest, *, device: torch.device, dtype: torch.dtype) -> Conditioning:
        conditions = [
            self.runner.components.conditioner.encode(case, device=device, dtype=dtype) for case in self.requests
        ]
        if any(not isinstance(condition, Conditioning) for condition in conditions):
            raise TypeError("conditioner.encode must return Conditioning")
        return Conditioning(
            **{
                branch: _merge_conditions([getattr(condition, branch) for condition in conditions], self.sizes)
                for branch in ("positive", "negative", "shared")
            }
        )

    def initialize(
        self, request: DiffusionRequest, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype
    ) -> Tensor | LatentInitialization:
        initializer = self.runner.components.latent_initializer
        encoder = self.runner.components.latent_encoder
        values = [
            initializer.initialize(case, generator=rng, device=device, dtype=dtype)
            if encoder is None or not isinstance(initializer, EncodedLatentInitializer)
            else initializer.initialize_with_encoder(
                case,
                latent_encoder=encoder,
                generator=rng,
                device=device,
                dtype=dtype,
            )
            for case, rng in zip(self.requests, self.generators)
        ]
        if all(isinstance(value, Tensor) for value in values):
            self.artifacts = tuple({} for _ in values)
            return cast(Tensor, _merge_conditions([{"latents": value} for value in values], self.sizes)["latents"])
        if not all(isinstance(value, LatentInitialization) for value in values):
            raise TypeError("request batching requires matching tensor or LatentInitialization values")
        initializations = cast(list[LatentInitialization], values)
        self.artifacts = tuple(value.artifacts for value in initializations)
        latents = cast(
            Tensor, _merge_conditions([{"latents": value.latents} for value in initializations], self.sizes)["latents"]
        )
        return LatentInitialization(
            latents,
            _merge_conditions([value.conditioning for value in initializations], self.sizes),
        )

    def expected_step_count(self, sampling: SamplingConfig) -> int:
        expected = getattr(self.runner.components.scheduler, "expected_step_count", None)
        return expected(sampling) if callable(expected) else sampling.num_inference_steps

    def schedule(
        self, sampling: SamplingConfig, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[SchedulerStep, ...]:
        self.schedules = tuple(
            tuple(self.runner.components.scheduler.schedule(case.sampling, device=device, dtype=dtype))
            for case in self.requests
        )
        # Audited schedulers construct a fresh solver in schedule(), or are
        # stateless. Matching policies produce the same immutable time grid.
        return self.schedules[0]

    def scale_model_input(self, latents: Tensor, step: SchedulerStep) -> Tensor:
        return torch.cat(
            [
                self.runner.components.scheduler.scale_model_input(part, schedule[step.index])
                for part, schedule in zip(latents.split(self.sizes), self.schedules)
            ]
        )

    def step(self, model_output: Tensor, step: SchedulerStep, latents: Tensor, *, generator: torch.Generator) -> Tensor:
        return torch.cat(
            [
                self.runner.components.scheduler.step(prediction, schedule[step.index], part, generator=rng)
                for prediction, part, schedule, rng in zip(
                    model_output.split(self.sizes), latents.split(self.sizes), self.schedules, self.generators
                )
            ]
        )

    def decode(self, latents: Tensor, request: DiffusionRequest) -> Tensor:
        samples = [
            self.runner.components.decoder.decode(part, case)
            for part, case in zip(latents.split(self.sizes), self.requests)
        ]
        return cast(Tensor, _merge_conditions([{"sample": sample} for sample in samples], self.sizes)["sample"])


@torch.no_grad()
def run_request_batch(
    runner: NativeDiffusionRunner, requests: Sequence[DiffusionRequest]
) -> tuple[DiffusionOutput, ...]:
    """Run compatible Wan requests through one native loop, in input order.

    Supports native T2V, image-conditioned and frozen-first-frame initialization
    with standard or dual-expert guidance and Euler/UniPC. Custom extensions,
    approximate caches and distributed execution need
    separate batching contracts. Validation and execution failures propagate;
    no partial outputs or singleton retries are returned.
    """
    from ..extensions.frozen_mask import FrozenLatentMaskExtension
    from ..models.initializers.wan.component import (
        WanImageToVideoLatentInitializer,
        WanTextImageToVideoLatentInitializer,
        WanTextToVideoLatentInitializer,
    )
    from ..schedulers.flow_match import FlowMatchEulerScheduler
    from ..schedulers.wan import WanFlowMatchEulerScheduler, WanFlowUniPCScheduler

    cases = tuple(requests)
    initializer = runner.components.latent_initializer
    image_initializer = type(initializer) in (WanImageToVideoLatentInitializer, WanTextImageToVideoLatentInitializer)
    validate_requests(cases, allow_images=image_initializer)
    masked = type(initializer) is WanTextImageToVideoLatentInitializer
    supported_extensions = (
        len(runner.extensions) == 1 and type(runner.extensions[0]) is FrozenLatentMaskExtension
        if masked
        else not runner.extensions
    )
    if (
        type(runner) not in (NativeDiffusionRunner, Wan22DualExpertGuidanceRunner)
        or not supported_extensions
        or runner.cfg_parallel_degree != 1
        or runner.cfg_gate_step != 1.0
    ):
        raise ValueError("request batching requires native guidance without custom extensions, CFG gating or CFG parallelism")
    if type(initializer) is not WanTextToVideoLatentInitializer and not image_initializer:
        raise ValueError("request batching requires a native Wan T2V, I2V or TI2V initializer")
    if image_initializer:
        if runner.components.latent_encoder is None:
            raise ValueError("image request batching requires a latent encoder")
        has_images = tuple(WanTextImageToVideoLatentInitializer._optional_image(case) is not None for case in cases)
        if len(set(has_images)) != 1 or (not masked and not has_images[0]):
            raise ValueError("request batching requires matching image-conditioning presence")
    if type(runner.components.scheduler) not in (
        FlowMatchEulerScheduler,
        WanFlowMatchEulerScheduler,
        WanFlowUniPCScheduler,
    ):
        raise ValueError("request batching requires a request-local native Euler or Wan UniPC scheduler")
    validate = getattr(runner.components.denoiser, "validate_request_batching", None)
    if not callable(validate):
        raise ValueError("denoiser does not support request batching")
    validate()

    batch = _RequestBatch(runner, cases)
    merged = replace(
        cases[0],
        prompt=tuple(prompt for case in cases for prompt in case.prompts),
        negative_prompt=tuple(
            prompt for case in cases for prompt in (case.negative_prompts or ("",) * case.batch_size)
        ),
        metadata={},
    )
    receipt = _BatchReceipt(batch.sizes)
    components = RunnerComponents(
        denoiser=runner.components.denoiser,
        conditioner=batch,
        latent_initializer=batch,
        scheduler=batch,
        decoder=batch,
    )
    if isinstance(runner, Wan22DualExpertGuidanceRunner):
        batched_runner = Wan22DualExpertGuidanceRunner(
            model_id=runner.model_id,
            components=components,
            device=runner.device,
            dtype=runner.dtype,
            guidance_mode=runner.guidance_mode,
            extensions=(*runner.extensions, receipt),
            boundary_ratio=runner.boundary_ratio,
            low_noise_guidance_scale=runner.low_noise_guidance_scale,
            high_noise_guidance_scale=runner.high_noise_guidance_scale,
            num_train_timesteps=runner.num_train_timesteps,
        )
    else:
        batched_runner = NativeDiffusionRunner(
            model_id=runner.model_id,
            components=components,
            device=runner.device,
            dtype=runner.dtype,
            guidance_mode=runner.guidance_mode,
            extensions=(*runner.extensions, receipt),
        )
    result = batched_runner.run(merged)
    outputs: list[DiffusionOutput] = []
    offset = 0
    for index, (case, sample, latents) in enumerate(
        zip(cases, result.sample.split(batch.sizes), result.latents.split(batch.sizes))
    ):
        metadata = dict(result.metadata)
        metadata.update(
            seed=case.sampling.seed,
            extensions=[extension.extension_id for extension in runner.extensions],
            request_metadata=dict(case.metadata),
            request_batching={
                "execution": "batched" if len(cases) > 1 else "singleton",
                "batch_id": receipt.request_id,
                "request_count": len(cases),
                "sample_count": merged.batch_size,
                "request_index": index,
                "sample_range": [offset, offset + case.batch_size],
                "denoiser_calls": receipt.calls,
                "receipt_scope": "batch",
            },
        )
        # Each caller owns its storage, including when a decoder returns the
        # latents directly. Retaining one output must not retain another's rows.
        outputs.append(
            DiffusionOutput(
                sample=sample.clone(),
                latents=latents.clone(),
                artifacts=batch.artifacts[index],
                metadata=metadata,
            )
        )
        offset += case.batch_size
    return tuple(outputs)
