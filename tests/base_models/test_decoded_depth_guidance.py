# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Native depth-guidance gradients, request isolation and cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DenoiserOutput,
    DiffusionRequest,
    SamplingConfig,
    SchedulerStep,
)
from worldfoundry.base_models.diffusion_model.extensions.base import DiffusionRunContext
from worldfoundry.base_models.diffusion_model.extensions.depth_guidance import DecodedDepthGuidance
from worldfoundry.base_models.diffusion_model.runners.base import RunnerComponents

pytestmark = pytest.mark.unit


def test_guidance_recomputes_real_gradients_and_keeps_state_request_local() -> None:
    network = torch.nn.Conv2d(1, 1, 1, bias=False).requires_grad_(False)
    network.weight.data.fill_(0.5)
    latent = torch.ones(1, 1, 2, 2)

    class Denoiser:
        def __call__(self, inputs: DenoiserInput) -> DenoiserOutput:
            return DenoiserOutput(network(inputs.latents))

    class Scheduler:
        def step(
            self, prediction: torch.Tensor, step: SchedulerStep, latents: torch.Tensor, *, generator: torch.Generator
        ) -> torch.Tensor:
            return latents - prediction

    class Decoder:
        def decode_for_guidance(self, values: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
            return values / 200 + 0.001

    request = DiffusionRequest(
        prompt="",
        sampling=SamplingConfig(num_inference_steps=4),
        inputs={
            "target_depth": torch.full_like(latent, 0.001),
            "align_mask": torch.ones_like(latent).bool(),
            "guidance_steps": 2,
        },
    )
    components = cast(RunnerComponents, SimpleNamespace(denoiser=Denoiser(), scheduler=Scheduler(), decoder=Decoder()))
    context = DiffusionRunContext(
        request=request,
        components=components,
        conditioning=Conditioning(positive={}),
        generator=torch.Generator().manual_seed(0),
        step=SchedulerStep(2, torch.tensor(5), torch.tensor(1)),
    )
    extension = DecodedDepthGuidance()
    inputs = DenoiserInput(latent, torch.tensor(5), torch.tensor(1), {}, 2, 4)
    baseline = components.denoiser(inputs)
    with torch.no_grad():
        extension.before_denoiser(context, inputs)
        actual = extension.after_denoiser(context, baseline)
    # The gradient is normalized to twice the prediction norm at this step.
    torch.testing.assert_close(actual.sample, torch.full_like(latent, 1.5))
    assert not actual.sample.requires_grad and network.weight.grad is None
    assert context.state == {}
    extension.before_denoiser(context, inputs)
    extension.on_run_error(context, RuntimeError("cancelled"))
    assert context.state == {}
    unanchored = DiffusionRunContext(
        request=DiffusionRequest(prompt="", sampling=SamplingConfig(num_inference_steps=4)),
        components=components,
        conditioning=Conditioning(positive={}),
        generator=torch.Generator(),
    )
    unanchored.step = context.step
    extension.before_denoiser(unanchored, inputs)
    assert extension.after_denoiser(unanchored, baseline) is baseline
