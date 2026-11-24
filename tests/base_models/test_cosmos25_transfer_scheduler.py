# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Regression test for Cosmos-Transfer2.5 shifted rectified-flow UniPC scheduling."""

from __future__ import annotations

from typing import cast

import torch

from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKind,
    RuntimePolicy,
)
from worldfoundry.base_models.diffusion_model.contracts import SamplingConfig
from worldfoundry.base_models.diffusion_model.recipes.cosmos2p5 import cosmos25_transfer_2b_recipe
from worldfoundry.base_models.diffusion_model.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from worldfoundry.base_models.diffusion_model.schedulers.wan import WanFlowUniPCScheduler


def test_cosmos_transfer_2b_recipe_uses_rectified_flow_unipc_without_karras() -> None:
    recipe = cosmos25_transfer_2b_recipe()
    scheduler_spec = next(spec for spec in recipe.components if spec.key.kind is ComponentKind.SCHEDULER)

    assert scheduler_spec.options["use_karras_sigma"] is False
    assert scheduler_spec.options["shift"] == 5.0

    context = ComponentBuildContext(
        model_id=recipe.model_id,
        key=scheduler_spec.key,
        policy=RuntimePolicy(),
        component_options=scheduler_spec.options,
    )
    scheduler = cast(WanFlowUniPCScheduler, scheduler_spec.factory(context))
    schedule = scheduler.schedule(
        SamplingConfig(num_inference_steps=4, scheduler_options={"shift": 5.0}),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reference = FlowUniPCMultistepScheduler(**dict(scheduler.solver.config))
    reference.set_timesteps(4, device=torch.device("cpu"), shift=5.0, use_kerras_sigma=False)

    assert len(schedule) == 4
    torch.testing.assert_close(
        torch.stack([step.timestep for step in schedule]),
        reference.timesteps,
    )
