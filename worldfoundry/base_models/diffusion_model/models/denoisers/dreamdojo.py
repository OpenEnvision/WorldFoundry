# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""DreamDojo 2B GR-1 checkpoint mapping and native velocity contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.cosmos2p5.action import ActionChunkCosmosTransformer
from .cosmos2p5 import COSMOS25_2B_CONFIG, convert_cosmos25_state_dict


def convert_dreamdojo_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Select EMA tensors and reuse the canonical Cosmos checkpoint mapping."""
    if any(key.startswith("net_ema.") for key in state_dict):
        state_dict = {
            key.removeprefix("net_ema."): value for key, value in state_dict.items() if key.startswith("net_ema.")
        }
    return convert_cosmos25_state_dict(state_dict)


class DreamDojoDenoiser:
    """Action-conditioned velocity prediction with the observed frame frozen."""

    def __init__(self, model: ActionChunkCosmosTransformer) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        latents = model_input.latents
        context, actions, condition, mask, noise = (
            values.get(name) for name in ("context", "actions", "condition_latents", "condition_mask", "initial_noise")
        )
        if not all(isinstance(value, torch.Tensor) for value in (context, actions, condition, mask, noise)):
            raise TypeError("DreamDojo requires text, robot actions, and an encoded observation")
        context, actions = cast(torch.Tensor, context), cast(torch.Tensor, actions)
        dtype = next(self.model.parameters()).dtype
        condition, mask, noise = (cast(torch.Tensor, value).to(latents) for value in (condition, mask, noise))
        model_latents = mask * condition + (1 - mask) * latents
        prediction = self.model(
            model_latents.to(dtype),
            model_input.timestep.to(latents).reshape(-1, 1).div(1000).to(dtype),
            context.to(device=latents.device, dtype=dtype),
            action=actions.to(device=latents.device, dtype=dtype),
            condition_mask=mask,
            padding_mask=values.get("padding_mask"),
        ).float()
        return DenoiserOutput(sample=(noise - condition) * mask + prediction * (1 - mask))


def build_dreamdojo_denoiser(context: ComponentBuildContext) -> DreamDojoDenoiser:
    """Load the released action-chunk 2B network through native checkpoint IO."""
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=ActionChunkCosmosTransformer,
            config={**COSMOS25_2B_CONFIG, "rope_scale": (1.0, 3.0, 3.0), "action_dim": 384},
            state_dict_converter=convert_dreamdojo_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.LayerNorm: AutoWrappedModule,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="transformer_blocks",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, ActionChunkCosmosTransformer):
        raise TypeError("DreamDojo loader must build an ActionChunkCosmosTransformer")
    return DreamDojoDenoiser(model)
