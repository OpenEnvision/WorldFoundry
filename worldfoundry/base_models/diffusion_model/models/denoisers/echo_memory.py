"""Echo-Memory denoiser assembled on the shared Wan :class:`~...contracts.Denoiser` contract.

Echo-Memory is a Wan2.1-T2V-1.3B DiT whose blocks are *promoted* to
checkpoint-declared extra slots (action MLP, action attention, block-wise
SSM, VideoSSM hybrid, spatial memory adapter).  This module does not own
sampling: the runner still calls ``__call__(DenoiserInput) -> DenoiserOutput``.

``DenoiserInput.conditioning`` mapping
--------------------------------------
- ``context`` (required tensor): UMT5 prompt embeddings from the Wan
  :class:`~...contracts.ConditionEncoder`.
- ``actions`` (required): planar camera / embodiment actions; rank-2
  inputs are unsqueezed to a batch.
- ``num_context_frames`` (required positive int): how many latent frames
  are treated as memory context (suffix packing).

Weights load through :class:`~...loaders.module.NativeModuleLoader` after
:func:`.echo_memory_checkpoint.inspect_echo_checkpoint` validates the
safetensors header against the recipe in :mod:`.echo_memory_spec`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from worldfoundry.core.nn import RMSNorm

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeCheckpointResolver, NativeModuleLoader
from ..networks.echo_memory.architecture import validate_echo_checkpoint_architecture
from ..networks.echo_memory.blocks import EchoWanAttentionBlock, EchoWanMemoryAdapter
from ..networks.echo_memory.schema import EchoMemoryRecipe
from ..networks.wan.model import WanModel
from .echo_memory_checkpoint import (
    EchoCheckpointArchitecture,
    inspect_echo_checkpoint,
    normalize_echo_state_dict,
)
from .echo_memory_spec import EchoMemoryModelSpec, get_echo_memory_model_spec
from .wan import WAN21_T2V_1P3B_CONFIG


def _install_echo_memory_modules(
    model: torch.nn.Module,
    *,
    recipe: EchoMemoryRecipe,
    architecture: EchoCheckpointArchitecture,
) -> None:
    """Promote canonical Wan blocks to the checkpoint-declared Echo topology."""

    validate_echo_checkpoint_architecture(architecture, recipe, num_blocks=len(model.blocks))
    model_trainable = any(parameter.requires_grad for parameter in model.parameters())
    action_attention = bool(architecture.action_attention_blocks)
    new_blocks = torch.nn.ModuleList()
    for index, old_block in enumerate(model.blocks):
        parameter = next(old_block.parameters())
        block = EchoWanAttentionBlock(
            model.has_image_input,
            model.dim,
            old_block.num_heads,
            old_block.ffn_dim,
            eps=float(old_block.norm1.eps),
            action_attention=action_attention,
            block_state_space=index in architecture.block_ssm_blocks,
            video_ssm_hybrid=index in architecture.video_ssm_blocks,
        ).to(device=parameter.device, dtype=parameter.dtype)
        result = block.load_state_dict(old_block.state_dict(), strict=False)
        allowed_missing = (
            "action_mlp.",
            "self_attn_with_action.",
            "block_wise_ssm.",
            "videossm_hybrid.",
        )
        invalid_missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "canonical Wan block could not be promoted to Echo block: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        block.train(old_block.training)
        block.requires_grad_(any(value.requires_grad for value in old_block.parameters()))
        new_blocks.append(block)
    model.blocks = new_blocks

    parameter = next(model.parameters())
    model.memory_adapter = EchoWanMemoryAdapter(
        dim=model.dim,
        recipe=recipe,
        architecture=architecture,
    ).to(device=parameter.device, dtype=parameter.dtype)
    model.memory_adapter.train(model.training)
    model.memory_adapter.requires_grad_(model_trainable)


class EchoWanModel(WanModel):
    """Wan2.1-T2V-1.3B graph plus Echo memory / action / SSM slots.

    Construction first builds the canonical Wan stack, then replaces each
    attention block and installs ``memory_adapter`` so the checkpoint can
    load extra keys that a plain :class:`WanModel` would reject.
    """

    def __init__(
        self,
        *,
        echo_recipe: EchoMemoryRecipe,
        echo_architecture: EchoCheckpointArchitecture,
        **config,
    ) -> None:
        super().__init__(**config)
        _install_echo_memory_modules(
            self,
            recipe=echo_recipe,
            architecture=echo_architecture,
        )


class EchoMemoryDenoiser:
    """:class:`~...contracts.Denoiser` adapter for one Echo-Memory recipe.

    Maps ``DenoiserInput.latents`` / ``timestep`` / ``conditioning`` onto
    :meth:`EchoWanModel.forward` and returns velocity/flow as
    ``DenoiserOutput.sample``.  Autocast uses the recipe policy dtype.
    """

    def __init__(
        self,
        model: EchoWanModel,
        spec: EchoMemoryModelSpec,
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.spec = spec
        self.compute_dtype = compute_dtype

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        """Evaluate one CFG branch; TeaCache is not used (variable memory path)."""
        conditioning = model_input.conditioning
        context = conditioning.get("context")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Echo-Memory denoising requires tensor 'context' conditioning")
        actions = conditioning.get("actions")
        if actions is None:
            raise KeyError("Echo-Memory denoising requires 'actions' in request.inputs")
        actions = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        context_frames = int(conditioning.get("num_context_frames", 0))
        if context_frames <= 0:
            raise ValueError("Echo-Memory requires a positive num_context_frames")

        timestep = model_input.timestep.to(
            device=model_input.latents.device,
            dtype=torch.float32,
        ).reshape(-1)
        if timestep.numel() == 1 and model_input.latents.shape[0] != 1:
            timestep = timestep.expand(model_input.latents.shape[0])
        if timestep.numel() != model_input.latents.shape[0]:
            raise ValueError("Echo-Memory timestep must be scalar or match the latent batch")

        memory_context: Mapping[str, object] = {
            "actions": actions.to(
                device=model_input.latents.device,
                dtype=model_input.latents.dtype,
            ),
            "num_context_frames": context_frames,
            "context_position": "suffix",
            "recipe_id": self.spec.recipe_id,
        }
        autocast_enabled = self.compute_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(
            device_type=model_input.latents.device.type,
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            sample = self.model(
                x=model_input.latents,
                timestep=timestep,
                context=context,
                memory_context=memory_context,
            )
        return DenoiserOutput(sample=sample)


def build_echo_memory_denoiser(context: ComponentBuildContext) -> EchoMemoryDenoiser:
    """Load a full Echo DiT with exact keys through the shared module loader."""

    spec = get_echo_memory_model_spec(context.model_id)
    checkpoint = context.require_checkpoint("weights")
    materialized = NativeCheckpointResolver().materialize(checkpoint)
    if len(materialized.paths) != 1:
        raise ValueError("Echo-Memory denoiser requires exactly one safetensors checkpoint")
    inspection = inspect_echo_checkpoint(
        materialized.paths[0],
        spec.recipe,
        num_blocks=int(WAN21_T2V_1P3B_CONFIG["num_layers"]),
    )

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    weight_dtype = context.component_options.get("weight_dtype", torch.float32)
    if not isinstance(weight_dtype, torch.dtype):
        raise TypeError(f"Echo denoiser weight dtype must be a torch.dtype, got {weight_dtype!r}")
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=EchoWanModel,
            config={
                **WAN21_T2V_1P3B_CONFIG,
                "echo_recipe": spec.recipe,
                "echo_architecture": inspection.architecture,
            },
            state_dict_converter=normalize_echo_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.Conv1d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            layer_container="blocks",
        ),
        checkpoint,
        replace(context.policy, dtype=weight_dtype),
    )
    if not isinstance(model, EchoWanModel):
        raise TypeError(f"expected EchoWanModel, got {type(model).__name__}")
    model.requires_grad_(False)
    return EchoMemoryDenoiser(model, spec, compute_dtype=context.policy.dtype)


__all__ = [
    "EchoMemoryDenoiser",
    "EchoWanModel",
    "build_echo_memory_denoiser",
]
