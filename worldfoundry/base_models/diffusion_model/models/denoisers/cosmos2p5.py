"""Cosmos Predict 2.5 denoiser component and official checkpoint mapping.

Maps NVIDIA training keys onto the native inference DiT and exposes a
``Denoiser`` for Predict 2.5 and Transfer 2.5 recipes.  The runner never
sees the vendor checkpoint layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from ...components import BuildPurpose, ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.cosmos2p5.model import Cosmos25Transfer3DModel, Cosmos25Transformer3DModel

_KEY_REPLACEMENTS = {
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn.q_proj": "attn1.to_q",
    "self_attn.q_norm": "attn1.norm_q",
    "self_attn.k_proj": "attn1.to_k",
    "self_attn.k_norm": "attn1.norm_k",
    "self_attn.v_proj": "attn1.to_v",
    "self_attn.output_proj": "attn1.to_out.0",
    "cross_attn.q_proj": "attn2.to_q",
    "cross_attn.q_norm": "attn2.norm_q",
    "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.k_norm": "attn2.norm_k",
    "cross_attn.v_proj": "attn2.to_v",
    "cross_attn.output_proj": "attn2.to_out.0",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "control_embedder.proj.1": "control_embedder.proj",
    "t_embedder.1.linear_1": "time_embed.t_embedder.linear_1",
    "t_embedder.1.linear_2": "time_embed.t_embedder.linear_2",
    "t_embedding_norm": "time_norm",
    "crossattn_proj.0": "text_embed.0",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}


def _convert_key(source: str) -> str | None:
    key = source.removeprefix("net.").removeprefix("model.")
    if key.endswith("_extra_state") or key.startswith(("accum_", "pos_embedder.", "loss.")):
        return None
    if key.startswith("blocks."):
        key = f"transformer_blocks.{key.removeprefix('blocks.')}"
    for old, new in _KEY_REPLACEMENTS.items():
        key = key.replace(old, new)
    return key


def convert_cosmos25_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map NVIDIA's training checkpoint names onto the native inference DiT."""

    converted: dict[str, object] = {}
    for source, value in state_dict.items():
        key = _convert_key(source)
        if key is None:
            continue
        converted[key] = value
    return converted


def convert_cosmos25_transfer_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map NVIDIA's combined base/VACE checkpoint onto the native Transfer DiT."""

    return convert_cosmos25_state_dict(state_dict)


COSMOS25_2B_CONFIG = {
    "in_channels": 17,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "rope_enable_fps_modulation": False,
}
COSMOS25_14B_CONFIG = {**COSMOS25_2B_CONFIG, "num_attention_heads": 40, "num_layers": 36}
COSMOS25_TRANSFER_2B_CONFIG = {
    **COSMOS25_2B_CONFIG,
    "rope_scale": (1.0, 3.0, 3.0),
    "num_max_modalities": 8,
    "control_block_every_n": 7,
    "rope_enable_fps_modulation": False,
}


class Cosmos25Denoiser:
    """Predict 2.5 DiT adapter that returns a velocity-style ``DenoiserOutput``."""

    def __init__(self, model: Cosmos25Transformer3DModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        context = values.get("context")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Cosmos2.5 denoising requires a context tensor")
        latents = model_input.latents
        model_dtype = next(self.model.parameters(), latents).dtype
        condition_latents = values.get("condition_latents")
        condition_mask = values.get("condition_mask")
        condition_indicator = values.get("condition_indicator")
        initial_noise = values.get("initial_noise")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (condition_latents, condition_mask, condition_indicator, initial_noise)
        ):
            raise TypeError("Cosmos2.5 latent initialization conditions are missing")
        condition_latents = condition_latents.to(device=latents.device, dtype=latents.dtype)
        condition_mask = condition_mask.to(device=latents.device, dtype=latents.dtype)
        condition_indicator = condition_indicator.to(device=latents.device, dtype=latents.dtype)
        initial_noise = initial_noise.to(device=latents.device, dtype=latents.dtype)
        model_latents = condition_mask * condition_latents + (1 - condition_mask) * latents
        timestep = model_input.timestep.to(device=latents.device, dtype=latents.dtype).reshape(-1, 1) / 1000.0
        if timestep.shape[0] == 1:
            timestep = timestep.expand(latents.shape[0], -1)
        elif timestep.shape[0] != latents.shape[0]:
            raise ValueError("Cosmos2.5 timestep must be scalar or match the latent batch")
        conditional_timestep = float(values.get("conditional_frame_timestep", -1.0))
        if conditional_timestep >= 0.0:
            indicator = condition_indicator[:, 0, :, 0, 0]
            timestep = indicator * (conditional_timestep / 1000.0) + (1 - indicator) * timestep
        padding_mask = values.get("padding_mask")
        prediction = self.model(
            model_latents.to(model_dtype),
            timestep.to(model_dtype),
            context.to(model_dtype),
            fps=float(values.get("fps", 16.0)),
            condition_mask=condition_mask,
            padding_mask=padding_mask if isinstance(padding_mask, torch.Tensor) else None,
            control_hidden_states=values.get("control_hidden_states"),
        )
        ground_truth_velocity = initial_noise - condition_latents
        prediction = ground_truth_velocity * condition_mask + prediction * (1 - condition_mask)
        return DenoiserOutput(sample=prediction)


class Cosmos25TransferDenoiser(Cosmos25Denoiser):
    def __init__(self, model: Cosmos25Transfer3DModel) -> None:
        self.model = model

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        context = values.get("context")
        latent_control_input = values.get("latent_control_input")
        if not isinstance(context, torch.Tensor):
            raise TypeError("Cosmos Transfer2.5 denoising requires a context tensor")
        if not isinstance(latent_control_input, torch.Tensor):
            raise TypeError("Cosmos Transfer2.5 requires an encoded control video")
        latents = model_input.latents
        model_dtype = next(self.model.parameters(), latents).dtype
        conditions = tuple(
            values.get(name) for name in ("condition_latents", "condition_mask", "condition_indicator", "initial_noise")
        )
        if not all(isinstance(value, torch.Tensor) for value in conditions):
            raise TypeError("Cosmos Transfer2.5 latent initialization conditions are missing")
        condition_latents, condition_mask, condition_indicator, initial_noise = (
            value.to(device=latents.device, dtype=latents.dtype) for value in conditions
        )
        model_latents = condition_mask * condition_latents + (1 - condition_mask) * latents
        timestep = model_input.timestep.to(device=latents.device, dtype=latents.dtype).reshape(-1, 1) / 1000.0
        if timestep.shape[0] == 1:
            timestep = timestep.expand(latents.shape[0], -1)
        elif timestep.shape[0] != latents.shape[0]:
            raise ValueError("Cosmos Transfer2.5 timestep must be scalar or match the latent batch")
        conditional_timestep = float(values.get("conditional_frame_timestep", -1.0))
        if conditional_timestep >= 0.0:
            indicator = condition_indicator[:, 0, :, 0, 0]
            timestep = indicator * (conditional_timestep / 1000.0) + (1 - indicator) * timestep
        padding_mask = values.get("padding_mask")
        prediction = self.model(
            model_latents.to(model_dtype),
            timestep.to(model_dtype),
            context.to(model_dtype),
            latent_control_input=latent_control_input.to(device=latents.device, dtype=model_dtype),
            fps=float(values.get("fps", 16.0)),
            condition_mask=condition_mask,
            padding_mask=padding_mask if isinstance(padding_mask, torch.Tensor) else None,
            control_context_scale=values.get("control_context_scale", 1.0),
        )
        ground_truth_velocity = initial_noise - condition_latents
        prediction = ground_truth_velocity * condition_mask + prediction * (1 - condition_mask)
        return DenoiserOutput(sample=prediction)


def _build(context: ComponentBuildContext, config: Mapping[str, object]) -> Cosmos25Denoiser:
    from worldfoundry.core.distributed.model_parallel.inference_parallel import (
        InferenceParallelContext,
        build_inference_parallel_context,
        validate_inference_parallel_policy,
        validate_parallel_degrees,
    )
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    tp, cp = validate_parallel_degrees(
        context.component_options.get("tensor_parallel", 1),
        context.component_options.get("context_parallel", 1),
    )
    parallel = context.component_options.get("parallel_context")
    owns_context = False
    load_policy = context.policy
    if tp * cp > 1:
        validate_inference_parallel_policy(context.policy)
        if context.purpose is not BuildPurpose.INFERENCE:
            raise ValueError("Cosmos model parallelism supports inference builds only")
        if config["num_attention_heads"] % tp:
            raise ValueError("tensor_parallel must evenly divide Cosmos2.5 attention heads")
        if parallel is None:
            parallel = build_inference_parallel_context(
                tensor_parallel=tp, context_parallel=cp, device=context.policy.device,
            )
            owns_context = True
        if not isinstance(parallel, InferenceParallelContext):
            raise TypeError("parallel_context must be an InferenceParallelContext")
        if (parallel.tp_size, parallel.cp_size, parallel.device) != (tp, cp, context.policy.device):
            raise ValueError("parallel_context must match the component degrees and device")
        load_policy = replace(context.policy, device=torch.device("cpu"))
    elif parallel is not None:
        raise ValueError("parallel_context requires explicit multi-rank component degrees")

    try:
        if parallel is not None:
            parallel.agree(("cosmos25-load", dict(config), str(context.policy.dtype), str(context.require_checkpoint())))
        model = NativeModuleLoader().load(
            ModuleLoadSpec(
                module_class=Cosmos25Transformer3DModel,
                config=config,
                state_dict_converter=convert_cosmos25_state_dict,
                vram_module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.RMSNorm: AutoWrappedModule,
                },
                layer_container="transformer_blocks",
            ),
            context.require_checkpoint("weights"),
            load_policy,
        )
        if not isinstance(model, Cosmos25Transformer3DModel):
            raise TypeError(f"expected Cosmos25Transformer3DModel, got {type(model).__name__}")
        if parallel is not None:
            model.parallelize(parallel).to(device=context.policy.device)
    except BaseException:
        if owns_context:
            try:
                parallel.close()
            except Exception:
                pass
        raise
    return Cosmos25Denoiser(model)


def build_cosmos25_2b_denoiser(context: ComponentBuildContext) -> Cosmos25Denoiser:
    return _build(context, COSMOS25_2B_CONFIG)


def build_cosmos25_14b_denoiser(context: ComponentBuildContext) -> Cosmos25Denoiser:
    return _build(context, COSMOS25_14B_CONFIG)


def build_cosmos25_transfer_2b_denoiser(context: ComponentBuildContext) -> Cosmos25TransferDenoiser:
    from worldfoundry.core.distributed.model_parallel.inference_parallel import validate_parallel_degrees
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    tp, cp = validate_parallel_degrees(
        context.component_options.get("tensor_parallel", 1), context.component_options.get("context_parallel", 1),
    )
    if tp * cp > 1 or context.component_options.get("parallel_context") is not None:
        raise ValueError("multi-rank Cosmos Transfer inference is not supported")
    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos25Transfer3DModel,
            config=COSMOS25_TRANSFER_2B_CONFIG,
            state_dict_converter=convert_cosmos25_transfer_state_dict,
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
    if not isinstance(model, Cosmos25Transfer3DModel):
        raise TypeError(f"expected Cosmos25Transfer3DModel, got {type(model).__name__}")
    return Cosmos25TransferDenoiser(model)


__all__ = [
    "COSMOS25_2B_CONFIG",
    "COSMOS25_14B_CONFIG",
    "COSMOS25_TRANSFER_2B_CONFIG",
    "Cosmos25Denoiser",
    "Cosmos25TransferDenoiser",
    "build_cosmos25_2b_denoiser",
    "build_cosmos25_14b_denoiser",
    "build_cosmos25_transfer_2b_denoiser",
    "convert_cosmos25_state_dict",
    "convert_cosmos25_transfer_state_dict",
]
