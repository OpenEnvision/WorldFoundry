"""Native Cosmos Predict1 / GEN3C denoiser role.

EDM-preconditioned wrapper around the native 3D transformer.  The runner
supplies ``DenoiserInput``; this class maps GEN3C video-pose conditions and
returns clean-x0 style ``DenoiserOutput.sample``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch

from ...components import ComponentBuildContext
from ...contracts import DenoiserInput, DenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader
from ..networks.cosmos1 import Cosmos1Transformer3DModel

COSMOS1_GEN3C_7B_CONFIG = {
    "in_channels": 81,
    "out_channels": 16,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "num_layers": 28,
    "rope_scale": (2.0, 1.0, 1.0),
}


def _convert_block_key(key: str) -> str | None:
    match = re.match(r"blocks\.block(\d+)\.blocks\.([012])\.(.*)", key)
    if match is None:
        return None
    layer, subblock, suffix = match.groups()
    prefix = f"transformer_blocks.{layer}"
    norm = {"0": "norm1", "1": "norm2", "2": "norm3"}[subblock]
    if suffix.startswith("adaLN_modulation.1"):
        return f"{prefix}.{norm}.linear_1{suffix.removeprefix('adaLN_modulation.1')}"
    if suffix.startswith("adaLN_modulation.2"):
        return f"{prefix}.{norm}.linear_2{suffix.removeprefix('adaLN_modulation.2')}"
    if subblock in {"0", "1"} and suffix.startswith("block.attn."):
        attention = "attn1" if subblock == "0" else "attn2"
        suffix = suffix.removeprefix("block.attn.")
        replacements = {
            "to_q.0": "to_q",
            "to_q.1": "norm_q",
            "to_k.0": "to_k",
            "to_k.1": "norm_k",
            "to_v.0": "to_v",
            "to_out.0": "to_out.0",
        }
        for old, new in replacements.items():
            if suffix.startswith(old):
                return f"{prefix}.{attention}.{new}{suffix.removeprefix(old)}"
    if subblock == "2" and suffix.startswith("block.layer1"):
        return f"{prefix}.ff.net.0.proj{suffix.removeprefix('block.layer1')}"
    if subblock == "2" and suffix.startswith("block.layer2"):
        return f"{prefix}.ff.net.2{suffix.removeprefix('block.layer2')}"
    return None


def convert_cosmos1_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Map NVIDIA's GEN3C checkpoint onto the native inference-only module."""

    converted: dict[str, object] = {}
    direct = {
        "x_embedder.proj.1.weight": "patch_embed.proj.weight",
        "extra_pos_embedder.pos_emb_h": "extra_position.pos_emb_h",
        "extra_pos_embedder.pos_emb_w": "extra_position.pos_emb_w",
        "extra_pos_embedder.pos_emb_t": "extra_position.pos_emb_t",
        "t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
        "t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.adaLN_modulation.1.weight": "norm_out.linear_1.weight",
        "final_layer.adaLN_modulation.2.weight": "norm_out.linear_2.weight",
        "affline_norm.weight": "time_norm.weight",
    }
    for source, value in state_dict.items():
        key = source.removeprefix("model.").removeprefix("net.")
        target = direct.get(key) or _convert_block_key(key)
        if target is not None:
            converted[target] = value
    return converted


class Cosmos1Gen3CDenoiser:
    """GEN3C Video2World denoiser with EDM preconditioning and pose control."""
    def __init__(self, model: Cosmos1Transformer3DModel, *, sigma_data: float = 0.5) -> None:
        self.model = model
        self.sigma_data = float(sigma_data)

    def __call__(self, model_input: DenoiserInput) -> DenoiserOutput:
        values = model_input.conditioning
        context = values.get("context")
        condition_latents = values.get("condition_latents")
        indicator = values.get("condition_indicator")
        input_mask = values.get("condition_video_input_mask")
        pose = values.get("condition_video_pose")
        condition_noise = values.get("condition_noise")
        required = (context, condition_latents, indicator, input_mask, pose, condition_noise)
        if not all(isinstance(value, torch.Tensor) for value in required):
            raise TypeError("Cosmos1 GEN3C conditioning is incomplete")

        latents = model_input.latents
        context, condition_latents, indicator, input_mask, pose, condition_noise = (
            value.to(device=latents.device, dtype=latents.dtype) for value in required
        )
        sigma = model_input.timestep.to(device=latents.device, dtype=latents.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(latents.shape[0])
        elif sigma.numel() != latents.shape[0]:
            raise ValueError("Cosmos1 GEN3C sigma must be scalar or match the latent batch")
        if torch.any(sigma <= 0):
            raise ValueError("Cosmos1 GEN3C sigma must be positive")
        c_noise = sigma.log().mul(0.25)
        sigma = sigma.reshape(-1, 1, 1, 1, 1)
        sigma_data = self.sigma_data
        c_in = torch.rsqrt(sigma.square() + sigma_data**2)
        augment_sigma = float(values.get("condition_augment_sigma", 0.001))
        augmented = condition_latents + condition_noise * augment_sigma
        conditioned_input = augmented / (augment_sigma**2 + sigma_data**2) ** 0.5
        active_indicator = indicator * (sigma > augment_sigma).to(dtype=indicator.dtype)
        model_latents = active_indicator * conditioned_input + (1 - active_indicator) * (latents * c_in)
        branch_pose = torch.zeros_like(pose) if model_input.branch == "negative" else pose
        network_input = torch.cat((model_latents, input_mask, branch_pose), dim=1)
        padding_mask = values.get("padding_mask")
        prediction = self.model(
            network_input,
            c_noise,
            context,
            attention_mask=values.get("context_mask") if isinstance(values.get("context_mask"), torch.Tensor) else None,
            fps=values.get("fps", 24.0),
            padding_mask=padding_mask if isinstance(padding_mask, torch.Tensor) else None,
        )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(
                "Cosmos1 GEN3C denoiser produced non-finite values "
                f"for branch={model_input.branch!r}, sigma={float(sigma.flatten()[0])}"
            )

        c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)
        clean = c_skip * latents + c_out * prediction
        clean = active_indicator * condition_latents + (1 - active_indicator) * clean
        if not bool(torch.isfinite(clean).all()):
            raise FloatingPointError(
                "Cosmos1 GEN3C EDM preconditioning produced non-finite clean latents "
                f"for branch={model_input.branch!r}, sigma={float(sigma.flatten()[0])}"
            )
        return DenoiserOutput(sample=clean)


def build_cosmos1_gen3c_denoiser(context: ComponentBuildContext) -> Cosmos1Gen3CDenoiser:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos1Transformer3DModel,
            config=COSMOS1_GEN3C_7B_CONFIG,
            state_dict_converter=convert_cosmos1_state_dict,
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
    if not isinstance(model, Cosmos1Transformer3DModel):
        raise TypeError(f"expected Cosmos1Transformer3DModel, got {type(model).__name__}")
    return Cosmos1Gen3CDenoiser(
        model,
        sigma_data=float(context.component_options.get("sigma_data", 0.5)),
    )


__all__ = [
    "COSMOS1_GEN3C_7B_CONFIG",
    "Cosmos1Gen3CDenoiser",
    "build_cosmos1_gen3c_denoiser",
    "convert_cosmos1_state_dict",
]
