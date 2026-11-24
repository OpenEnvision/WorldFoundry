# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""Split DiffVAE checkpoint conversion, shared by resident and lazy disk loads.

Derived from Lightricks/LTX-2 at a95ab856 (LTX-2.x Community License;
see THIRD-PARTY-NOTICES). Static gates fold in FP32 and round once to the
stored dtype; QKV payloads split without changing their values.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import torch

from worldfoundry.core.model_loading.checkpoints.transforms import TensorTransform, TransformedStateDict

from .transformer.config import DiffVAEMode


def _split_qkv(value: torch.Tensor, *, index: int) -> torch.Tensor:
    if value.shape[0] % 3:
        raise ValueError("fused DiffVAE QKV leading dimension must be divisible by three")
    return value.chunk(3, dim=0)[index].detach().clone()


def _fold_gate(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    scale = gate.to(device=value.device, dtype=torch.float32)
    if value.ndim == 2:
        scale = scale.unsqueeze(1)
    return (scale * value.float()).to(value.dtype)


def _zero_type_embedding(weight: torch.Tensor) -> torch.Tensor:
    return torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)


def _softmax_bound(weight: torch.Tensor) -> torch.Tensor:
    from ltx_kernels.vae import softmax_row_bound

    return softmax_row_bound(weight)


def _fused_context(
    weight: torch.Tensor,
    bias: torch.Tensor,
    upsample_weight: torch.Tensor,
    upsample_bias: torch.Tensor,
    *,
    index: int,
) -> torch.Tensor:
    from ltx_kernels.vae import pack_fused_ctx_weights

    return pack_fused_ctx_weights(upsample_weight, upsample_bias, weight, bias)[index]


def convert_ltx_diffusion_decoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    mode: str = "chunked_eager",
) -> Mapping[str, torch.Tensor]:
    """Map released decoder keys, fold gates, split QKV, and bind deferred stage four."""
    names: dict[str, str] = {}
    for key in state_dict:
        if key.startswith("vae.decoder."):
            names[key.removeprefix("vae.decoder.")] = key
        elif key.startswith("decoder."):
            names[key.removeprefix("decoder.")] = key
        elif key.startswith("vae.per_channel_statistics."):
            names[key.removeprefix("vae.")] = key
        elif key.startswith("per_channel_statistics."):
            names[key] = key
    source = TransformedStateDict(state_dict, names)
    keys: dict[str, str | TensorTransform] = {}
    gates = {".attn.proj": ".gate_msa", ".mlp.w_down": ".gate_mlp", ".context_proj": ".gate_ctx"}
    for key in source:
        if key.startswith("coarse_") or key.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")):
            continue
        if key.startswith("per_channel_statistics.") and key.rsplit(".", 1)[-1] not in {
            "mean-of-means",
            "std-of-means",
        }:
            continue
        destination = key.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        destination = destination.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        if key.endswith((".qkv.weight", ".qkv.bias")):
            stem, leaf = destination.rsplit(".", 1)
            for index, projection in enumerate(("to_q", "to_k", "to_v")):
                keys[f"{stem}.{projection}.{leaf}"] = TensorTransform((key,), partial(_split_qkv, index=index))
            continue
        keys[destination] = key
        for target, suffix in gates.items():
            for leaf in (".weight", ".bias"):
                if key.endswith(target + leaf):
                    gate = key.removesuffix(target + leaf) + suffix
                    if gate in source:
                        keys[destination] = TensorTransform((key, gate), _fold_gate)
    if "type_emb" not in keys:
        keys["type_emb"] = TensorTransform(("conv_in.weight",), _zero_type_embedding)
    folded = TransformedStateDict(source, keys)
    final: dict[str, str | TensorTransform] = {key: key for key in folded}
    resolved_mode = DiffVAEMode(mode)
    if resolved_mode in {DiffVAEMode.CHUNKED_EAGER, DiffVAEMode.CHUNKED_COMPILE, DiffVAEMode.BLACKWELL_DSL}:
        blocks = {key.split(".")[1] for key in folded if key.startswith("diff_blocks.")}
        for index in blocks:
            for leaf in ("weight", "bias"):
                final[f"diff_blocks.{index}.stage4_upsample.proj.{leaf}"] = f"upsamples.3.proj.{leaf}"
    if resolved_mode is DiffVAEMode.BLACKWELL_DSL:
        from .transformer.dsl_kernels import FUSED_CTX_BIAS_BUFFER, FUSED_CTX_WEIGHT_BUFFER, NA_SOFTMAX_BOUND_BUFFER

        for key in folded:
            if key.endswith(".attn.k_norm.weight"):
                final[key.removesuffix("k_norm.weight") + NA_SOFTMAX_BOUND_BUFFER] = TensorTransform(
                    (key,), _softmax_bound
                )
            if key.endswith(".context_proj.weight"):
                stem = key.removesuffix(".context_proj.weight")
                sources = (key, stem + ".context_proj.bias", "upsamples.3.proj.weight", "upsamples.3.proj.bias")
                for index, name in enumerate((FUSED_CTX_WEIGHT_BUFFER, FUSED_CTX_BIAS_BUFFER)):
                    final[f"{stem}.{name}"] = TensorTransform(sources, partial(_fused_context, index=index))
    converted = TransformedStateDict(folded, final)
    return TransformedStateDict(converted, {f"decoder.{key}": key for key in converted})
