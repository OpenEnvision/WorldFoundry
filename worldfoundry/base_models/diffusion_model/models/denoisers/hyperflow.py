# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
# Adapted from Video-Rebirth/hyperflow at 1dd2f342aba5ab51da02b62885939655e8e268da.
"""Released HyperFlow and Diffusers-format H3 keys mapped to canonical H3 variants."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...schedulers.hyperflow import DEFAULT_SIGMAS_8STEP
from ..networks.minimax_h3.config import MiniMaxH3DiTArchConfig
from ..networks.minimax_h3.loading import _shard_paths
from ..networks.minimax_h3.variants.hyperflow.conditioning import HyperFlowLinear, HyperFlowTwoTimeEmbedder
from ..networks.minimax_h3.variants.hyperflow.model import HyperFlowDiTModel


@dataclass(frozen=True)
class HyperFlowMetadata:
    gate: float
    alpha: float
    rank: int
    sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float

    @classmethod
    def from_header(cls, header: Mapping[str, str]) -> HyperFlowMetadata:
        if header.get("hyperflow", "").lower() != "true":
            raise ValueError("Not a HyperFlow safetensors file.")
        return cls(
            gate=float(header["hyperflow_gate"]),
            alpha=float(header["lora_alpha"]),
            rank=int(header["lora_rank"]),
            sigmas=tuple(json.loads(header["hyperflow_sigmas"]))
            if "hyperflow_sigmas" in header
            else DEFAULT_SIGMAS_8STEP,
            video_shift=float(header.get("hyperflow_video_shift", "12.0")),
            audio_shift=float(header.get("hyperflow_audio_shift", "3.0")),
        )


def hyperflow_arch_config(config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
    """Translate the released architecture fields; no runtime settings enter the graph."""
    names = {
        "num_refiner_layers": "token_refiner_num_layers",
        "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim",
        "audio_in_channels": "audio_latents_dim",
        "freq_dim": "timestep_input_dim",
        "time_embed_hidden_dim": "time_embed_hidden_size",
        "rope_freq_dim": "rope_inv_freq_len",
    }
    options = {
        names.get(name, name): value
        for name, value in config.items()
        if not name.startswith("_") and name != "rope_theta"
    }
    hidden = int(options.get("hidden_size", 5376))
    options.update(adaln_out_features=18 * hidden, final_adaln_out_features=2 * hidden)
    return MiniMaxH3DiTArchConfig(**options)


def hyperflow_foundation_key(key: str) -> str:
    """One-to-one released H3 key mapping, retaining unfused projection row order."""
    for source, target in (
        ("audio_proj_in.", "audio_patch_proj."),
        ("proj_in.", "video_patch_proj."),
        ("audio_proj_out.", "final_layer.audio_out."),
        ("proj_out.", "final_layer.video_out."),
        ("context_embedder.", "condition_proj."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("norm_out.norm.", "final_layer.norm."),
    ):
        if key.startswith(source):
            key = target + key[len(source) :]
            break
    for source, target in (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
        ("time_embedder.linear_1", "time_embedder.proj_in"),
        ("time_embedder.linear_2", "time_embedder.proj_out"),
        (".attn.to_q.", ".attn.qkv_proj.q."),
        (".attn.to_k.", ".attn.qkv_proj.k."),
        (".attn.to_v.", ".attn.qkv_proj.v."),
        (".attn.to_out.0.", ".attn.out_proj."),
        (".attn.norm_q.", ".attn.q_norm."),
        (".attn.norm_k.", ".attn.k_norm."),
        (".ff.net.0.proj.", ".mlp.fc1.proj."),
        (".ff.net.2.", ".mlp.fc2."),
    ):
        key = key.replace(source, target)
    return key


def hyperflow_adapter_plan(keys: Iterable[str]) -> dict[str, dict[str, str]]:
    """Map each released A/B pair onto exactly one canonical projection."""
    plan: dict[str, dict[str, str]] = {}
    for key in keys:
        match = re.fullmatch(r"transformer\.(.+)\.lora_([AB])\.weight", key)
        if match is None:
            raise ValueError(f"Unexpected HyperFlow adapter key: {key}")
        module, matrix = match.groups()
        if module.startswith("endpoint_time_embedder."):
            module = "time_embedder.endpoint." + module.removeprefix("endpoint_time_embedder.")
        elif module.startswith("time_embedder."):
            module = "time_embedder.base." + module.removeprefix("time_embedder.")
        module = module.replace(".linear_1", ".proj_in").replace(".linear_2", ".proj_out")
        module = hyperflow_foundation_key(module + ".weight").removesuffix(".weight")
        plan.setdefault(module, {})[matrix] = key
    if not plan or any(set(pair) != {"A", "B"} for pair in plan.values()):
        raise ValueError("HyperFlow requires complete, nonempty LoRA A/B pairs.")
    return plan


def load_hyperflow_adapter(model: HyperFlowDiTModel, path: str | Path) -> HyperFlowMetadata:
    """Load the local two-time adapter without merging LoRA into the base weights."""
    from safetensors import safe_open

    if isinstance(model.time_embedder, HyperFlowTwoTimeEmbedder):
        raise ValueError("HyperFlow owns one adapter; load it onto a fresh H3 model.")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = HyperFlowMetadata.from_header(handle.metadata() or {})
        model.time_embedder = HyperFlowTwoTimeEmbedder(model.time_embedder, metadata.gate)
        plan = hyperflow_adapter_plan(handle.keys())
        for target, pair in plan.items():
            base = model.get_submodule(target)
            if not isinstance(base, nn.Linear):
                raise ValueError(f"HyperFlow adapter target is not a linear projection: {target}")
            expected = {"A": [metadata.rank, base.in_features], "B": [base.out_features, metadata.rank]}
            for matrix, key in pair.items():
                if handle.get_slice(key).get_shape() != expected[matrix]:
                    raise ValueError(f"HyperFlow shape mismatch for {key}; expected {expected[matrix]}")
            adapter = HyperFlowLinear(base, metadata.rank, metadata.alpha)
            for matrix, key in pair.items():
                projection = getattr(adapter, f"lora_{matrix}")
                projection.weight = nn.Parameter(handle.get_tensor(key).to(base.weight), requires_grad=False)
            parent, _, name = target.rpartition(".")
            setattr(model.get_submodule(parent), name, adapter)
    return metadata


def load_hyperflow_transformer(
    component_dir: str | Path,
    weights_path: str | Path,
    *,
    attention_backend: str = "native",
    sol_attn: bool = False,
    degree: int = 1,
    rank: int = 0,
) -> tuple[HyperFlowDiTModel, HyperFlowMetadata]:
    """Strictly load a local transformer/transformer_ref snapshot and its HyperFlow adapter."""
    from safetensors.torch import load_file

    directory = Path(component_dir)
    config = json.loads((directory / "config.json").read_text())
    with torch.device("meta"):
        model = HyperFlowDiTModel(
            hyperflow_arch_config(config),
            attention_backend=attention_backend,
            sol_attn=sol_attn,
            degree=degree,
            rank=rank,
            rope_theta=float(config.get("rope_theta", 10000.0)),
        )
    state: dict[str, torch.Tensor] = {}
    for shard in _shard_paths(directory):
        for name, value in load_file(str(shard)).items():
            target = hyperflow_foundation_key(name)
            if target in state:
                raise ValueError(f"Duplicate HyperFlow foundation tensor: {target}")
            state[target] = value
    model.load_state_dict(state, strict=True, assign=True)
    model.post_load_weights()
    metadata = load_hyperflow_adapter(model, weights_path)
    return model.eval().requires_grad_(False), metadata
