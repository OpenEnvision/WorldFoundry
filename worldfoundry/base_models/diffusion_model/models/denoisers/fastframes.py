# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Adapted from apple-aiml-research/ml-fastframes, revision
# a65c59240e01cf1de86197fa771ca09b38cf592a (Apple Sample Code License).
"""Strict checkpoint loaders and component assembly for FastFrames-S and FastFrames-L."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.fastframes import (
    FASTFRAMES_WAN_CONFIGS,
    BridgedPixelDecoder,
    FastFramesCausalWanModel,
    FastFramesWanConfig,
    KVRefinementNetwork,
    PixelDecoder,
)
from worldfoundry.base_models.diffusion_model.schedulers.fastframes import (
    FastFramesVariantConfig,
    resolve_fastframes_variant,
)
from worldfoundry.core.io.paths import resolve_local_checkpoint_file, resolve_local_hf_model_path

__all__ = [
    "FastFramesComponents",
    "build_fastframes_decoder",
    "build_fastframes_dit",
    "build_fastframes_refiner",
    "load_fastframes_components",
    "load_fastframes_state_dict",
    "load_fastframes_text_encoder",
]

_STATE_DICT_PREFIXES: tuple[str, ...] = (
    "model._fsdp_wrapped_module.",
    "_fsdp_wrapped_module.",
    "model.",
)


@dataclass(frozen=True)
class FastFramesComponents:
    """Assembled inference components for a FastFrames variant."""

    config: FastFramesVariantConfig
    dit: FastFramesCausalWanModel
    refiner: KVRefinementNetwork
    decoder: PixelDecoder | BridgedPixelDecoder
    text_encoder: Any | None = None


def _strip_prefix(name: str) -> str:
    for prefix in _STATE_DICT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def load_fastframes_state_dict(
    source: str | Path | Mapping[str, torch.Tensor],
    key: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load a tensor state dict from ``.safetensors``, sharded index JSON, or ``.pt``.

    For PyTorch checkpoints (``.pt`` / ``.pth``), extracts nested ``state[key]`` when
    ``key`` is provided (e.g., ``"generator_ema"`` or ``"adapter"``), then strips
    FSDP and ``model.`` wrapper prefixes.
    """
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
        if key is not None and key in raw and isinstance(raw[key], Mapping):
            raw = raw[key]
        return {_strip_prefix(str(k)): v for k, v in raw.items()}

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"FastFrames checkpoint file not found: {path}")

    if path.name.endswith(".safetensors.index.json"):
        from safetensors.torch import load_file

        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"Invalid safetensors index file (missing weight_map): {path}")
        state: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            state.update(load_file(str(path.parent / shard_name)))
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Expected mapping checkpoint in {path}, got {type(loaded).__name__}")
        if key is not None:
            if key not in loaded:
                raise KeyError(f"Expected key {key!r} in checkpoint {path}; found {sorted(loaded.keys())[:10]}")
            nested = loaded[key]
            if not isinstance(nested, Mapping):
                raise ValueError(f"Checkpoint key {key!r} in {path} is not a state dict mapping")
            state = dict(nested)
        else:
            state = dict(loaded)
    return {_strip_prefix(str(k)): v for k, v in state.items()}


def _count_indexed_blocks(state_dict: Mapping[str, torch.Tensor], prefix: str, default: int) -> int:
    indices: set[int] = set()
    for name in state_dict:
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        index_text, sep, _ = remainder.partition(".")
        if sep and index_text.isdigit():
            indices.add(int(index_text))
    return len(indices) if indices else default


def build_fastframes_dit(
    wan_model: str | FastFramesWanConfig,
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> FastFramesCausalWanModel:
    """Strictly load a ``FastFramesCausalWanModel`` without materializing random weights."""
    wan_cfg = wan_model if isinstance(wan_model, FastFramesWanConfig) else FASTFRAMES_WAN_CONFIGS[wan_model]
    with torch.device("meta"):
        model = FastFramesCausalWanModel(wan_cfg)
    model.load_state_dict(dict(state_dict), strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def build_fastframes_refiner(
    wan_model: str | FastFramesWanConfig,
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    stem_n_heads: int | None = None,
) -> KVRefinementNetwork:
    """Strictly load a ``KVRefinementNetwork`` inferring stem width and block count from weights."""
    wan_cfg = wan_model if isinstance(wan_model, FastFramesWanConfig) else FASTFRAMES_WAN_CONFIGS[wan_model]
    if "stem_in.weight" not in state_dict:
        raise KeyError("KV refinement state dict is missing required key 'stem_in.weight'")
    d_a = int(state_dict["stem_in.weight"].shape[0])
    freq_dim = int(state_dict["t_embed.mlp.0.weight"].shape[1]) if "t_embed.mlp.0.weight" in state_dict else 256
    n_stem = _count_indexed_blocks(state_dict, "stem_blocks.", default=4)
    stem_mlp_mult = (
        int(state_dict["stem_blocks.0.mlp.0.weight"].shape[0]) // d_a
        if "stem_blocks.0.mlp.0.weight" in state_dict
        else 4
    )
    head_mlp_mult = (
        int(state_dict["heads_grouped.gemms.0.m0_w"].shape[2]) // d_a
        if "heads_grouped.gemms.0.m0_w" in state_dict
        else 2
    )
    resolved_stem_heads = stem_n_heads if stem_n_heads is not None else (8 if d_a % 8 == 0 else 4)
    refiner = KVRefinementNetwork(
        dim=wan_cfg.dim,
        num_heads=wan_cfg.num_heads,
        head_dim=wan_cfg.dim // wan_cfg.num_heads,
        num_blocks=wan_cfg.num_layers,
        d_a=d_a,
        n_stem=n_stem,
        stem_n_heads=resolved_stem_heads,
        stem_mlp_mult=stem_mlp_mult,
        head_mlp_mult=head_mlp_mult,
        freq_dim=freq_dim,
    ).to(dtype=dtype)
    refiner.load_state_dict(dict(state_dict), strict=True)
    return refiner.to(device=device, dtype=dtype).eval().requires_grad_(False)


def build_fastframes_decoder(
    wan_dim: int,
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
) -> PixelDecoder | BridgedPixelDecoder:
    """Strictly load either a ``PixelDecoder`` (FF-S) or ``BridgedPixelDecoder`` (FF-L)."""
    has_bridge = any(k.startswith("bridge.") for k in state_dict)
    if has_bridge:
        bridge_weight = state_dict["bridge.weight"]
        decoder_dim = int(bridge_weight.shape[0])
        in_dim = int(bridge_weight.shape[1])
        if in_dim != wan_dim:
            raise ValueError(f"BridgedPixelDecoder bridge input dim ({in_dim}) does not match wan_dim ({wan_dim})")
        n_low = _count_indexed_blocks(state_dict, "pixel_head.conv_low.", default=2)
        n_high = _count_indexed_blocks(state_dict, "pixel_head.conv_high.", default=1)
        ffn_mult = (
            int(state_dict["pixel_head.conv_low.0.pw1.weight"].shape[0]) // decoder_dim
            if "pixel_head.conv_low.0.pw1.weight" in state_dict
            else 4
        )
        decoder: PixelDecoder | BridgedPixelDecoder = BridgedPixelDecoder(
            in_dim=wan_dim,
            dim=decoder_dim,
            n_low_blocks=n_low,
            n_high_blocks=n_high,
            ffn_mult=ffn_mult,
        )
        decoder.load_state_dict(dict(state_dict), strict=True)
    else:
        normalized = {k.removeprefix("pixel_head."): v for k, v in state_dict.items()}
        n_low = _count_indexed_blocks(normalized, "conv_low.", default=2)
        n_high = _count_indexed_blocks(normalized, "conv_high.", default=1)
        ffn_mult = (
            int(normalized["conv_low.0.pw1.weight"].shape[0]) // wan_dim if "conv_low.0.pw1.weight" in normalized else 4
        )
        decoder = PixelDecoder(
            wan_dim,
            n_low_blocks=n_low,
            n_high_blocks=n_high,
            ffn_mult=ffn_mult,
        )
        decoder.load_state_dict(normalized, strict=True)
    return decoder.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_fastframes_text_encoder(
    wan_dir: str | Path,
    *,
    text_len: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> Any:
    """Load the canonical WorldFoundry Wan2.1 UMT5-XXL prompt encoder from ``wan_dir``."""
    from worldfoundry.base_models.diffusion_model.models.encoders.wan.component import WanUMT5PromptEncoder

    root = Path(wan_dir).expanduser()
    checkpoint_path = root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = root / "google" / "umt5-xxl"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Wan2.1 UMT5 checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Wan2.1 UMT5 tokenizer directory not found: {tokenizer_path}")
    return WanUMT5PromptEncoder(
        checkpoint_path=str(checkpoint_path),
        tokenizer_path=str(tokenizer_path),
        text_length=text_len,
        dtype=dtype,
        device=device,
    )


def _resolve_component_paths(
    variant_cfg: FastFramesVariantConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    wan_dir: str | Path | None = None,
    dit_path: str | Path | Mapping[str, torch.Tensor] | None = None,
    refiner_path: str | Path | Mapping[str, torch.Tensor] | None = None,
    decoder_path: str | Path | Mapping[str, torch.Tensor] | None = None,
) -> tuple[
    Path | None,
    Path | Mapping[str, torch.Tensor],
    Path | Mapping[str, torch.Tensor],
    Path | Mapping[str, torch.Tensor],
]:
    root = None if checkpoint_dir is None else Path(checkpoint_dir).expanduser()

    def _resolve_file(
        explicit: str | Path | Mapping[str, torch.Tensor] | None,
        rel_name: str,
        repo_id: str,
    ) -> Path | Mapping[str, torch.Tensor]:
        if isinstance(explicit, Mapping):
            return explicit
        if explicit is not None:
            candidate = Path(explicit).expanduser()
            if candidate.is_dir():
                nested = candidate / rel_name
                if nested.is_file():
                    return nested
                leaf = candidate / Path(rel_name).name
                if leaf.is_file():
                    return leaf
            return candidate
        if root is not None:
            direct = root / rel_name
            if direct.is_file():
                return direct
            leaf = root / Path(rel_name).name
            if leaf.is_file():
                return leaf
        return resolve_local_checkpoint_file(repo_id, rel_name)

    resolved_dit = _resolve_file(dit_path, variant_cfg.dit_file, variant_cfg.dit_repo_id)
    resolved_refiner = _resolve_file(refiner_path, variant_cfg.refiner_file, variant_cfg.hf_repo_id)
    resolved_decoder = _resolve_file(decoder_path, variant_cfg.decoder_file, variant_cfg.hf_repo_id)

    resolved_wan: Path | None = None
    if wan_dir is not None:
        resolved_wan = Path(wan_dir).expanduser()
    elif root is not None and (root / variant_cfg.wan_dir).is_dir():
        resolved_wan = root / variant_cfg.wan_dir
    else:
        try:
            resolved_wan = resolve_local_hf_model_path(
                variant_cfg.wan_repo_id,
                required_files=("models_t5_umt5-xxl-enc-bf16.pth",),
            )
        except FileNotFoundError:
            resolved_wan = root / variant_cfg.wan_dir if root is not None else None

    return resolved_wan, resolved_dit, resolved_refiner, resolved_decoder


def load_fastframes_components(
    *,
    variant: str | FastFramesVariantConfig = "fastframes-s",
    checkpoint_dir: str | Path | None = None,
    wan_dir: str | Path | None = None,
    dit_path: str | Path | Mapping[str, torch.Tensor] | None = None,
    refiner_path: str | Path | Mapping[str, torch.Tensor] | None = None,
    decoder_path: str | Path | Mapping[str, torch.Tensor] | None = None,
    wan_config: FastFramesWanConfig | None = None,
    device: str | torch.device = "cuda",
    dit_dtype: torch.dtype = torch.bfloat16,
    decoder_dtype: torch.dtype | None = None,
    load_text_encoder: bool = True,
    stem_n_heads: int | None = None,
) -> FastFramesComponents:
    """Load and strictly verify FastFrames DiT, KV refiner, pixel decoder, and UMT5 encoder."""
    variant_cfg = resolve_fastframes_variant(variant)
    wan_cfg = wan_config if wan_config is not None else FASTFRAMES_WAN_CONFIGS[variant_cfg.wan_model]

    resolved_wan, resolved_dit, resolved_refiner, resolved_decoder = _resolve_component_paths(
        variant_cfg,
        checkpoint_dir=checkpoint_dir,
        wan_dir=wan_dir,
        dit_path=dit_path,
        refiner_path=refiner_path,
        decoder_path=decoder_path,
    )

    dit_state = load_fastframes_state_dict(resolved_dit, key="generator_ema")
    dit = build_fastframes_dit(wan_cfg, dit_state, dtype=dit_dtype, device=device)

    refiner_state = load_fastframes_state_dict(resolved_refiner, key="adapter")
    refiner = build_fastframes_refiner(
        wan_cfg,
        refiner_state,
        dtype=dit_dtype,
        device=device,
        stem_n_heads=stem_n_heads,
    )

    decoder_state = load_fastframes_state_dict(resolved_decoder)
    effective_decoder_dtype = variant_cfg.decoder_dtype if decoder_dtype is None else decoder_dtype
    decoder = build_fastframes_decoder(
        wan_cfg.dim,
        decoder_state,
        dtype=effective_decoder_dtype,
        device=device,
    )

    text_encoder = None
    if load_text_encoder:
        if resolved_wan is None:
            raise FileNotFoundError(
                f"Wan2.1 snapshot ({variant_cfg.wan_repo_id}) not found locally; pass wan_dir or stage checkpoints."
            )
        text_encoder = load_fastframes_text_encoder(
            resolved_wan,
            text_len=wan_cfg.text_len,
            dtype=dit_dtype,
            device=device,
        )

    return FastFramesComponents(
        config=variant_cfg,
        dit=dit,
        refiner=refiner,
        decoder=decoder,
        text_encoder=text_encoder,
    )
