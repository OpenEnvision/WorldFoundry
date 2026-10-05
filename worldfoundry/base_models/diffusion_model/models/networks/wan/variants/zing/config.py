# Adapted from seedleap/zing-world-model (Apache-2.0).
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GeneratorConfig:
    patch_size: tuple[int, int, int]
    in_dim: int
    out_dim: int
    dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    freq_dim: int
    text_dim: int
    qk_norm: bool
    cross_attn_norm: bool
    eps: float
    rope_max_seq_len: int
    compile_fusion: bool
    local_attn_size: int
    sink_size: int


@dataclass(frozen=True)
class ActionConfig:
    embed_dim: int
    hidden_dim: int
    kernel_size: int
    non_proj_bias: bool


@dataclass(frozen=True)
class VaeConfig:
    z_dim: int
    spatial_scale: int
    temporal_scale: int


@dataclass(frozen=True)
class TextEncoderConfig:
    max_length: int


@dataclass(frozen=True)
class InferenceConfig:
    num_timesteps: int
    timestep_shift: float
    denoising_steps: tuple[int, ...]
    frames_per_block: int
    output_fps: int
    deterministic_attention: bool


@dataclass(frozen=True)
class ZingConfig:
    generator: GeneratorConfig
    action: ActionConfig
    vae: VaeConfig
    text_encoder: TextEncoderConfig
    inference: InferenceConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ValueError(f"{name} keys mismatch: missing={sorted(missing)} extra={sorted(extra)}")


def _validate_cache_window(local_attn_size: int, sink_size: int, frames_per_block: int) -> None:
    if local_attn_size == -1:
        if sink_size != 0:
            raise ValueError("generator.sink_size requires a local attention window")
        return
    local_blocks = (local_attn_size + frames_per_block - 1) // frames_per_block
    sink_blocks = (sink_size + frames_per_block - 1) // frames_per_block
    if local_attn_size < 1 or sink_size < 0 or local_blocks - sink_blocks < 2:
        raise ValueError("generator local attention window is invalid")


def load_config(path: str | Path) -> ZingConfig:
    with Path(path).open(encoding="utf-8") as source:
        root = _mapping(yaml.safe_load(source), "config")
    _keys(root, {"generator", "action", "vae", "text_encoder", "inference"}, "config")

    generator = _mapping(root["generator"], "generator")
    generator_keys = {
        "patch_size",
        "in_dim",
        "out_dim",
        "dim",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "freq_dim",
        "text_dim",
        "qk_norm",
        "cross_attn_norm",
        "eps",
        "rope_max_seq_len",
        "compile_fusion",
        "local_attn_size",
        "sink_size",
    }
    _keys(generator, generator_keys, "generator")
    patch_size = tuple(int(value) for value in generator["patch_size"])
    if len(patch_size) != 3 or min(patch_size) < 1:
        raise ValueError("generator.patch_size must contain three positive integers")
    generator_config = GeneratorConfig(
        patch_size=patch_size, **{key: generator[key] for key in generator_keys - {"patch_size"}}
    )
    if generator_config.dim % generator_config.num_heads:
        raise ValueError("generator.dim must be divisible by generator.num_heads")
    generator_values = (
        *generator_config.patch_size,
        generator_config.in_dim,
        generator_config.out_dim,
        generator_config.dim,
        generator_config.num_heads,
        generator_config.num_layers,
        generator_config.ffn_dim,
        generator_config.freq_dim,
        generator_config.text_dim,
        generator_config.rope_max_seq_len,
    )
    if min(generator_values) < 1:
        raise ValueError("generator dimensions must be positive")
    action = _mapping(root["action"], "action")
    action_keys = {"embed_dim", "hidden_dim", "kernel_size", "non_proj_bias"}
    _keys(action, action_keys, "action")
    action_config = ActionConfig(**action)
    if (
        action_config.embed_dim % 8
        or min(action_config.embed_dim, action_config.hidden_dim, action_config.kernel_size) < 1
    ):
        raise ValueError("action dimensions are invalid")

    vae = _mapping(root["vae"], "vae")
    vae_keys = {"z_dim", "spatial_scale", "temporal_scale"}
    _keys(vae, vae_keys, "vae")
    vae_config = VaeConfig(**vae)
    if min(vae_config.z_dim, vae_config.spatial_scale, vae_config.temporal_scale) < 1:
        raise ValueError("vae values must be positive")

    text_encoder = _mapping(root["text_encoder"], "text_encoder")
    _keys(text_encoder, {"max_length"}, "text_encoder")
    text_encoder_config = TextEncoderConfig(**text_encoder)
    if text_encoder_config.max_length < 512:
        raise ValueError("text_encoder.max_length must be at least 512")

    inference = _mapping(root["inference"], "inference")
    inference_keys = {
        "num_timesteps",
        "timestep_shift",
        "denoising_steps",
        "frames_per_block",
        "output_fps",
        "deterministic_attention",
    }
    _keys(inference, inference_keys, "inference")
    denoising_steps = tuple(int(value) for value in inference["denoising_steps"])
    inference_config = InferenceConfig(
        denoising_steps=denoising_steps,
        **{key: inference[key] for key in inference_keys - {"denoising_steps"}},
    )
    if len(denoising_steps) != 4 or any(a <= b for a, b in zip(denoising_steps, denoising_steps[1:])):
        raise ValueError("inference.denoising_steps must be strictly descending")
    if denoising_steps[0] > inference_config.num_timesteps or denoising_steps[-1] < 0:
        raise ValueError("inference.denoising_steps are outside the diffusion range")
    if (
        min(
            inference_config.num_timesteps,
            inference_config.timestep_shift,
            inference_config.frames_per_block,
            inference_config.output_fps,
        )
        < 1
    ):
        raise ValueError("inference frame values must be positive")
    _validate_cache_window(
        generator_config.local_attn_size,
        generator_config.sink_size,
        inference_config.frames_per_block,
    )
    return ZingConfig(generator_config, action_config, vae_config, text_encoder_config, inference_config)


def with_cache_window(config: ZingConfig, local_attn_size: int | None, sink_size: int | None) -> ZingConfig:
    local = config.generator.local_attn_size if local_attn_size is None else int(local_attn_size)
    sink = config.generator.sink_size if sink_size is None else int(sink_size)
    _validate_cache_window(local, sink, config.inference.frames_per_block)
    return replace(config, generator=replace(config.generator, local_attn_size=local, sink_size=sink))
