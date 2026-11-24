"""Strict local construction for the AHA-WAM inference architecture."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import WanTextEncoder
from worldfoundry.base_models.diffusion_model.models.networks.gamma_world.multiagent_rope import (
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
)

from .action_dit import ActionDiT
from .ahawam import AHAWAM
from .mot import MoT
from .wan_video_dit import WanVideoDiT
from .wan_video_text_encoder import HuggingfaceTokenizer
from .wan_video_vae import WanVideoVAE38


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _tensor_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(state, Mapping) and isinstance(state.get("model_state"), Mapping):
            state = state["model_state"]
    if not isinstance(state, Mapping) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"weight file must contain a flat tensor mapping: {path}")
    return dict(state)


def _strict_assign(module: nn.Module, state: Mapping[str, torch.Tensor], name: str) -> None:
    try:
        module.load_state_dict(dict(state), strict=True, assign=True)
    except RuntimeError as error:
        raise RuntimeError(f"strict {name} checkpoint restoration failed: {error}") from error


def _build_meta_policy(architecture: Mapping[str, Any], dtype: torch.dtype) -> AHAWAM:
    video_config = _mapping(architecture.get("video_expert"), "architecture.video_expert")
    action_config = _mapping(architecture.get("action_expert"), "architecture.action_expert")
    policy_config = _mapping(architecture.get("policy"), "architecture.policy")
    vae_config = _mapping(architecture.get("vae"), "architecture.vae")
    text_encoder_config = _mapping(architecture.get("text_encoder"), "architecture.text_encoder")
    scheduler_config = _mapping(architecture.get("scheduler"), "architecture.scheduler")
    video_scheduler = _mapping(scheduler_config.get("video"), "architecture.scheduler.video")
    action_scheduler = _mapping(scheduler_config.get("action"), "architecture.scheduler.action")

    with torch.device("meta"):
        video_expert = WanVideoDiT(**video_config)
        action_expert = ActionDiT(**action_config)
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("video and action experts must use the same attention head count")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("video and action experts must use the same attention head dimension")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("video and action experts must use the same layer count")
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
        )
        vae = WanVideoVAE38(**vae_config)
        text_encoder = WanTextEncoder(**text_encoder_config)
        model = AHAWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=int(video_config["text_dim"]),
            proprio_dim=int(policy_config["proprio_dim"]),
            device="meta",
            torch_dtype=dtype,
            video_infer_shift=float(video_scheduler["shift"]),
            video_num_timesteps=int(video_scheduler["num_timesteps"]),
            action_infer_shift=float(action_scheduler["shift"]),
            action_num_timesteps=int(action_scheduler["num_timesteps"]),
        )
        model.configure_action_chunking(
            action_horizon=int(policy_config["action_horizon"]),
            action_chunk_size=int(policy_config["action_chunk_size"]),
        )
        model.configure_chunk_obs_context(
            action_obs_downsample_factor=int(policy_config["action_obs_downsample_factor"]),
            chunk_kv_editor_num_queries=int(policy_config["chunk_kv_editor_num_queries"]),
            chunk_kv_delta_gate=bool(policy_config["chunk_kv_delta_gate"]),
            chunk_kv_gate_init=float(policy_config["chunk_kv_gate_init"]),
        )
        model.configure_chunk_history(
            num_history_frames=int(policy_config["num_history_frames"]),
            action_video_read_mode=str(policy_config["action_video_read_mode"]),
            video_rope_frame_stride=int(policy_config["video_rope_frame_stride"]),
        )
        model.max_action_offset = int(policy_config["max_action_offset"])
    return model


def restore_ahawam_model(
    *,
    policy_checkpoint: Path,
    vae_checkpoint: Path,
    text_encoder_checkpoint: Path,
    tokenizer_path: Path,
    architecture: Mapping[str, Any],
    policy_device: torch.device,
    policy_dtype: torch.dtype,
    vae_device: torch.device,
    vae_dtype: torch.dtype,
    text_encoder_device: torch.device,
    text_encoder_dtype: torch.dtype,
) -> AHAWAM:
    """Restore all released assets strictly without a remote-code or network path."""

    model = _build_meta_policy(architecture, policy_dtype)
    payload = torch.load(
        policy_checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"policy checkpoint must contain a mapping: {policy_checkpoint}")
    required = (
        "mot",
        "proprio_encoder",
        "action_obs_visual_proj",
        "chunk_obs_query_encoder",
    )
    missing = [name for name in required if not isinstance(payload.get(name), Mapping)]
    if missing:
        raise ValueError(f"policy checkpoint is missing required inference states: {missing}")
    _strict_assign(model.mot, payload["mot"], "mot")
    if model.proprio_encoder is None:
        raise RuntimeError("released AHA-WAM architecture requires a proprio encoder")
    _strict_assign(model.proprio_encoder, payload["proprio_encoder"], "proprio encoder")
    _strict_assign(model.action_obs_visual_proj, payload["action_obs_visual_proj"], "visual projection")
    _strict_assign(model.chunk_obs_query_encoder, payload["chunk_obs_query_encoder"], "chunk query encoder")
    del payload
    gc.collect()

    vae_state = _tensor_state(vae_checkpoint)
    if vae_state and not next(iter(vae_state)).startswith("model."):
        vae_state = {f"model.{key}": value for key, value in vae_state.items()}
    _strict_assign(model.vae, vae_state, "Wan VAE")
    del vae_state
    gc.collect()

    text_state = _tensor_state(text_encoder_checkpoint)
    _strict_assign(model.text_encoder, text_state, "UMT5 text encoder")
    del text_state
    gc.collect()

    model.mot.to(device=policy_device, dtype=policy_dtype)
    model.proprio_encoder.to(device=policy_device, dtype=policy_dtype)
    model.action_obs_visual_proj.to(device=policy_device, dtype=policy_dtype)
    model.chunk_obs_query_encoder.to(device=policy_device, dtype=policy_dtype)
    model.vae.to(device=vae_device, dtype=vae_dtype)
    model.text_encoder.to(device=text_encoder_device, dtype=text_encoder_dtype)
    model.tokenizer = HuggingfaceTokenizer(
        name=str(tokenizer_path),
        seq_len=int(_mapping(architecture.get("text"), "architecture.text")["tokenizer_max_length"]),
        clean="whitespace",
        local_files_only=True,
    )
    model.video_expert.freqs = tuple(
        frequency.to(device=policy_device)
        for frequency in precompute_freqs_cis_3d(
            model.video_expert.attn_head_dim,
            end=model.video_expert.rope_max_length,
            theta=model.video_expert.rope_theta,
        )
    )
    model.action_expert.freqs = precompute_freqs_cis(
        model.action_expert.attn_head_dim,
        end=model.action_expert.rope_max_length,
        theta=model.action_expert.rope_theta,
    ).to(device=policy_device)
    model.device = policy_device
    model.torch_dtype = policy_dtype
    model.eval()
    return model


__all__ = ["restore_ahawam_model"]
