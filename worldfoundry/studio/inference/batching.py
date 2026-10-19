# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Lightweight admission for native Wan resident-worker request batches."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

_NATIVE_T2V_MODELS = frozenset({"wan2.1-t2v-1.3b", "wan2.2-t2v-a14b"})
_NATIVE_IMAGE_MODELS = frozenset({"wan2.2-ti2v-5b"})
_BATCH_LOAD_OPTIONS = frozenset(
    {
        "model_path",
        "checkpoint_path",
        "checkpoint_dir",
        "pretrained_model_path",
        "checkpoint_overrides",
        "device",
        "torch_dtype",
        "dtype",
        "weight_dtype",
        "dit_weight_dtype",
        "offload_mode",
        "attention_backend",
        "compile",
        "torch_compile",
        "compile_scope",
        "compile_mode",
        "compile_backend",
        "compile_fullgraph",
        "compile_dynamic",
        "compile_options",
        "static_cross_kv",
        "fuse_qkv",
        "qkv_strategy",
        "qkv_split_threshold",
        "fused_rope",
        "rope_precision",
        "rms_norm_precision",
        "cuda_graph",
        "inplace_residual",
        "warmup_requests",
        "vae_weight_dtype",
        "vae_decode_autocast",
        "vae_channels_last_3d",
        "vae_spatial_tiling",
        "vae_temporal_chunk_size",
        "vae_tile_size",
        "vae_tile_stride",
        "vae_tiled_decode",
    }
)


def native_batch_key(run_kwargs: Mapping[str, Any]) -> str | None:
    """Return a compatibility key, or None to retain ordinary singleton serving.

    The worker repeats admission before loading. Native request validation owns
    tensor compatibility; this control-plane key never imports Torch or models.
    """
    model_id = run_kwargs.get("model_id")
    image_model = model_id in _NATIVE_IMAGE_MODELS
    if model_id not in _NATIVE_T2V_MODELS | _NATIVE_IMAGE_MODELS or run_kwargs.get("action") != "run":
        return None
    if run_kwargs.get("backend") not in ("auto", "from_pretrained"):
        return None
    if any(
        run_kwargs.get(key)
        for key in (
            "image",
            "video",
            "last_frame",
            "reference_files",
            "interactions_text",
            "camera_view_text",
            "intrinsics_text",
            "meta_path",
            "panorama_path",
            "api_key",
        )
    ):
        return None
    if not isinstance(run_kwargs.get("prompt", ""), str):
        return None
    try:
        load = json.loads(run_kwargs.get("load_kwargs_text") or "{}")
        call = json.loads(run_kwargs.get("call_kwargs_text") or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(load, dict) or not isinstance(call, dict):
        return None
    if set(load) - _BATCH_LOAD_OPTIONS:
        return None
    if load.get("offload_mode") not in ("resident", "none") or load.get("attention_backend") != "torch":
        return None
    if load.get("compile", load.get("torch_compile", False)) and load.get("compile_scope") != "regional":
        return None
    if load.get("fuse_qkv") or load.get("fused_rope"):
        return None
    if load.get("rope_precision", "fp64") != "fp64" or load.get("rms_norm_precision", "fp32") != "fp32":
        return None
    if image_model and (load.get("rope_precision") != "fp64" or load.get("rms_norm_precision") != "fp32"):
        # The 5B catalog defaults do not select this numerical profile.
        return None
    if not isinstance(call.get("prompt", ""), str) or call.get("return_dict") is False:
        return None
    input_path = run_kwargs.get("input_path", "")
    if image_model:
        # Workspace supplies one image through its generic input picker.
        # Explicit call aliases stay singleton so precedence is unchanged.
        if not isinstance(input_path, str) or any(
            name in call for name in ("images", "image_path", "input_path", "ref_image_path", "video", "video_path")
        ):
            return None
        if input_path and PurePath(input_path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return None
    elif input_path:
        return None
    key = dict(run_kwargs)
    key.pop("prompt", None)
    if image_model:
        key.pop("input_path", None)
        key["image_conditioned"] = bool(input_path)
    key["load_kwargs_text"] = load
    key["call_kwargs_text"] = {
        name: value
        for name, value in call.items()
        if name not in {"prompt", "negative_prompt", "seed", "output_path", "output_dir"}
    }
    return json.dumps(key, sort_keys=True, ensure_ascii=True)
