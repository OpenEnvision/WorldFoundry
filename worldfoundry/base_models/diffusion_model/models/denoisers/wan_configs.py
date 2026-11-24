# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical architecture configs and CivitAI key-hash lookup for native Wan denoisers.

Extracted from :mod:`.wan` so pure configuration dictionaries stay decoupled from
runtime denoiser adapters, CUDA Graph wrapping, and checkpoint builders. All
public symbols are re-exported by :mod:`.wan` for full backwards compatibility.
"""

from __future__ import annotations

WAN21_T2V_1P3B_CONFIG: dict[str, object] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}

WAN21_T2V_14B_CONFIG: dict[str, object] = {
    **WAN21_T2V_1P3B_CONFIG,
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "num_layers": 40,
}

WAN22_T2V_A14B_CONFIG: dict[str, object] = dict(WAN21_T2V_14B_CONFIG)

WAN21_I2V_14B_CONFIG: dict[str, object] = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": True,
    "in_dim": 36,
}

# Some Wan2.1 research checkpoints (for example WoW 1.3B) train the
# first-frame VAE condition without adding the CLIP cross-attention branch.
# This is still the canonical Wan graph: only its explicit conditioning roles
# differ from the released 14B I2V checkpoint.
WAN21_VAE_I2V_1P3B_CONFIG: dict[str, object] = {
    **WAN21_T2V_1P3B_CONFIG,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

# Wan2.2 A14B I2V conditions the latent stream with the VAE mask/latent
# tensor but does not own the Wan2.1 CLIP image branch.  Keeping that split in
# one canonical graph is required by Fun-Control and FantasyWorld releases.
WAN22_I2V_A14B_CONFIG: dict[str, object] = {
    **WAN21_T2V_14B_CONFIG,
    "has_image_input": False,
    "in_dim": 36,
    "require_vae_embedding": True,
    "require_clip_embedding": False,
}

WAN22_TI2V_5B_CONFIG: dict[str, object] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "per_token_timestep": True,
}

# Self-Forcing's generator is the Wan2.1 T2V 1.3B geometry on the causal
# Self-Forcing graph, with global attention (no local window or sink).
SELF_FORCING_WAN21_T2V_1P3B_CONFIG: dict[str, object] = {
    "model_type": "t2v",
    "patch_size": (1, 2, 2),
    "text_len": 512,
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "local_attn_size": -1,
    "sink_size": 0,
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

# Approximate long-rollout preset, after FlashDreams' Self-Forcing
# ``sink5-window7-rerope`` pipeline: the first five latent frames stay in the
# cache as a static sink, a rolling window holds the seven newest (the recent
# four plus the current block of three), and RoPE positions are cache slots.
SELF_FORCING_LONG_ROLLOUT_KV_WINDOW = "sink5-window7-rerope"
SELF_FORCING_LONG_ROLLOUT_SINK_FRAMES = 5
SELF_FORCING_LONG_ROLLOUT_WINDOW_FRAMES = 7

SKYREELS_V2_DF_1P3B_CONFIG: dict[str, object] = {
    **WAN21_T2V_1P3B_CONFIG,
    "inject_sample_info": True,
}

SKYREELS_V3_R2V_14B_CONFIG: dict[str, object] = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 40,
    "num_layers": 40,
    "eps": 1e-6,
}

WAN_CIVITAI_CONFIGS_BY_HASH: dict[str, dict[str, object]] = {
    "9269f8db9040a9d860eaca435be61814": WAN21_T2V_1P3B_CONFIG,
    "aafcfd9672c3a2456dc46e1cb6e52c70": WAN21_T2V_14B_CONFIG,
    "6bfcfb3b342cb286ce886889d519a77e": WAN21_I2V_14B_CONFIG,
    "6d6ccde6845b95ad9114ab993d917893": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 36,
    },
    "349723183fc063b2bfc10bb2835cf677": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "efa44cddf936c70abd0ea28b6cbe946c": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
    },
    "3ef3b1f8e1dab83d5b71fd7b617f859f": {
        **WAN21_I2V_14B_CONFIG,
        "has_image_pos_emb": True,
    },
    "70ddad9d3a133785da5ea371aae09504": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "26bde73488a92e64cc20b0a7485b9e5b": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 48,
        "has_ref_conv": True,
    },
    "ac6a5aa74f4a0aab6f64eb9a72f19901": {
        **WAN21_T2V_1P3B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
    "b61c605c2adbd23124d152ed28e049ae": {
        **WAN21_T2V_14B_CONFIG,
        "has_image_input": True,
        "in_dim": 32,
        "add_control_adapter": True,
        "in_dim_control_adapter": 24,
    },
}

__all__ = [
    "SELF_FORCING_LONG_ROLLOUT_KV_WINDOW",
    "SELF_FORCING_LONG_ROLLOUT_SINK_FRAMES",
    "SELF_FORCING_LONG_ROLLOUT_WINDOW_FRAMES",
    "SELF_FORCING_WAN21_T2V_1P3B_CONFIG",
    "SKYREELS_V2_DF_1P3B_CONFIG",
    "SKYREELS_V3_R2V_14B_CONFIG",
    "WAN21_I2V_14B_CONFIG",
    "WAN21_T2V_1P3B_CONFIG",
    "WAN21_T2V_14B_CONFIG",
    "WAN21_VAE_I2V_1P3B_CONFIG",
    "WAN22_I2V_A14B_CONFIG",
    "WAN22_T2V_A14B_CONFIG",
    "WAN22_TI2V_5B_CONFIG",
    "WAN_CIVITAI_CONFIGS_BY_HASH",
]
