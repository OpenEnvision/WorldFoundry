# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free component, parity, key-mapping, loader, and scheduler contracts for H3-Tiny."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.denoisers.h3_tiny import (
    detect_h3_tiny_endpoint_mode,
    h3_tiny_arch_config,
    h3_tiny_diffusers_to_wf_key,
    h3_tiny_wf_to_diffusers_key,
    load_h3_tiny_transformer,
    read_h3_tiny_config,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
    MiniMaxH3DiTBlock,
    MiniMaxH3TokenRefinerBlock,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_tiny import (
    H3TinyDiTModel,
    H3TinyMeanFlowTimeEmbedder,
    LowRankAdalnProj,
)
from worldfoundry.base_models.diffusion_model.schedulers.h3_tiny import (
    H3_TINY_AUDIO_SHIFT,
    H3_TINY_VIDEO_SHIFT,
    H3TinySchedule,
    h3_tiny_euler_step,
    h3_tiny_row_time_pairs,
    h3_tiny_shifted_sigmas,
)
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)

pytestmark = pytest.mark.unit

_FP32_DIFFUSERS_PREFIXES = (
    "proj_in.",
    "audio_proj_in.",
    "proj_out.",
    "audio_proj_out.",
    "time_embedder.",
    "r_time_embedder.",
)


def _tiny_diffusers_config(
    *,
    num_layers: int = 2,
    num_refiner_layers: int = 1,
    ffn_dim: int = 40,
    refiner_ffn_dim: int | None = 64,
    adaln_rank: int | None = 4,
    rope_theta: float = 10000.0,
) -> dict[str, Any]:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": num_layers,
        "num_refiner_layers": num_refiner_layers,
        "ffn_dim": ffn_dim,
        "refiner_ffn_dim": refiner_ffn_dim,
        "in_channels": 4,
        "audio_in_channels": 8,
        "patch_size": [1, 2, 2],
        "text_dim": 12,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "rope_theta": rope_theta,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
        "adaln_rank": adaln_rank,
    }


def _tiny_inputs(seed: int = 42) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, 5, 16, generator=g, dtype=torch.float32),
        "audio_hidden_states": torch.randn(1, 4, 8, generator=g, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(1, 3, 12, generator=g, dtype=torch.float32),
        "timestep": torch.tensor([0.15, 0.45, 0.999], dtype=torch.float32),
        "endpoint_timestep": torch.tensor([0.35, 0.70, 0.999], dtype=torch.float32),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0], dtype=torch.long),
        "token_tags": torch.tensor([1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0], dtype=torch.long),
        "position_ids": torch.randn(12, 3, generator=g, dtype=torch.float32),
        "video_indices": torch.arange(7, 12, dtype=torch.long),
        "audio_indices": torch.arange(3, 7, dtype=torch.long),
        "text_indices": torch.arange(3, dtype=torch.long),
    }


def _make_diffusers_state(
    config: dict[str, Any],
    *,
    endpoint_mode: str,
    block_dtype: torch.dtype,
    seed: int = 101,
) -> dict[str, torch.Tensor]:
    arch, variant_kwargs = h3_tiny_arch_config(config)
    torch.manual_seed(seed)
    model = H3TinyDiTModel(arch, endpoint_mode=endpoint_mode, **variant_kwargs)
    state: dict[str, torch.Tensor] = {}
    g = torch.Generator().manual_seed(seed)
    for wf_key, param in model.state_dict().items():
        diff_key = h3_tiny_wf_to_diffusers_key(wf_key)
        is_fp32 = diff_key.startswith(_FP32_DIFFUSERS_PREFIXES)
        dtype = torch.float32 if is_fp32 else block_dtype
        if diff_key.endswith(".weight") and param.ndim == 1:
            tensor = (1.0 + 0.05 * torch.randn(param.shape, generator=g, dtype=torch.float32)).to(dtype)
        else:
            tensor = (0.1 * torch.randn(param.shape, generator=g, dtype=torch.float32)).to(dtype)
        state[diff_key] = tensor.contiguous()
    return state


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _sinusoidal_time_proj_ref(t: torch.Tensor, freq_dim: int) -> torch.Tensor:
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _upstream_h3tiny_forward_ref(
    diffusers_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    *,
    endpoint_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct implementation of upstream `h3tiny/modeling.py` forward math for parity checks."""
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    hidden_size = int(config["hidden_size"])
    rope_freq_dim = int(config["rope_freq_dim"])
    rope_theta = float(config.get("rope_theta", 10000.0))
    norm_eps = float(config.get("norm_eps", 1e-5))
    qk_norm_eps = float(config.get("qk_norm_eps", 1e-5))
    final_norm_eps = float(config.get("final_norm_eps", 1e-5))
    freq_dim = int(config["freq_dim"])

    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = diffusers_state[f"{prefix}.weight"]
        b = diffusers_state.get(f"{prefix}.bias")
        return F.linear(x.to(w.dtype), w, b)

    # 3D RoPE matching Diffusers MiniMaxH3RotaryPosEmbed
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
    pos = inputs["position_ids"].float()
    per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    half_freqs = torch.cat((per_axis[:, 0], per_axis[:, 1], per_axis[:, 2]), dim=-1)
    block_dtype = diffusers_state["context_embedder.weight"].dtype
    cos_half = torch.cos(half_freqs).to(block_dtype)
    sin_half = torch.sin(half_freqs).to(block_dtype)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    rot_dim = cos.shape[-1]

    # Input projections + token refiner
    video_embeds = linear(inputs["hidden_states"][0], "proj_in").to(block_dtype)
    audio_embeds = linear(inputs["audio_hidden_states"][0], "audio_proj_in").to(block_dtype)
    text_embeds = linear(inputs["encoder_hidden_states"][0], "context_embedder")
    for idx in range(int(config["num_refiner_layers"])):
        p = f"token_refiner.refiner_blocks.{idx}"
        h = _rms_norm_ref(text_embeds, diffusers_state[f"{p}.norm1.weight"], norm_eps)
        q = _rms_norm_ref(
            linear(h, f"{p}.attn.to_q").view(-1, num_heads, head_dim),
            diffusers_state[f"{p}.attn.norm_q.weight"],
            qk_norm_eps,
        )
        k = _rms_norm_ref(
            linear(h, f"{p}.attn.to_k").view(-1, num_heads, head_dim),
            diffusers_state[f"{p}.attn.norm_k.weight"],
            qk_norm_eps,
        )
        v = linear(h, f"{p}.attn.to_v").view(-1, num_heads, head_dim)
        attn_out = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=False,
                scale=head_dim**-0.5,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(-1, num_heads * head_dim)
        )
        text_embeds = text_embeds + linear(attn_out, f"{p}.attn.to_out.0")
        h2 = _rms_norm_ref(text_embeds, diffusers_state[f"{p}.norm2.weight"], norm_eps)
        up, gate = linear(h2, f"{p}.ff.net.0.proj").chunk(2, dim=-1)
        ff_out = linear(F.silu(gate) * up, f"{p}.ff.net.2")
        text_embeds = text_embeds + ff_out
    text_embeds = _rms_norm_ref(text_embeds, diffusers_state["token_refiner.final_norm.weight"], final_norm_eps)

    seq_len = int(inputs["position_ids"].shape[0])
    hidden = text_embeds.new_zeros((seq_len, hidden_size))
    hidden = hidden.index_copy(0, inputs["text_indices"], text_embeds)
    hidden = hidden.index_copy(0, inputs["video_indices"], video_embeds)
    hidden = hidden.index_copy(0, inputs["audio_indices"], audio_embeds)

    # MeanFlow time_embedding(t, r) = emb_t(t) + emb_r(r - t)
    t = inputs["timestep"].float()
    temb = linear(
        F.silu(linear(_sinusoidal_time_proj_ref(t, freq_dim), "time_embedder.linear_1")), "time_embedder.linear_2"
    )
    if endpoint_mode != "none":
        r = (
            t
            if endpoint_mode == "r_equals_t" or inputs.get("endpoint_timestep") is None
            else inputs["endpoint_timestep"].float()
        )
        interval = r - t
        remb = linear(
            F.silu(linear(_sinusoidal_time_proj_ref(interval, freq_dim), "r_time_embedder.linear_1")),
            "r_time_embedder.linear_2",
        )
        temb = temb + remb

    adaln_in = F.silu(temb).to(block_dtype)
    adaln_indices = inputs["timestep_indices"] * 3 + inputs["token_tags"]

    for idx in range(int(config["num_layers"])):
        p = f"transformer_blocks.{idx}"
        if f"{p}.adaln_proj.linear_1.weight" in diffusers_state:
            mod = linear(linear(adaln_in, f"{p}.adaln_proj.linear_1"), f"{p}.adaln_proj.linear_2")
        else:
            mod = linear(adaln_in, f"{p}.adaln_proj.linear")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.view(-1, 6 * hidden_size).chunk(6, dim=-1)

        h = _rms_norm_ref(hidden, diffusers_state[f"{p}.norm1.weight"], norm_eps)
        h = (h * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)).to(
            block_dtype
        )
        q = _rms_norm_ref(
            linear(h, f"{p}.attn.to_q").view(-1, num_heads, head_dim),
            diffusers_state[f"{p}.attn.norm_q.weight"],
            qk_norm_eps,
        )
        k = _rms_norm_ref(
            linear(h, f"{p}.attn.to_k").view(-1, num_heads, head_dim),
            diffusers_state[f"{p}.attn.norm_k.weight"],
            qk_norm_eps,
        )
        v = linear(h, f"{p}.attn.to_v").view(-1, num_heads, head_dim)
        q = torch.cat(((q[..., :rot_dim] * cos) + (_rotate_half_ref(q[..., :rot_dim]) * sin), q[..., rot_dim:]), dim=-1)
        k = torch.cat(((k[..., :rot_dim] * cos) + (_rotate_half_ref(k[..., :rot_dim]) * sin), k[..., rot_dim:]), dim=-1)
        attn_out = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=False,
                scale=head_dim**-0.5,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(-1, num_heads * head_dim)
        )
        hidden = (hidden + gate_msa.index_select(0, adaln_indices) * linear(attn_out, f"{p}.attn.to_out.0")).to(
            block_dtype
        )

        h2 = _rms_norm_ref(hidden, diffusers_state[f"{p}.norm2.weight"], norm_eps)
        h2 = (h2 * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)).to(
            block_dtype
        )
        up, gate = linear(h2, f"{p}.ff.net.0.proj").chunk(2, dim=-1)
        ff_out = linear(F.silu(gate) * up, f"{p}.ff.net.2")
        hidden = (hidden + gate_mlp.index_select(0, adaln_indices) * ff_out).to(block_dtype)

    shift_out, scale_out = linear(adaln_in, "norm_out.linear").view(-1, 2 * hidden_size).chunk(2, dim=-1)
    h_out = _rms_norm_ref(hidden, diffusers_state["norm_out.norm.weight"], final_norm_eps)
    h_out = (
        h_out * (1.0 + scale_out.index_select(0, inputs["timestep_indices"]))
        + shift_out.index_select(0, inputs["timestep_indices"])
    ).to(torch.float32)
    video_out = linear(h_out, "proj_out").index_select(0, inputs["video_indices"])[None]
    audio_out = linear(h_out, "audio_proj_out").index_select(0, inputs["audio_indices"])[None]
    return video_out, audio_out


@pytest.mark.parametrize("endpoint_mode", ["meanflow", "r_equals_t", "none"])
@pytest.mark.parametrize("adaln_rank", [4, None])
@pytest.mark.parametrize("block_dtype", [torch.float32, torch.bfloat16])
def test_h3_tiny_model_numerical_parity_against_upstream_reference(
    tmp_path: Path,
    endpoint_mode: str,
    adaln_rank: int | None,
    block_dtype: torch.dtype,
) -> None:
    config = _tiny_diffusers_config(ffn_dim=40, refiner_ffn_dim=64, adaln_rank=adaln_rank)
    diff_state = _make_diffusers_state(config, endpoint_mode=endpoint_mode, block_dtype=block_dtype)
    ckpt_dir = tmp_path / f"ckpt_{endpoint_mode}_{adaln_rank}_{block_dtype}"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    save_file(diff_state, str(ckpt_dir / "model.safetensors"))

    model = load_h3_tiny_transformer(ckpt_dir, endpoint_mode=endpoint_mode, device="cpu")
    assert all(type(block) is MiniMaxH3DiTBlock for block in model.blocks)
    first_block = model.blocks[0]
    first_refiner_block = model.token_refiner.blocks[0]
    assert isinstance(first_block, MiniMaxH3DiTBlock)
    assert isinstance(first_refiner_block, MiniMaxH3TokenRefinerBlock)
    block_fc2 = getattr(first_block.mlp, "fc2")
    refiner_fc2 = getattr(first_refiner_block.mlp, "fc2")
    assert isinstance(block_fc2, torch.nn.Linear) and block_fc2.in_features == 40
    assert isinstance(refiner_fc2, torch.nn.Linear) and refiner_fc2.in_features == 64

    inputs = _tiny_inputs()
    with torch.no_grad():
        actual_video, actual_audio = model(**inputs)
        ref_video, ref_audio = _upstream_h3tiny_forward_ref(diff_state, config, inputs, endpoint_mode=endpoint_mode)

    tol = 1e-5 if block_dtype == torch.float32 else 2e-3
    torch.testing.assert_close(actual_video, ref_video, atol=tol, rtol=tol)
    torch.testing.assert_close(actual_audio, ref_audio, atol=tol, rtol=tol)


def test_meanflow_time_embedder_validation_and_modes() -> None:
    arch, _ = h3_tiny_arch_config(_tiny_diffusers_config())
    embedder = H3TinyMeanFlowTimeEmbedder(arch, endpoint_mode="meanflow").eval()
    t = torch.tensor([0.2, 0.6], dtype=torch.float32)
    r = torch.tensor([0.5, 0.9], dtype=torch.float32)

    expected = embedder.forward_with_endpoint(t, r)
    with embedder.endpoint_context(r):
        actual = embedder(t)
    assert torch.equal(actual, expected)
    assert embedder._endpoint_timestep is None

    # Rejects r < t
    with pytest.raises(ValueError, match="r >= t"):
        embedder.forward_with_endpoint(t, torch.tensor([0.1, 0.9], dtype=torch.float32))

    # Rejects shape mismatch and clears context on failure
    with pytest.raises(ValueError, match="endpoint_timestep"), embedder.endpoint_context(r[:1]):
        embedder(t)
    assert embedder._endpoint_timestep is None

    # r_equals_t pins endpoint to timestep (evaluates r_time_embedder at 0)
    pinned = H3TinyMeanFlowTimeEmbedder(arch, endpoint_mode="r_equals_t").eval()
    assert torch.equal(pinned.forward_with_endpoint(t, r), pinned.forward_with_endpoint(t, t))

    # LowRankAdalnProj standalone apply_silu=True matches silu(temb) -> linear_1 -> linear_2
    low_rank = LowRankAdalnProj(16, 32, 4, apply_silu=True, dtype=torch.float32).eval()
    temb = torch.randn(3, 16)
    chunks = low_rank(temb)
    manual = low_rank.linear_2(low_rank.linear_1(F.silu(temb))).view(9, 6 * 32).chunk(6, dim=-1)
    assert all(torch.equal(a, b) for a, b in zip(chunks, manual, strict=True))


def test_bijective_key_mapping_matches_exact_630_fewstep_and_626_pruned_counts() -> None:
    prod_config = {
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "hidden_size": 5376,
        "num_layers": 42,
        "num_refiner_layers": 2,
        "ffn_dim": 9792,
        "refiner_ffn_dim": 14336,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 5120,
        "freq_dim": 256,
        "time_embed_hidden_dim": 5376,
        "time_embed_dim": 2688,
        "rope_freq_dim": 16,
        "adaln_rank": 8,
    }
    arch, variant_kwargs = h3_tiny_arch_config(prod_config)
    for endpoint_mode, expected_count in (("meanflow", 630), ("none", 626)):
        with torch.device("meta"):
            model = H3TinyDiTModel(arch, endpoint_mode=endpoint_mode, **variant_kwargs)
        wf_keys = sorted(model.state_dict().keys())
        assert len(wf_keys) == expected_count
        diff_keys = [h3_tiny_wf_to_diffusers_key(k) for k in wf_keys]
        assert len(set(diff_keys)) == expected_count
        roundtrip_wf = [h3_tiny_diffusers_to_wf_key(k) for k in diff_keys]
        assert roundtrip_wf == wf_keys
        # Verify refiner vs pruned block FFN dimensions are distinct in 14B config
        assert tuple(model.state_dict()["token_refiner.blocks.0.mlp.fc1.proj.weight"].shape) == (2 * 14336, 5376)
        assert tuple(model.state_dict()["blocks.0.mlp.fc1.proj.weight"].shape) == (2 * 9792, 5376)
        assert tuple(model.state_dict()["blocks.0.adaln_proj.linear_1.weight"].shape) == (8, 2688)
        assert tuple(model.state_dict()["blocks.0.adaln_proj.linear_2.weight"].shape) == (18 * 5376, 8)
        if endpoint_mode == "meanflow":
            assert "r_time_embedder.linear_1.weight" in diff_keys
            assert "r_time_embedder.linear_2.bias" in diff_keys
        else:
            assert not any(k.startswith("r_time_embedder.") for k in diff_keys)

    with pytest.raises(ValueError, match="Unsupported"):
        h3_tiny_diffusers_to_wf_key("unknown_layer.weight")
    with pytest.raises(ValueError, match="Unsupported"):
        h3_tiny_wf_to_diffusers_key("unknown_layer.weight")
    with pytest.raises(ValueError, match="AdaLN"):
        h3_tiny_diffusers_to_wf_key("token_refiner.refiner_blocks.0.adaln_proj.linear_1.weight")


def test_loader_supports_sharded_diffusers_metadata_and_subdir_and_preserves_dtypes(tmp_path: Path) -> None:
    config = _tiny_diffusers_config(ffn_dim=40, refiner_ffn_dim=64, adaln_rank=4)

    # 1) Sharded Diffusers directory (FewStep auto-detects "meanflow")
    sharded_dir = tmp_path / "sharded_fewstep"
    sharded_dir.mkdir()
    fewstep_state = _make_diffusers_state(config, endpoint_mode="meanflow", block_dtype=torch.bfloat16)
    keys = sorted(fewstep_state)
    split = len(keys) // 2
    part1 = {k: fewstep_state[k] for k in keys[:split]}
    part2 = {k: fewstep_state[k] for k in keys[split:]}
    save_file(part1, str(sharded_dir / "model-00001-of-00002.safetensors"))
    save_file(part2, str(sharded_dir / "model-00002-of-00002.safetensors"))
    weight_map = {
        **{k: "model-00001-of-00002.safetensors" for k in part1},
        **{k: "model-00002-of-00002.safetensors" for k in part2},
    }
    (sharded_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (sharded_dir / "config.json").write_text(json.dumps(config))

    assert detect_h3_tiny_endpoint_mode(sharded_dir) == "meanflow"
    assert read_h3_tiny_config(sharded_dir)["ffn_dim"] == 40
    loaded = load_h3_tiny_transformer(sharded_dir, endpoint_mode=None, device="cpu")
    assert loaded.endpoint_mode == "meanflow"
    assert loaded.video_patch_proj.weight.dtype == torch.float32
    assert loaded.time_embedder.proj_in.weight.dtype == torch.float32
    assert loaded.time_embedder.r_time_embedder is not None
    r_proj_out = getattr(loaded.time_embedder.r_time_embedder, "proj_out")
    assert isinstance(r_proj_out, torch.nn.Linear) and r_proj_out.weight.dtype == torch.float32
    assert loaded.final_layer.video_out.weight.dtype == torch.float32
    loaded_block0 = loaded.blocks[0]
    assert isinstance(loaded_block0, MiniMaxH3DiTBlock)
    assert isinstance(loaded_block0.adaln_proj, LowRankAdalnProj)
    assert loaded_block0.adaln_proj.linear_1.weight.dtype == torch.bfloat16
    assert loaded.rope.inv_freq.dtype == torch.float32
    assert not loaded.rope.inv_freq.is_meta

    # 2) metadata.json + payload layout (Pruned auto-detects "none")
    meta_dir = tmp_path / "metadata_pruned"
    meta_dir.mkdir()
    pruned_state = _make_diffusers_state(config, endpoint_mode="none", block_dtype=torch.bfloat16)
    save_file(pruned_state, str(meta_dir / "payload.safetensors"))
    (meta_dir / "metadata.json").write_text(json.dumps({"model_config": config, "payload_file": "payload.safetensors"}))
    assert detect_h3_tiny_endpoint_mode(meta_dir) == "none"
    loaded_pruned = load_h3_tiny_transformer(meta_dir, endpoint_mode=None, device="cpu")
    assert loaded_pruned.endpoint_mode == "none"
    assert loaded_pruned.time_embedder.r_time_embedder is None

    # 3) transformer_ref subdir layout
    subdir_root = tmp_path / "with_subdir"
    (subdir_root / "transformer_ref").mkdir(parents=True)
    save_file(pruned_state, str(subdir_root / "transformer_ref" / "diffusion_pytorch_model.safetensors"))
    (subdir_root / "transformer_ref" / "config.json").write_text(json.dumps(config))
    loaded_sub = load_h3_tiny_transformer(subdir_root, endpoint_mode=None, device="cpu")
    assert loaded_sub.endpoint_mode == "none"

    # 4) Strict failure on missing keys
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text(json.dumps(config))
    save_file({"proj_in.weight": fewstep_state["proj_in.weight"]}, str(broken_dir / "model.safetensors"))
    with pytest.raises(RuntimeError, match="do not match"):
        load_h3_tiny_transformer(broken_dir, endpoint_mode="none", device="cpu")


@pytest.mark.parametrize("steps", [4, 6, 50])
def test_shifted_sigmas_and_row_time_pairs_match_upstream_math(steps: int) -> None:
    for shift in (H3_TINY_VIDEO_SHIFT, H3_TINY_AUDIO_SHIFT):
        actual = h3_tiny_shifted_sigmas(steps, shift)
        grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
        expected = (shift * grid / (1.0 + (shift - 1.0) * grid)).to(torch.float32)
        assert actual.dtype == torch.float32
        assert len(actual) == steps + 1
        assert float(actual[0]) == 1.0
        assert float(actual[-1]) == 0.0
        assert torch.equal(actual, expected)

    # Step 0 edge case: both video and audio start at t=0.0 but have different endpoints r_v != r_a
    v_sigmas = h3_tiny_shifted_sigmas(6, H3_TINY_VIDEO_SHIFT)
    a_sigmas = h3_tiny_shifted_sigmas(6, H3_TINY_AUDIO_SHIFT)
    t_v, r_v = float(1.0 - v_sigmas[0]), float(1.0 - v_sigmas[1])
    t_a, r_a = float(1.0 - a_sigmas[0]), float(1.0 - a_sigmas[1])
    assert t_v == t_a == 0.0
    assert r_v != r_a

    video_idx = torch.tensor([4, 5, 6], dtype=torch.long)
    audio_idx = torch.tensor([2, 3], dtype=torch.long)
    ts, rs, inv = h3_tiny_row_time_pairs(
        video_indices=video_idx,
        audio_indices=audio_idx,
        num_condition_video_rows=1,
        num_condition_audio_rows=1,
        num_text_tokens=2,
        video_time=(t_v, r_v),
        audio_time=(t_a, r_a),
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    # Four distinct (t, r) rows: (0, r_v), (0, r_a), (0.999, 0.999), (1.0, 1.0)
    assert len(ts) == 4
    expected_t = torch.tensor([t_v, t_v, 1.0, t_a, 0.999, t_v, t_v], dtype=torch.float32)
    expected_r = torch.tensor([r_v, r_v, 1.0, r_a, 0.999, r_v, r_v], dtype=torch.float32)
    assert torch.equal(ts[inv], expected_t)
    assert torch.equal(rs[inv], expected_r)


def test_euler_step_sign_convention_matches_minimax_h3_and_reconstructs_clean_data() -> None:
    torch.manual_seed(7)
    x0 = torch.randn(4, 16)
    noise = torch.randn(4, 16)
    velocity = x0 - noise  # H3 / H3-Tiny velocity convention: dx/dt = x0 - noise

    # Single jump from sigma=1 -> sigma=0 recovers x0 exactly
    recovered = h3_tiny_euler_step(noise, velocity, sigma_curr=1.0, sigma_next=0.0)
    torch.testing.assert_close(recovered, x0, atol=1e-6, rtol=1e-6)

    # Full 6-step trajectory matches exact linear interpolation and minimax_h3_euler_eta0_step
    schedule = H3TinySchedule(num_steps=6)
    for is_audio, sigmas in ((False, schedule.video_sigmas), (True, schedule.audio_sigmas)):
        state_h3tiny = noise.clone()
        state_canonical = noise.clone()
        for step in range(6):
            s_curr = float(sigmas[step])
            s_next = float(sigmas[step + 1])
            state_h3tiny = schedule.step(velocity, state_h3tiny, step, audio=is_audio)
            denoised = minimax_h3_rf_v_to_x0(state_canonical, velocity, torch.tensor(1.0 - s_curr))
            state_canonical = minimax_h3_euler_eta0_step(
                state_canonical, denoised, sigma_curr=s_curr, sigma_next=s_next
            )
            torch.testing.assert_close(state_h3tiny, state_canonical, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(state_h3tiny, x0, atol=1e-5, rtol=1e-5)
