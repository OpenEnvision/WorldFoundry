# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free component parity and strict checkpoint loading tests for pi-H3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from worldfoundry.base_models.diffusion_model.models.denoisers.pi_h3 import (
    load_pi_h3_transformer,
    pi_h3_arch_config,
    pi_h3_foundation_key,
    split_pi_h3_adapter_state_dict,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import MiniMaxH3DiTBlock
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3 import (
    PiH3DiTModel,
    PiH3DWDAdaLayerNormOut,
    PiH3PrunedTimeEmbedder,
    RGBColorEncoder,
    SolH3Block,
    SolH3DWDHead,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3.sol_dwd_head import (
    _global_rows,
    _rope,
    _window_rows,
)
from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)
from worldfoundry.base_models.diffusion_model.schedulers.pi_h3 import (
    PI_H3_DEFAULT_AUDIO_SHIFT,
    PI_H3_DEFAULT_VIDEO_SHIFT,
    PI_H3_MAX_RAW_T,
    pi_h3_flow_euler_sigmas,
    pi_h3_flow_euler_step,
)

pytestmark = pytest.mark.unit


def _tiny_config(*, num_layers: int = 4, with_dwd: bool = False) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": num_layers,
        "num_refiner_layers": 1,
        "ffn_dim": 64,
        "in_channels": 4,
        "audio_in_channels": 8,
        "patch_size": [1, 2, 2],
        "text_dim": 12,
        "freq_dim": 8,
        "time_embed_hidden_dim": 32,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
        "adaln_rank": 8,
        "time_table_size": 17,
    }
    if with_dwd:
        cfg["dwd_head_config"] = {
            "pixel_patch_size": [4, 8, 8],
            "first_frame_pixel_patch_size": [1, 8, 8],
            "first_frame_period": 5,
            "pixel_channels": 3,
            "linear_head": False,
            "pixel_refiner": {
                "type": "SolH3DWDHead",
                "num_layers": 2,
                "total_vae_layers": 24,
                "num_heads": 2,
                "head_dim": 16,
                "ffn_mult": 2,
                "num_register_tokens": 4,
                "window_size": 4,
                "block_size": 5,
                "t_shift": 2,
                "time_norm": 5,
                "spatial_patch": 4,
                "patch_t": 4,
                "frames_to_trim": 3,
            },
        }
    return cfg


def _tiny_packed_inputs(
    *,
    video_tokens: int = 2,
    audio_tokens: int = 4,
    text_tokens: int = 3,
    seed: int = 101,
) -> dict[str, object]:
    gen = torch.Generator().manual_seed(seed)
    total = text_tokens + audio_tokens + video_tokens
    text_pos = torch.arange(0, text_tokens, dtype=torch.long)
    audio_pos = torch.arange(text_tokens, text_tokens + audio_tokens, dtype=torch.long)
    img_pos = torch.arange(text_tokens + audio_tokens, total, dtype=torch.long)
    x = torch.zeros(1, total, 16, dtype=torch.float32)
    audio_x = torch.zeros(1, total, 8, dtype=torch.float32)
    x[0, img_pos] = torch.randn(video_tokens, 16, generator=gen)
    audio_x[0, audio_pos] = torch.randn(audio_tokens, 8, generator=gen)
    token_tags = torch.cat(
        [
            torch.ones(text_tokens, dtype=torch.long),
            torch.full((audio_tokens,), 2, dtype=torch.long),
            torch.zeros(video_tokens, dtype=torch.long),
        ]
    )
    inverse_indices = torch.cat(
        [
            torch.zeros(text_tokens, dtype=torch.long),
            torch.ones(audio_tokens, dtype=torch.long),
            torch.zeros(video_tokens, dtype=torch.long),
        ]
    )
    return {
        "x": x,
        "audio_x": audio_x,
        "img_position_ids": torch.randn(1, total, 3, generator=gen),
        "unique_timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
        "inverse_indices": inverse_indices,
        "update_mask": torch.ones(video_tokens, dtype=torch.bool),
        "token_tags": token_tags,
        "skip_mask_out_condition": True,
        "prompt_embeds": torch.randn(text_tokens, 12, generator=gen),
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": img_pos},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, total], dtype=torch.int32)},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, text_tokens], dtype=torch.int32)},
    }


def _to_diffusers_pruned_state(model: PiH3DiTModel, *, use_folded_bias: bool = True) -> dict[str, torch.Tensor]:
    """Invert ``pi_h3_foundation_key`` on a tiny model to synthesize a realistic MiniMax-H3-Pruned checkpoint."""

    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.startswith(("dwd_norm_out.", "dwd_pixel_refiner.")):
            continue
        raw = key
        for target, source in (
            ("audio_patch_proj.", "audio_proj_in."),
            ("video_patch_proj.", "proj_in."),
            ("final_layer.audio_out.", "audio_proj_out."),
            ("final_layer.video_out.", "proj_out."),
            ("condition_proj.", "context_embedder."),
            ("final_layer.adaln_proj.linear.", "norm_out.linear."),
            ("final_layer.norm.", "norm_out.norm."),
        ):
            if raw.startswith(target):
                raw = source + raw[len(target) :]
                break
        for target, source in (
            ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
            (".attn.qkv_proj.q.", ".attn.to_q."),
            (".attn.qkv_proj.k.", ".attn.to_k."),
            (".attn.qkv_proj.v.", ".attn.to_v."),
            (".attn.out_proj.", ".attn.to_out.0."),
            (".attn.q_norm.", ".attn.norm_q."),
            (".attn.k_norm.", ".attn.norm_k."),
            (".mlp.fc1.proj.", ".ff.net.0.proj."),
            (".mlp.fc2.", ".ff.net.2."),
        ):
            raw = raw.replace(target, source)
        if raw.startswith("blocks."):
            raw = "transformer_blocks." + raw[len("blocks.") :]
        if use_folded_bias and (raw.endswith(".adaln_proj.linear.bias") or raw == "norm_out.linear.bias"):
            raw = raw[: -len("linear.bias")] + "folded_bias"
        out[raw] = value.detach().clone().cpu()
    return out


def test_pruned_time_embedder_linear_interpolation_parity() -> None:
    torch.manual_seed(1)
    embedder = PiH3PrunedTimeEmbedder(table_size=1025, adaln_rank=8)
    table = torch.randn(1025, 8, dtype=torch.float32)
    embedder.table.copy_(table)

    t = torch.tensor([-0.25, 0.0, 1.0 / 1024.0, 0.371234, 0.999, 1.0, 1.5], dtype=torch.float32)
    actual = embedder(t)

    # Upstream MiniMaxH3PrunedTimeEmbedder reference math.
    steps = 1024
    pos = t.float().clamp(0.0, 1.0) * steps
    lower = pos.floor().clamp(max=steps - 1).long()
    expected = torch.lerp(table[lower], table[lower + 1], (pos - lower).unsqueeze(-1))
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], table[0])
    assert torch.equal(actual[1], table[0])
    assert torch.equal(actual[5], table[-1])
    assert torch.equal(actual[6], table[-1])

    # bf16 table still interpolates cleanly in fp32.
    embedder_bf16 = PiH3PrunedTimeEmbedder(table_size=1025, adaln_rank=8)
    embedder_bf16.table = table.to(torch.bfloat16)
    out_bf16 = embedder_bf16(t)
    assert out_bf16.dtype == torch.float32
    torch.testing.assert_close(out_bf16, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_steps", [2, 4])
@pytest.mark.parametrize("shift", [PI_H3_DEFAULT_VIDEO_SHIFT, PI_H3_DEFAULT_AUDIO_SHIFT])
def test_flow_euler_sigmas_and_step_parity_with_upstream_math(num_steps: int, shift: float) -> None:
    sigmas = pi_h3_flow_euler_sigmas(num_steps, shift=shift)
    max_raw_t = PI_H3_MAX_RAW_T[num_steps]
    raw_end = max_raw_t * 0.6 / (num_steps - 1 + 0.6)
    raw = np.linspace(max_raw_t, raw_end, num_steps, dtype=np.float32)
    ref_t = torch.from_numpy(raw)
    ref_sigmas = torch.cat([shift * ref_t / (1.0 + (shift - 1.0) * ref_t), torch.zeros(1)])

    assert len(sigmas) == num_steps + 1
    assert sigmas[-1] == 0.0
    torch.testing.assert_close(torch.tensor(sigmas), ref_sigmas, atol=0.0, rtol=0.0)
    assert all(left > right for left, right in zip(sigmas, sigmas[1:], strict=False))

    # Verify exact ODE step parity across LakonLab (u = -v) and canonical MiniMax-H3 (v -> x0 -> eta0).
    torch.manual_seed(7)
    sample = torch.randn(4, 16, dtype=torch.float32)
    raw_velocity = torch.randn(4, 16, dtype=torch.float32)
    for idx in range(num_steps):
        s_curr, s_next = sigmas[idx], sigmas[idx + 1]
        step_v = pi_h3_flow_euler_step(sample, raw_velocity, sigma=s_curr, sigma_next=s_next, prediction_type="v")
        step_u = pi_h3_flow_euler_step(sample, -raw_velocity, sigma=s_curr, sigma_next=s_next, prediction_type="u")
        assert torch.equal(step_v, step_u)

        # Canonical H3: t = 1 - sigma, x0 = xt + (1 - t) * v, out = (s_next/s_curr)*xt + (1 - s_next/s_curr)*x0
        x0 = minimax_h3_rf_v_to_x0(sample, raw_velocity, torch.tensor(1.0 - s_curr, dtype=torch.float32))
        step_x0 = pi_h3_flow_euler_step(sample, x0, sigma=s_curr, sigma_next=s_next, prediction_type="x0")
        step_h3 = minimax_h3_euler_eta0_step(sample, x0, sigma_curr=s_curr, sigma_next=s_next)
        torch.testing.assert_close(step_v, step_x0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(step_v, step_h3, atol=1e-6, rtol=1e-6)


def test_rgb_color_encoder_affine_and_unit_range_parity() -> None:
    encoder = RGBColorEncoder(
        use_affine_norm=True,
        mean=(-0.03, -0.088, -0.188),
        std=(0.458, 0.448, 0.45),
    )
    torch.manual_seed(3)
    raw = torch.randn(1, 3, 5, 8, 8)
    decoded_m11 = encoder.decode(raw)
    unit = encoder.decode_to_unit_range(raw)

    # ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) satisfies 2*im - 1 == affine.
    im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    expected_unit = (raw * im_std + im_mean).clamp(0.0, 1.0)
    torch.testing.assert_close(unit, expected_unit, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(decoded_m11.add(1.0).mul(0.5), unit, atol=1e-6, rtol=1e-6)

    # Round-trip on interior [-1, 1] values.
    rgb_m11 = torch.rand(1, 3, 2, 4, 4) * 1.6 - 0.8
    torch.testing.assert_close(encoder.decode(encoder.encode(rgb_m11)), rgb_m11, atol=1e-6, rtol=1e-6)


def test_sol_dwd_head_tiny_forward_and_padded_vs_varlen_index_parity() -> None:
    torch.manual_seed(5)
    head = SolH3DWDHead(
        in_channels=32,
        src_patch_size=(4, 8, 8),
        first_frame_src_patch_size=(1, 8, 8),
        first_frame_period=5,
        residual=False,
        pixel_channels=3,
        num_layers=2,
        total_vae_layers=24,
        num_heads=2,
        head_dim=16,
        ffn_mult=2,
        num_register_tokens=4,
        window_size=4,
        block_size=5,
        t_shift=2,
        time_norm=5,
        spatial_patch=4,
        patch_t=4,
        frames_to_trim=3,
    ).eval()
    with torch.no_grad():
        head.register_tokens.normal_(0.0, 0.02)
        for blk in head.blocks:
            assert isinstance(blk, SolH3Block)
            blk.scale1.fill_(0.5)
            blk.scale2.fill_(0.5)

    # Verify exact upstream checkpoint parameter names are preserved.
    param_names = set(head.state_dict())
    expected_prefixes = {
        "feature_expand.weight",
        "feature_expand.bias",
        "register_tokens",
        "blocks.0.norm1.weight",
        "blocks.0.norm2.weight",
        "blocks.0.scale1",
        "blocks.0.scale2",
        "blocks.0.attn.to_qkv.weight",
        "blocks.0.attn.to_out.weight",
        "blocks.0.ff.w1.weight",
        "blocks.0.ff.w2.weight",
        "norm_out.weight",
        "norm_out.bias",
        "proj_out.weight",
        "proj_out.bias",
        "first_frame_proj_out.weight",
        "first_frame_proj_out.bias",
    }
    assert expected_prefixes.issubset(param_names)

    # Test both single-tail (T=2 -> 5 frames) and multi-block (T=7 -> 17 + 5 = 22 frames).
    for latent_t, expected_frames in ((2, 5), (7, 22)):
        feats = torch.randn(1, latent_t * 2 * 3, 32)
        with torch.no_grad():
            pixels = head(None, feats, (latent_t, 2, 3))
        assert pixels.shape == (1, expected_frames, 3, 2 * 8, 3 * 8)
        assert torch.isfinite(pixels).all()

    # Verify padded SDPA indices and packed varlen indices compute identical attention outputs.
    B, T, H, W, nb, S, heads, dh = 1, 7, 5, 6, 2, 5, 2, 16
    N = B * T * H * W
    q = torch.randn(N + B * nb * S, heads, dh)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def _varlen_sdpa_ref(q_p, k_p, v_p, cu_q, cu_k):
        out = torch.empty_like(q_p)
        for i in range(cu_q.numel() - 1):
            qs, qe = int(cu_q[i]), int(cu_q[i + 1])
            ks, ke = int(cu_k[i]), int(cu_k[i + 1])
            if qe == qs:
                continue
            seg = F.scaled_dot_product_attention(
                q_p[qs:qe].transpose(0, 1).unsqueeze(0),
                k_p[ks:ke].transpose(0, 1).unsqueeze(0),
                v_p[ks:ke].transpose(0, 1).unsqueeze(0),
                is_causal=False,
            )
            out[qs:qe] = seg.squeeze(0).transpose(0, 1)
        return out

    # Global attention parity.
    g_rows_pad, g_mask_pad = _global_rows(B, T, H, W, nb, S, torch.device("cpu"), padded=True, block=5)
    og_pad = (
        F.scaled_dot_product_attention(
            q[N:].unflatten(0, (B * nb, S)).transpose(1, 2),
            k[g_rows_pad].transpose(1, 2),
            v[g_rows_pad].transpose(1, 2),
            attn_mask=g_mask_pad[:, None, None, :],
        )
        .transpose(1, 2)
        .flatten(0, 1)
    )
    g_rows_var, cu_gq, cu_gk = _global_rows(B, T, H, W, nb, S, torch.device("cpu"), padded=False, block=5)
    og_var = _varlen_sdpa_ref(q[N:], k[g_rows_var], v[g_rows_var], cu_gq, cu_gk)
    torch.testing.assert_close(og_pad, og_var, atol=1e-5, rtol=1e-5)

    # Window attention parity for both unshifted and shifted layers.
    for s, st in ((0, 0), (2, 2)):
        q_r_p, k_r_p, mask_p, out_r_p = _window_rows(
            B, T, H, W, nb, S, s, st, torch.device("cpu"), padded=True, block=5, window=4
        )
        o_pad = (
            F.scaled_dot_product_attention(
                q[q_r_p].transpose(1, 2),
                k[k_r_p].transpose(1, 2),
                v[k_r_p].transpose(1, 2),
                attn_mask=mask_p[:, None, None, :],
            )
            .transpose(1, 2)
            .flatten(0, 1)[out_r_p]
        )
        q_r_v, k_r_v, cu_wq, cu_wk = _window_rows(
            B, T, H, W, nb, S, s, st, torch.device("cpu"), padded=False, block=5, window=4
        )
        o_var_packed = _varlen_sdpa_ref(q[q_r_v], k[k_r_v], v[k_r_v], cu_wq, cu_wk)
        o_var = torch.empty_like(o_var_packed).index_copy_(0, q_r_v, o_var_packed)
        torch.testing.assert_close(o_pad, o_var, atol=1e-5, rtol=1e-5)

    # Verify full-scale RoPE matches upstream arange(0, 1, 6/48).
    cos_ref, sin_ref = _rope(5, 4, 4, 0, 0, torch.device("cpu"))
    assert cos_ref.shape == (1, 5 * 4 * 4, 1, 24)
    assert sin_ref.shape == (1, 5 * 4 * 4, 1, 24)


def test_dwd_adaln_out_matches_diffusers_adalayernorm_continuous() -> None:
    from diffusers.models.normalization import AdaLayerNormContinuous

    torch.manual_seed(11)
    ours = PiH3DWDAdaLayerNormOut(32, 8, eps=1e-5, dtype=torch.float32)
    ref = AdaLayerNormContinuous(32, 8, elementwise_affine=False, eps=1e-5, norm_type="layer_norm")
    ref.linear.weight.data.copy_(ours.linear.weight.data)
    ref.linear.bias.data.copy_(ours.linear.bias.data)

    hidden = torch.randn(6, 32)
    temb = torch.randn(2, 8)
    idx = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)

    actual = ours(hidden, temb, idx)
    # Match row-by-row against Diffusers AdaLayerNormContinuous.
    expected = torch.cat([ref(hidden[i : i + 1, None, :], temb[idx[i] : idx[i] + 1])[:, 0, :] for i in range(6)], dim=0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_pi_h3_dit_forward_dwd_and_non_dwd_no_double_silu() -> None:
    torch.manual_seed(19)
    cfg = _tiny_config(num_layers=2, with_dwd=True)
    model = PiH3DiTModel(
        pi_h3_arch_config(cfg),
        adaln_rank=8,
        time_table_size=17,
        dwd_head_config=cfg["dwd_head_config"],
        dtype=torch.float32,
    ).eval()
    assert all(type(block) is MiniMaxH3DiTBlock for block in model.blocks)
    assert model.has_dwd_head

    # Populate strongly negative table values so any accidental extra SiLU would diverge sharply.
    assert model.dwd_pixel_refiner is not None
    with torch.no_grad():
        model.time_embedder.table.fill_(-2.5)
        model.dwd_pixel_refiner.register_tokens.normal_(0.0, 0.02)
        for blk in model.dwd_pixel_refiner.blocks:
            assert isinstance(blk, SolH3Block)
            blk.scale1.fill_(0.5)
            blk.scale2.fill_(0.5)

    inputs = _tiny_packed_inputs(video_tokens=2 * 2 * 2, audio_tokens=4, text_tokens=3)
    with torch.no_grad():
        v_out, a_out = model(**inputs, return_features=False)
        (v_out_f, a_out_f), feats = model(**inputs, return_features=True)
        pixels = model.forward_dwd_head(feats["image"], (2, 2, 2))

    assert torch.equal(v_out, v_out_f)
    assert torch.equal(a_out, a_out_f)
    assert feats["image"].shape == (8, 32)
    assert feats["audio"].shape == (4, 32)
    assert pixels.shape == (1, 3, 5, 16, 16)
    assert torch.isfinite(pixels).all()

    # Check directly that block 0 AdaLN receives raw table values (-2.5), NOT silu(-2.5).
    raw_temb = model.time_embedder(inputs["unique_timesteps"])
    assert torch.allclose(raw_temb, torch.full_like(raw_temb, -2.5))
    blk0 = model.blocks[0]
    assert isinstance(blk0, MiniMaxH3DiTBlock)
    shift_raw, *_ = blk0.adaln_proj(raw_temb)
    shift_silu, *_ = blk0.adaln_proj(F.silu(raw_temb))
    assert not torch.allclose(shift_raw, shift_silu)

    # Verify non-DWD variant works and rejects forward_dwd_head.
    non_dwd = PiH3DiTModel(
        pi_h3_arch_config(cfg), adaln_rank=8, time_table_size=17, dwd_head_config=None, dtype=torch.float32
    ).eval()
    assert not non_dwd.has_dwd_head
    with torch.no_grad():
        (v2, a2), feats2 = non_dwd(**inputs, return_features=True)
    assert v2.shape == (8, 16) and a2.shape == (4, 8) and feats2["image"].shape == (8, 32)
    with pytest.raises(RuntimeError, match="SolH3DWDHead"):
        non_dwd.forward_dwd_head(feats2["image"], (2, 2, 2))


def test_strict_loader_dwd_and_non_dwd_with_lora_and_overrides(tmp_path: Path) -> None:
    torch.manual_seed(31)
    base_cfg = _tiny_config(num_layers=4, with_dwd=False)
    dwd_cfg = _tiny_config(num_layers=4, with_dwd=True)

    base_model = PiH3DiTModel(
        pi_h3_arch_config(base_cfg),
        adaln_rank=8,
        time_table_size=17,
        dwd_head_config=None,
        dtype=torch.float32,
    ).eval()
    base_dir = tmp_path / "base_transformer"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(json.dumps(base_cfg))
    base_diffusers_sd = _to_diffusers_pruned_state(base_model, use_folded_bias=True)
    assert "transformer_blocks.0.adaln_proj.folded_bias" in base_diffusers_sd
    assert "norm_out.folded_bias" in base_diffusers_sd
    assert "rope.inv_freq" not in base_diffusers_sd
    save_file(base_diffusers_sd, base_dir / "diffusion_pytorch_model.safetensors")

    # ---- Case 1: 2step DWD adapter (LoRA on blocks 0..1, full override on blocks 2..3, + DWD head) ----
    dwd_ref = PiH3DiTModel(
        pi_h3_arch_config(dwd_cfg),
        adaln_rank=8,
        time_table_size=17,
        dwd_head_config=dwd_cfg["dwd_head_config"],
        dtype=torch.float32,
    ).eval()
    assert dwd_ref.dwd_pixel_refiner is not None
    with torch.no_grad():
        dwd_ref.dwd_pixel_refiner.register_tokens.normal_(0.0, 0.02)
        for blk in dwd_ref.dwd_pixel_refiner.blocks:
            assert isinstance(blk, SolH3Block)
            blk.scale1.fill_(0.5)
            blk.scale2.fill_(0.5)

    dwd_adapter_sd: dict[str, torch.Tensor] = {}
    rank = 4
    expected_lora_deltas: dict[str, torch.Tensor] = {}
    for blk_idx in (0, 1):
        # LoRA targets matching Lakonik/pi-H3 (to_out.0, ff.net.0.proj, ff.net.2)
        for suffix, canonical_target, in_f, out_f in (
            ("attn.to_out.0", f"blocks.{blk_idx}.attn.out_proj", 32, 32),
            ("ff.net.0.proj", f"blocks.{blk_idx}.mlp.fc1.proj", 32, 128),
            ("ff.net.2", f"blocks.{blk_idx}.mlp.fc2", 64, 32),
        ):
            lora_a = torch.randn(rank, in_f) * 0.05
            lora_b = torch.randn(out_f, rank) * 0.05
            dwd_adapter_sd[f"transformer.transformer_blocks.{blk_idx}.{suffix}.lora_A.weight"] = lora_a
            dwd_adapter_sd[f"transformer.transformer_blocks.{blk_idx}.{suffix}.lora_B.weight"] = lora_b
            expected_lora_deltas[f"{canonical_target}.weight"] = lora_b @ lora_a
        # Per-block adaln_proj.linear.weight override
        dwd_adapter_sd[f"transformer.transformer_blocks.{blk_idx}.adaln_proj.linear.weight"] = (
            torch.randn(18 * 32, 8) * 0.1
        )

    # Full weight overrides for tail blocks 2..3 (mirroring blocks 46..49 in full H3)
    tail_diffusers_sd = _to_diffusers_pruned_state(dwd_ref, use_folded_bias=False)
    for k, v in tail_diffusers_sd.items():
        if k.startswith(
            ("transformer_blocks.2.", "transformer_blocks.3.", "norm_out.", "proj_out.", "audio_proj_out.")
        ):
            dwd_adapter_sd[f"transformer.{k}"] = v.clone()
    for k, v in dwd_ref.state_dict().items():
        if k.startswith(("dwd_norm_out.", "dwd_pixel_refiner.")):
            dwd_adapter_sd[f"transformer.{k}"] = v.detach().clone().cpu()

    dwd_adapter_dir = tmp_path / "pi_h3_2step_dwd_adapter"
    dwd_adapter_dir.mkdir()
    (dwd_adapter_dir / "config.json").write_text(json.dumps(dwd_cfg))
    save_file(dwd_adapter_sd, dwd_adapter_dir / "diffusion_pytorch_model.safetensors")

    loaded_dwd = load_pi_h3_transformer(base_dir, dwd_adapter_dir, device="cpu", dtype=torch.float32)
    assert loaded_dwd.has_dwd_head
    assert loaded_dwd.rope.inv_freq.dtype == torch.float32
    assert loaded_dwd.rope.inv_freq.device.type == "cpu"

    # Check LoRA fusion on blocks 0..1 (including ff.net.0.proj without row corruption).
    for weight_key, delta in expected_lora_deltas.items():
        expected_w = base_model.state_dict()[weight_key] + delta
        torch.testing.assert_close(loaded_dwd.state_dict()[weight_key], expected_w, atol=1e-6, rtol=1e-6)

    # Check folded_bias landed properly on block 0 and full overrides landed on block 3.
    loaded_dwd_sd = loaded_dwd.state_dict()
    dwd_ref_sd = dwd_ref.state_dict()
    torch.testing.assert_close(
        loaded_dwd_sd["blocks.0.adaln_proj.linear.bias"],
        base_diffusers_sd["transformer_blocks.0.adaln_proj.folded_bias"],
    )
    torch.testing.assert_close(
        loaded_dwd_sd["blocks.3.attn.out_proj.weight"],
        dwd_ref_sd["blocks.3.attn.out_proj.weight"],
    )
    assert loaded_dwd.dwd_pixel_refiner is not None
    torch.testing.assert_close(
        loaded_dwd.dwd_pixel_refiner.feature_expand.weight,
        dwd_ref.dwd_pixel_refiner.feature_expand.weight,
    )

    # bf16 load preserves fp32 rope.inv_freq and runs cleanly.
    loaded_dwd_bf16 = load_pi_h3_transformer(base_dir, dwd_adapter_dir, device="cpu", dtype=torch.bfloat16)
    assert loaded_dwd_bf16.rope.inv_freq.dtype == torch.float32
    assert loaded_dwd_bf16.state_dict()["blocks.0.attn.out_proj.weight"].dtype == torch.bfloat16

    # ---- Case 2: 2step non-DWD adapter (LoRA on all 4 blocks + adaln/out overrides, no DWD head) ----
    non_dwd_adapter_sd: dict[str, torch.Tensor] = {}
    for blk_idx in range(4):
        for suffix, in_f, out_f in (
            ("attn.to_out.0", 32, 32),
            ("ff.net.0.proj", 32, 128),
            ("ff.net.2", 64, 32),
        ):
            non_dwd_adapter_sd[f"transformer_blocks.{blk_idx}.{suffix}.lora_A.default.weight"] = (
                torch.ones(2, in_f) * 0.01
            )
            non_dwd_adapter_sd[f"transformer_blocks.{blk_idx}.{suffix}.lora_B.default.weight"] = (
                torch.ones(out_f, 2) * 0.01
            )
        non_dwd_adapter_sd[f"transformer_blocks.{blk_idx}.adaln_proj.linear.weight"] = torch.zeros(18 * 32, 8)
    non_dwd_adapter_sd["proj_out.bias"] = torch.full((16,), 0.25)

    non_dwd_dir = tmp_path / "pi_h3_2step_adapter"
    non_dwd_dir.mkdir()
    (non_dwd_dir / "config.json").write_text(json.dumps(base_cfg))
    save_file(non_dwd_adapter_sd, non_dwd_dir / "diffusion_pytorch_model.safetensors")

    loaded_vae = load_pi_h3_transformer(base_dir, non_dwd_dir, device="cpu", dtype=torch.float32)
    assert not loaded_vae.has_dwd_head
    torch.testing.assert_close(loaded_vae.final_layer.video_out.bias, torch.full((16,), 0.25))
    delta_fc1 = (
        loaded_vae.state_dict()["blocks.3.mlp.fc1.proj.weight"]
        - base_model.state_dict()["blocks.3.mlp.fc1.proj.weight"]
    )
    torch.testing.assert_close(delta_fc1, torch.full_like(delta_fc1, 2 * 0.01 * 0.01), atol=1e-6, rtol=1e-6)

    # ---- Strict error checks ----
    assert pi_h3_foundation_key("transformer_blocks.0.adaln_proj.folded_bias") == "blocks.0.adaln_proj.linear.bias"
    with pytest.raises(ValueError, match="Incomplete"):
        split_pi_h3_adapter_state_dict({"transformer_blocks.0.attn.to_out.0.lora_A.weight": torch.zeros(2, 32)})

    bad_dir = tmp_path / "bad_adapter"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text(json.dumps(base_cfg))
    save_file({"unexpected_extra_tensor": torch.zeros(4)}, bad_dir / "diffusion_pytorch_model.safetensors")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_pi_h3_transformer(base_dir, bad_dir, device="cpu", dtype=torch.float32)
