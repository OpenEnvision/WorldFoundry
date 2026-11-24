# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and CUDA tests for the MiniMax H3 SOL-Engine acceleration stack.

Covers:
1. ``AttentionBackendConfigurable`` and ``TransformerBlockStack`` protocol seams,
   ``dense_layers`` pinning, refiner isolation, and custom ``attention_fn`` hooks.
2. Lossless Diffusers/HyperFlow QKV & SwiGLU projection fusion and idempotency.
3. Zero-sync ``_varlen_attention`` tiers (single-doc, trailing-pad skip, multi-doc
   ``aten._flash_attention_forward``, and empty-sequence boundary).
4. Fused strided Triton QK-norm+RoPE, tiled wide-row AdaLN/RMSNorm+AdaLN, and
   fused SwiGLU + rowwise FP8 activation quantization (``FusedFP8SwiGLUMLP``).
5. Hopper rowwise FP8 W8A8 quantization, sensitive/edge-layer exclusions, and
   low-precision execution receipts.
6. Whole-stack and piecewise block CUDA Graph capture/replay, multi-signature
   replays, graph invalidation, and ``sol_attn`` / ``cudnn_fp8`` composition.
7. ``MiniMaxH3VideoVAE`` and ``NativeMiniMaxH3Pipeline`` acceleration helpers,
   request-static denoise-loop caching, ``H3TinyDiTModel``, and ``SolH3DWDHead``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.vae_vit import ViT3DDecoder
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
    DEFAULT_MINIMAX_H3_FP8_EXCLUDES,
    MiniMaxH3AccelerationConfig,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTModel,
    accelerate_minimax_h3_dit,
    fuse_minimax_h3_projections,
    install_minimax_h3_cuda_graph,
    minimax_h3_sol_profile,
    ops,
    quantize_minimax_h3_fp8,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
    _varlen_attention,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_fp8_swiglu import (
    FusedFP8SwiGLUMLP,
    swiglu_quantize_rowwise_fp8_triton,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.h3_tiny import H3TinyDiTModel
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.conditioning import (
    HyperFlowQKVProjection,
    HyperFlowSwiGLUProjection,
)
from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3 import (
    PiH3DiTModel,
    SolH3DWDHead,
)
from worldfoundry.core.attention.backends.dispatch import (
    attention_provider_runtime_report,
    reset_attention_provider_runtime,
)
from worldfoundry.core.attention.backends.probe import attention_backend_capability
from worldfoundry.core.nn.transformer.contracts import AttentionBackendConfigurable, TransformerBlockStack
from worldfoundry.pipelines.minimax._minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from worldfoundry.pipelines.minimax.pipeline_minimax_h3 import NativeMiniMaxH3Pipeline

_CUDA = torch.cuda.is_available()
_SM_MAJOR = torch.cuda.get_device_capability(0)[0] if _CUDA else 0
_HOPPER_OR_NEWER = _CUDA and _SM_MAJOR >= 9


def _tiny_arch(
    *,
    num_layers: int = 4,
    hidden_size: int = 256,
    num_heads: int = 2,
    head_dim: int = 128,
    ffn_hidden_size: int = 512,
) -> MiniMaxH3DiTArchConfig:
    return MiniMaxH3DiTArchConfig(
        num_layers=num_layers,
        token_refiner_num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        ffn_hidden_size=ffn_hidden_size,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=64,
        timestep_input_dim=256,
        time_embed_hidden_size=256,
        time_embed_dim=128,
        adaln_out_features=18 * hidden_size,
        final_adaln_out_features=2 * hidden_size,
        rope_inv_freq_len=16,
    )


def _zero_adaln(model: MiniMaxH3DiTModel) -> None:
    with torch.no_grad():
        for block in model.blocks:
            for param in cast(Any, block).adaln_proj.parameters():
                param.zero_()
        for param in model.final_layer.adaln_proj.parameters():
            param.zero_()


def _build_packed_inputs(
    cfg: MiniMaxH3DiTArchConfig,
    device: torch.device,
    *,
    text_len: int = 4,
    audio_len: int = 6,
    video_len: int = 14,
    pad_len: int = 0,
    ignore_trailing_pad: bool = False,
    num_timesteps: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    used = text_len + audio_len + video_len
    seq_len = used + pad_len
    video_row = cfg.video_patch_output_dim

    text_pos = torch.arange(0, text_len, device=device)
    audio_pos = torch.arange(text_len, text_len + audio_len, device=device)
    img_pos = torch.arange(text_len + audio_len, used, device=device)

    x = torch.randn(1, seq_len, video_row, generator=gen, dtype=torch.float32).to(device)
    audio_x = torch.randn(1, seq_len, cfg.audio_latents_dim, generator=gen, dtype=torch.float32).to(device)
    img_position_ids = torch.zeros(1, seq_len, 3, device=device, dtype=torch.float32)
    img_position_ids[0, :used, 0] = torch.arange(used, device=device, dtype=torch.float32)
    img_position_ids[0, :used, 1] = torch.arange(used, device=device, dtype=torch.float32) % 4
    img_position_ids[0, :used, 2] = torch.arange(used, device=device, dtype=torch.float32) % 7

    token_tags = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    token_tags[text_pos] = 1
    token_tags[audio_pos] = 2
    token_tags[img_pos] = 0

    if num_timesteps == 1:
        unique_timesteps = torch.tensor([0.5], device=device, dtype=torch.float32)
        inverse_indices = torch.zeros(seq_len, dtype=torch.long, device=device)
    else:
        unique_timesteps = torch.linspace(0.25, 0.75, num_timesteps, device=device, dtype=torch.float32)
        inverse_indices = torch.arange(seq_len, device=device, dtype=torch.long) % num_timesteps

    prompt_embeds = torch.randn(text_len, cfg.text_dim, generator=gen, dtype=torch.bfloat16).to(device)
    update_mask = torch.ones(video_len, device=device, dtype=torch.float32)

    cu_list = [0, used, seq_len] if pad_len > 0 else [0, used]
    packed: dict[str, object] = {
        "cu_seqlens_q": torch.tensor(cu_list, dtype=torch.int32, device=device),
        "cu_seqlens_q_host": tuple(cu_list),
        "max_seqlen_q": used,
        "ignore_trailing_pad": ignore_trailing_pad,
    }
    refiner: dict[str, object] = {
        "cu_seqlens_q": torch.tensor([0, text_len], dtype=torch.int32, device=device),
        "cu_seqlens_q_host": (0, text_len),
        "max_seqlen_q": text_len,
    }
    return {
        "x": x,
        "audio_x": audio_x,
        "img_position_ids": img_position_ids,
        "unique_timesteps": unique_timesteps,
        "inverse_indices": inverse_indices,
        "update_mask": update_mask,
        "token_tags": token_tags,
        "prompt_embeds": prompt_embeds,
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": img_pos},
        "packed_seq_params": packed,
        "refiner_packed_seq_params": refiner,
    }


def _make_rope_cache(seq_len: int, rot_dim: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(7)
    half = rot_dim // 2
    angles = torch.randn(seq_len, half, generator=gen, dtype=torch.float32).to(device)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(torch.bfloat16).contiguous()


# --------------------------------------------------------------------------- #
# 1. Protocol & attention backend seam tests (CPU)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_protocol_conformance_across_minimax_h3_family() -> None:
    cfg = _tiny_arch(num_layers=3)
    base = MiniMaxH3DiTModel(cfg)
    pi = PiH3DiTModel(cfg, adaln_rank=8, time_table_size=17)
    tiny = H3TinyDiTModel(cfg, adaln_rank=8, endpoint_mode="meanflow")

    for model in (base, pi, tiny):
        assert isinstance(model, AttentionBackendConfigurable)
        assert isinstance(model, TransformerBlockStack)
        assert isinstance(model.token_refiner, AttentionBackendConfigurable)
        assert isinstance(model.token_refiner.blocks[0], AttentionBackendConfigurable)
        assert isinstance(model.blocks[0], AttentionBackendConfigurable)
        assert isinstance(model.blocks[0].attn, AttentionBackendConfigurable)


@pytest.mark.unit
def test_set_attention_backend_dense_layers_refiner_and_reset() -> None:
    model = MiniMaxH3DiTModel(_tiny_arch(num_layers=4))
    model.set_attention_backend(
        "sol_attn",
        backend_options={"tau": 0.75},
        dense_layers=(0, -1),
        include_refiner=False,
    )
    assert model.attention_backend == "sol_attn"
    assert model.blocks[0].attention_backend == "torch"
    assert model.blocks[1].attention_backend == "sol_attn"
    assert model.blocks[2].attention_backend == "sol_attn"
    assert model.blocks[3].attention_backend == "torch"
    assert cast(Any, model.blocks[1]).attn.attention_backend_options == {"tau": 0.75}
    assert model.token_refiner.attention_backend is None

    model.set_attention_backend("torch", include_refiner=True)
    assert model.token_refiner.attention_backend == "torch"

    model.set_attention_backend(None)
    assert model.attention_backend is None
    assert model.token_refiner.attention_backend is None
    assert all(b.attention_backend is None for b in model.blocks)

    with pytest.raises(ValueError, match="Unknown attention backend"):
        model.set_attention_backend("nonexistent_backend_xyz")


@pytest.mark.unit
def test_sol_attn_auto_prefix_sink_injection_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.attention.backends.dispatch as dispatch_mod

    seen: list[dict[str, object] | None] = []

    def fake_attention_forward(q, k, v, **kwargs):
        del k, v
        seen.append(kwargs.get("backend_options"))
        return q

    monkeypatch.setattr(dispatch_mod, "attention_forward", fake_attention_forward)
    torch.manual_seed(29)
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    inputs = _build_packed_inputs(cfg, torch.device("cpu"), text_len=4, audio_len=6, video_len=14)
    inputs["prompt_embeds"] = inputs["prompt_embeds"].float()  # type: ignore[union-attr]

    model.set_attention_backend("sol_attn", backend_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True})
    with torch.no_grad():
        model(**inputs)
    assert len(seen) == 2
    assert all(opt is not None and opt.get("sink_tokens") == 10 and opt.get("sink_start") == 0 for opt in seen)

    # Explicit sink_tokens=0 overrides automatic prefix sink protection.
    seen.clear()
    model.set_attention_backend("sol_attn", backend_options={"tau": 0.5, "compress_kv": True, "sink_tokens": 0})
    with torch.no_grad():
        model(**inputs)
    assert len(seen) == 2
    assert all(opt is not None and opt.get("sink_tokens") == 0 for opt in seen)


@pytest.mark.unit
def test_set_attention_backend_preserves_custom_attention_fn_hook() -> None:
    torch.manual_seed(5)
    model = MiniMaxH3DiTModel(_tiny_arch(num_layers=2)).float().eval()
    called = {"count": 0}

    def custom_attn(q, k, v, *, cu_seqlens, softmax_scale):
        del cu_seqlens, softmax_scale
        called["count"] += 1
        return v

    cast(Any, model.blocks[0]).attn.attention_fn = custom_attn
    model.set_attention_backend("sol_attn")
    inputs = _build_packed_inputs(model.arch, torch.device("cpu"))
    inputs["prompt_embeds"] = inputs["prompt_embeds"].float()
    # Run only block 0 via forward_blocks / full forward would hit block 1's sol_attn on CPU;
    # set block 1 back to torch so full forward runs on CPU while block 0 keeps custom hook.
    cast(Any, model.blocks[1]).set_attention_backend("torch")
    with torch.no_grad():
        video, audio = model(**inputs)
    assert called["count"] == 1
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


# --------------------------------------------------------------------------- #
# 2. Lossless projection fusion tests (CPU)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("head_dim", [None, 32])
@pytest.mark.parametrize("bias", [True, False])
def test_fuse_hyperflow_qkv_and_swiglu_exact_parity(head_dim: int | None, bias: bool) -> None:
    torch.manual_seed(11)
    wrapper: Any = nn.Module()
    wrapper.qkv = HyperFlowQKVProjection(
        nn.Linear(128, 3 * 128, bias=bias, dtype=torch.float32),
        head_dim=head_dim,
    )
    wrapper.fc1 = HyperFlowSwiGLUProjection(nn.Linear(128, 2 * 256, bias=bias, dtype=torch.float32))
    x = torch.randn(17, 128, dtype=torch.float32)

    with torch.no_grad():
        ref_qkv = wrapper.qkv(x)
        ref_fc1 = wrapper.fc1(x)

    counts = fuse_minimax_h3_projections(wrapper)
    assert counts == {"qkv_fused": 1, "swiglu_fused": 1}
    assert isinstance(wrapper.qkv, nn.Linear)
    assert isinstance(wrapper.fc1, nn.Linear)

    with torch.no_grad():
        got_qkv = wrapper.qkv(x)
        got_fc1 = wrapper.fc1(x)

    assert torch.allclose(got_qkv, ref_qkv, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got_fc1, ref_fc1, atol=1e-5, rtol=1e-5)
    # Idempotent on second call.
    assert fuse_minimax_h3_projections(wrapper) == {"qkv_fused": 0, "swiglu_fused": 0}


@pytest.mark.unit
def test_fuse_minimax_h3_projections_on_pi_h3_and_h3_tiny() -> None:
    cfg = _tiny_arch(num_layers=2)
    variants: tuple[tuple[Any, dict[str, Any]], ...] = (
        (PiH3DiTModel, {"adaln_rank": 8, "time_table_size": 17}),
        (H3TinyDiTModel, {"adaln_rank": 8, "endpoint_mode": "meanflow"}),
    )
    for model_cls, kwargs in variants:
        torch.manual_seed(17)
        model = model_cls(cfg, **kwargs).float().eval()
        inputs = _build_packed_inputs(cfg, torch.device("cpu"), seed=3)
        inputs["prompt_embeds"] = inputs["prompt_embeds"].float()
        with torch.no_grad():
            ref_v, ref_a = model(**inputs)
        counts = fuse_minimax_h3_projections(model)
        assert counts["qkv_fused"] >= cfg.num_layers
        assert counts["swiglu_fused"] >= cfg.num_layers
        with torch.no_grad():
            got_v, got_a = model(**inputs)
        assert torch.allclose(got_v, ref_v, atol=1e-4, rtol=1e-4)
        assert torch.allclose(got_a, ref_a, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# 3. Zero-sync _varlen_attention tests (CPU + CUDA)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_varlen_attention_empty_and_ignore_trailing_pad_cpu() -> None:
    torch.manual_seed(21)
    empty = torch.empty(0, 2, 64, dtype=torch.float32)
    cu_empty = torch.tensor([0, 0], dtype=torch.int32)
    out_empty = _varlen_attention(empty, empty, empty, cu_seqlens=cu_empty, softmax_scale=64**-0.5)
    assert out_empty.shape == (0, 2, 64)

    used, total = 19, 24
    q = torch.randn(total, 2, 64, dtype=torch.float32)
    k = torch.randn(total, 2, 64, dtype=torch.float32)
    v = torch.randn(total, 2, 64, dtype=torch.float32)
    cu = torch.tensor([0, used, total], dtype=torch.int32)

    ref = _varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=64**-0.5)
    fast = _varlen_attention(
        q,
        k,
        v,
        cu_seqlens=cu,
        cu_seqlens_host=(0, used, total),
        softmax_scale=64**-0.5,
        ignore_trailing_pad=True,
    )
    assert torch.allclose(fast[:used], ref[:used], atol=1e-6, rtol=1e-6)
    assert torch.equal(fast[used:], torch.zeros_like(fast[used:]))

    # Degenerate boundary: used == total and used == 0.
    cu_full = torch.tensor([0, total, total], dtype=torch.int32)
    fast_full = _varlen_attention(
        q, k, v, cu_seqlens=cu_full, cu_seqlens_host=(0, total, total), softmax_scale=64**-0.5, ignore_trailing_pad=True
    )
    ref_full = _varlen_attention(
        q, k, v, cu_seqlens=torch.tensor([0, total], dtype=torch.int32), softmax_scale=64**-0.5
    )
    assert torch.allclose(fast_full, ref_full, atol=1e-6, rtol=1e-6)

    cu_zero = torch.tensor([0, 0, total], dtype=torch.int32)
    fast_zero = _varlen_attention(
        q, k, v, cu_seqlens=cu_zero, cu_seqlens_host=(0, 0, total), softmax_scale=64**-0.5, ignore_trailing_pad=True
    )
    assert torch.equal(fast_zero, torch.zeros_like(q))


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_varlen_attention_multidoc_flash_attention_parity_cuda() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    bounds = [0, 17, 41, 64]
    total = bounds[-1]
    q = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)

    # Zero-sync packed varlen via aten._flash_attention_forward (max_seqlen supplied).
    packed = _varlen_attention(
        q, k, v, cu_seqlens=cu, cu_seqlens_host=tuple(bounds), max_seqlen=24, softmax_scale=128**-0.5
    )
    # Portable per-segment SDPA reference (max_seqlen=None forces per-segment path).
    ref = _varlen_attention(
        q, k, v, cu_seqlens=cu, cu_seqlens_host=tuple(bounds), max_seqlen=None, softmax_scale=128**-0.5
    )
    assert torch.isfinite(packed).all()
    assert torch.allclose(packed.float(), ref.float(), atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# 4. Fused Triton kernel tests (CUDA)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("rot_dim", [96, 128])
def test_strided_fused_qknorm_rope_on_qkv_split_cuda(rot_dim: int) -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    seq_len, num_heads, head_dim = 65, 8, 128
    inner_dim = num_heads * head_dim
    # Simulate fused QKV projection output where q/k/v slices are NON-CONTIGUOUS!
    qkv = torch.randn(seq_len, 3 * inner_dim, device=device, dtype=torch.bfloat16)
    q_flat, k_flat, _ = qkv.split(inner_dim, dim=-1)
    q = q_flat.view(seq_len, num_heads, head_dim)
    k = k_flat.view(seq_len, num_heads, head_dim)
    assert not q.is_contiguous() and q.stride(-1) == 1

    cache = _make_rope_cache(seq_len, rot_dim, device)
    positions = torch.arange(seq_len, device=device)
    q_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).to(device)
    k_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        q_norm.weight.uniform_(0.5, 1.5)
        k_norm.weight.uniform_(0.5, 1.5)

        assert ops.can_use_fused_qknorm_rope(q, k, cache) is True
        got_q, got_k = ops.qk_norm_rope(q, k, q_norm, k_norm, cache, positions)
        ref_q, ref_k = ops.apply_qk_norm(q, k, q_norm, k_norm, head_dim)
        ref_q, ref_k = ops.apply_rope_qk(ref_q, ref_k, cache, positions)

        assert got_q.is_contiguous() and got_k.is_contiguous()
        assert torch.allclose(got_q.float(), ref_q.float(), atol=3.5e-2, rtol=3.5e-2)
        assert torch.allclose(got_k.float(), ref_k.float(), atol=3.5e-2, rtol=3.5e-2)

        # Empty T=0 boundary.
        empty_q = torch.empty(0, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        empty_cache = torch.empty(0, rot_dim, device=device, dtype=torch.bfloat16)
        eq, ek = ops.qk_norm_rope(
            empty_q, empty_q, q_norm, k_norm, empty_cache, torch.empty(0, device=device, dtype=torch.long)
        )
        assert eq.shape == (0, num_heads, head_dim) and ek.shape == (0, num_heads, head_dim)


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_tiled_wide_row_rmsnorm_adaln_and_gate_cuda() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    # Real H3 hidden width 5376 exercises multi-tile BLOCK_N=1024 loop + partial tail tile.
    rows, hidden = 131, 5376
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    x_orig = x.clone()
    shift = torch.randn(6, hidden, device=device, dtype=torch.bfloat16) * 0.2
    scale = torch.randn(6, hidden, device=device, dtype=torch.bfloat16) * 0.2
    gate = torch.randn(6, hidden, device=device, dtype=torch.bfloat16) * 0.2
    other = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, 6, (rows,), device=device, dtype=torch.long)

    norm = ops.make_rms_norm(hidden, eps=1e-6, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)

        got_fused = ops.rms_norm_indexed_scale_shift(x, norm, shift, scale, idx, dtype=torch.bfloat16)
        ref_fused = (norm(x) * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)).to(torch.bfloat16)
        assert torch.allclose(got_fused.float(), ref_fused.float(), atol=3.5e-2, rtol=3.5e-2)
        assert torch.equal(x, x_orig)

        got_gate = ops.indexed_gate(x, gate, other, idx, dtype=torch.bfloat16)
        ref_gate = (x + gate.index_select(0, idx) * other).to(torch.bfloat16)
        assert torch.allclose(got_gate.float(), ref_gate.float(), atol=3.5e-2, rtol=3.5e-2)
        assert torch.equal(x, x_orig)

        # Empty row boundary.
        empty_x = torch.empty(0, hidden, device=device, dtype=torch.bfloat16)
        empty_idx = torch.empty(0, device=device, dtype=torch.long)
        empty_out = ops.rms_norm_indexed_scale_shift(empty_x, norm, shift, scale, empty_idx, dtype=torch.bfloat16)
        assert empty_out.shape == (0, hidden)

        # Row-strided slices from AdalnProj.chunk(6, dim=-1) (non-contiguous, stride(-1)==1) must be eligible.
        packed_adaln = torch.randn(6, 6 * hidden, device=device, dtype=torch.bfloat16) * 0.2
        s_msa, sc_msa, g_msa, _, _, _ = packed_adaln.chunk(6, dim=-1)
        assert not s_msa.is_contiguous() and s_msa.stride(-1) == 1
        assert ops.can_use_fused_indexed_modulation(x, norm.weight, s_msa, sc_msa) is True
        got_strided = ops.rms_norm_indexed_scale_shift(x, norm, s_msa, sc_msa, idx, dtype=torch.bfloat16)
        ref_strided = (norm(x) * (1.0 + sc_msa.index_select(0, idx)) + s_msa.index_select(0, idx)).to(torch.bfloat16)
        assert torch.allclose(got_strided.float(), ref_strided.float(), atol=3.5e-2, rtol=3.5e-2)
        got_strided_gate = ops.indexed_gate(x, g_msa, other, idx, dtype=torch.bfloat16)
        ref_strided_gate = (x + g_msa.index_select(0, idx) * other).to(torch.bfloat16)
        assert torch.allclose(got_strided_gate.float(), ref_strided_gate.float(), atol=3.5e-2, rtol=3.5e-2)


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("ffn_hidden", [1500, 14336])
def test_swiglu_quantize_rowwise_fp8_triton_exact_match_cuda(ffn_hidden: int) -> None:
    from worldfoundry.core.acceleration.quantization.linear import _quantize_rowwise_fp8
    from worldfoundry.core.kernels.triton_diffusion import silu_and_mul as triton_silu_and_mul

    torch.manual_seed(41)
    device = torch.device("cuda")
    rows = 37
    hidden = torch.randn(rows, 2 * ffn_hidden, device=device, dtype=torch.bfloat16)

    got_codes, got_scales = swiglu_quantize_rowwise_fp8_triton(hidden, torch.float8_e4m3fn)
    # Exact match against Triton silu_and_mul + rowwise FP8 quantization (same single-round boundary).
    triton_act = triton_silu_and_mul(hidden)
    triton_codes, triton_scales = _quantize_rowwise_fp8(triton_act, torch.float8_e4m3fn)
    assert got_codes.dtype == torch.float8_e4m3fn
    assert got_scales.shape == (rows, 1)
    assert torch.allclose(got_scales, triton_scales, atol=1e-6, rtol=1e-5)
    exact_match = float((got_codes.view(torch.uint8) == triton_codes.view(torch.uint8)).float().mean().item())
    assert exact_match > 0.999, f"expected >99.9% exact FP8 code match vs Triton split, got {exact_match:.6f}"

    # Also verify dequantized output closely matches PyTorch eager F.silu(gate) * up.
    gate, up = hidden.chunk(2, dim=-1)
    eager_ref = (torch.nn.functional.silu(gate) * up).float()
    dequant = got_codes.float() * got_scales
    rel_l2 = float((dequant - eager_ref).norm() / eager_ref.norm().clamp_min(1e-6))
    assert rel_l2 < 0.03, f"dequantized FP8 SwiGLU drifted from eager ref: rel_l2={rel_l2:.4f}"


# --------------------------------------------------------------------------- #
# 5. Hopper FP8 W8A8 + FusedFP8SwiGLUMLP tests (SM90+)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _HOPPER_OR_NEWER, reason="requires SM90+ Hopper/Blackwell CUDA GPU for rowwise FP8")
def test_quantize_minimax_h3_fp8_exclusions_and_fused_ffn_receipts() -> None:
    from worldfoundry.core.acceleration.quantization.linear import Float8Linear

    torch.manual_seed(43)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=1024, num_heads=8, head_dim=128, ffn_hidden_size=2048)
    model = PiH3DiTModel(cfg, adaln_rank=8, time_table_size=17).to(device).eval()
    _zero_adaln(model)
    fuse_minimax_h3_projections(model)

    inputs = _build_packed_inputs(cfg, device, text_len=8, audio_len=16, video_len=104, seed=9)
    with torch.inference_mode():
        ref_v, ref_a = model(**inputs)

    q_receipt = quantize_minimax_h3_fp8(
        model,
        scaling="rowwise",
        use_fast_accum=True,
        keep_dense_fallback=False,
        exclude_edge_blocks=True,
        fuse_fp8_ffn=True,
    )
    # 2 interior blocks (blocks 1 and 2) x 4 linears each (qkv, out, fc1, fc2) = 8 Float8Linears.
    assert q_receipt["fp8_linears"] == 8
    assert q_receipt["fused_fp8_ffns"] == 2

    # Edge blocks 0 and 3 and every sensitive layer stay unquantized.
    blocks_any: Any = model.blocks
    assert not isinstance(blocks_any[0].attn.qkv_proj, Float8Linear)
    assert not isinstance(blocks_any[3].attn.qkv_proj, Float8Linear)
    assert isinstance(blocks_any[1].attn.qkv_proj, Float8Linear)
    assert isinstance(blocks_any[1].mlp, FusedFP8SwiGLUMLP)
    for name, module in model.named_modules():
        if any(pattern in name for pattern in DEFAULT_MINIMAX_H3_FP8_EXCLUDES):
            assert not isinstance(module, Float8Linear), f"sensitive module {name} must not be FP8 quantized"

    model.reset_acceleration_window()
    with torch.inference_mode():
        got_v, got_a = model(**inputs)

    assert torch.isfinite(got_v).all() and torch.isfinite(got_a).all()
    rel_l2_v = float((got_v - ref_v).norm() / ref_v.norm().clamp_min(1e-6))
    assert rel_l2_v < 0.08, f"FP8 forward drifted too far from BF16 reference: rel_l2={rel_l2_v:.4f}"

    report = model.acceleration_report()
    assert report["quantization"]["low_precision_kernel_calls"] > 0
    assert report["quantization"]["dense_fallback_calls"] == 0
    assert report["fused_fp8_swiglu"]["layers"] == 2
    assert report["fused_fp8_swiglu"]["fused_calls"] == 2
    assert report["fused_fp8_swiglu"]["fallback_calls"] == 0

    # Verify fail-fast when low_precision_enabled=False without dense fallback, and
    # clean dense fallback when keep_dense_fallback=True.
    mlp = model.blocks[1].mlp
    assert isinstance(mlp, FusedFP8SwiGLUMLP)
    mlp.fc2.low_precision_enabled = False
    x_probe = torch.randn(16, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no dense fallback was retained"):
        _ = mlp(x_probe)
    assert mlp.fallback_calls >= 1

    fc1_fb = nn.Linear(cfg.hidden_size, 2 * cfg.ffn_hidden_size, bias=False, device=device, dtype=torch.bfloat16)
    fc2_dense = nn.Linear(cfg.ffn_hidden_size, cfg.hidden_size, bias=False, device=device, dtype=torch.bfloat16)
    fc2_fp8_fb = Float8Linear.from_linear(fc2_dense, keep_dense_fallback=True, scaling="rowwise")
    fc2_fp8_fb.low_precision_enabled = False
    mlp_fb = FusedFP8SwiGLUMLP(fc1_fb, fc2_fp8_fb)
    with torch.inference_mode():
        out_fb = mlp_fb(x_probe)
    assert torch.isfinite(out_fb).all()
    assert mlp_fb.fallback_calls == 1 and mlp_fb.fused_calls == 0


# --------------------------------------------------------------------------- #
# 6. CUDA Graph & Piecewise Block Graph tests (CUDA)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", ["whole", "piecewise"])
def test_cuda_graph_capture_replay_multisignature_and_invalidation(graph_mode: str) -> None:
    torch.manual_seed(53)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)

    inputs_m1 = _build_packed_inputs(cfg, device, pad_len=4, ignore_trailing_pad=True, num_timesteps=1, seed=1)
    inputs_m2 = _build_packed_inputs(cfg, device, pad_len=4, ignore_trailing_pad=True, num_timesteps=2, seed=2)
    with torch.inference_mode():
        ref_v1, ref_a1 = model(**inputs_m1)
        ref_v2, ref_a2 = model(**inputs_m2)

    install_receipt = install_minimax_h3_cuda_graph(
        model,
        mode=cast(Any, graph_mode),
        warmup=1,
        head_blocks=1,
        tail_blocks=1,
    )
    assert install_receipt["mode"] == graph_mode

    # Call 1 = warmup, Call 2 = capture, Call 3 = replay (for both M=1 and M=2).
    with torch.inference_mode():
        for _ in range(2):
            model(**inputs_m1)
        model.reset_acceleration_window()
        got_v1, got_a1 = model(**inputs_m1)
        for _ in range(3):
            got_v2, got_a2 = model(**inputs_m2)

    assert torch.allclose(got_v1, ref_v1, atol=2e-3, rtol=2e-3)
    assert torch.allclose(got_a1, ref_a1, atol=2e-3, rtol=2e-3)
    assert torch.allclose(got_v2, ref_v2, atol=2e-3, rtol=2e-3)
    assert torch.allclose(got_a2, ref_a2, atol=2e-3, rtol=2e-3)

    cg_report = model.acceleration_report()["cuda_graph"]
    assert cg_report is not None
    if graph_mode == "whole":
        assert cg_report["graphs"] == 2
        assert cg_report["lifetime"]["capture"] >= 2
        assert cg_report["replay"] >= 2
        assert cg_report["capture_failed"] == 0
    else:
        assert len(cg_report["segments"]) == 3
        for seg in cg_report["segments"].values():
            assert seg["graphs"] == 2
            assert seg["lifetime"]["capture"] >= 2
            assert seg["replay"] >= 2
            assert seg["capture_failed"] == 0

    # Changing attention backend must invalidate captured graphs cleanly.
    model.set_attention_backend("torch")
    cg_after = model.acceleration_report()["cuda_graph"]
    if graph_mode == "whole":
        assert cg_after["graphs"] == 0
    else:
        assert all(seg["graphs"] == 0 for seg in cg_after["segments"].values())


@pytest.mark.unit
def test_cuda_graph_invalid_mode_and_sol_attn_compile_conflict() -> None:
    model = MiniMaxH3DiTModel(_tiny_arch(num_layers=2))
    with pytest.raises(ValueError, match="unsupported MiniMax H3 cuda_graph mode"):
        install_minimax_h3_cuda_graph(model, mode=cast(Any, "bogus"))

    conflict_cfg = MiniMaxH3AccelerationConfig(attention_backend="sol_attn", compile=True)
    with pytest.raises(ValueError, match="Sol-Attn cannot be combined with torch.compile"):
        accelerate_minimax_h3_dit(model, conflict_cfg)


# --------------------------------------------------------------------------- #
# 7. Real SOL-Attn and cuDNN FP8 attention routing tests (CUDA SM90+)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("sol_attn").usable),
    reason="requires Hopper CUDA GPU with sol_attn available",
)
def test_sol_attn_routing_and_cuda_graph_composition() -> None:
    torch.manual_seed(61)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    inputs = _build_packed_inputs(cfg, device, text_len=8, audio_len=16, video_len=104, seed=11)

    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn",
            attention_options={"tau": 0.5, "thresh_type": "diag", "compress_kv": True},
            dense_attention_layers=(0,),
            quantization="none",
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            piecewise_head_blocks=1,
            piecewise_tail_blocks=1,
        ),
    )

    reset_attention_provider_runtime()
    with torch.inference_mode():
        # Warmup + capture + replay.
        v_eager, a_eager = model(**inputs)
        _ = model(**inputs)
        v_replay, a_replay = model(**inputs)

    assert torch.isfinite(v_replay).all() and torch.isfinite(a_replay).all()
    assert torch.allclose(v_replay, v_eager, atol=2e-3, rtol=2e-3)
    assert torch.allclose(a_replay, a_eager, atol=2e-3, rtol=2e-3)

    report = model.acceleration_report()
    assert report["block_attention_backends"] == {"torch": 1, "sol_attn": 3}
    prov = attention_provider_runtime_report()
    assert prov.get("sol_attn", {}).get("successes", 0) > 0
    for seg in report["cuda_graph"]["segments"].values():
        assert seg["capture_failed"] == 0
        assert seg["replay"] >= 1


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("cudnn_fp8").usable),
    reason="requires Hopper CUDA GPU with cudnn_fp8 available",
)
def test_cudnn_fp8_attention_routing_eager_and_graph_cuda() -> None:
    torch.manual_seed(67)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=2, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    inputs = _build_packed_inputs(cfg, device, text_len=8, audio_len=16, video_len=104, seed=13)

    with torch.inference_mode():
        ref_v, ref_a = model(**inputs)

    reset_attention_provider_runtime()
    model.set_attention_backend("cudnn_fp8")
    with torch.inference_mode():
        fp8_v, fp8_a = model(**inputs)

    assert torch.isfinite(fp8_v).all() and torch.isfinite(fp8_a).all()
    rel_l2 = float((fp8_v - ref_v).norm() / ref_v.norm().clamp_min(1e-6))
    assert rel_l2 < 0.08, f"cudnn_fp8 attention drifted unexpectedly: rel_l2={rel_l2:.4f}"
    prov = attention_provider_runtime_report()
    assert prov.get("cudnn_fp8", {}).get("successes", 0) >= 2

    # Static descale under CUDA Graph capture must succeed without host sync.
    install_minimax_h3_cuda_graph(model, mode="whole", warmup=1)
    with torch.inference_mode():
        _ = model(**inputs)
        _ = model(**inputs)
        cg_v, cg_a = model(**inputs)
    assert torch.isfinite(cg_v).all() and torch.isfinite(cg_a).all()
    cg_rep = model.acceleration_report()["cuda_graph"]
    assert cg_rep["capture_failed"] == 0 and cg_rep["replay"] >= 1


# --------------------------------------------------------------------------- #
# 8. H3-Tiny zero-sync MeanFlow, SolH3DWDHead RoPE cache, Video VAE & Pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_h3_tiny_skip_host_validation_and_exact_profile_cpu() -> None:
    torch.manual_seed(71)
    cfg = _tiny_arch(num_layers=2)
    model = H3TinyDiTModel(cfg, adaln_rank=8, endpoint_mode="meanflow").float().eval()
    inputs = _build_packed_inputs(cfg, torch.device("cpu"), seed=5)
    inputs["prompt_embeds"] = inputs["prompt_embeds"].float()  # type: ignore[union-attr]

    t = torch.tensor([0.25], dtype=torch.float32)
    r = torch.tensor([0.75], dtype=torch.float32)
    with torch.no_grad():
        ref_emb = model.time_embedder.forward_with_endpoint(t, r)
    receipt = accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True, quantization="none", cuda_graph="off", skip_host_validation=True
        ),
    )
    assert receipt["projection_fusion"]["qkv_fused"] >= 2
    assert model.time_embedder._skip_host_validation is True
    with torch.no_grad():
        fast_emb = model.time_embedder.forward_with_endpoint(t, r)
        v, a = model(**inputs)
    assert torch.equal(fast_emb, ref_emb)
    assert torch.isfinite(v).all() and torch.isfinite(a).all()


@pytest.mark.unit
def test_sol_dwd_head_rope_cache_determinism() -> None:
    torch.manual_seed(73)
    head = SolH3DWDHead(
        in_channels=64,
        src_patch_size=(4, 16, 16),
        first_frame_src_patch_size=(1, 16, 16),
        first_frame_period=4,
        num_layers=2,
        total_vae_layers=2,
        num_heads=2,
        head_dim=32,
        ffn_mult=2,
        num_register_tokens=2,
        window_size=2,
        block_size=4,
        t_shift=1,
        spatial_patch=8,
        patch_t=4,
        frames_to_trim=3,
    ).eval()
    with torch.no_grad():
        nn.init.normal_(head.register_tokens, std=0.02)
        for blk in head.blocks:
            blk_any: Any = blk
            nn.init.ones_(blk_any.scale1)
            nn.init.ones_(blk_any.scale2)
    x = torch.randn(1, 2 * 2 * 2, 64, dtype=torch.float32)
    with torch.no_grad():
        out1 = head(None, x, (2, 2, 2), force_padded=True)
        out2 = head(None, x, (2, 2, 2), force_padded=True)
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2)


@pytest.mark.unit
def test_vit3d_decoder_prepare_autocast_weights_preserves_fp32_io() -> None:
    dec = ViT3DDecoder(
        in_channels=24,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=2,
        heads=2,
        dim_head=64,
        rope_dim_ratio=0.75,
        ffn_use_gated=True,
    )
    # Also verify HyperFlow wrappers are handled cleanly if present.
    tblocks: Any = dec.transformer_blocks
    tblocks[0].attn.to_qkv = HyperFlowQKVProjection(
        nn.Linear(128, 3 * 128, bias=True, dtype=torch.float32),
        head_dim=64,
    )
    converted = dec.prepare_autocast_linear_weights(torch.bfloat16)
    assert converted >= 6
    assert dec.x_embedder.weight.dtype == torch.float32
    assert dec.proj_out.weight.dtype == torch.float32
    assert tblocks[1].attn.to_qkv.weight.dtype == torch.bfloat16
    assert cast(Any, tblocks[0].attn.to_qkv).q.weight.dtype == torch.bfloat16


@pytest.mark.unit
def test_sol_profiles_and_pipeline_configure_acceleration() -> None:
    for tier in ("exact", "fast", "sol_high_fidelity", "sol_balanced", "sol_turbo"):
        cfg_sm90 = minimax_h3_sol_profile(tier, sm_major=9)
        assert cfg_sm90.fuse_projections is True
        if tier != "exact":
            assert cfg_sm90.quantization == "fp8"
        cfg_sm80 = minimax_h3_sol_profile(tier, sm_major=8)
        assert cfg_sm80.quantization == "none"

    with pytest.raises(ValueError, match="unsupported MiniMax H3 acceleration tier"):
        minimax_h3_sol_profile("unknown_tier")

    # Verify NativeMiniMaxH3Pipeline.from_pretrained auto-aligns VAE flags on profile string.
    dit = MiniMaxH3DiTModel(_tiny_arch(num_layers=2))

    class _FakeVAE:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def configure_acceleration(self, **kwargs):
            self.calls.append(kwargs)
            return dict(kwargs)

        def acceleration_report(self):
            return self.calls[-1] if self.calls else {}

    vae = _FakeVAE()
    pipe = NativeMiniMaxH3Pipeline.from_pretrained(
        transformer=dit,
        video_vae=vae,
        device="cpu",
        acceleration="exact",
    )
    assert len(vae.calls) == 1
    assert vae.calls[0]["autocast_dtype"] == torch.bfloat16
    assert vae.calls[0]["channels_last_3d"] is True
    assert vae.calls[0]["stack_tiling"] is True
    report = pipe.acceleration_report()
    assert "transformer" in report and "video_vae" in report

    # Explicit vae_decode_autocast=False must be respected.
    vae2 = _FakeVAE()
    _ = NativeMiniMaxH3Pipeline.from_pretrained(
        transformer=MiniMaxH3DiTModel(_tiny_arch(num_layers=2)),
        video_vae=vae2,
        device="cpu",
        acceleration="exact",
        vae_decode_autocast=False,
    )
    assert vae2.calls[0]["autocast_dtype"] is False


@pytest.mark.unit
def test_denoise_loop_request_static_rope_and_refiner_cache_matches_uncached() -> None:
    from worldfoundry.pipelines.minimax._minimax_h3 import minimax_h3_packed_sequence

    torch.manual_seed(79)
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)

    text_len, audio_t = 3, 4
    latent_t, latent_h, latent_w = 2, 4, 4
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    prompt = torch.randn(text_len, cfg.text_dim, dtype=torch.float32)
    video_len = int(packed["img_pos"].shape[0])
    audio_len = int(packed["audio_pos"].shape[0])
    video_init = torch.randn(video_len, cfg.video_patch_output_dim, dtype=torch.float32)
    audio_init = torch.randn(audio_len, cfg.audio_latents_dim, dtype=torch.float32)

    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=prompt,
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    sigmas = [1.0, 0.5, 0.0]

    with torch.no_grad():
        got_v, got_a = minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=video_init.clone(),
            initial_audio_rows=audio_init.clone(),
            keyframe_cond_rows=None,
            sigmas_video=sigmas,
            sigmas_audio=sigmas,
            device=torch.device("cpu"),
        )

    # Compare against an uncached reference using a thin wrapper that bypasses
    # isinstance(..., MiniMaxH3DiTModel) precomputation and full trailing-pad attention.
    class _UncachedWrapper(nn.Module):
        def __init__(self, inner: MiniMaxH3DiTModel) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, **kwargs):
            return self.inner(**kwargs)

    branch_full = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=prompt,
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    branch_full.static_kwargs["packed_seq_params"]["ignore_trailing_pad"] = False
    with torch.no_grad():
        ref_v, ref_a = minimax_h3_denoise_loop(
            model=_UncachedWrapper(model),
            positive=branch_full,
            initial_video_rows=video_init.clone(),
            initial_audio_rows=audio_init.clone(),
            keyframe_cond_rows=None,
            sigmas_video=sigmas,
            sigmas_audio=sigmas,
            device=torch.device("cpu"),
        )

    assert torch.allclose(got_v, ref_v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(got_a, ref_a, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# 9. Adversarial & regression tests (review hardening)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_ops_autograd_eps_none_1d_and_dtype_safety_cuda() -> None:
    device = torch.device("cuda")
    rows, heads, head_dim = 8, 2, 128
    hidden = 64

    # 1. nn.RMSNorm(eps=None) must not raise TypeError in qk_norm_rope or rms_norm_indexed_scale_shift.
    q = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    cs = _make_rope_cache(rows, 96, device)
    pos = torch.arange(rows, device=device)
    qn_none = nn.RMSNorm(head_dim, eps=None, dtype=torch.bfloat16).to(device)
    kn_none = nn.RMSNorm(head_dim, eps=None, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        qo, ko = ops.qk_norm_rope(q, k, qn_none, kn_none, cs, pos)
    assert qo.shape == q.shape and ko.shape == k.shape and torch.isfinite(qo).all()

    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    shift = torch.randn(2, hidden, device=device, dtype=torch.bfloat16)
    scale = torch.randn(2, hidden, device=device, dtype=torch.bfloat16)
    idx = torch.zeros(rows, device=device, dtype=torch.long)
    norm_none = nn.RMSNorm(hidden, eps=None, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        mo = ops.rms_norm_indexed_scale_shift(x, norm_none, shift, scale, idx, dtype=torch.bfloat16)
    assert mo.shape == x.shape and torch.isfinite(mo).all()

    # 2. Autograd must never be silently detached when requires_grad=True.
    qg = q.detach().clone().requires_grad_(True)
    kg = k.detach().clone().requires_grad_(True)
    qog, kog = ops.qk_norm_rope(qg, kg, qn_none, kn_none, cs, pos)
    assert qog.requires_grad and kog.requires_grad
    (qog.float().sum() + kog.float().sum()).backward()
    assert qg.grad is not None and kg.grad is not None

    xg = x.detach().clone().requires_grad_(True)
    sg = shift.detach().clone().requires_grad_(True)
    scg = scale.detach().clone().requires_grad_(True)
    mog = ops.rms_norm_indexed_scale_shift(xg, norm_none, sg, scg, idx, dtype=torch.bfloat16)
    assert mog.requires_grad
    mog.float().sum().backward()
    assert xg.grad is not None and sg.grad is not None and scg.grad is not None

    # 3. FP32 cos_sin_cache must preserve PyTorch fp32 promotion instead of silently truncating to bf16.
    cs_fp32 = cs.float()
    with torch.no_grad():
        qo_fp32, _ = ops.qk_norm_rope(q, k, qn_none, kn_none, cs_fp32, pos)
    assert qo_fp32.dtype == torch.float32

    # 4. CPU indices or invalid 1D shift/scale must fall back cleanly instead of crashing Triton or reading OOB.
    idx_cpu = torch.zeros(rows, device="cpu", dtype=torch.long)
    with torch.no_grad(), pytest.raises(RuntimeError):
        # Eager fallback raises standard PyTorch device-mismatch error rather than Triton pointer crash.
        ops.indexed_scale_shift(x, shift, scale, idx_cpu, dtype=torch.bfloat16)
    shift_1d = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    scale_1d = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), pytest.raises(RuntimeError):
        # Previously Triton silently accepted 1D shift/scale and read out-of-bounds GPU memory;
        # now it falls back to PyTorch eager which cleanly rejects the mismatched broadcast shape.
        ops.indexed_scale_shift(x, shift_1d, scale_1d, idx, dtype=torch.bfloat16)


@pytest.mark.unit
def test_fl2va_prefix_sink_tokens_protects_text_cond_and_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.attention.backends.dispatch as dispatch_mod
    from worldfoundry.pipelines.minimax._minimax_h3 import minimax_h3_packed_sequence

    seen: list[dict[str, object] | None] = []

    def fake_attention_forward(q, k, v, **kwargs):
        del k, v
        seen.append(kwargs.get("backend_options"))
        return q

    monkeypatch.setattr(dispatch_mod, "attention_forward", fake_attention_forward)
    torch.manual_seed(83)
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    packed = minimax_h3_packed_sequence(
        text_len=5,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=True,
        keyframe_frame_indices=(0,),
        frame_count=5,
    )
    cond_rows = int((~packed["update_mask"]).sum())
    audio_rows = int(packed["audio_pos"].shape[0])
    expected_prefix = 5 + cond_rows + audio_rows
    assert expected_prefix == int(packed["img_pos"][packed["update_mask"]][0])

    prompt = torch.randn(5, cfg.text_dim, dtype=torch.float32)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=prompt,
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    step_t = branch.prepare_timestep_plan(
        video_timesteps=[0.5],
        audio_timesteps=[0.5],
        imgvid_cond_noise_aug=0.999,
        audio_ref_cond_noise_aug=1.0,
    )[0]
    fk = branch.forward_kwargs(
        video_rows=torch.randn(int(packed["img_pos"].shape[0]), cfg.video_patch_output_dim),
        audio_rows=torch.randn(audio_rows, cfg.audio_latents_dim),
        step_timesteps=step_t,
    )
    model.set_attention_backend("sol_attn", backend_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True})
    with torch.no_grad():
        model(**fk)
    assert len(seen) == 2
    assert all(
        opt is not None and opt.get("sink_tokens") == expected_prefix and opt.get("sink_start") == 0 for opt in seen
    )


@pytest.mark.unit
def test_validation_before_mutation_sticky_refiner_and_varlen_zero_tail() -> None:
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg)

    # Alias 'sol' + compile must raise BEFORE mutating model state.
    with pytest.raises(ValueError, match="Sol-Attn cannot be combined with torch.compile"):
        accelerate_minimax_h3_dit(
            model,
            MiniMaxH3AccelerationConfig(
                attention_backend="sol",
                quantization="fp8",
                fp8_min_features=128,
                compile=True,
            ),
        )
    assert model.attention_backend is None
    assert all(b.attention_backend is None for b in model.blocks)

    # Switching from include_refiner=True to include_refiner=False must clear refiner backend.
    model.set_attention_backend("sol_attn", include_refiner=True)
    assert model.token_refiner.attention_backend == "sol_attn"
    model.set_attention_backend("torch", include_refiner=False)
    assert model.token_refiner.attention_backend is None

    # Uncovered tail when bounds[-1] < total must be zero-initialized, never uninitialized NaN garbage.
    q = torch.randn(16, 2, 64, dtype=torch.float32)
    cu = torch.tensor([0, 10], dtype=torch.int32)
    out = _varlen_attention(q, q, q, cu_seqlens=cu, cu_seqlens_host=(0, 10), softmax_scale=64**-0.5)
    assert torch.isfinite(out).all()
    assert torch.equal(out[10:], torch.zeros_like(out[10:]))


@pytest.mark.unit
def test_step_adaptive_dense_initial_steps_in_denoise_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.core.attention.backends.dispatch as dispatch_mod
    from worldfoundry.pipelines.minimax._minimax_h3 import minimax_h3_packed_sequence

    sol_calls = {"count": 0}

    def fake_attention_forward(q, k, v, **kwargs):
        del k, v
        if kwargs.get("backend") == "sol_attn":
            sol_calls["count"] += 1
        return q

    monkeypatch.setattr(dispatch_mod, "attention_forward", fake_attention_forward)
    torch.manual_seed(89)
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)
    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=False,
            attention_backend="sol_attn",
            attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True},
            dense_initial_steps=2,
            quantization="none",
            cuda_graph="off",
        ),
    )

    packed = minimax_h3_packed_sequence(
        text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=3, include_keyframe_cond=False
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(4, cfg.text_dim, dtype=torch.float32),
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    # 4 steps total: steps 0 & 1 run dense torch (0 sol calls), steps 2 & 3 run sol_attn (2 steps * 2 layers = 4 calls).
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    with torch.no_grad():
        minimax_h3_denoise_loop(
            model=model,
            positive=branch,
            initial_video_rows=torch.randn(int(packed["img_pos"].shape[0]), cfg.video_patch_output_dim),
            initial_audio_rows=torch.randn(int(packed["audio_pos"].shape[0]), cfg.audio_latents_dim),
            keyframe_cond_rows=None,
            sigmas_video=sigmas,
            sigmas_audio=sigmas,
            device=torch.device("cpu"),
        )
    assert sol_calls["count"] == 4


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("cudnn_fp8").usable),
    reason="requires Hopper CUDA GPU with cudnn_fp8 available",
)
def test_cudnn_fp8_per_layer_warmup_calibration_under_cuda_graph_cuda() -> None:
    torch.manual_seed(97)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=2, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    # Scale QKV weights so V amax >> 14 (matching real 50-layer H3 where V amax reaches 836).
    with torch.no_grad():
        for block in model.blocks:
            cast(Any, block).attn.qkv_proj.weight.mul_(25.0)
    inputs = _build_packed_inputs(cfg, device, text_len=8, audio_len=16, video_len=104, seed=17)

    with torch.inference_mode():
        ref_v, _ = model(**inputs)

    model.set_attention_backend("cudnn_fp8")
    with torch.inference_mode():
        eager_v, _ = model(**inputs)

    install_minimax_h3_cuda_graph(model, mode="whole", warmup=1)
    with torch.inference_mode():
        _ = model(**inputs)
        _ = model(**inputs)
        model.reset_acceleration_window()
        cg_v, _ = model(**inputs)

    assert all(cast(Any, b).attn._cudnn_fp8_calibrated_descales is not None for b in model.blocks)
    cg_vs_eager = float((cg_v - eager_v).norm() / eager_v.norm().clamp_min(1e-6))
    cg_vs_ref = float((cg_v - ref_v).norm() / ref_v.norm().clamp_min(1e-6))
    assert cg_vs_eager < 0.02, f"calibrated cudnn_fp8 CUDA Graph drifted from eager: {cg_vs_eager:.4f}"
    assert cg_vs_ref < 0.03, f"calibrated cudnn_fp8 CUDA Graph drifted from BF16 ref: {cg_vs_ref:.4f}"

    # Verify pure-replay graph telemetry captures cudnn_fp8 executions.
    rep = model.acceleration_report()
    assert rep["cuda_graph"]["replay"] == 1
    assert rep["attention_provider_runtime_effective"].get("cudnn_fp8", 0) == 2


@pytest.mark.skipif(not _HOPPER_OR_NEWER, reason="requires SM90+ Hopper/Blackwell CUDA GPU for rowwise FP8")
def test_cuda_graph_pure_replay_telemetry_receipts_fp8_and_ffn() -> None:
    torch.manual_seed(101)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=1024, num_heads=8, head_dim=128, ffn_hidden_size=2048)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    inputs = _build_packed_inputs(cfg, device, text_len=8, audio_len=16, video_len=104, seed=19)

    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="torch",
            quantization="fp8",
            fp8_exclude_edge_blocks=True,
            fuse_fp8_ffn=True,
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            piecewise_head_blocks=1,
            piecewise_tail_blocks=1,
        ),
    )

    with torch.inference_mode():
        _ = model(**inputs)
        _ = model(**inputs)
        model.reset_acceleration_window()
        _ = model(**inputs)
        _ = model(**inputs)

    rep = model.acceleration_report()
    # 2 interior blocks x 4 FP8 linears x 2 pure-replay steps = 16; 2 interior FFNs x 2 steps = 4.
    assert rep["quantization"]["graph_replayed_low_precision_kernel_calls"] == 16
    assert rep["quantization"]["effective_low_precision_kernel_calls"] == 16
    assert rep["fused_fp8_swiglu"]["graph_replayed_fused_calls"] == 4
    assert rep["fused_fp8_swiglu"]["effective_fused_calls"] == 4


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_video_vae_disable_autocast_restores_fp32_and_encode_channels_last_3d_cuda() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video import MiniMaxH3VideoVAE
    from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.config import (
        MiniMaxH3VideoVAEArchConfig,
        MiniMaxH3VideoVAEConfig,
    )

    device = torch.device("cuda")
    vae = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig(arch_config=MiniMaxH3VideoVAEArchConfig(latent_channels=24)))
    vae.decoder.transformer_blocks = vae.decoder.transformer_blocks[:2]
    vae = vae.to(device).eval()

    z = torch.randn(2, 24, 1, 2, 2, device=device, dtype=torch.float32)
    vae.configure_acceleration(autocast_dtype=torch.bfloat16, channels_last_3d=True, stack_tiling=True)
    out_bf16 = vae.decode(z)
    assert torch.isfinite(out_bf16).all()

    # Disabling autocast must restore FP32 linear weights cleanly (no Float vs BFloat16 crash).
    rep = vae.configure_acceleration(autocast_dtype=None, channels_last_3d=True, stack_tiling=True)
    assert rep["decode_autocast"]["enabled"] is False
    assert rep["decode_autocast"]["restored_linears"] > 0
    out_fp32 = vae.decode(z)
    assert torch.isfinite(out_fp32).all() and out_fp32.shape == out_bf16.shape


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_sol_h3_swiglu_triton_fast_path_parity_cuda() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.pi_h3.sol_dwd_head import (
        SolH3SwiGLU,
    )

    torch.manual_seed(107)
    device = torch.device("cuda")
    ff = SolH3SwiGLU(256, 512).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(16, 256, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        got = ff(x)
        h = ff.w1(x)
        g, u = h.chunk(2, dim=-1)
        ref = ff.w2(torch.nn.functional.silu(g) * u)
    assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2)

    # Autograd must stay intact.
    xg = x.detach().clone().requires_grad_(True)
    out_g = ff(xg)
    assert out_g.requires_grad
    out_g.float().sum().backward()
    assert xg.grad is not None


# --------------------------------------------------------------------------- #
# 10. Wave 2: SM90 kv_splits, Direct/Varlen FA3, Tiled QK-Norm, Step Cache, CFG=2
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sm90_sol_kv_splits_clamping_and_profile_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
        _resolve_sm90_sol_kv_splits,
    )

    # Non-SM90 (CPU) always clamps kv_splits > 1 back to 1 and drops allow_multi_split.
    cpu_opts = _resolve_sm90_sol_kv_splits(
        {"tau": 0.5, "kv_splits": 4, "allow_multi_split": True},
        seq_len=38624,
        device=torch.device("cpu"),
    )
    assert cpu_opts["kv_splits"] == 1
    assert "allow_multi_split" not in cpu_opts

    # Simulate SM90 capability across route_groups = 1, 2, 4, 10.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (9, 0))
    fake_cuda = torch.device("cuda", 0)

    sm90_rg4 = _resolve_sm90_sol_kv_splits({"kv_splits": 4}, seq_len=14696, device=fake_cuda)
    assert sm90_rg4["kv_splits"] == 4 and sm90_rg4["allow_multi_split"] is True

    sm90_rg2 = _resolve_sm90_sol_kv_splits({"kv_splits": 4}, seq_len=8192, device=fake_cuda)
    assert sm90_rg2["kv_splits"] == 2 and sm90_rg2["allow_multi_split"] is True

    sm90_rg1 = _resolve_sm90_sol_kv_splits(
        {"kv_splits": 4, "allow_multi_split": True},
        seq_len=512,
        device=fake_cuda,
    )
    assert sm90_rg1["kv_splits"] == 1 and "allow_multi_split" not in sm90_rg1

    # Explicit allow_multi_split=False must not be silently overridden.
    explicit_false = _resolve_sm90_sol_kv_splits(
        {"kv_splits": 2, "allow_multi_split": False},
        seq_len=14696,
        device=fake_cuda,
    )
    assert explicit_false["kv_splits"] == 2 and explicit_false["allow_multi_split"] is False

    # minimax_h3_sol_profile kv_splits validation and SM90/SM80 routing.
    cfg_kv4 = minimax_h3_sol_profile("sol_balanced", sm_major=9, kv_splits=4)
    assert cfg_kv4.attention_options is not None
    assert cfg_kv4.attention_options["kv_splits"] == 4
    assert cfg_kv4.attention_options["allow_multi_split"] is True
    assert cfg_kv4.fp8_use_fast_accum is False

    cfg_sm80 = minimax_h3_sol_profile("sol_balanced", sm_major=8, kv_splits=4)
    assert cfg_sm80.attention_options is not None
    assert "kv_splits" not in cfg_sm80.attention_options

    for bad_splits in (0, 3, 8, True):
        with pytest.raises(ValueError, match="kv_splits must be 1, 2, or 4"):
            minimax_h3_sol_profile("sol_balanced", sm_major=9, kv_splits=cast(Any, bad_splits))


@pytest.mark.skipif(
    not (
        _CUDA
        and tuple(torch.cuda.get_device_capability(0)) == (9, 0)
        and attention_backend_capability("sol_attn").usable
    ),
    reason="requires Hopper SM90 CUDA GPU with sol_attn available",
)
def test_sol_attn_sm90_kv_splits_2_and_4_piecewise_graph_cuda() -> None:
    torch.manual_seed(109)
    device = torch.device("cuda")
    # Use seq_len=12352 (route_groups=4) with tiny 2-head model so kv_splits=1, 2, 4 all execute natively.
    cfg = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    inputs = _build_packed_inputs(cfg, device, text_len=16, audio_len=32, video_len=12304, pad_len=0, seed=21)

    outputs: dict[int, torch.Tensor] = {}
    for splits in (1, 2, 4):
        accelerate_minimax_h3_dit(
            model,
            MiniMaxH3AccelerationConfig(
                fuse_projections=False,
                attention_backend="sol_attn",
                attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True, "kv_splits": splits},
                dense_attention_backend="torch",
                dense_attention_layers=(0, -1),
                quantization="none",
                cuda_graph="piecewise",
                cuda_graph_warmup=1,
                piecewise_head_blocks=1,
                piecewise_tail_blocks=1,
            ),
        )
        reset_attention_provider_runtime()
        with torch.inference_mode():
            _ = model(**inputs)
            _ = model(**inputs)
            model.reset_acceleration_window()
            v_out, a_out = model(**inputs)
        assert torch.isfinite(v_out).all() and torch.isfinite(a_out).all()
        rep = model.acceleration_report()
        assert rep["attention_provider_runtime_effective"].get("sol_attn", 0) == 2
        for seg in rep["cuda_graph"]["segments"].values():
            assert seg["capture_failed"] == 0 and seg["replay"] >= 1
        outputs[splits] = v_out

    for splits in (2, 4):
        rel_l2 = float((outputs[splits] - outputs[1]).norm() / outputs[1].norm().clamp_min(1e-6))
        assert rel_l2 < 0.03, f"SM90 sol_attn kv_splits={splits} drifted from kv_splits=1: rel_l2={rel_l2:.4f}"


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("flash_attention_3").usable),
    reason="requires Hopper SM90+ CUDA GPU with flash_attention_3 available",
)
def test_flash_attention_3_direct_inplace_pad_varlen_autograd_and_graph_cuda() -> None:
    torch.manual_seed(113)
    device = torch.device("cuda")

    # 1. Single-doc + trailing pad (in-place out_buf=out[:used]) exact parity vs torch SDPA.
    used, total = 19, 24
    qkv = torch.randn(total, 3 * 4 * 128, device=device, dtype=torch.bfloat16)
    q, k, v = [t.view(total, 4, 128) for t in qkv.split(4 * 128, dim=-1)]
    cu_pad = torch.tensor([0, used, total], dtype=torch.int32, device=device)

    reset_attention_provider_runtime()
    with torch.inference_mode():
        ref_pad = _varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_pad,
            cu_seqlens_host=(0, used, total),
            softmax_scale=128**-0.5,
            backend="torch",
            ignore_trailing_pad=True,
        )
        got_pad = _varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_pad,
            cu_seqlens_host=(0, used, total),
            softmax_scale=128**-0.5,
            backend="flash_attention_3",
            ignore_trailing_pad=True,
        )
    assert torch.allclose(got_pad[:used].float(), ref_pad[:used].float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(got_pad[used:], torch.zeros_like(got_pad[used:]))
    assert attention_provider_runtime_report().get("flash_attention_3", {}).get("successes", 0) >= 1

    # 2. Multi-document packed varlen single-launch FA3 parity vs per-segment torch SDPA.
    bounds = [0, 17, 41, 64]
    qm = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
    km = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
    vm = torch.randn(64, 4, 128, device=device, dtype=torch.bfloat16)
    cu_m = torch.tensor(bounds, dtype=torch.int32, device=device)
    with torch.inference_mode():
        ref_m = _varlen_attention(
            qm,
            km,
            vm,
            cu_seqlens=cu_m,
            cu_seqlens_host=tuple(bounds),
            max_seqlen=None,
            softmax_scale=128**-0.5,
            backend="torch",
        )
        got_m = _varlen_attention(
            qm,
            km,
            vm,
            cu_seqlens=cu_m,
            cu_seqlens_host=tuple(bounds),
            max_seqlen=24,
            softmax_scale=128**-0.5,
            backend="flash_attention_3",
        )
    assert torch.isfinite(got_m).all()
    assert torch.allclose(got_m.float(), ref_m.float(), atol=2e-2, rtol=2e-2)

    # 3. Autograd preservation on both single-doc and multi-doc varlen FA3.
    qg = qm.detach().clone().requires_grad_(True)
    kg = km.detach().clone().requires_grad_(True)
    vg = vm.detach().clone().requires_grad_(True)
    out_g = _varlen_attention(
        qg,
        kg,
        vg,
        cu_seqlens=cu_m,
        cu_seqlens_host=tuple(bounds),
        max_seqlen=24,
        softmax_scale=128**-0.5,
        backend="flash_attention_3",
    )
    assert out_g.requires_grad
    out_g.float().sum().backward()
    assert qg.grad is not None and kg.grad is not None and vg.grad is not None
    assert torch.isfinite(qg.grad).all()

    # 4. Dense edge blocks & dense_initial_steps routing to FA3 alongside sol_attn under piecewise CUDA Graph.
    if attention_backend_capability("sol_attn").usable:
        cfg = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
        model = MiniMaxH3DiTModel(cfg).to(device).eval()
        _zero_adaln(model)
        inputs = _build_packed_inputs(
            cfg, device, text_len=8, audio_len=16, video_len=104, pad_len=4, ignore_trailing_pad=True
        )
        accelerate_minimax_h3_dit(
            model,
            MiniMaxH3AccelerationConfig(
                fuse_projections=True,
                attention_backend="sol_attn",
                attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True},
                dense_attention_backend="flash_attention_3",
                dense_attention_layers=(0, -1),
                dense_initial_steps=1,
                quantization="none",
                cuda_graph="piecewise",
                cuda_graph_warmup=1,
                piecewise_head_blocks=1,
                piecewise_tail_blocks=1,
            ),
        )
        with torch.inference_mode():
            # Warmup + capture both force_dense_attention=True and False.
            for dense_flag in (True, False):
                for _ in range(2):
                    model(**inputs, force_dense_attention=dense_flag)
            model.reset_acceleration_window()
            v_dense, _ = model(**inputs, force_dense_attention=True)
            v_sparse, _ = model(**inputs, force_dense_attention=False)
        assert torch.isfinite(v_dense).all() and torch.isfinite(v_sparse).all()
        rep = model.acceleration_report()
        assert rep["dense_attention_backend"] == "flash_attention_3"
        assert rep["block_attention_backends"] == {"flash_attention_3": 2, "sol_attn": 2}
        # Dense step: 4 FA3 blocks; sparse step: 2 FA3 edge blocks + 2 sol_attn blocks.
        assert rep["attention_provider_runtime_effective"].get("flash_attention_3", 0) == 6
        assert rep["attention_provider_runtime_effective"].get("sol_attn", 0) == 2
        for seg in rep["cuda_graph"]["segments"].values():
            assert seg["capture_failed"] == 0


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
@pytest.mark.parametrize("num_heads", [1, 3, 7, 56])
def test_tiled_qknorm_rope_arbitrary_head_counts_cuda(num_heads: int) -> None:
    torch.manual_seed(127 + num_heads)
    device = torch.device("cuda")
    seq_len, head_dim = 33, 128
    inner_dim = num_heads * head_dim
    qkv = torch.randn(seq_len, 3 * inner_dim, device=device, dtype=torch.bfloat16)
    q_flat, k_flat, _ = qkv.split(inner_dim, dim=-1)
    q = q_flat.view(seq_len, num_heads, head_dim)
    k = k_flat.view(seq_len, num_heads, head_dim)
    cache = _make_rope_cache(seq_len, 96, device)
    positions = torch.arange(seq_len, device=device)
    q_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).to(device)
    k_norm = ops.make_rms_norm(head_dim, eps=1e-5, dtype=torch.bfloat16).to(device)
    with torch.inference_mode():
        got_q, got_k = ops.qk_norm_rope(q, k, q_norm, k_norm, cache, positions)
        ref_q, ref_k = ops.apply_qk_norm(q, k, q_norm, k_norm, head_dim)
        ref_q, ref_k = ops.apply_rope_qk(ref_q, ref_k, cache, positions)
    assert got_q.is_contiguous() and got_k.is_contiguous()
    assert torch.allclose(got_q.float(), ref_q.float(), atol=3.5e-2, rtol=3.5e-2)
    assert torch.allclose(got_k.float(), ref_k.float(), atol=3.5e-2, rtol=3.5e-2)


@pytest.mark.unit
def test_step_cache_config_validation_and_boundary_protection_cpu() -> None:
    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3StepCacheConfig,
        minimax_h3_packed_sequence,
    )

    # Invalid configs fail fast.
    with pytest.raises(ValueError, match="threshold"):
        MiniMaxH3StepCacheConfig(threshold=-0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        MiniMaxH3StepCacheConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="probe_stride"):
        MiniMaxH3StepCacheConfig(probe_stride=0)
    with pytest.raises(ValueError, match="mode"):
        MiniMaxH3StepCacheConfig(mode=cast(Any, "invalid"))
    with pytest.raises(ValueError, match="teacache_coefficients"):
        MiniMaxH3StepCacheConfig(mode="teacache", teacache_coefficients=())

    torch.manual_seed(131)
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)
    packed = minimax_h3_packed_sequence(
        text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=3, include_keyframe_cond=False
    )
    prompt = torch.randn(4, cfg.text_dim, dtype=torch.float32)
    video_init = torch.randn(int(packed["img_pos"].shape[0]), cfg.video_patch_output_dim, dtype=torch.float32)
    audio_init = torch.randn(int(packed["audio_pos"].shape[0]), cfg.audio_latents_dim, dtype=torch.float32)
    sigmas = [1.0 - i / 10.0 for i in range(11)]  # 10 denoising steps

    def _run(
        sc: MiniMaxH3StepCacheConfig | None, *, dense_init: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        model._dense_initial_steps = dense_init
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=prompt,
            token_tags=packed["token_tags"].clone(),
            device=torch.device("cpu"),
        )
        rep: dict[str, Any] = {}
        with torch.no_grad():
            v, a = minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video_init.clone(),
                initial_audio_rows=audio_init.clone(),
                keyframe_cond_rows=None,
                sigmas_video=sigmas,
                sigmas_audio=sigmas,
                device=torch.device("cpu"),
                step_cache=sc,
                step_cache_report=rep,
            )
        return v, a, rep

    ref_v, ref_a, _ = _run(None)

    # threshold=0.0 must execute every step exactly (0 skips, bitwise parity).
    v0, a0, rep0 = _run(MiniMaxH3StepCacheConfig(threshold=0.0, warmup_steps=2, dense_last_steps=2))
    assert rep0["enabled"] is True and rep0["cached_steps"] == 0 and rep0["computed_steps"] == 10
    assert torch.equal(v0, ref_v) and torch.equal(a0, ref_a)

    # EasyCache with generous threshold must skip interior steps while strictly respecting
    # warmup, dense_last, max_consecutive_skips, and dense->sparse transition boundaries.
    block_calls = {"count": 0}
    orig_exec = model._execute_transformer_blocks

    def _counting_exec(*args: Any, **kwargs: Any) -> torch.Tensor:
        block_calls["count"] += 1
        return orig_exec(*args, **kwargs)

    model._execute_transformer_blocks = _counting_exec  # type: ignore[method-assign]
    vc, ac, repc = _run(
        MiniMaxH3StepCacheConfig(
            threshold=10.0,
            warmup_steps=2,
            dense_last_steps=2,
            max_consecutive_skips=2,
            probe_stride=2,
            mode="easycache",
        ),
        dense_init=1,
    )
    decisions = repc["branches"]["positive"]["decisions"]
    assert len(decisions) == 10
    # Warmup steps 0, 1 and final boundary steps 8, 9 must never skip.
    assert decisions[0]["hit"] is False and decisions[0]["reason"] == "warmup"
    assert decisions[1]["hit"] is False and decisions[1]["reason"] == "warmup"
    assert decisions[8]["hit"] is False and decisions[8]["reason"] == "dense_last"
    assert decisions[9]["hit"] is False and decisions[9]["reason"] == "dense_last"
    assert repc["cached_steps"] > 0
    assert block_calls["count"] == repc["computed_steps"]
    assert torch.isfinite(vc).all() and torch.isfinite(ac).all()

    # TeaCache polynomial mode also skips cleanly and produces finite outputs.
    vt, at, rept = _run(
        MiniMaxH3StepCacheConfig(
            threshold=10.0,
            warmup_steps=2,
            dense_last_steps=2,
            max_consecutive_skips=2,
            mode="teacache",
            teacache_coefficients=(0.5, 0.0),
        )
    )
    assert rept["cached_steps"] > 0 and torch.isfinite(vt).all() and torch.isfinite(at).all()


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_step_cache_with_piecewise_cuda_graph_and_pipeline_cuda() -> None:
    from worldfoundry.pipelines.minimax._minimax_h3 import MiniMaxH3StepCacheConfig

    torch.manual_seed(137)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)
    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="torch",
            quantization="none",
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            piecewise_head_blocks=1,
            piecewise_tail_blocks=1,
        ),
    )
    pipe = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
    prompt = torch.randn(6, cfg.text_dim, device=device, dtype=torch.bfloat16)

    # Warmup graphs first.
    _ = pipe.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=11,
        seed=5,
        return_latent_rows=True,
    )
    model.reset_acceleration_window()
    res = pipe.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=11,
        seed=5,
        return_latent_rows=True,
        step_cache=MiniMaxH3StepCacheConfig(
            threshold=5.0,
            warmup_steps=2,
            dense_last_steps=2,
            max_consecutive_skips=2,
            probe_stride=2,
        ),
    )
    assert "step_cache" in res
    sc_rep = res["step_cache"]
    assert sc_rep["total_steps"] == 10 and sc_rep["cached_steps"] >= 3
    cg_rep = model.acceleration_report()["cuda_graph"]
    for seg in cg_rep["segments"].values():
        assert seg["capture_failed"] == 0
        # Replays must equal computed_steps (cached steps skipped the CUDA graph!).
        assert seg["replay"] == sc_rep["computed_steps"]
    assert pipe.acceleration_report()["step_cache"]["cached_steps"] == sc_rep["cached_steps"]


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_cfg_parallel_2gpu_exact_parity_with_mock_process_group_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3CFGParallelConfig,
        MiniMaxH3StepCacheConfig,
    )

    torch.manual_seed(149)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=2, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)

    pos_prompt = torch.randn(6, cfg.text_dim, device=device, dtype=torch.bfloat16)
    neg_prompt = torch.randn(4, cfg.text_dim, device=device, dtype=torch.bfloat16)
    pipe_serial = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")

    # 1. Serial single-GPU CFG reference.
    ref = pipe_serial.generate(
        prompt_embeds=pos_prompt,
        negative_prompt_embeds=neg_prompt,
        cfg_scale=3.5,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=7,
        seed=11,
        return_latent_rows=True,
    )

    # 2. Mock 2-rank process group running Rank 0 (positive) & Rank 1 (negative) concurrently on GPU 4.
    barrier = threading.Barrier(2)
    slots: list[torch.Tensor | None] = [None, None]
    rank_forwards = [0, 0]
    gather_calls = [0, 0]

    def _make_all_gather(rank: int):
        def _gather(out_list: list[torch.Tensor], src: torch.Tensor, _grp: Any) -> None:
            gather_calls[rank] += 1
            slots[rank] = src.detach().clone()
            barrier.wait(timeout=10.0)
            for idx in range(2):
                peer = slots[idx]
                assert peer is not None
                out_list[idx].copy_(peer)
            barrier.wait(timeout=10.0)

        return _gather

    class _CountingModel(nn.Module):
        def __init__(self, inner: MiniMaxH3DiTModel, rank: int) -> None:
            super().__init__()
            self.inner = inner
            self.rank = rank

        def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            rank_forwards[self.rank] += 1
            return self.inner(**kwargs)

    rank_results: list[dict[str, Any] | None] = [None, None]
    errors: list[BaseException] = []

    def _worker(rank: int) -> None:
        try:
            pipe_r = NativeMiniMaxH3Pipeline(transformer=_CountingModel(model, rank), device="cuda:0")
            rank_results[rank] = pipe_r.generate(
                prompt_embeds=pos_prompt,
                negative_prompt_embeds=neg_prompt,
                cfg_scale=3.5,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=7,
                seed=11,
                return_latent_rows=True,
                cfg_parallel=MiniMaxH3CFGParallelConfig(
                    enabled=True,
                    rank=rank,
                    world_size=2,
                    all_gather_fn=_make_all_gather(rank),
                ),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    assert not errors, f"CFG parallel worker failed: {errors[0]!r}"
    assert rank_results[0] is not None and rank_results[1] is not None

    # Each rank must execute ONLY 1 branch per step (6 forwards instead of 12) with 1 collective per step.
    assert rank_forwards == [6, 6]
    assert gather_calls == [6, 6]
    # Both ranks must match each other bitwise and match serial CFG exactly!
    assert torch.equal(rank_results[0]["video_rows"], rank_results[1]["video_rows"])
    assert torch.equal(rank_results[0]["audio_rows"], rank_results[1]["audio_rows"])
    assert torch.allclose(rank_results[0]["video_rows"], ref["video_rows"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(rank_results[0]["audio_rows"], ref["audio_rows"], atol=1e-5, rtol=1e-5)

    # 3. Verify CFG=2 parallel + step_cache compose without deadlock across asymmetric hit/miss decisions.
    barrier2 = threading.Barrier(2)
    slots2: list[torch.Tensor | None] = [None, None]
    sc_results: list[dict[str, Any] | None] = [None, None]
    sc_errors: list[BaseException] = []

    def _sc_worker(rank: int) -> None:
        try:

            def _gather2(out_list: list[torch.Tensor], src: torch.Tensor, _grp: Any) -> None:
                slots2[rank] = src.detach().clone()
                barrier2.wait(timeout=10.0)
                for idx in range(2):
                    peer = slots2[idx]
                    assert peer is not None
                    out_list[idx].copy_(peer)
                barrier2.wait(timeout=10.0)

            pipe_r = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
            # Asymmetric thresholds force Rank 0 to skip while Rank 1 computes on some steps.
            sc_cfg = MiniMaxH3StepCacheConfig(
                threshold=5.0 if rank == 0 else 0.0,
                warmup_steps=2,
                dense_last_steps=1,
                max_consecutive_skips=2,
            )
            sc_results[rank] = pipe_r.generate(
                prompt_embeds=pos_prompt,
                negative_prompt_embeds=neg_prompt,
                cfg_scale=3.5,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=8,
                seed=11,
                return_latent_rows=True,
                step_cache=sc_cfg,
                cfg_parallel=MiniMaxH3CFGParallelConfig(enabled=True, rank=rank, world_size=2, all_gather_fn=_gather2),
            )
        except BaseException as exc:
            sc_errors.append(exc)

    threads2 = [threading.Thread(target=_sc_worker, args=(r,)) for r in (0, 1)]
    for th in threads2:
        th.start()
    for th in threads2:
        th.join(timeout=30.0)
    assert not sc_errors, f"CFG parallel + step_cache failed: {sc_errors[0]!r}"
    assert sc_results[0] is not None and sc_results[1] is not None
    assert sc_results[0]["step_cache"]["cached_steps"] > 0
    assert sc_results[1]["step_cache"]["cached_steps"] == 0
    assert torch.equal(sc_results[0]["video_rows"], sc_results[1]["video_rows"])

    # 4. Verify default torch.distributed resolution path via monkeypatch + error boundaries.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        pipe_serial.generate(
            prompt_embeds=pos_prompt,
            negative_prompt_embeds=neg_prompt,
            cfg_scale=3.5,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=3,
            seed=11,
            return_latent_rows=True,
            cfg_parallel=True,
        )

    # world_size=4 must fail fast; world_size=1 must degrade cleanly to serial CFG.
    with pytest.raises(ValueError, match="cfg_world_size == 2"):
        pipe_serial.generate(
            prompt_embeds=pos_prompt,
            negative_prompt_embeds=neg_prompt,
            cfg_scale=3.5,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=3,
            seed=11,
            return_latent_rows=True,
            cfg_parallel=MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=4),
        )

    ws1 = pipe_serial.generate(
        prompt_embeds=pos_prompt,
        negative_prompt_embeds=neg_prompt,
        cfg_scale=3.5,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=7,
        seed=11,
        return_latent_rows=True,
        cfg_parallel=MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=1),
    )
    assert torch.equal(ws1["video_rows"], ref["video_rows"])

    # cfg_scale=1.0 (no CFG) cleanly ignores cfg_parallel without calling any collective.
    nocfg_called = {"count": 0}

    def _should_not_call(_out: list[torch.Tensor], _src: torch.Tensor, _grp: Any) -> None:
        nocfg_called["count"] += 1

    _ = pipe_serial.generate(
        prompt_embeds=pos_prompt,
        cfg_scale=1.0,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=3,
        seed=11,
        return_latent_rows=True,
        cfg_parallel=MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=2, all_gather_fn=_should_not_call),
    )
    assert nocfg_called["count"] == 0


@pytest.mark.unit
def test_wave2_review_guards_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
        _can_use_direct_flash_attention_3,
    )
    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3CFGParallelConfig,
        MiniMaxH3StepCacheConfig,
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    # 1. Blackwell SM100 must NOT receive SM90-only kv_splits or Hopper-only FA3 dense_backend.
    cfg_sm100 = minimax_h3_sol_profile("sol_balanced", sm_major=10, kv_splits=4)
    assert cfg_sm100.attention_options is not None and "kv_splits" not in cfg_sm100.attention_options
    assert cfg_sm100.dense_attention_backend == "torch"

    # 2. _can_use_direct_flash_attention_3 rejects SM100 even if CUDA device is mocked.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: (10, 0))
    dummy = torch.zeros(4, 2, 128, dtype=torch.bfloat16)
    assert _can_use_direct_flash_attention_3(dummy, dummy, dummy) is False

    # 3. accelerate_minimax_h3_dit honors dense_attention_backend even when attention_backend=None.
    cfg = _tiny_arch(num_layers=2)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)
    model.set_attention_backend("torch", dense_backend="torch")
    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=False,
            attention_backend=None,
            dense_attention_backend="flash_attention_3",
            quantization="none",
            cuda_graph="off",
        ),
    )
    assert model.dense_attention_backend == "flash_attention_3"
    assert all(cast(Any, block).attn.dense_attention_backend == "flash_attention_3" for block in model.blocks)

    # 4. MiniMaxH3CFGParallelConfig field validation (bool rank/world_size, partial rank/world_size).
    with pytest.raises(ValueError, match="rank"):
        MiniMaxH3CFGParallelConfig(enabled=True, rank=cast(Any, True), world_size=2)
    with pytest.raises(ValueError, match="world_size"):
        MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=cast(Any, True))
    with pytest.raises(ValueError, match="specified together"):
        MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=None)

    # 5. cfg_parallel=True with cfg_scale=1.0 must NOT require initialized torch.distributed.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    pipe = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")
    pe = torch.randn(6, cfg.text_dim, dtype=torch.float32)
    res_nocfg = pipe.generate(
        prompt_embeds=pe,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=3,
        seed=1,
        return_latent_rows=True,
        cfg_scale=1.0,
        cfg_parallel=True,
    )
    assert "cfg_parallel" not in res_nocfg
    assert pipe.acceleration_report()["cfg_parallel"] == {
        "enabled": True,
        "active": False,
        "rank": 0,
        "world_size": 1,
        "collectives": 0,
    }

    # 6. Explicit rank/world_size=2 with no all_gather_fn must fail fast BEFORE running step 0.
    fwd_calls = {"count": 0}
    orig_fwd = model.forward

    def _counting_fwd(**kw: Any) -> tuple[torch.Tensor, torch.Tensor]:
        fwd_calls["count"] += 1
        return orig_fwd(**kw)

    model.forward = _counting_fwd  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        pipe.generate(
            prompt_embeds=pe,
            negative_prompt_embeds=pe,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=3,
            seed=1,
            return_latent_rows=True,
            cfg_scale=3.0,
            cfg_parallel=MiniMaxH3CFGParallelConfig(enabled=True, rank=0, world_size=2),
        )
    assert fwd_calls["count"] == 0
    model.forward = orig_fwd  # type: ignore[method-assign]

    # 7. configure_acceleration clears stale step_cache report & validates cfg_parallel type.
    pipe.configure_acceleration(step_cache=MiniMaxH3StepCacheConfig(threshold=5.0, warmup_steps=1, dense_last_steps=1))
    _ = pipe.generate(
        prompt_embeds=pe,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=5,
        seed=1,
        return_latent_rows=True,
    )
    assert pipe.acceleration_report()["step_cache"]["enabled"] is True
    pipe.configure_acceleration(step_cache={"enabled": False})
    assert "step_cache" not in pipe.acceleration_report()
    with pytest.raises(TypeError, match="cfg_parallel"):
        pipe.configure_acceleration(cfg_parallel=cast(Any, "invalid"))

    # 8. fl2va and ref2va step_cache probe_start excludes static text + conditioning/reference rows.
    packed_fl = minimax_h3_packed_sequence(
        text_len=6,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=2,
        include_keyframe_cond=True,
        keyframe_frame_indices=(0, -1),
        frame_count=5,
    )
    kf_rows = torch.randn(int((~packed_fl["update_mask"]).sum()), 96, dtype=torch.float32)
    res_fl = pipe.generate(
        prompt_embeds=pe,
        task="fl2va",
        height=64,
        width=64,
        num_frames=5,
        keyframe_frame_indices=(0, -1),
        keyframe_cond_rows=kf_rows,
        num_inference_steps=6,
        seed=2,
        return_latent_rows=True,
        step_cache=MiniMaxH3StepCacheConfig(threshold=5.0, warmup_steps=1, dense_last_steps=1, probe_stride=1),
    )
    assert res_fl["step_cache"]["branches"]["positive"]["probe_start"] == 6 + int(kf_rows.shape[0])

    ref_blocks = [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 2}]
    packed_ref = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=6, latent_t=2, latent_h=4, latent_w=4, audio_t=2, ref_blocks=ref_blocks
    )
    ref_img_rows = torch.randn(int((~packed_ref["update_mask"]).sum()), 96, dtype=torch.float32)
    ref_aud_rows = torch.randn(int((~packed_ref["audio_update_mask"]).sum()), 32, dtype=torch.float32)
    res_ref = pipe.generate(
        prompt_embeds=pe,
        task="ref2va",
        height=64,
        width=64,
        num_frames=5,
        ref_blocks=ref_blocks,
        keyframe_cond_rows=ref_img_rows,
        audio_ref_rows=ref_aud_rows,
        num_inference_steps=6,
        seed=2,
        return_latent_rows=True,
        step_cache=MiniMaxH3StepCacheConfig(threshold=5.0, warmup_steps=1, dense_last_steps=1, probe_stride=1),
    )
    assert res_ref["step_cache"]["branches"]["positive"]["probe_start"] == (
        6 + int(ref_img_rows.shape[0]) + int(ref_aud_rows.shape[0])
    )

    # 9. NativeMiniMaxH3Pipeline.__call__ accepts negative_prompt and encodes both prompts.
    encoded: list[str] = []

    class _FakeTokenizer:
        def __call__(self, text: str, **_kw: Any) -> dict[str, torch.Tensor]:
            encoded.append(text)
            return {"input_ids": torch.arange(1, 5, dtype=torch.long).unsqueeze(0)}

    class _FakeTextEncoder:
        def encode_ids(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.ones((int(ids.shape[0]), cfg.text_dim), dtype=torch.float32)

    pipe_call = NativeMiniMaxH3Pipeline(
        transformer=model,
        text_encoder=_FakeTextEncoder(),
        tokenizer=_FakeTokenizer(),
        device="cpu",
    )
    out_call = pipe_call(
        "positive prompt",
        negative_prompt="negative prompt",
        cfg_scale=2.5,
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=3,
        seed=1,
        return_latent_rows=True,
    )
    assert encoded == ["positive prompt", "negative prompt"]
    assert torch.isfinite(out_call["video_rows"]).all()


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("flash_attention_3").usable),
    reason="requires Hopper SM90 CUDA GPU with flash_attention_3 available",
)
def test_wave2_fa3_varlen_edge_cases_and_qknorm_strided_cuda() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
        _can_use_direct_flash_attention_3,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.triton_qknorm_rope import (
        fused_qknorm_rope,
    )

    torch.manual_seed(211)
    device = torch.device("cuda")

    # 1. Unsupported head_dim=256 falls back cleanly to exact PyTorch SDPA instead of crashing FA3.
    q256 = torch.randn(32, 2, 256, device=device, dtype=torch.bfloat16)
    k256 = torch.randn(32, 2, 256, device=device, dtype=torch.bfloat16)
    v256 = torch.randn(32, 2, 256, device=device, dtype=torch.bfloat16)
    cu32 = torch.tensor([0, 32], dtype=torch.int32, device=device)
    with torch.inference_mode():
        ref256 = _varlen_attention(
            q256, k256, v256, cu_seqlens=cu32, cu_seqlens_host=(0, 32), softmax_scale=256**-0.5, backend="torch"
        )
        got256 = _varlen_attention(
            q256,
            k256,
            v256,
            cu_seqlens=cu32,
            cu_seqlens_host=(0, 32),
            softmax_scale=256**-0.5,
            backend="flash_attention_3",
        )
    assert _can_use_direct_flash_attention_3(q256, k256, v256) is False
    assert torch.allclose(got256.float(), ref256.float(), atol=2e-2, rtol=2e-2)

    # 2. Uncovered trailing rows in multi-doc varlen must stay zero (no NaN memory reuse leak).
    total = 64
    q = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 16, 40], dtype=torch.int32, device=device)
    with torch.inference_mode():
        ref = _varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu,
            cu_seqlens_host=(0, 16, 40),
            max_seqlen=None,
            softmax_scale=128**-0.5,
            backend="torch",
        )
        for backend_name, host_b in (
            ("flash_attention_3", (0, 16, 40)),
            ("flash_attention_3", None),
            ("torch", (0, 16, 40)),
        ):
            poison = torch.full((total, 4, 128), float("nan"), device=device, dtype=torch.bfloat16)
            del poison
            got = _varlen_attention(
                q,
                k,
                v,
                cu_seqlens=cu,
                cu_seqlens_host=host_b,
                max_seqlen=24,
                softmax_scale=128**-0.5,
                backend=backend_name,
            )
            assert torch.isfinite(got).all()
            assert torch.equal(got[40:], torch.zeros_like(got[40:]))
            assert torch.allclose(got[:40].float(), ref[:40].float(), atol=2e-2, rtol=2e-2)

    # 3. Non-contiguous transposed q/k/v with ignore_trailing_pad=True must not crash FA3 out_buf.
    q_nc = torch.randn(24, 128, 4, device=device, dtype=torch.bfloat16).transpose(1, 2)
    k_nc = torch.randn(24, 128, 4, device=device, dtype=torch.bfloat16).transpose(1, 2)
    v_nc = torch.randn(24, 128, 4, device=device, dtype=torch.bfloat16).transpose(1, 2)
    cu_pad = torch.tensor([0, 19, 24], dtype=torch.int32, device=device)
    with torch.inference_mode():
        ref_nc = _varlen_attention(
            q_nc,
            k_nc,
            v_nc,
            cu_seqlens=cu_pad,
            cu_seqlens_host=(0, 19, 24),
            softmax_scale=128**-0.5,
            backend="torch",
            ignore_trailing_pad=True,
        )
        got_nc = _varlen_attention(
            q_nc,
            k_nc,
            v_nc,
            cu_seqlens=cu_pad,
            cu_seqlens_host=(0, 19, 24),
            softmax_scale=128**-0.5,
            backend="flash_attention_3",
            ignore_trailing_pad=True,
        )
    assert got_nc.is_contiguous()
    assert torch.allclose(got_nc[:19].float(), ref_nc[:19].float(), atol=2e-2, rtol=2e-2)
    assert torch.equal(got_nc[19:], torch.zeros_like(got_nc[19:]))

    # 4. Direct fused_qknorm_rope handles non-unit innermost stride & rejects non-bf16 cleanly.
    cs = _make_rope_cache(24, 96, device)
    w = torch.ones(128, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        gq, gk = fused_qknorm_rope(q_nc, k_nc, w, w, cs)
        rq, rk = fused_qknorm_rope(q_nc.contiguous(), k_nc.contiguous(), w, w, cs)
    assert torch.equal(gq, rq) and torch.equal(gk, rk)
    with pytest.raises(ValueError, match="bf16"):
        fused_qknorm_rope(q_nc.float(), k_nc.float(), w, w, cs)

    # 5. sol_attn kv_splits=None raises clean ValueError (not TypeError).
    if attention_backend_capability("sol_attn").usable:
        q_s = torch.randn(64, 2, 128, device=device, dtype=torch.bfloat16)
        cu_s = torch.tensor([0, 64], dtype=torch.int32, device=device)
        with torch.inference_mode(), pytest.raises(ValueError, match="kv_splits"):
            _varlen_attention(
                q_s,
                q_s,
                q_s,
                cu_seqlens=cu_s,
                cu_seqlens_host=(0, 64),
                softmax_scale=128**-0.5,
                backend="sol_attn",
                backend_options={"tau": 0.5, "kv_splits": None},
            )

    # 6. invalidate_cuda_graphs() + torch.cuda.empty_cache() must release stale graph pools
    # so subsequent piecewise captures (e.g., switching t2va -> fl2va) succeed with 0 failures.
    cfg_g = _tiny_arch(num_layers=4, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model_g = MiniMaxH3DiTModel(cfg_g).to(device).eval()
    _zero_adaln(model_g)
    accelerate_minimax_h3_dit(
        model_g,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="flash_attention_3",
            dense_attention_backend="flash_attention_3",
            quantization="none",
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            piecewise_head_blocks=1,
            piecewise_tail_blocks=1,
        ),
    )
    pipe_g = NativeMiniMaxH3Pipeline(transformer=model_g, device="cuda:0")
    pe_g = torch.randn(6, cfg_g.text_dim, device=device, dtype=torch.bfloat16)
    _ = pipe_g.generate(
        prompt_embeds=pe_g,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=4,
        seed=1,
        return_latent_rows=True,
    )
    model_g.invalidate_cuda_graphs()
    torch.cuda.empty_cache()
    kf_g = torch.randn(4, 96, device=device, dtype=torch.float32)
    _ = pipe_g.generate(
        prompt_embeds=pe_g,
        task="fl2va",
        height=64,
        width=64,
        num_frames=5,
        keyframe_frame_indices=(0,),
        keyframe_cond_rows=kf_g,
        num_inference_steps=4,
        seed=1,
        return_latent_rows=True,
    )
    for seg in model_g.acceleration_report()["cuda_graph"]["segments"].values():
        assert seg["capture_failed"] == 0 and seg["capture"] >= 1


# --------------------------------------------------------------------------- #
# 11. Wave 3: FirstBlockCache (FBCache) & Ulysses Context Parallel (SP=2/4)
# --------------------------------------------------------------------------- #


class _MockSPComm:
    """Thread-synchronized in-process mock of ``all_to_all_single`` and ``all_gather``."""

    def __init__(self, sp_size: int) -> None:
        import threading

        self.sp_size = int(sp_size)
        self._a2a_barrier = threading.Barrier(self.sp_size)
        self._a2a_slots: list[torch.Tensor | None] = [None] * self.sp_size
        self._ag_barrier = threading.Barrier(self.sp_size)
        self._ag_slots: list[torch.Tensor | None] = [None] * self.sp_size

    def make_all_to_all(self, rank: int):
        def _a2a(recv: torch.Tensor, send: torch.Tensor, _grp: Any) -> None:
            # Clone to ensure cross-thread read safety on both CPU and CUDA.
            self._a2a_slots[rank] = send.detach().clone()
            self._a2a_barrier.wait(timeout=15.0)
            for src in range(self.sp_size):
                peer = self._a2a_slots[src]
                assert peer is not None
                recv[src].copy_(peer[rank])
            self._a2a_barrier.wait(timeout=15.0)

        return _a2a

    def make_all_gather(self, rank: int):
        def _ag(out_list: list[torch.Tensor], src: torch.Tensor, _grp: Any) -> None:
            self._ag_slots[rank] = src.detach().clone()
            self._ag_barrier.wait(timeout=15.0)
            for idx in range(self.sp_size):
                peer = self._ag_slots[idx]
                assert peer is not None
                out_list[idx].copy_(peer)
            self._ag_barrier.wait(timeout=15.0)

        return _ag

    def config_for(self, rank: int, *, allow_cuda_graph: bool = False):
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
            MiniMaxH3ContextParallelConfig,
        )

        return MiniMaxH3ContextParallelConfig(
            sp_size=self.sp_size,
            rank=rank,
            all_to_all_fn=self.make_all_to_all(rank),
            all_gather_fn=self.make_all_gather(rank),
            allow_cuda_graph=allow_cuda_graph,
        )


@pytest.mark.unit
def test_firstblock_cache_exact_semantics_guards_and_boundaries_cpu() -> None:
    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3ResidualCache,
        MiniMaxH3StepCacheConfig,
        minimax_h3_packed_sequence,
    )

    torch.manual_seed(307)
    cfg = _tiny_arch(num_layers=4, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)

    packed = minimax_h3_packed_sequence(
        text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=3, include_keyframe_cond=False
    )
    prompt = torch.randn(4, cfg.text_dim, dtype=torch.float32)
    video_init = torch.randn(int(packed["img_pos"].shape[0]), cfg.video_patch_output_dim, dtype=torch.float32)
    audio_init = torch.randn(int(packed["audio_pos"].shape[0]), cfg.audio_latents_dim, dtype=torch.float32)
    sigmas = [1.0 - i / 10.0 for i in range(11)]  # 10 steps

    block_calls = [0] * len(model.blocks)
    orig_forwards = [b.forward for b in model.blocks]
    for idx, blk in enumerate(model.blocks):

        def _make_wrap(i: int, fn: Any):
            def _wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
                block_calls[i] += 1
                return fn(*args, **kwargs)

            return _wrapped

        blk.forward = _make_wrap(idx, orig_forwards[idx])  # type: ignore[method-assign]

    def _run(sc: MiniMaxH3StepCacheConfig | None, *, dense_init: int = 0):
        for i in range(len(block_calls)):
            block_calls[i] = 0
        model._dense_initial_steps = dense_init
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=prompt,
            token_tags=packed["token_tags"].clone(),
            device=torch.device("cpu"),
        )
        rep: dict[str, Any] = {}
        with torch.no_grad():
            v, a = minimax_h3_denoise_loop(
                model=model,
                positive=branch,
                initial_video_rows=video_init.clone(),
                initial_audio_rows=audio_init.clone(),
                keyframe_cond_rows=None,
                sigmas_video=sigmas,
                sigmas_audio=sigmas,
                device=torch.device("cpu"),
                step_cache=sc,
                step_cache_report=rep,
            )
        return v, a, rep, list(block_calls)

    ref_v, ref_a, _, ref_counts = _run(None)
    assert ref_counts == [10, 10, 10, 10]

    # 1. threshold=0.0 executes all blocks every step with bitwise parity vs uncached.
    v0, a0, rep0, counts0 = _run(MiniMaxH3StepCacheConfig(mode="firstblock", threshold=0.0))
    assert rep0["cached_steps"] == 0 and rep0["computed_steps"] == 10
    assert counts0 == [10, 10, 10, 10]
    assert torch.equal(v0, ref_v) and torch.equal(a0, ref_a)

    # 2. Active FBCache: block 0 runs EVERY step; blocks 1..3 run ONLY on cache misses;
    # warmup, dense_last, max_consecutive_skips, and attention_mode_transition are strictly enforced.
    vc, ac, repc, countsc = _run(
        MiniMaxH3StepCacheConfig(
            mode="firstblock",
            threshold=10.0,
            warmup_steps=1,
            dense_last_steps=2,
            max_consecutive_skips=2,
            probe_stride=1,
        ),
        dense_init=1,
    )
    decisions = repc["branches"]["positive"]["decisions"]
    assert len(decisions) == 10
    assert decisions[0]["hit"] is False and decisions[0]["reason"] == "warmup"
    # Step 1 switches from dense_init (force_dense=True) to sparse (False): must NOT skip!
    assert decisions[1]["hit"] is False and decisions[1]["reason"] == "attention_mode_transition"
    # Steps 2 & 3 skip (max_consecutive_skips=2), step 4 must compute!
    assert decisions[2]["hit"] is True and decisions[2]["reason"] == "cache_hit"
    assert decisions[3]["hit"] is True and decisions[3]["reason"] == "cache_hit"
    assert decisions[4]["hit"] is False and decisions[4]["reason"] == "max_consecutive_skips"
    # Final 2 steps are dense_last boundary steps.
    assert decisions[8]["hit"] is False and decisions[8]["reason"] == "dense_last"
    assert decisions[9]["hit"] is False and decisions[9]["reason"] == "dense_last"
    # Block 0 ran all 10 steps; blocks 1..3 skipped every cached step!
    assert countsc[0] == 10
    assert countsc[1] == countsc[2] == countsc[3] == repc["computed_steps"]
    assert repc["cached_steps"] == 4 and repc["computed_steps"] == 6
    assert torch.isfinite(vc).all() and torch.isfinite(ac).all()

    # 3. Direct anchor drift check: anchor stays fixed across consecutive hits (NOT updated on hit).
    fb_cache = MiniMaxH3ResidualCache(
        MiniMaxH3StepCacheConfig(
            mode="firstblock",
            threshold=0.25,
            warmup_steps=0,
            dense_last_steps=0,
            max_consecutive_skips=5,
            probe_stride=1,
        ),
        total_steps=6,
    )
    h1_base = torch.ones(8, 16)
    anchor_probe = torch.ones(8, 16)
    with torch.no_grad():
        # Step 0: seed (miss -> cached residual = +2.0).
        out0 = fb_cache.run(
            step=0,
            hidden_in=h1_base,
            probe=anchor_probe,
            active_tokens=8,
            force_compute=False,
            compute_fn=lambda: h1_base + 2.0,
        )
        assert torch.equal(out0, h1_base + 2.0)
        # Step 1: drift = 0.15 < 0.25 -> HIT (reuses +2.0 residual; anchor stays at 1.0!).
        out1 = fb_cache.run(
            step=1,
            hidden_in=h1_base * 3.0,
            probe=anchor_probe * 1.15,
            active_tokens=8,
            force_compute=False,
            compute_fn=lambda: (_ for _ in ()).throw(AssertionError("should not compute on hit")),
        )
        assert torch.allclose(out1, h1_base * 3.0 + 2.0)
        assert fb_cache.decisions[-1]["hit"] is True
        assert abs(fb_cache.decisions[-1]["step_drift"] - 0.15) < 1e-4
        # Step 2: drift from adjacent step 1 is only 0.15/1.15=0.13, but drift from ANCHOR (1.0) is 0.30 >= 0.25 -> MISS!
        miss_called = {"n": 0}

        def _miss_compute() -> torch.Tensor:
            miss_called["n"] += 1
            return h1_base + 5.0

        out2 = fb_cache.run(
            step=2,
            hidden_in=h1_base,
            probe=anchor_probe * 1.30,
            active_tokens=8,
            force_compute=False,
            compute_fn=_miss_compute,
        )
        assert miss_called["n"] == 1
        assert torch.equal(out2, h1_base + 5.0)
        assert fb_cache.decisions[-1]["hit"] is False and fb_cache.decisions[-1]["reason"] == "threshold"
        assert abs(fb_cache.decisions[-1]["step_drift"] - 0.30) < 1e-4

        # 4. Non-finite probe clears cache safely without throwing or caching NaN.
        bad_probe = anchor_probe.clone()
        bad_probe[0, 0] = float("nan")
        _ = fb_cache.run(
            step=3,
            hidden_in=h1_base,
            probe=bad_probe,
            active_tokens=8,
            force_compute=False,
            compute_fn=lambda: h1_base + 1.0,
        )
        assert fb_cache.decisions[-1]["hit"] is False and fb_cache.decisions[-1]["reason"] == "nonfinite_probe"

    # 5. Autograd mode never caches and propagates gradients through ALL blocks (0..3).
    inputs_g = _build_packed_inputs(cfg, torch.device("cpu"), seed=19)
    inputs_g["prompt_embeds"] = inputs_g["prompt_embeds"].float().requires_grad_(True)  # type: ignore[union-attr]
    inputs_g["x"] = inputs_g["x"].requires_grad_(True)
    fb_grad = MiniMaxH3ResidualCache(
        MiniMaxH3StepCacheConfig(mode="firstblock", threshold=100.0, warmup_steps=0, dense_last_steps=0),
        total_steps=4,
    )
    for i in range(len(block_calls)):
        block_calls[i] = 0
    vg, ag = model(**inputs_g, step_cache=fb_grad, step_index=1)
    assert block_calls == [1, 1, 1, 1]
    assert fb_grad.decisions[-1]["reason"] == "autograd"
    (vg.sum() + ag.sum()).backward()
    assert inputs_g["x"].grad is not None and inputs_g["prompt_embeds"].grad is not None

    # 6. 1-layer model boundary (no blocks[1:] to skip).
    cfg1 = _tiny_arch(num_layers=1, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=256)
    model1 = MiniMaxH3DiTModel(cfg1).float().eval()
    inputs1 = _build_packed_inputs(cfg1, torch.device("cpu"), seed=23)
    inputs1["prompt_embeds"] = inputs1["prompt_embeds"].float()  # type: ignore[union-attr]
    fb1 = MiniMaxH3ResidualCache(
        MiniMaxH3StepCacheConfig(mode="firstblock", threshold=10.0, warmup_steps=0, dense_last_steps=0),
        total_steps=3,
    )
    with torch.no_grad():
        v1_ref, a1_ref = model1(**inputs1)
        v1_0, a1_0 = model1(**inputs1, step_cache=fb1, step_index=0)
        v1_1, a1_1 = model1(**inputs1, step_cache=fb1, step_index=1)
    assert torch.equal(v1_0, v1_ref) and torch.equal(a1_0, a1_ref)
    assert torch.equal(v1_1, v1_ref) and torch.equal(a1_1, a1_ref)
    assert fb1.decisions[1]["hit"] is True


@pytest.mark.unit
@pytest.mark.parametrize("sp_size", [2, 4])
@pytest.mark.parametrize("video_len", [14, 17])  # seq_len=24 (even) and seq_len=27 (uneven SP padding!)
def test_ulysses_context_parallel_sp2_and_sp4_parity_and_sharding_cpu(sp_size: int, video_len: int) -> None:
    import threading

    torch.manual_seed(311 + sp_size * 10 + video_len)
    cfg = _tiny_arch(num_layers=2, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)

    inputs = _build_packed_inputs(
        cfg,
        torch.device("cpu"),
        text_len=4,
        audio_len=6,
        video_len=video_len,
        pad_len=0,
        seed=7,
    )
    inputs["prompt_embeds"] = inputs["prompt_embeds"].float()  # type: ignore[union-attr]
    seq_len = int(inputs["x"].shape[1])
    expected_local = (seq_len + sp_size - 1) // sp_size

    with torch.no_grad():
        ref_v, ref_a = model(**inputs)

    # Instrument linear & attention shapes to prove compute scales by 1/sp_size.
    seen_qkv_rows: list[int] = []
    seen_mlp_rows: list[int] = []
    seen_attn_shapes: list[tuple[int, int, int]] = []
    lock = threading.Lock()
    blk0: Any = model.blocks[0]

    def _qkv_hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        with lock:
            seen_qkv_rows.append(int(inp[0].shape[0]))

    def _mlp_hook(_mod: nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        with lock:
            seen_mlp_rows.append(int(inp[0].shape[0]))

    orig_attn_fn = blk0.attn.attention_fn

    def _spy_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kw: Any) -> torch.Tensor:
        with lock:
            seen_attn_shapes.append((int(q.shape[0]), int(q.shape[1]), int(q.shape[2])))
        return orig_attn_fn(q, k, v, **kw)

    h1 = blk0.attn.qkv_proj.register_forward_hook(_qkv_hook)
    h2 = blk0.mlp.fc1.register_forward_hook(_mlp_hook)
    blk0.attn.attention_fn = _spy_attn

    comm = _MockSPComm(sp_size)
    outputs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * sp_size
    errors: list[BaseException] = []

    def _worker(rank: int) -> None:
        try:
            with torch.no_grad():
                outputs[rank] = model(**inputs, context_parallel=comm.config_for(rank))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(r,)) for r in range(sp_size)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    h1.remove()
    h2.remove()
    blk0.attn.attention_fn = orig_attn_fn

    assert not errors, f"Ulysses CP worker failed: {errors[0]!r}"
    # Linear / FFN rows scaled by 1/sp_size; attention saw FULL seq_len & num_heads/sp_size!
    assert seen_qkv_rows == [expected_local] * sp_size
    assert seen_mlp_rows == [expected_local] * sp_size
    assert seen_attn_shapes == [(seq_len, cfg.num_attention_heads // sp_size, cfg.attention_head_dim)] * sp_size

    for rank in range(sp_size):
        out_r = outputs[rank]
        assert out_r is not None
        got_v, got_a = out_r
        assert torch.allclose(got_v, ref_v, atol=1e-5, rtol=1e-5)
        assert torch.allclose(got_a, ref_a, atol=1e-5, rtol=1e-5)
        if rank > 0:
            out_0 = outputs[0]
            assert out_0 is not None
            assert torch.equal(got_v, out_0[0])
            assert torch.equal(got_a, out_0[1])


@pytest.mark.unit
def test_ulysses_cp_autograd_empty_multidoc_and_pi_h3_cpu() -> None:
    import threading

    torch.manual_seed(317)
    cfg = _tiny_arch(num_layers=2, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=256)
    model_ref = MiniMaxH3DiTModel(cfg).float()
    _zero_adaln(model_ref)
    inputs_base = _build_packed_inputs(cfg, torch.device("cpu"), text_len=4, audio_len=4, video_len=12, seed=13)
    inputs_base["prompt_embeds"] = inputs_base["prompt_embeds"].float()  # type: ignore[union-attr]

    # 1. Differentiable autograd backward parity vs 1-GPU (input & parameter gradients).
    x_ref = inputs_base["x"].detach().clone().requires_grad_(True)
    p_ref = inputs_base["prompt_embeds"].detach().clone().requires_grad_(True)
    in_ref = {**inputs_base, "x": x_ref, "prompt_embeds": p_ref}
    v_ref, a_ref = model_ref(**in_ref)
    (v_ref.square().mean() + a_ref.square().mean()).backward()
    assert x_ref.grad is not None and p_ref.grad is not None
    w_ref_grad = cast(Any, model_ref.blocks[0]).attn.qkv_proj.weight.grad.detach().clone()

    comm = _MockSPComm(2)
    rank_models = [MiniMaxH3DiTModel(cfg).float() for _ in range(2)]
    for m in rank_models:
        m.load_state_dict(model_ref.state_dict())
    x_grads: list[torch.Tensor | None] = [None, None]
    p_grads: list[torch.Tensor | None] = [None, None]
    w_grads: list[torch.Tensor | None] = [None, None]
    errors: list[BaseException] = []

    def _grad_worker(rank: int) -> None:
        try:
            xr = inputs_base["x"].detach().clone().requires_grad_(True)
            pr = inputs_base["prompt_embeds"].detach().clone().requires_grad_(True)
            inr = {**inputs_base, "x": xr, "prompt_embeds": pr}
            vr, ar = rank_models[rank](**inr, context_parallel=comm.config_for(rank))
            (vr.square().mean() + ar.square().mean()).backward()
            assert xr.grad is not None and pr.grad is not None
            x_grads[rank] = xr.grad.detach().clone()
            p_grads[rank] = pr.grad.detach().clone()
            w_grads[rank] = cast(Any, rank_models[rank].blocks[0]).attn.qkv_proj.weight.grad.detach().clone()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_grad_worker, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30.0)
    assert not errors, f"Ulysses autograd worker failed: {errors[0]!r}"
    assert x_grads[0] is not None and p_grads[0] is not None
    assert torch.allclose(x_grads[0], x_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(p_grads[0], p_ref.grad, atol=1e-4, rtol=1e-4)
    # Each SP rank computes parameter grad over its sequence shard; summing across SP ranks matches 1-GPU!
    assert w_grads[0] is not None and w_grads[1] is not None
    assert torch.allclose(w_grads[0] + w_grads[1], w_ref_grad, atol=1e-4, rtol=1e-4)

    # 2. Trailing-pad skip (ignore_trailing_pad=True) & PiH3DiTModel standalone _run_transformer_blocks CP parity.
    pi_model = PiH3DiTModel(cfg, adaln_rank=8, time_table_size=17).float().eval()
    in_pad = _build_packed_inputs(
        cfg, torch.device("cpu"), text_len=4, audio_len=4, video_len=12, pad_len=4, ignore_trailing_pad=True, seed=17
    )
    in_pad["prompt_embeds"] = in_pad["prompt_embeds"].float()  # type: ignore[union-attr]
    with torch.no_grad():
        pi_ref_v, pi_ref_a = pi_model(**in_pad)

    comm_pi = _MockSPComm(2)
    pi_out: list[tuple[torch.Tensor, torch.Tensor] | None] = [None, None]
    pi_err: list[BaseException] = []

    def _pi_worker(rank: int) -> None:
        try:
            with torch.no_grad():
                pi_out[rank] = pi_model(**in_pad, context_parallel=comm_pi.config_for(rank))
        except BaseException as exc:
            pi_err.append(exc)

    threads_pi = [threading.Thread(target=_pi_worker, args=(r,)) for r in (0, 1)]
    for th in threads_pi:
        th.start()
    for th in threads_pi:
        th.join(timeout=30.0)
    assert not pi_err, f"PiH3 CP worker failed: {pi_err[0]!r}"
    assert pi_out[0] is not None and pi_out[1] is not None
    assert torch.allclose(pi_out[0][0], pi_ref_v, atol=1e-4, rtol=1e-4)
    assert torch.allclose(pi_out[0][1], pi_ref_a, atol=1e-4, rtol=1e-4)
    assert torch.equal(pi_out[0][0], pi_out[1][0])

    # 3. Empty sequence (T=0) boundary inside Ulysses attention collectives.
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
        _cp_all_to_all_out,
        _cp_all_to_all_qkv,
        resolve_minimax_h3_context_parallel,
    )

    cp0 = resolve_minimax_h3_context_parallel(_MockSPComm(2).config_for(0), num_heads=4)
    assert cp0 is not None
    empty = torch.empty(0, 4, 32, dtype=torch.float32)
    eq, ek, ev = _cp_all_to_all_qkv(empty, empty, empty, cp0)
    assert eq.shape == (0, 2, 32) and ek.shape == (0, 2, 32) and ev.shape == (0, 2, 32)
    eout = _cp_all_to_all_out(eq, cp0)
    assert eout.shape == (0, 4, 32)
    assert cp0.all_to_all_calls == 0


@pytest.mark.unit
def test_ulysses_cp_with_step_caches_and_2d_cfg2_x_sp2_cpu() -> None:
    import threading

    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3CFGParallelConfig,
        MiniMaxH3StepCacheConfig,
    )

    torch.manual_seed(331)
    cfg = _tiny_arch(num_layers=3, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(model)
    pos_prompt = torch.randn(6, cfg.text_dim, dtype=torch.float32)
    neg_prompt = torch.randn(4, cfg.text_dim, dtype=torch.float32)
    pipe_ref = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")

    # 1. SP=2 with cfg_scale=1.0 (distilled / non-CFG) matches 1-GPU exactly.
    ref_nocfg = pipe_ref.generate(
        prompt_embeds=pos_prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=6,
        seed=5,
        cfg_scale=1.0,
        return_latent_rows=True,
    )
    comm_sp2 = _MockSPComm(2)
    sp2_res: list[dict[str, Any] | None] = [None, None]
    sp2_err: list[BaseException] = []

    def _sp2_worker(rank: int) -> None:
        try:
            p = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")
            sp2_res[rank] = p.generate(
                prompt_embeds=pos_prompt,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=6,
                seed=5,
                cfg_scale=1.0,
                return_latent_rows=True,
                context_parallel=comm_sp2.config_for(rank),
            )
        except BaseException as exc:
            sp2_err.append(exc)

    ths = [threading.Thread(target=_sp2_worker, args=(r,)) for r in (0, 1)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=30.0)
    assert not sp2_err, f"SP=2 pipeline failed: {sp2_err[0]!r}"
    assert sp2_res[0] is not None and sp2_res[1] is not None
    assert torch.equal(sp2_res[0]["video_rows"], sp2_res[1]["video_rows"])
    assert torch.allclose(sp2_res[0]["video_rows"], ref_nocfg["video_rows"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(sp2_res[0]["audio_rows"], ref_nocfg["audio_rows"], atol=1e-4, rtol=1e-4)
    cp_rep = sp2_res[0]["context_parallel"]
    assert cp_rep["active"] is True and cp_rep["sp_size"] == 2 and cp_rep["local_heads"] == 2
    # 5 steps x 3 layers x 4 A2As/layer = 60 A2As; 5 final logit gathers.
    assert cp_rep["all_to_all_calls"] == 5 * 3 * 4
    assert cp_rep["gather_calls"] == 5

    # 2. SP=2 + FirstBlockCache, EasyCache, and TeaCache — including an adversarial fl2va
    # case with large text_len=40 so probe_start > T_local (Rank 0 has 0 active probe tokens!).
    large_prompt = torch.randn(40, cfg.text_dim, dtype=torch.float32)
    for cache_mode in ("firstblock", "easycache", "teacache"):
        sc_cfg = MiniMaxH3StepCacheConfig(
            mode=cast(Any, cache_mode),
            threshold=5.0,
            warmup_steps=2,
            dense_last_steps=1,
            max_consecutive_skips=2,
            probe_stride=3,
        )
        ref_sc = pipe_ref.generate(
            prompt_embeds=large_prompt,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=8,
            seed=9,
            return_latent_rows=True,
            step_cache=sc_cfg,
        )
        comm_c = _MockSPComm(2)
        c_res: list[dict[str, Any] | None] = [None, None]
        c_err: list[BaseException] = []

        def _cache_worker(rank: int, comm_obj=_MockSPComm(2), cfg_obj=sc_cfg) -> None:
            try:
                p = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")
                c_res[rank] = p.generate(
                    prompt_embeds=large_prompt,
                    task="t2va",
                    height=64,
                    width=64,
                    num_frames=5,
                    num_inference_steps=8,
                    seed=9,
                    return_latent_rows=True,
                    step_cache=cfg_obj,
                    context_parallel=comm_obj.config_for(rank),
                )
            except BaseException as exc:
                c_err.append(exc)

        ths_c = [threading.Thread(target=_cache_worker, args=(r, comm_c, sc_cfg)) for r in (0, 1)]
        for th in ths_c:
            th.start()
        for th in ths_c:
            th.join(timeout=30.0)
        assert not c_err, f"SP=2 + {cache_mode} failed: {c_err[0]!r}"
        assert c_res[0] is not None and c_res[1] is not None
        assert c_res[0]["step_cache"]["cached_steps"] == ref_sc["step_cache"]["cached_steps"] > 0
        assert (
            c_res[0]["step_cache"]["branches"]["positive"]["decisions"]
            == c_res[1]["step_cache"]["branches"]["positive"]["decisions"]
        )
        assert torch.allclose(c_res[0]["video_rows"], ref_sc["video_rows"], atol=1e-4, rtol=1e-4)
        # On FirstBlockCache hits, only Block 0 runs A2As (4 per hit vs 12 per miss)!
        if cache_mode == "firstblock":
            hits = c_res[0]["step_cache"]["cached_steps"]
            misses = c_res[0]["step_cache"]["computed_steps"]
            expected_a2a = misses * 3 * 4 + hits * 1 * 4
            assert c_res[0]["context_parallel"]["all_to_all_calls"] == expected_a2a

    # 3. 4-rank 2D hybrid CFG=2 x SP=2 with FirstBlockCache matches serial 1-GPU CFG exactly!
    ref_cfg = pipe_ref.generate(
        prompt_embeds=pos_prompt,
        negative_prompt_embeds=neg_prompt,
        cfg_scale=3.5,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=6,
        seed=11,
        return_latent_rows=True,
    )
    sp_groups = [_MockSPComm(2), _MockSPComm(2)]  # cfg_rank 0 -> {0,1}, cfg_rank 1 -> {2,3}
    cfg_groups = [_MockSPComm(2), _MockSPComm(2)]  # sp_rank 0 -> {0,2}, sp_rank 1 -> {1,3}
    grid_res: list[dict[str, Any] | None] = [None] * 4
    grid_err: list[BaseException] = []

    def _grid_worker(global_rank: int) -> None:
        try:
            cfg_r = global_rank // 2
            sp_r = global_rank % 2
            p = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")
            grid_res[global_rank] = p.generate(
                prompt_embeds=pos_prompt,
                negative_prompt_embeds=neg_prompt,
                cfg_scale=3.5,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=6,
                seed=11,
                return_latent_rows=True,
                cfg_parallel=MiniMaxH3CFGParallelConfig(
                    enabled=True,
                    rank=cfg_r,
                    world_size=2,
                    all_gather_fn=cfg_groups[sp_r].make_all_gather(cfg_r),
                ),
                context_parallel=sp_groups[cfg_r].config_for(sp_r),
            )
        except BaseException as exc:
            grid_err.append(exc)

    ths4 = [threading.Thread(target=_grid_worker, args=(r,)) for r in range(4)]
    for th in ths4:
        th.start()
    for th in ths4:
        th.join(timeout=30.0)
    assert not grid_err, f"2D CFG=2 x SP=2 failed: {grid_err[0]!r}"
    for r in range(4):
        res_r = grid_res[r]
        assert res_r is not None
        assert res_r["cfg_parallel"]["active"] is True
        assert res_r["context_parallel"]["active"] is True
        assert torch.allclose(res_r["video_rows"], ref_cfg["video_rows"], atol=1e-4, rtol=1e-4)
        assert torch.allclose(res_r["audio_rows"], ref_cfg["audio_rows"], atol=1e-4, rtol=1e-4)
        if r > 0:
            res_0 = grid_res[0]
            assert res_0 is not None
            assert torch.equal(res_r["video_rows"], res_0["video_rows"])


@pytest.mark.unit
def test_wave3_cp_and_sol_profile_validation_guards_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
        MiniMaxH3ContextParallelConfig,
    )

    # 1. SOL profiles default to piecewise_head_blocks=1, tail=2 (FBCache head=0:1, body=1:48, tail=48:50).
    for tier in ("sol_high_fidelity", "sol_balanced", "sol_turbo"):
        p1 = minimax_h3_sol_profile(tier, sm_major=9)
        assert p1.piecewise_head_blocks == 1 and p1.piecewise_tail_blocks == 2
        assert p1.context_parallel_size == 1 and p1.cuda_graph == "piecewise"
        p2 = minimax_h3_sol_profile(tier, sm_major=9, context_parallel_size=2)
        assert p2.context_parallel_size == 2 and p2.cuda_graph == "off"
        p2_cg = minimax_h3_sol_profile(tier, sm_major=9, context_parallel_size=2, cuda_graph="piecewise")
        assert p2_cg.context_parallel_size == 2 and p2_cg.cuda_graph == "piecewise"

    with pytest.raises(ValueError, match="context_parallel_size"):
        minimax_h3_sol_profile("sol_balanced", context_parallel_size=cast(Any, True))
    with pytest.raises(ValueError, match="context_parallel_size"):
        minimax_h3_sol_profile("sol_balanced", context_parallel_size=0)

    # 2. MiniMaxH3ContextParallelConfig & accelerate_minimax_h3_dit validate before mutating.
    with pytest.raises(ValueError, match="sp_size"):
        MiniMaxH3ContextParallelConfig(sp_size=0)
    with pytest.raises(ValueError, match="sp_size"):
        MiniMaxH3ContextParallelConfig(sp_size=cast(Any, True))
    with pytest.raises(ValueError, match="rank"):
        MiniMaxH3ContextParallelConfig(sp_size=2, rank=2)

    cfg = _tiny_arch(num_layers=2, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).float().eval()
    with pytest.raises(ValueError, match="divisible by context_parallel_size"):
        accelerate_minimax_h3_dit(
            model,
            MiniMaxH3AccelerationConfig(
                attention_backend="torch",
                context_parallel_size=4,
                cuda_graph="off",
            ),
        )
    assert model.attention_backend is None and model._context_parallel is None

    # 3. Pipeline alias merging & error boundaries.
    pipe = NativeMiniMaxH3Pipeline(transformer=model, device="cpu")
    with pytest.raises(ValueError, match="not True"):
        pipe.configure_acceleration(context_parallel=True)
    with pytest.raises(ValueError, match="conflicting"):
        pipe.configure_acceleration(context_parallel=2, context_parallel_size=4)
    rec = pipe.configure_acceleration(context_parallel_size=2)
    assert rec["context_parallel"]["enabled"] is True and rec["context_parallel"]["sp_size"] == 2
    assert pipe.acceleration_report()["context_parallel"]["sp_size"] == 2
    # Resetting back to 1 disables cleanly.
    rec_off = pipe.configure_acceleration(context_parallel=1)
    assert rec_off["context_parallel"]["enabled"] is False

    # 4. Uninitialized torch.distributed fails fast BEFORE step 0; world_size=2 rejects 2D CFG+SP when cfg>1.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    pe = torch.randn(6, cfg.text_dim, dtype=torch.float32)
    fwd_calls = {"count": 0}
    orig_fwd = model.forward

    def _cnt(**kw: Any):
        fwd_calls["count"] += 1
        return orig_fwd(**kw)

    model.forward = _cnt  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        pipe.generate(
            prompt_embeds=pe,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=3,
            seed=1,
            return_latent_rows=True,
            context_parallel=2,
        )
    assert fwd_calls["count"] == 0

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    with pytest.raises(ValueError, match="world_size == 4"):
        pipe.generate(
            prompt_embeds=pe,
            negative_prompt_embeds=pe,
            cfg_scale=3.0,
            task="t2va",
            height=64,
            width=64,
            num_frames=5,
            num_inference_steps=3,
            seed=1,
            return_latent_rows=True,
            cfg_parallel=True,
            context_parallel=2,
        )
    assert fwd_calls["count"] == 0
    model.forward = orig_fwd  # type: ignore[method-assign]


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_firstblock_cache_with_piecewise_cuda_graph_and_sol_attn_cuda() -> None:
    from worldfoundry.pipelines.minimax._minimax_h3 import MiniMaxH3StepCacheConfig

    torch.manual_seed(337)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=6, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=512)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)

    use_sol = _HOPPER_OR_NEWER and attention_backend_capability("sol_attn").usable
    use_fa3 = _HOPPER_OR_NEWER and attention_backend_capability("flash_attention_3").usable
    dense_be = "flash_attention_3" if use_fa3 else "torch"

    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn" if use_sol else dense_be,
            attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True} if use_sol else None,
            dense_attention_backend=dense_be,
            dense_attention_layers=(0, 1, -1),
            dense_initial_steps=1 if use_sol else 0,
            quantization="none",
            cuda_graph="piecewise",
            cuda_graph_warmup=1,
            piecewise_head_blocks=1,
            piecewise_tail_blocks=2,
        ),
    )
    # Segments must be head=(0,1), body=(1,4), tail=(4,6).
    assert model._piecewise_segment_bounds == {"head": (0, 1), "body": (1, 4), "tail": (4, 6)}

    pipe = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
    prompt = torch.randn(6, cfg.text_dim, device=device, dtype=torch.bfloat16)

    # Warmup & capture all 3 segments first (no cache).
    _ = pipe.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=11,
        seed=5,
        return_latent_rows=True,
    )
    model.reset_acceleration_window()
    res = pipe.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=11,
        seed=5,
        return_latent_rows=True,
        step_cache=MiniMaxH3StepCacheConfig(
            mode="firstblock",
            threshold=5.0,
            warmup_steps=2,
            dense_last_steps=2,
            max_consecutive_skips=2,
            probe_stride=2,
        ),
    )
    assert torch.isfinite(res["video_rows"]).all() and torch.isfinite(res["audio_rows"]).all()
    sc = res["step_cache"]
    assert sc["total_steps"] == 10 and sc["cached_steps"] >= 3
    cg = model.acceleration_report()["cuda_graph"]
    for seg_name, seg in cg["segments"].items():
        assert seg["capture_failed"] == 0, f"segment {seg_name} had capture failures"
    # Head (block 0) replays on EVERY step (0 skips); body & tail skip on every cache hit!
    assert cg["segments"]["head"]["replay"] == 10
    assert cg["segments"]["head"]["request_window"]["cache_skips"] == 0
    assert cg["segments"]["body"]["replay"] == sc["computed_steps"]
    assert cg["segments"]["body"]["request_window"]["cache_skips"] == sc["cached_steps"]
    assert cg["segments"]["tail"]["replay"] == sc["computed_steps"]
    assert cg["segments"]["tail"]["request_window"]["cache_skips"] == sc["cached_steps"]


@pytest.mark.skipif(not _HOPPER_OR_NEWER, reason="requires SM90+ Hopper/Blackwell CUDA GPU")
def test_ulysses_cp2_with_fp8_sol_attn_and_firstblock_cuda() -> None:
    import threading

    from worldfoundry.pipelines.minimax._minimax_h3 import MiniMaxH3StepCacheConfig

    torch.manual_seed(347)
    device = torch.device("cuda")
    cfg = _tiny_arch(num_layers=4, hidden_size=1024, num_heads=8, head_dim=128, ffn_hidden_size=2048)
    model = MiniMaxH3DiTModel(cfg).to(device).eval()
    _zero_adaln(model)

    use_sol = attention_backend_capability("sol_attn").usable
    use_fa3 = attention_backend_capability("flash_attention_3").usable
    dense_be = "flash_attention_3" if use_fa3 else "torch"

    accelerate_minimax_h3_dit(
        model,
        MiniMaxH3AccelerationConfig(
            fuse_projections=True,
            attention_backend="sol_attn" if use_sol else dense_be,
            attention_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True} if use_sol else None,
            dense_attention_backend=dense_be,
            dense_attention_layers=(0, -1),
            dense_initial_steps=1 if use_sol else 0,
            quantization="fp8",
            fp8_scaling="rowwise",
            fp8_use_fast_accum=False,
            fp8_exclude_edge_blocks=True,
            fuse_fp8_ffn=True,
            cuda_graph="off",
        ),
    )

    pipe_ref = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
    prompt = torch.randn(8, cfg.text_dim, device=device, dtype=torch.bfloat16)

    # 1. Exact parity vs 1-GPU (FP8 exact accum + Sol-Attn + FA3) without step cache.
    ref = pipe_ref.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=5,
        seed=3,
        return_latent_rows=True,
    )
    # Pre-compile Sol-Attn's CuTe MLIR kernel for local_heads=4 on the main thread
    # (in real torchrun each rank is a separate OS process, whereas this mock test
    # shares one MLIRContext across two Python threads).
    if use_sol:
        from worldfoundry.core.attention.backends.dispatch import attention_forward

        _w = torch.randn(
            1, 32, cfg.num_attention_heads // 2, cfg.attention_head_dim, device=device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            _ = attention_forward(
                _w,
                _w,
                _w,
                q_pattern="b s n d",
                k_pattern="b s n d",
                v_pattern="b s n d",
                out_pattern="b s n d",
                scale=cfg.attention_head_dim**-0.5,
                backend="sol_attn",
                backend_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True},
            )
    comm1 = _MockSPComm(2)
    out1: list[dict[str, Any] | None] = [None, None]
    err1: list[BaseException] = []

    def _w1(rank: int) -> None:
        try:
            p = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
            out1[rank] = p.generate(
                prompt_embeds=prompt,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=5,
                seed=3,
                return_latent_rows=True,
                context_parallel=comm1.config_for(rank),
            )
        except BaseException as exc:
            err1.append(exc)

    ths1 = [threading.Thread(target=_w1, args=(r,)) for r in (0, 1)]
    for th in ths1:
        th.start()
    for th in ths1:
        th.join(timeout=60.0)
    assert not err1, f"CUDA SP=2 FP8+SolAttn failed: {err1[0]!r}"
    assert out1[0] is not None and out1[1] is not None
    assert torch.equal(out1[0]["video_rows"], out1[1]["video_rows"])
    cos_v = float(
        torch.nn.functional.cosine_similarity(
            out1[0]["video_rows"].flatten().float(), ref["video_rows"].flatten().float(), dim=0
        ).item()
    )
    assert cos_v > 0.9995, f"SP=2 FP8+SolAttn drifted from 1-GPU: cos={cos_v:.6f}"
    assert out1[0]["context_parallel"]["local_heads"] == 4

    # 2. Compose SP=2 + FP8 + Sol-Attn + FirstBlockCache with zero deadlocks & exact decision parity.
    fb_cfg = MiniMaxH3StepCacheConfig(
        mode="firstblock",
        threshold=5.0,
        warmup_steps=2,
        dense_last_steps=1,
        max_consecutive_skips=2,
        probe_stride=2,
    )
    ref_fb = pipe_ref.generate(
        prompt_embeds=prompt,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=8,
        seed=3,
        return_latent_rows=True,
        step_cache=fb_cfg,
    )
    comm2 = _MockSPComm(2)
    out2: list[dict[str, Any] | None] = [None, None]
    err2: list[BaseException] = []

    def _w2(rank: int) -> None:
        try:
            p = NativeMiniMaxH3Pipeline(transformer=model, device="cuda:0")
            out2[rank] = p.generate(
                prompt_embeds=prompt,
                task="t2va",
                height=64,
                width=64,
                num_frames=5,
                num_inference_steps=8,
                seed=3,
                return_latent_rows=True,
                step_cache=fb_cfg,
                context_parallel=comm2.config_for(rank),
            )
        except BaseException as exc:
            err2.append(exc)

    ths2 = [threading.Thread(target=_w2, args=(r,)) for r in (0, 1)]
    for th in ths2:
        th.start()
    for th in ths2:
        th.join(timeout=60.0)
    assert not err2, f"CUDA SP=2 + FirstBlockCache failed: {err2[0]!r}"
    assert out2[0] is not None and out2[1] is not None
    assert out2[0]["step_cache"]["cached_steps"] == ref_fb["step_cache"]["cached_steps"] > 0
    assert (
        out2[0]["step_cache"]["branches"]["positive"]["decisions"]
        == out2[1]["step_cache"]["branches"]["positive"]["decisions"]
    )
    cos_fb = float(
        torch.nn.functional.cosine_similarity(
            out2[0]["video_rows"].flatten().float(), ref_fb["video_rows"].flatten().float(), dim=0
        ).item()
    )
    assert cos_fb > 0.999, f"SP=2 + FirstBlockCache drifted from 1-GPU: cos={cos_fb:.6f}"
    assert torch.isfinite(out2[0]["video_rows"]).all()


@pytest.mark.unit
def test_wave3_review_regression_guards_and_edge_cases_cpu() -> None:
    import threading

    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
        _cp_gather_dim0_impl,
        resolve_minimax_h3_context_parallel,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.variants.hyperflow.model import (
        HyperFlowDiTModel,
    )
    from worldfoundry.pipelines.minimax._minimax_h3 import (
        MiniMaxH3ResidualCache,
        MiniMaxH3StepCacheConfig,
    )

    torch.manual_seed(353)
    cfg = _tiny_arch(num_layers=2, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=256)

    # 1. Explicit cuda_graph='piecewise' + context_parallel_size=2 must enable allow_cuda_graph on CP config.
    m1 = MiniMaxH3DiTModel(cfg).float().eval()
    rec_cg = accelerate_minimax_h3_dit(
        m1,
        minimax_h3_sol_profile("sol_balanced", sm_major=9, context_parallel_size=2, cuda_graph="piecewise"),
    )
    assert m1._context_parallel is not None and m1._context_parallel.allow_cuda_graph is True
    assert rec_cg["context_parallel"]["allow_cuda_graph"] is True
    assert m1._piecewise_graph is not None
    m1_default = MiniMaxH3DiTModel(cfg).float().eval()
    rec_off = accelerate_minimax_h3_dit(
        m1_default,
        minimax_h3_sol_profile("sol_balanced", sm_major=9, context_parallel_size=2),
    )
    assert m1_default._context_parallel is not None and m1_default._context_parallel.allow_cuda_graph is False
    assert rec_off["context_parallel"]["allow_cuda_graph"] is False

    # 2. Pipeline configure_acceleration keeps _context_parallel_config synced with acceleration profiles.
    m2 = MiniMaxH3DiTModel(cfg).float().eval()
    pipe = NativeMiniMaxH3Pipeline(transformer=m2, device="cpu")
    rec_p2 = pipe.configure_acceleration(
        acceleration=minimax_h3_sol_profile("exact", sm_major=0, context_parallel_size=2, cuda_graph="off")
    )
    assert rec_p2["context_parallel"]["sp_size"] == 2
    assert pipe.acceleration_report()["context_parallel"]["sp_size"] == 2
    # Resetting back to SP=1 via acceleration profile clears stale pipeline CP state.
    pipe.configure_acceleration(
        acceleration=minimax_h3_sol_profile("exact", sm_major=0, context_parallel_size=1, cuda_graph="off")
    )
    assert pipe._context_parallel_config is None and m2._context_parallel is None
    pe = torch.randn(4, cfg.text_dim, dtype=torch.float32)
    out_single = pipe.generate(
        prompt_embeds=pe,
        task="t2va",
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=2,
        seed=0,
        return_latent_rows=True,
    )
    assert torch.isfinite(out_single["video_rows"]).all()
    # Conflicting CP kwargs fail BEFORE mutating transformer.
    m2_guard = MiniMaxH3DiTModel(cfg).float().eval()
    pipe_guard = NativeMiniMaxH3Pipeline(transformer=m2_guard, device="cpu")
    with pytest.raises(ValueError, match="conflicting"):
        pipe_guard.configure_acceleration(
            acceleration=minimax_h3_sol_profile("exact", sm_major=0, cuda_graph="off"),
            context_parallel=2,
            context_parallel_size=4,
        )
    assert m2_guard.attention_backend is None

    # 3. Direct forward with ignore_trailing_pad=True and omitted cu_seqlens_q_host matches explicit host bounds.
    m_fb = MiniMaxH3DiTModel(cfg).float().eval()
    _zero_adaln(m_fb)
    in_pad = _build_packed_inputs(
        cfg, torch.device("cpu"), text_len=4, audio_len=4, video_len=10, pad_len=5, ignore_trailing_pad=True, seed=29
    )
    in_pad["prompt_embeds"] = in_pad["prompt_embeds"].float()  # type: ignore[union-attr]
    in_pad_no_host = {
        **in_pad,
        "packed_seq_params": {
            "cu_seqlens_q": in_pad["packed_seq_params"]["cu_seqlens_q"],
            "ignore_trailing_pad": True,
        },
    }
    c_host = MiniMaxH3ResidualCache(
        MiniMaxH3StepCacheConfig(mode="firstblock", threshold=0.05, warmup_steps=0, dense_last_steps=0, probe_stride=1),
        total_steps=4,
    )
    c_no_host = MiniMaxH3ResidualCache(
        MiniMaxH3StepCacheConfig(mode="firstblock", threshold=0.05, warmup_steps=0, dense_last_steps=0, probe_stride=1),
        total_steps=4,
    )
    with torch.no_grad():
        m_fb(**in_pad, step_cache=c_host, step_index=0)
        m_fb(**{**in_pad, "x": in_pad["x"] * 1.1}, step_cache=c_host, step_index=1)
        m_fb(**in_pad_no_host, step_cache=c_no_host, step_index=0)
        m_fb(**{**in_pad_no_host, "x": in_pad["x"] * 1.1}, step_cache=c_no_host, step_index=1)
    assert c_host.decisions[1]["step_drift"] == c_no_host.decisions[1]["step_drift"]

    # 4. HyperFlowDiTModel(degree > 1) rejects Ulysses context_parallel > 1 before mutation.
    hf = HyperFlowDiTModel(cfg, degree=2, rank=0).float().eval()
    with pytest.raises(ValueError, match="double-shard"):
        hf.set_context_parallel(2)
    with pytest.raises(ValueError, match="double-shard"):
        accelerate_minimax_h3_dit(
            hf, minimax_h3_sol_profile("exact", sm_major=0, context_parallel_size=2, cuda_graph="off")
        )
    assert hf.attention_backend is None and hf._context_parallel is None

    # 5. Uneven seq_len (pad_len > 0) autograd backward & multi-doc SP=4 parity.
    m_grad_ref = MiniMaxH3DiTModel(cfg).float()
    _zero_adaln(m_grad_ref)
    in_uneven = _build_packed_inputs(
        cfg, torch.device("cpu"), text_len=4, audio_len=4, video_len=15, pad_len=0, seed=31
    )  # seq_len = 23 -> pad_len = 1 under SP=4
    in_uneven["prompt_embeds"] = in_uneven["prompt_embeds"].float()  # type: ignore[union-attr]
    xr_ref = in_uneven["x"].detach().clone().requires_grad_(True)
    pr_ref = in_uneven["prompt_embeds"].detach().clone().requires_grad_(True)
    vr, ar = m_grad_ref(**{**in_uneven, "x": xr_ref, "prompt_embeds": pr_ref})
    (vr.square().mean() + ar.square().mean()).backward()
    assert xr_ref.grad is not None and pr_ref.grad is not None
    w_qkv_ref = cast(Any, m_grad_ref.blocks[0]).attn.qkv_proj.weight.grad.detach().clone()

    comm4 = _MockSPComm(4)
    rank_models = [MiniMaxH3DiTModel(cfg).float() for _ in range(4)]
    for rm in rank_models:
        rm.load_state_dict(m_grad_ref.state_dict())
    xg: list[torch.Tensor | None] = [None] * 4
    pg: list[torch.Tensor | None] = [None] * 4
    wg: list[torch.Tensor | None] = [None] * 4
    errs4: list[BaseException] = []

    def _w4_grad(rank: int) -> None:
        try:
            xr = in_uneven["x"].detach().clone().requires_grad_(True)
            pr = in_uneven["prompt_embeds"].detach().clone().requires_grad_(True)
            v, a = rank_models[rank](
                **{**in_uneven, "x": xr, "prompt_embeds": pr}, context_parallel=comm4.config_for(rank)
            )
            (v.square().mean() + a.square().mean()).backward()
            assert xr.grad is not None and pr.grad is not None
            xg[rank] = xr.grad.detach().clone()
            pg[rank] = pr.grad.detach().clone()
            wg[rank] = cast(Any, rank_models[rank].blocks[0]).attn.qkv_proj.weight.grad.detach().clone()
        except BaseException as exc:
            errs4.append(exc)

    ths4 = [threading.Thread(target=_w4_grad, args=(r,)) for r in range(4)]
    for th in ths4:
        th.start()
    for th in ths4:
        th.join(timeout=30.0)
    assert not errs4, f"Uneven SP=4 autograd failed: {errs4[0]!r}"
    assert xg[0] is not None and pg[0] is not None and all(g is not None for g in wg)
    assert torch.allclose(xg[0], xr_ref.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(pg[0], pr_ref.grad, atol=1e-4, rtol=1e-4)
    w_sum = torch.stack(cast(list[torch.Tensor], wg), dim=0).sum(dim=0)
    assert torch.allclose(w_sum, w_qkv_ref, atol=1e-4, rtol=1e-4)

    # 6. Rank-1-only non-finite probe in SP=2 + FirstBlockCache synchronizes cleanly on BOTH ranks.
    comm_nf = _MockSPComm(2)
    nf_decs: list[list[dict[str, Any]] | None] = [None, None]
    nf_errs: list[BaseException] = []

    def _w_nf(rank: int) -> None:
        try:
            cp_ctx = resolve_minimax_h3_context_parallel(comm_nf.config_for(rank), num_heads=4)
            assert cp_ctx is not None
            cache = MiniMaxH3ResidualCache(
                MiniMaxH3StepCacheConfig(
                    mode="firstblock",
                    threshold=1.0,
                    warmup_steps=0,
                    dense_last_steps=0,
                    max_consecutive_skips=4,
                    probe_stride=1,
                ),
                total_steps=4,
            )
            h = torch.ones(4, 8)
            with torch.no_grad():
                cache.run(
                    step=0,
                    hidden_in=h,
                    probe=torch.ones(4, 8),
                    active_tokens=8,
                    force_compute=False,
                    compute_fn=lambda: h + 1.0,
                    sp_ctx=cp_ctx,
                    total_tokens=8,
                    row_start=rank * 4,
                    row_stop=(rank + 1) * 4,
                )
                p1 = torch.ones(4, 8)
                if rank == 1:
                    p1[0, 0] = float("nan")
                cache.run(
                    step=1,
                    hidden_in=h,
                    probe=p1,
                    active_tokens=8,
                    force_compute=False,
                    compute_fn=lambda: h + 2.0,
                    sp_ctx=cp_ctx,
                    total_tokens=8,
                    row_start=rank * 4,
                    row_stop=(rank + 1) * 4,
                )
                nf_decs[rank] = list(cache.decisions)
        except BaseException as exc:
            nf_errs.append(exc)

    ths_nf = [threading.Thread(target=_w_nf, args=(r,)) for r in (0, 1)]
    for th in ths_nf:
        th.start()
    for th in ths_nf:
        th.join(timeout=15.0)
    assert not nf_errs, f"Non-finite SP=2 worker failed: {nf_errs[0]!r}"
    assert nf_decs[0] == nf_decs[1] and nf_decs[0] is not None
    assert nf_decs[0][1]["hit"] is False and nf_decs[0][1]["reason"] == "nonfinite_probe"

    # 7. FL2VA (2 keyframes: first + last) under SP=2 + FirstBlockCache matches 1-GPU exactly.
    pe_fl = torch.randn(10, cfg.text_dim, dtype=torch.float32)
    kf2 = torch.randn(8, 96, dtype=torch.float32)
    sc_fb = MiniMaxH3StepCacheConfig(
        mode="firstblock", threshold=5.0, warmup_steps=1, dense_last_steps=1, max_consecutive_skips=2, probe_stride=2
    )
    ref_fl = NativeMiniMaxH3Pipeline(transformer=m_fb, device="cpu").generate(
        prompt_embeds=pe_fl,
        task="fl2va",
        height=64,
        width=64,
        num_frames=9,
        keyframe_frame_indices=(0, -1),
        keyframe_cond_rows=kf2,
        num_inference_steps=6,
        seed=5,
        return_latent_rows=True,
        step_cache=sc_fb,
    )
    comm_fl = _MockSPComm(2)
    fl_res: list[dict[str, Any] | None] = [None, None]
    fl_errs: list[BaseException] = []

    def _w_fl(rank: int) -> None:
        try:
            p = NativeMiniMaxH3Pipeline(transformer=m_fb, device="cpu")
            fl_res[rank] = p.generate(
                prompt_embeds=pe_fl,
                task="fl2va",
                height=64,
                width=64,
                num_frames=9,
                keyframe_frame_indices=(0, -1),
                keyframe_cond_rows=kf2,
                num_inference_steps=6,
                seed=5,
                return_latent_rows=True,
                step_cache=sc_fb,
                context_parallel=comm_fl.config_for(rank),
            )
        except BaseException as exc:
            fl_errs.append(exc)

    ths_fl = [threading.Thread(target=_w_fl, args=(r,)) for r in (0, 1)]
    for th in ths_fl:
        th.start()
    for th in ths_fl:
        th.join(timeout=30.0)
    assert not fl_errs, f"FL2VA SP=2 + FBCache failed: {fl_errs[0]!r}"
    assert fl_res[0] is not None and fl_res[1] is not None
    assert torch.equal(fl_res[0]["video_rows"], fl_res[1]["video_rows"])
    assert torch.allclose(fl_res[0]["video_rows"], ref_fl["video_rows"], atol=1e-4, rtol=1e-4)
    assert fl_res[0]["step_cache"]["cached_steps"] == ref_fl["step_cache"]["cached_steps"] > 0

    # 8. Empty sequence gather boundary skips collectives cleanly.
    cp0 = resolve_minimax_h3_context_parallel(_MockSPComm(2).config_for(0), num_heads=4)
    assert cp0 is not None
    gathered_empty = _cp_gather_dim0_impl(torch.empty(0, 16), cp0, orig_seq_len=0)
    assert gathered_empty.shape == (0, 16) and cp0.gather_calls == 0


@pytest.mark.unit
def test_wave3_2d_topology_group_caching_and_cfg1_fallback_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import worldfoundry.core.distributed.sequence_parallel.runtime as sp_rt
    from worldfoundry.pipelines.minimax._minimax_h3 import denoise_loop

    cfg = _tiny_arch(num_layers=2, hidden_size=512, num_heads=4, head_dim=128, ffn_hidden_size=256)
    model = MiniMaxH3DiTModel(cfg).float().eval()

    class _FakeGroup:
        def __init__(self, ranks: list[int]) -> None:
            self.ranks = list(ranks)

    fake_world = _FakeGroup([0, 1, 2, 3])
    created: list[list[int]] = []
    fake_sp_slot: dict[str, Any] = {"sp": None, "cfg": None}

    monkeypatch.setattr(denoise_loop, "_MINIMAX_H3_2D_GROUPS", {})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=fake_world))
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: len(fake_world.ranks) if group is None else len(group.ranks),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda group=None: 0 if group is None else group.ranks.index(0),
    )

    def _fake_init_sp(sp_size: int) -> None:
        g = _FakeGroup(list(range(sp_size)))
        created.append(g.ranks)
        fake_sp_slot["sp"] = g

    def _fake_init_cfg(sp_size: int, cfg_size: int) -> None:
        g = _FakeGroup([i * sp_size for i in range(cfg_size)])
        created.append(g.ranks)
        fake_sp_slot["cfg"] = g

    monkeypatch.setattr(sp_rt, "get_sequence_parallel_group", lambda: fake_sp_slot["sp"])
    monkeypatch.setattr(sp_rt, "get_cfg_parallel_group", lambda: fake_sp_slot["cfg"])
    monkeypatch.setattr(sp_rt, "initialize_sequence_parallel_group", _fake_init_sp)
    monkeypatch.setattr(sp_rt, "initialize_cfg_parallel_group", _fake_init_cfg)

    # 1. Fresh 4-rank process with cfg_parallel=True, cp=2, use_cfg=False (cfg_scale=1.0) initializes 2D SP group!
    _, cfg_active_1, _, _, _, _, _, resolved_cp_1, _ = denoise_loop._resolve_parallel_topology(
        cfg_parallel=True,
        context_parallel=2,
        model=model,
        use_cfg=False,
    )
    assert cfg_active_1 is False
    assert resolved_cp_1 is not None and resolved_cp_1.sp_size == 2
    assert len(created) == 2  # one SP init + one CFG init

    # 2. Subsequent 2D CFG=2 x SP=2 (use_cfg=True) reuses cached groups with ZERO new group creations!
    _, cfg_active_2, _, _, cfg_grp_2, _, _, resolved_cp_2, _ = denoise_loop._resolve_parallel_topology(
        cfg_parallel=True,
        context_parallel=2,
        model=model,
        use_cfg=True,
    )
    assert cfg_active_2 is True and cfg_grp_2 is not None
    assert resolved_cp_2 is not None and resolved_cp_2.process_group is resolved_cp_1.process_group
    assert len(created) == 2

    # 3. Even if an external caller overwrites global SP group to size 4, 2D CFG=2 x SP=2 STILL reuses cache!
    fake_sp_slot["sp"] = _FakeGroup([0, 1, 2, 3])
    _, cfg_active_3, _, _, cfg_grp_3, _, _, resolved_cp_3, _ = denoise_loop._resolve_parallel_topology(
        cfg_parallel=True,
        context_parallel=2,
        model=model,
        use_cfg=True,
    )
    assert cfg_active_3 is True and cfg_grp_3 is cfg_grp_2
    assert resolved_cp_3 is not None and resolved_cp_3.process_group is resolved_cp_1.process_group
    assert len(created) == 2

    # 4. On world_size=2 with cfg_parallel=True, cp=2, use_cfg=False (cfg_scale=1.0), uses WORLD cleanly.
    fake_world_2 = _FakeGroup([0, 1])
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=fake_world_2))
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: len(fake_world_2.ranks) if group is None else len(group.ranks),
    )
    fake_sp_slot["sp"] = None
    fake_sp_slot["cfg"] = None
    _, cfg_active_4, _, _, _, _, _, resolved_cp_4, _ = denoise_loop._resolve_parallel_topology(
        cfg_parallel=True,
        context_parallel=2,
        model=model,
        use_cfg=False,
    )
    assert cfg_active_4 is False and resolved_cp_4 is not None and resolved_cp_4.sp_size == 2


@pytest.mark.skipif(
    not (_HOPPER_OR_NEWER and attention_backend_capability("sol_attn").usable),
    reason="requires Hopper SM90+ CUDA GPU with sol_attn available",
)
def test_ulysses_cp_uneven_sol_attn_kv_splits_cuda() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import _varlen_attention

    device = torch.device("cuda")
    # Uneven orig_len (not divisible by 4 or 64) with trailing pad + kv_splits=4 + prefix_sink_tokens.
    for orig_len, used in ((131, 125), (4100, 4097)):
        q = torch.randn(orig_len, 14, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(orig_len, 14, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn(orig_len, 14, 128, device=device, dtype=torch.bfloat16)
        cu = torch.tensor([0, used, orig_len], dtype=torch.int32, device=device)
        with torch.inference_mode():
            out = _varlen_attention(
                q,
                k,
                v,
                cu_seqlens=cu,
                cu_seqlens_host=(0, used, orig_len),
                softmax_scale=128**-0.5,
                backend="sol_attn",
                backend_options={"tau": 0.5, "thresh_type": "exact", "compress_kv": True, "kv_splits": 4},
                ignore_trailing_pad=True,
                prefix_sink_tokens=16,
            )
        assert out.shape == q.shape
        assert torch.isfinite(out).all()
        assert torch.equal(out[used:], torch.zeros_like(out[used:]))
