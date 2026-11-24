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
