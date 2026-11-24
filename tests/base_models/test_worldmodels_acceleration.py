"""Unit and CUDA parity tests for HY-WorldPlay, HunyuanVideo, Matrix-Game-2.0, and LongLive accelerations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from worldfoundry.base_models.diffusion_model.models.denoisers.hunyuan_video import (
    HunyuanVideo15Denoiser,
    HunyuanVideoDenoiser,
    HunyuanVideoI2VDenoiser,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.action_camera import (
    ARHunyuanVideo_1_5_DiffusionTransformer,
    _apply_fast_interleaved_rope,
    _apply_prope_mat,
    _precompute_prope_mats,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_attention import (
    _FP8_ATTN_SCALES,
    prepare_static_kv_arena,
    reset_fp8_attention_scales,
    sequence_parallel_attention_vision,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_modulation import (
    apply_gate,
    modulate,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.i2v.model import (
    HYVideoDiffusionTransformer,
)
from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original import HunyuanVideoDiT
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm.prope import prope_qkv
from worldfoundry.core.attention.rotary.rope_nd import (
    apply_nd_rotary_embedding as apply_rotary_emb,
)
from worldfoundry.core.attention.rotary.rope_nd import (
    get_nd_rotary_pos_embed,
)
from worldfoundry.core.nn.transformer.contracts import AttentionBackendConfigurable
from worldfoundry.pipelines.longlive.pipeline_longlive import (
    NativeLongLivePipeline,
    _resolve_longlive_layout,
    load_interactive_prompt_jsonl,
    merge_longlive_lora,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.scheduler import (
    FlowMatchScheduler,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _init_zero_params(module: nn.Module, *, std: float = 0.02) -> None:
    """Populate zero-initialized modulation/final-layer weights so parity checks are non-vacuous."""
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point() and float(param.abs().max()) == 0.0:
                nn.init.normal_(param, std=std)


def _tiny_hy15_config(*, attn_mode: str = "torch", single_blocks: int = 1) -> dict[str, Any]:
    return {
        "patch_size": [1, 1, 1],
        "in_channels": 4,
        "concat_condition": False,
        "out_channels": 4,
        "hidden_size": 64,
        "heads_num": 2,
        "mlp_width_ratio": 2.0,
        "mlp_act_type": "gelu_tanh",
        "mm_double_blocks_depth": 2,
        "mm_single_blocks_depth": single_blocks,
        "rope_dim_list": [8, 12, 12],
        "qkv_bias": True,
        "qk_norm": True,
        "qk_norm_type": "rms",
        "guidance_embed": False,
        "text_projection": "single_refiner",
        "use_attention_mask": True,
        "text_states_dim": 32,
        "text_states_dim_2": 32,
        "rope_theta": 256,
        "attn_mode": attn_mode,
        "glyph_byT5_v2": False,
        "use_meanflow": False,
    }


def _tiny_worldplay_config(*, attn_mode: str = "torch") -> dict[str, Any]:
    cfg = _tiny_hy15_config(attn_mode=attn_mode, single_blocks=0)
    cfg["mm_single_blocks_depth"] = 0
    cfg["in_channels"] = 8
    cfg["out_channels"] = 4
    cfg["concat_condition"] = False
    return cfg


def _tiny_i2v_model() -> HYVideoDiffusionTransformer:
    args = SimpleNamespace(
        i2v_condition_type="latent_concat",
        text_states_dim=32,
        text_states_dim_2=32,
        gradient_checkpoint=False,
        gradient_checkpoint_layers=0,
    )
    return HYVideoDiffusionTransformer(
        args,
        patch_size=[1, 1, 1],
        in_channels=4,
        out_channels=4,
        hidden_size=64,
        heads_num=2,
        mlp_width_ratio=2.0,
        mm_double_blocks_depth=1,
        mm_single_blocks_depth=1,
        rope_dim_list=[8, 12, 12],
    )


def test_hunyuan_video_family_attention_backend_configurable() -> None:
    models: list[nn.Module] = [
        _tiny_i2v_model(),
        HunyuanVideo_1_5_DiffusionTransformer(**_tiny_hy15_config()),
        ARHunyuanVideo_1_5_DiffusionTransformer(**_tiny_worldplay_config()),
    ]
    if torch.cuda.is_available():
        models.insert(
            0,
            HunyuanVideoDiT(
                in_channels=4,
                hidden_size=192,
                text_dim=32,
                num_double_blocks=1,
                num_single_blocks=1,
            ),
        )

    for model in models:
        model_any: Any = model
        assert isinstance(model, AttentionBackendConfigurable)
        model.set_attention_backend("torch")
        assert model.attention_backend == "torch"
        for blk in model_any.double_blocks:
            assert blk.attention_backend == "torch"
        if hasattr(model, "single_blocks"):
            for blk in model_any.single_blocks:
                assert blk.attention_backend == "torch"
        if torch.cuda.is_available():
            model_cuda: Any = model.to("cuda")
            model_cuda.set_attention_backend("sol_attn")
            assert model_cuda.attention_backend == "sol_attn"
            for blk in model_cuda.double_blocks:
                assert blk.attention_backend == "sol_attn"
            if hasattr(model_cuda, "single_blocks"):
                for blk in model_cuda.single_blocks:
                    assert blk.attention_backend == "sol_attn"
        model.set_attention_backend(None)
        assert model.attention_backend == "torch"


def test_hunyuan_video_denoisers_accept_piecewise_graph_mode() -> None:
    i2v = _tiny_i2v_model()
    h15 = HunyuanVideo_1_5_DiffusionTransformer(**_tiny_hy15_config())
    denoisers: list[Any] = [
        HunyuanVideoI2VDenoiser(i2v, enable_cuda_graph="piecewise"),
        HunyuanVideo15Denoiser(h15, image_to_video=False, enable_cuda_graph="piecewise"),
    ]
    if torch.cuda.is_available():
        orig = HunyuanVideoDiT(
            in_channels=4,
            hidden_size=192,
            text_dim=32,
            num_double_blocks=1,
            num_single_blocks=1,
        )
        denoisers.insert(0, HunyuanVideoDenoiser(orig, enable_cuda_graph="piecewise"))

    for denoiser in denoisers:
        assert denoiser._piecewise_graph is not None
        report = denoiser.graph_report()
        assert report is not None
        assert report["mode"] == "piecewise-block-graph"


def test_hy15_fuse_qkv_exact_parity() -> None:
    torch.manual_seed(0)
    model = HunyuanVideo_1_5_DiffusionTransformer(**_tiny_hy15_config()).eval()
    _init_zero_params(model)
    x = torch.randn(1, 4, 2, 2, 2)
    t = torch.tensor([500.0])
    txt = torch.randn(1, 6, 32)
    mask = torch.ones(1, 6, dtype=torch.long)

    with torch.no_grad():
        ref = model(
            hidden_states=x,
            timestep=t,
            text_states=txt,
            text_states_2=None,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        assert float(ref.abs().max()) > 0.01
        fused_count = model.fuse_qkv()
        assert fused_count == 5  # 2 double blocks * 2 + 1 single block * 1
        assert model.fuse_qkv() == 0  # idempotent
        cand = model(
            hidden_states=x,
            timestep=t,
            text_states=txt,
            text_states_2=None,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]

    assert torch.allclose(ref, cand, atol=1e-5, rtol=1e-5)


def test_camera_modulation_compact_broadcast_parity() -> None:
    torch.manual_seed(1)
    bs, frames, spatial, dim = 2, 4, 15, 32
    x = torch.randn(bs, frames * spatial, dim)
    shift_compact = torch.randn(bs * frames, dim)
    scale_compact = torch.randn(bs * frames, dim)
    gate_compact = torch.randn(bs * frames, dim)

    shift_rep = (
        shift_compact.reshape(bs, frames, 1, dim).expand(bs, frames, spatial, dim).reshape(bs * frames * spatial, dim)
    )
    scale_rep = (
        scale_compact.reshape(bs, frames, 1, dim).expand(bs, frames, spatial, dim).reshape(bs * frames * spatial, dim)
    )
    gate_rep = (
        gate_compact.reshape(bs, frames, 1, dim).expand(bs, frames, spatial, dim).reshape(bs * frames * spatial, dim)
    )

    assert torch.equal(
        modulate(x, shift=shift_compact, scale=scale_compact),
        modulate(x, shift=shift_rep, scale=scale_rep),
    )
    assert torch.equal(apply_gate(x, gate=gate_compact), apply_gate(x, gate=gate_rep))


@requires_cuda
def test_worldplay_triton_prope_and_rope_exact_parity() -> None:
    device = torch.device("cuda")
    torch.manual_seed(7)
    bs, frames, spatial, heads, head_dim = 1, 4, 16, 4, 32
    seq_len = frames * spatial

    q = torch.randn(bs, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(bs, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(bs, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)

    # Non-identity SE(3) view matrices and camera intrinsics
    angles = torch.tensor([0.1, -0.25, 0.4, -0.15], device=device)
    viewmats = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(bs, frames, 1, 1)
    for i, a in enumerate(angles):
        c, s = torch.cos(a), torch.sin(a)
        viewmats[0, i, 0, 0] = c
        viewmats[0, i, 0, 2] = s
        viewmats[0, i, 2, 0] = -s
        viewmats[0, i, 2, 2] = c
        viewmats[0, i, 0, 3] = 0.2 * i
        viewmats[0, i, 2, 3] = -0.3 * i
    ks = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(bs, frames, 1, 1)
    ks[..., 0, 0] = 1.2
    ks[..., 1, 1] = 1.15
    ks[..., 0, 2] = 0.5
    ks[..., 1, 2] = 0.5

    # Exact FP32 parity against reference prope_qkv
    q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
    ref_q32, ref_k32, ref_v32, ref_apply_o32 = prope_qkv(
        q_f32.permute(0, 2, 1, 3), k_f32.permute(0, 2, 1, 3), v_f32.permute(0, 2, 1, 3), viewmats=viewmats, Ks=ks
    )
    p_mat, p_inv_t, p_t = _precompute_prope_mats(viewmats, ks)
    assert torch.allclose(_apply_prope_mat(q_f32, p_mat), ref_q32.permute(0, 2, 1, 3), atol=1e-5, rtol=1e-5)
    assert torch.allclose(_apply_prope_mat(k_f32, p_inv_t), ref_k32.permute(0, 2, 1, 3), atol=1e-5, rtol=1e-5)
    assert torch.allclose(_apply_prope_mat(v_f32, p_inv_t), ref_v32.permute(0, 2, 1, 3), atol=1e-5, rtol=1e-5)

    attn_out_f32 = torch.randn(bs, seq_len, heads * head_dim, device=device, dtype=torch.float32)
    ref_o32 = (
        ref_apply_o32(attn_out_f32.view(bs, seq_len, heads, head_dim).permute(0, 2, 1, 3))
        .permute(0, 2, 1, 3)
        .reshape_as(attn_out_f32)
    )
    assert torch.allclose(_apply_prope_mat(attn_out_f32, p_t), ref_o32, atol=1e-5, rtol=1e-5)

    # BF16 Triton kernel parity (1-ULP rounding vs BF16 einsum)
    fast_q = _apply_prope_mat(q, p_mat)
    assert torch.allclose(fast_q, ref_q32.permute(0, 2, 1, 3).to(torch.bfloat16), atol=2e-2, rtol=2e-2)

    # Interleaved real RoPE parity vs reference apply_rotary_emb
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        [8, 12, 12], [frames, 4, 4], theta=256, use_real=True, theta_rescale_factor=1
    )
    freqs_cis = (freqs_cos.to(device=device), freqs_sin.to(device=device))
    ref_rq, ref_rk = apply_rotary_emb(q, k, freqs_cis, head_first=False)
    fast_rq, fast_rk = _apply_fast_interleaved_rope(q, k, freqs_cis)
    assert torch.allclose(fast_rq, ref_rq, atol=1e-2, rtol=1e-2)
    assert torch.allclose(fast_rk, ref_rk, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_worldplay_static_kv_arena_and_cudnn_fp8_stream_split() -> None:
    device = torch.device("cuda")
    torch.manual_seed(3)
    bs, heads, head_dim = 1, 2, 64
    txt_len, vis_len, chunk_len = 8, 12, 16
    kv_cache = [
        {
            "k_txt": torch.randn(bs, heads, txt_len, head_dim, device=device, dtype=torch.bfloat16),
            "v_txt": torch.randn(bs, heads, txt_len, head_dim, device=device, dtype=torch.bfloat16),
            "k_vision": torch.randn(bs, heads, vis_len, head_dim, device=device, dtype=torch.bfloat16),
            "v_vision": torch.randn(bs, heads, vis_len, head_dim, device=device, dtype=torch.bfloat16),
        }
        for _ in range(3)
    ]
    q = torch.randn(bs, chunk_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(bs, chunk_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(bs, chunk_len, heads, head_dim, device=device, dtype=torch.bfloat16)

    arena: dict[Any, Any] = {}
    buffers, prefix_len, _ = prepare_static_kv_arena(kv_cache, chunk_len, arena=arena, dtype=torch.bfloat16)
    assert prefix_len == txt_len + vis_len

    ref_out, _ = sequence_parallel_attention_vision(q, k, v, block_idx=1, kv_cache=kv_cache, cache_vision=True)
    arena_out, _ = sequence_parallel_attention_vision(
        q,
        k,
        v,
        block_idx=1,
        kv_cache=None,
        cache_vision=False,
        static_kv_buffer=buffers[1],
        static_prefix_len=prefix_len,
    )
    assert torch.allclose(ref_out, arena_out, atol=1e-3, rtol=1e-3)

    # Verify cudnn_fp8 separates rope vs prope scale keys and preserves edge blocks in BF16
    reset_fp8_attention_scales()
    out_edge, _ = sequence_parallel_attention_vision(
        q,
        k,
        v,
        block_idx=0,
        kv_cache=None,
        cache_vision=False,
        attn_mode="cudnn_fp8",
        static_kv_buffer=buffers[0],
        static_prefix_len=prefix_len,
        stream_tag="rope",
        num_blocks=3,
    )
    assert len(_FP8_ATTN_SCALES) == 0  # block 0 is edge-protected in BF16
    assert torch.isfinite(out_edge).all()

    out_mid_rope, _ = sequence_parallel_attention_vision(
        q,
        k,
        v,
        block_idx=1,
        kv_cache=None,
        cache_vision=False,
        attn_mode="cudnn_fp8",
        static_kv_buffer=buffers[1],
        static_prefix_len=prefix_len,
        stream_tag="rope",
        num_blocks=3,
    )
    out_mid_prope, _ = sequence_parallel_attention_vision(
        q * 2.5,
        k * 2.5,
        v * 2.5,
        block_idx=1,
        kv_cache=None,
        cache_vision=False,
        attn_mode="cudnn_fp8",
        static_kv_buffer=buffers[1],
        static_prefix_len=prefix_len,
        stream_tag="prope",
        num_blocks=3,
    )
    assert (q.device, ("vision", "rope", 1)) in _FP8_ATTN_SCALES
    assert (q.device, ("vision", "prope", 1)) in _FP8_ATTN_SCALES
    assert _FP8_ATTN_SCALES[(q.device, ("vision", "rope", 1))] != _FP8_ATTN_SCALES[(q.device, ("vision", "prope", 1))]
    assert torch.isfinite(out_mid_rope).all() and torch.isfinite(out_mid_prope).all()


@requires_cuda
def test_worldplay_vision_cuda_graph_and_bi_forward() -> None:
    device = torch.device("cuda")
    torch.manual_seed(11)
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**_tiny_worldplay_config())
    model.add_prope_parameters()
    model.add_discrete_action_parameters()
    _init_zero_params(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model.fuse_qkv()

    bs, c_in, t_chunk, h, w = 1, 8, 2, 4, 4
    hidden = torch.randn(bs, c_in, t_chunk, h, w, device=device, dtype=torch.bfloat16)
    timestep = torch.full((bs, t_chunk), 500.0, device=device, dtype=torch.bfloat16)
    action = torch.zeros((bs, t_chunk), device=device, dtype=torch.long)
    viewmats = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(bs, t_chunk, 1, 1)
    ks = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(bs, t_chunk, 1, 1)
    txt = torch.randn(bs, 6, 32, device=device, dtype=torch.bfloat16)
    mask = torch.ones(bs, 6, device=device, dtype=torch.long)
    t_txt = torch.zeros((bs,), device=device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kv_txt = model.forward_txt(timestep_txt=t_txt, text_states=txt, encoder_attention_mask=mask, cache_txt=True)
        eager_out, _ = model.forward_vision(
            hidden_states=hidden,
            timestep=timestep,
            action=action,
            viewmats=viewmats,
            Ks=ks,
            kv_cache=kv_txt,
            cache_vision=False,
            rope_temporal_size=2,
            start_rope_start_idx=0,
        )

        model.enable_cuda_graph(warmup=1)
        for _ in range(3):
            graph_out, _ = model.forward_vision(
                hidden_states=hidden,
                timestep=timestep,
                action=action,
                viewmats=viewmats,
                Ks=ks,
                kv_cache=kv_txt,
                cache_vision=False,
                rope_temporal_size=2,
                start_rope_start_idx=0,
            )

        report = model.graph_report()
        assert report is not None
        assert report["enabled"] is True
        assert report["capture"] >= 1
        assert report["replay"] >= 1
        assert report["capture_failed"] == 0
        assert float(eager_out.abs().max()) > 0.01
        assert torch.allclose(eager_out, graph_out, atol=2e-2, rtol=2e-2)

        # Bidirectional path smoke + finiteness + non-vacuous output
        bi_out, _ = model.forward_bi(
            hidden_states=hidden,
            timestep=timestep.reshape(-1),
            timestep_txt=t_txt,
            text_states=txt,
            text_states_2=None,
            encoder_attention_mask=mask,
            viewmats=viewmats,
            Ks=ks,
            action=action,
        )
        assert bi_out.shape == (bs, 4, t_chunk, h, w)
        assert torch.isfinite(bi_out).all()
        assert float(bi_out.abs().max()) > 0.01


def test_matrix_game_scheduler_sigma_and_add_noise_parity() -> None:
    sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(1000, training=True)
    sample = torch.randn(1, 16, 3, 4, 4)
    noise = torch.randn_like(sample)
    t = torch.tensor([500.0])

    ref = sched.add_noise(sample, noise, t)
    sigma = sched.sigma_for_timestep(t, device=sample.device)
    fast = sched.add_noise_with_sigma(sample, noise, sigma)
    assert torch.allclose(ref, fast, atol=1e-6, rtol=1e-6)


@requires_cuda
def test_matrix_game_vae_wrapper_steady_cuda_graph_parity() -> None:
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
        VAEDecoderWrapper,
    )

    class _DummyCausalDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(16, 3, kernel_size=1)

        def forward(
            self, x: torch.Tensor, feat_cache: list[torch.Tensor | None]
        ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
            prev = feat_cache[0]
            if prev is None:
                out = self.conv(x)
                next_cache: list[torch.Tensor | None] = [x[:, :, -1:].clone()]
            else:
                out = self.conv(x + 0.1 * prev)
                next_cache = [x[:, :, -1:].clone()]
            return out, next_cache

    device = torch.device("cuda")
    torch.manual_seed(5)
    wrapper: Any = VAEDecoderWrapper()
    wrapper.conv2 = nn.Conv3d(16, 16, kernel_size=1)
    wrapper.decoder = _DummyCausalDecoder()
    wrapper = wrapper.to(device=device, dtype=torch.bfloat16).eval()

    x1 = torch.randn(1, 3, 16, 4, 4, device=device, dtype=torch.bfloat16)
    x2 = torch.randn(1, 3, 16, 4, 4, device=device, dtype=torch.bfloat16)
    x3 = torch.randn(1, 3, 16, 4, 4, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        e1, c_e = wrapper(x1, None)
        e2, c_e = wrapper(x2, *c_e)
        e3, c_e = wrapper(x3, *c_e)

        wrapper.enable_cuda_graph(warmup=1)
        g1, c_g = wrapper(x1, None)  # chunk 0 (eager, populates cache)
        g2, c_g = wrapper(x2, *c_g)  # capture + replay
        g3, c_g = wrapper(x3, *c_g)  # replay
        g4, _ = wrapper(x3, *c_g)  # replay

    assert torch.allclose(e1, g1, atol=1e-2, rtol=1e-2)
    assert torch.allclose(e2, g2, atol=1e-2, rtol=1e-2)
    assert torch.allclose(e3, g3, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(g4).all()
    report = wrapper.graph_report()
    assert report is not None
    assert report["enabled"] is True
    assert report["capture"] >= 1
    assert report["replay"] >= 2
    assert report["capture_failed"] == 0


def test_longlive_lora_merge_exactness_and_guards(tmp_path: Path) -> None:
    torch.manual_seed(17)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q = nn.Linear(16, 16, bias=True)
            self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(approximate="tanh"), nn.Linear(32, 16))

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block()])

    model = _TinyModel().to(dtype=torch.bfloat16)
    blk0: Any = model.blocks[0]
    orig_q = blk0.self_attn.q.weight.detach().clone().float()
    orig_f0 = blk0.ffn[0].weight.detach().clone().float()

    rank = 4
    a_q = torch.randn(rank, 16, dtype=torch.float32)
    b_q = torch.randn(16, rank, dtype=torch.float32)
    a_f0 = torch.randn(rank, 16, dtype=torch.float32)
    b_f0 = torch.randn(32, rank, dtype=torch.float32)

    lora_file = tmp_path / "lora.pt"
    torch.save(
        {
            "generator_lora": {
                "base_model.model.blocks.0.self_attn.q.lora_A.weight": a_q,
                "base_model.model.blocks.0.self_attn.q.lora_B.weight": b_q,
                "base_model.model.blocks.0.ffn.0.lora_A.weight": a_f0,
                "base_model.model.blocks.0.ffn.0.lora_B.weight": b_f0,
            }
        },
        lora_file,
    )

    report = merge_longlive_lora(model, lora_file, alpha=8.0)  # scale = 8 / 4 = 2.0
    assert report["applied"] is True
    assert report["merged_layers"] == 2
    assert report["ranks"] == [4]
    assert report["scale"] == 2.0

    expected_q = (orig_q + 2.0 * (b_q @ a_q)).to(torch.bfloat16)
    expected_f0 = (orig_f0 + 2.0 * (b_f0 @ a_f0)).to(torch.bfloat16)
    assert torch.equal(blk0.self_attn.q.weight, expected_q)
    assert torch.equal(blk0.ffn[0].weight, expected_f0)

    # Guard: non-Linear target (e.g. already fused/quantized) must raise TypeError
    blk0.self_attn.q = nn.Identity()
    with pytest.raises(TypeError, match="expected nn.Linear"):
        merge_longlive_lora(model, lora_file)

    # Guard: missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        merge_longlive_lora(model, tmp_path / "missing.pt")


def test_longlive_layout_discovery_and_jsonl_loader(tmp_path: Path) -> None:
    ll_dir = tmp_path / "LongLive-1.3B"
    (ll_dir / "models").mkdir(parents=True)
    (ll_dir / "models" / "longlive_base.pt").write_bytes(b"base")
    (ll_dir / "models" / "lora.pt").write_bytes(b"lora")

    wan_dir = tmp_path / "Wan2.1-T2V-1.3B"
    (wan_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (wan_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"t5")
    (wan_dir / "Wan2.1_VAE.pth").write_bytes(b"vae")

    model_p, comps, lora_p = _resolve_longlive_layout(
        str(ll_dir), None, wan_base_path=None, lora_path=None, apply_lora=True
    )
    assert model_p == str(ll_dir / "models" / "longlive_base.pt")
    assert lora_p == str(ll_dir / "models" / "lora.pt")
    assert comps["text-encoder"] == str(wan_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    assert comps["tokenizer"] == str(wan_dir / "google" / "umt5-xxl")
    assert comps["vae"] == str(wan_dir / "Wan2.1_VAE.pth")

    _, _, no_lora = _resolve_longlive_layout(str(ll_dir), None, wan_base_path=None, lora_path=None, apply_lora=False)
    assert no_lora is None

    jsonl = tmp_path / "interactive.jsonl"
    jsonl.write_text(
        json.dumps({"prompts": ["Scene A", "Scene B"]}) + "\n" + json.dumps({"prompt": "Scene C"}) + "\n",
        encoding="utf-8",
    )
    assert load_interactive_prompt_jsonl(jsonl) == ["Scene A", "Scene B", "Scene C"]


def test_longlive_inplace_crossattn_refresh_preserves_data_ptrs() -> None:
    torch.manual_seed(23)

    class _Cross(nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.norm_k = nn.Identity()

    class _Block(nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.cross_attn = _Cross(dim, heads)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_len = 8
            self.text_embedding = nn.Linear(16, 32, bias=False)
            self.blocks = nn.ModuleList([_Block(32, 4), _Block(32, 4)])

    model = _Model().eval()
    pipe = object.__new__(NativeLongLivePipeline)
    ctx_a = torch.randn(1, 8, 16)
    ctx_b = torch.randn(1, 8, 16)
    cross_cache: list[dict[str, Any]] = [
        {
            "is_init": True,
            "k": torch.zeros(1, 8, 4, 8),
            "v": torch.zeros(1, 8, 4, 8),
        }
        for _ in range(2)
    ]
    ptrs_before = [(c["k"].data_ptr(), c["v"].data_ptr()) for c in cross_cache]

    preserved_a = pipe._refresh_crossattn_cache_inplace(model, ctx_a, cross_cache)
    assert preserved_a is True
    assert [(c["k"].data_ptr(), c["v"].data_ptr()) for c in cross_cache] == ptrs_before
    snap_a = cross_cache[0]["k"].clone()

    preserved_b = pipe._refresh_crossattn_cache_inplace(model, ctx_b, cross_cache)
    assert preserved_b is True
    assert [(c["k"].data_ptr(), c["v"].data_ptr()) for c in cross_cache] == ptrs_before
    assert not torch.equal(snap_a, cross_cache[0]["k"])

    # Exact parity against direct projection
    emb_b = model.text_embedding(ctx_b)
    blk0_any: Any = model.blocks[0]
    expected_k = blk0_any.cross_attn.k(emb_b).view(1, 8, 4, 8)
    expected_v = blk0_any.cross_attn.v(emb_b).view(1, 8, 4, 8)
    assert torch.allclose(cross_cache[0]["k"], expected_k)
    assert torch.allclose(cross_cache[0]["v"], expected_v)


def test_longlive_switch_prompt_error_guards() -> None:
    pipe: Any = object.__new__(NativeLongLivePipeline)
    pipe._realtime_stream = None
    with pytest.raises(RuntimeError, match="not configured"):
        pipe.switch_prompt_realtime("next scene")

    pipe._realtime_stream = SimpleNamespace(_state=SimpleNamespace())
    with pytest.raises(ValueError, match="non-empty prompt"):
        pipe.switch_prompt_realtime("   ")


def test_hy15_denoiser_builder_fuses_qkv_via_post_load_hook_before_quantization(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.denoisers import hunyuan_video as hv_mod

    captured_specs: list[Any] = []

    class _FakeLoader:
        def load(self, spec: Any, checkpoint: Any, policy: Any) -> Any:
            del checkpoint, policy
            captured_specs.append(spec)
            model = HunyuanVideo_1_5_DiffusionTransformer(**_tiny_hy15_config()).eval()
            if spec.post_load_hook is not None:
                spec.post_load_hook(model)
            return model

    monkeypatch.setattr(hv_mod, "NativeModuleLoader", _FakeLoader)
    ctx: Any = SimpleNamespace(
        component_options={"fuse_qkv": True, "image_to_video": False},
        policy=SimpleNamespace(attention=hv_mod.AttentionBackend.AUTO, options={}),
        require_checkpoint=lambda role: role,
    )
    denoiser = hv_mod.build_hunyuan_video15_denoiser(ctx)
    assert len(captured_specs) == 1
    assert captured_specs[0].post_load_hook is not None
    # Confirm QKV projections were fused inside post_load_hook (before quantization would run)
    assert hasattr(denoiser.model.double_blocks[0], "img_attn_qkv")
    assert hasattr(denoiser.model.single_blocks[0], "linear1_qkv")


def test_hunyuan_video_denoiser_cuda_graph_arg_normalization() -> None:
    i2v = _tiny_i2v_model()
    h15 = HunyuanVideo_1_5_DiffusionTransformer(**_tiny_hy15_config())

    d_i2v = HunyuanVideoI2VDenoiser(i2v, enable_cuda_graph="off")
    d_h15 = HunyuanVideo15Denoiser(h15, image_to_video=False, enable_cuda_graph="off")
    assert d_i2v.graph_report() is None
    assert d_i2v._piecewise_graph is None
    assert d_h15.graph_report() is None
    assert d_h15._piecewise_graph is None

    with pytest.raises(ValueError, match="unsupported enable_cuda_graph"):
        HunyuanVideo15Denoiser(h15, image_to_video=False, enable_cuda_graph="bogus")
    with pytest.raises(ValueError, match="unsupported enable_cuda_graph"):
        HunyuanVideoI2VDenoiser(i2v, enable_cuda_graph="bogus")
    if torch.cuda.is_available():
        orig = HunyuanVideoDiT(
            in_channels=4,
            hidden_size=192,
            text_dim=32,
            num_double_blocks=1,
            num_single_blocks=1,
        )
        d_orig = HunyuanVideoDenoiser(orig, enable_cuda_graph="off")
        assert d_orig.graph_report() is None
        assert d_orig._piecewise_graph is None
        with pytest.raises(ValueError, match="unsupported enable_cuda_graph"):
            HunyuanVideoDenoiser(orig, enable_cuda_graph="bogus")


@requires_cuda
def test_hunyuan_video_piecewise_graphs_dynamic_txt_len_zero_capture_failures() -> None:
    from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(31)

    # 1. HunyuanVideoDiT (original)
    orig = (
        HunyuanVideoDiT(
            in_channels=16,
            hidden_size=3072,
            text_dim=32,
            num_double_blocks=1,
            num_single_blocks=1,
            guidance_embed=False,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    _init_zero_params(orig)
    d_orig_eager = HunyuanVideoDenoiser(orig, enable_cuda_graph="off")
    d_orig_pw = HunyuanVideoDenoiser(orig, enable_cuda_graph="piecewise")
    x_orig = torch.randn(1, 16, 2, 4, 4, device=device, dtype=dtype)
    t_orig = torch.tensor([500.0], device=device, dtype=dtype)
    pooled_orig = torch.randn(1, 768, device=device, dtype=dtype)
    with torch.no_grad():
        for txt_len in (6, 10):
            txt = torch.randn(1, txt_len, 32, device=device, dtype=dtype)
            mask = torch.ones(1, txt_len, device=device, dtype=torch.long)
            mask[0, -2:] = 0
            inp = DenoiserInput(
                latents=x_orig,
                timestep=t_orig,
                next_timestep=cast(Any, None),
                step_index=0,
                total_steps=1,
                conditioning={"text_states": txt, "text_states_2": pooled_orig, "text_mask": mask},
            )
            ref = d_orig_eager(inp).sample
            assert float(ref.abs().max()) > 0.01
            out = None
            for _ in range(4):
                out = d_orig_pw(inp).sample
            assert out is not None and torch.allclose(ref, out, atol=2e-2, rtol=2e-2)
    rep_orig = d_orig_pw.graph_report()
    assert rep_orig is not None
    assert rep_orig["capture"] >= 4 and rep_orig["replay"] >= 4 and rep_orig["capture_failed"] == 0

    # 2. HYVideoDiffusionTransformer (I2V)
    i2v = _tiny_i2v_model().to(device=device, dtype=dtype).eval()
    _init_zero_params(i2v)
    d_i2v_eager = HunyuanVideoI2VDenoiser(i2v, enable_cuda_graph="off")
    d_i2v_pw = HunyuanVideoI2VDenoiser(i2v, enable_cuda_graph="piecewise")
    x_i2v = torch.randn(1, 4, 2, 4, 4, device=device, dtype=dtype)
    t_i2v = torch.tensor([500.0], device=device, dtype=dtype)
    pooled_i2v = torch.randn(1, 32, device=device, dtype=dtype)
    with torch.no_grad():
        for txt_len in (6, 10):
            txt = torch.randn(1, txt_len, 32, device=device, dtype=dtype)
            mask = torch.ones(1, txt_len, device=device, dtype=torch.long)
            mask[0, -2:] = 0
            inp = DenoiserInput(
                latents=x_i2v,
                timestep=t_i2v,
                next_timestep=cast(Any, None),
                step_index=0,
                total_steps=1,
                conditioning={"text_states": txt, "text_states_2": pooled_i2v, "text_mask": mask},
            )
            ref = d_i2v_eager(inp).sample
            assert float(ref.abs().max()) > 0.01
            out = None
            for _ in range(4):
                out = d_i2v_pw(inp).sample
            assert out is not None and torch.allclose(ref, out, atol=2e-2, rtol=2e-2)
    rep_i2v = d_i2v_pw.graph_report()
    assert rep_i2v is not None
    assert rep_i2v["capture"] >= 4 and rep_i2v["replay"] >= 4 and rep_i2v["capture_failed"] == 0

    # 3. HunyuanVideo_1_5_DiffusionTransformer (H15)
    h15_cfg = _tiny_hy15_config()
    h15_cfg["in_channels"] = 2
    h15_cfg["concat_condition"] = True
    h15 = HunyuanVideo_1_5_DiffusionTransformer(**h15_cfg).to(device=device, dtype=dtype).eval()
    _init_zero_params(h15)
    d_h15_eager = HunyuanVideo15Denoiser(h15, image_to_video=False, enable_cuda_graph="off")
    d_h15_pw = HunyuanVideo15Denoiser(h15, image_to_video=False, enable_cuda_graph="piecewise")
    x_h15 = torch.randn(1, 2, 2, 2, 2, device=device, dtype=dtype)
    cond_h15 = torch.zeros(1, 3, 2, 2, 2, device=device, dtype=dtype)
    t_h15 = torch.tensor([500.0], device=device, dtype=dtype)
    byt5_states = torch.zeros(1, 4, 1472, device=device, dtype=dtype)
    byt5_mask = torch.zeros(1, 4, device=device, dtype=torch.long)
    with torch.no_grad():
        for txt_len in (6, 10):
            txt = torch.randn(1, txt_len, 32, device=device, dtype=dtype)
            mask = torch.ones(1, txt_len, device=device, dtype=torch.long)
            mask[0, -2:] = 0
            inp = DenoiserInput(
                latents=x_h15,
                timestep=t_h15,
                next_timestep=cast(Any, None),
                step_index=0,
                total_steps=1,
                conditioning={
                    "condition_latents": cond_h15,
                    "text_states": txt,
                    "text_mask": mask,
                    "byt5_text_states": byt5_states,
                    "byt5_text_mask": byt5_mask,
                },
            )
            ref = d_h15_eager(inp).sample
            assert float(ref.abs().max()) > 0.01
            out = None
            for _ in range(4):
                out = d_h15_pw(inp).sample
            assert out is not None and torch.allclose(ref, out, atol=2e-2, rtol=2e-2)
    rep_h15 = d_h15_pw.graph_report()
    assert rep_h15 is not None
    assert rep_h15["capture"] >= 4 and rep_h15["replay"] >= 4 and rep_h15["capture_failed"] == 0


@requires_cuda
def test_hy15_enable_fp8_protects_first_double_and_last_single_block() -> None:
    from worldfoundry.core.acceleration.quantization.linear import Float8Linear

    cfg = _tiny_hy15_config(single_blocks=2)
    cfg["hidden_size"] = 128
    cfg["heads_num"] = 2
    cfg["rope_dim_list"] = [16, 24, 24]
    model = HunyuanVideo_1_5_DiffusionTransformer(**cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    replaced = model.enable_fp8(exclude_edge_blocks=True, min_features=64)
    assert replaced > 0

    assert not any(isinstance(m, Float8Linear) for m in model.double_blocks[0].modules())
    assert any(isinstance(m, Float8Linear) for m in model.double_blocks[1].modules())
    assert any(isinstance(m, Float8Linear) for m in model.single_blocks[0].modules())
    assert not any(isinstance(m, Float8Linear) for m in model.single_blocks[1].modules())


@requires_cuda
def test_worldplay_kv_cache_mutation_prefix_split_backend_switch_and_inference_mode() -> None:
    device = torch.device("cuda")
    torch.manual_seed(43)
    cfg = _tiny_worldplay_config()
    cfg["hidden_size"] = 128
    cfg["heads_num"] = 2
    cfg["rope_dim_list"] = [16, 24, 24]
    cfg["mm_double_blocks_depth"] = 3
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model.add_prope_parameters()
    model.add_discrete_action_parameters()
    _init_zero_params(model)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.fuse_qkv()

    bs, c_in, h, w = 1, 8, 2, 2
    hidden1 = torch.randn(bs, c_in, 2, h, w, device=device, dtype=torch.bfloat16)
    t1 = torch.full((bs, 2), 500.0, device=device, dtype=torch.bfloat16)
    act1 = torch.zeros((bs, 2), device=device, dtype=torch.long)
    vm1 = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(bs, 2, 1, 1)
    ks1 = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(bs, 2, 1, 1)
    t_txt = torch.zeros((bs,), device=device, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        txt_a = torch.randn(bs, 8, 32, device=device, dtype=torch.bfloat16)
        mask_a = torch.ones(bs, 8, device=device, dtype=torch.long)
        kv = model.forward_txt(timestep_txt=t_txt, text_states=txt_a, encoder_attention_mask=mask_a, cache_txt=True)

        model.enable_cuda_graph(warmup=1)
        for _ in range(3):
            out_before, _ = model.forward_vision(
                hidden_states=hidden1,
                timestep=t1,
                action=act1,
                viewmats=vm1,
                Ks=ks1,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )

        # 1. In-place mutation of kv_cache tensor must refresh static arena
        for layer_slot in kv:
            layer_slot["v_txt"].add_(5.0)
            layer_slot["k_txt"].mul_(-1.0)
        out_mut_graph, _ = model.forward_vision(
            hidden_states=hidden1,
            timestep=t1,
            action=act1,
            viewmats=vm1,
            Ks=ks1,
            kv_cache=kv,
            rope_temporal_size=4,
            start_rope_start_idx=0,
        )
        model.disable_cuda_graph()
        out_mut_eager, _ = model.forward_vision(
            hidden_states=hidden1,
            timestep=t1,
            action=act1,
            viewmats=vm1,
            Ks=ks1,
            kv_cache=kv,
            rope_temporal_size=4,
            start_rope_start_idx=0,
        )
        assert not torch.allclose(out_before, out_mut_eager, atol=1e-2)
        assert torch.allclose(out_mut_graph, out_mut_eager, atol=2e-2, rtol=2e-2)

        # 2. Same total_len (16) with different (prefix_len, chunk_tokens): (8, 8) vs (4, 12)
        model.enable_cuda_graph(warmup=1)
        for _ in range(3):
            model.forward_vision(
                hidden_states=hidden1,
                timestep=t1,
                action=act1,
                viewmats=vm1,
                Ks=ks1,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )
        txt_b = torch.randn(bs, 4, 32, device=device, dtype=torch.bfloat16)
        mask_b = torch.ones(bs, 4, device=device, dtype=torch.long)
        kv_b = model.forward_txt(timestep_txt=t_txt, text_states=txt_b, encoder_attention_mask=mask_b, cache_txt=True)
        hidden2 = torch.randn(bs, c_in, 3, h, w, device=device, dtype=torch.bfloat16)
        t2 = torch.full((bs, 3), 500.0, device=device, dtype=torch.bfloat16)
        act2 = torch.zeros((bs, 3), device=device, dtype=torch.long)
        vm2 = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(bs, 3, 1, 1)
        ks2 = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(bs, 3, 1, 1)
        for _ in range(3):
            out_split_graph, _ = model.forward_vision(
                hidden_states=hidden2,
                timestep=t2,
                action=act2,
                viewmats=vm2,
                Ks=ks2,
                kv_cache=kv_b,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )
        model.disable_cuda_graph()
        out_split_eager, _ = model.forward_vision(
            hidden_states=hidden2,
            timestep=t2,
            action=act2,
            viewmats=vm2,
            Ks=ks2,
            kv_cache=kv_b,
            rope_temporal_size=4,
            start_rope_start_idx=0,
        )
        assert torch.allclose(out_split_graph, out_split_eager, atol=2e-2, rtol=2e-2)

        # 3. set_attention_backend("cudnn_fp8") invalidates old CUDA graphs & calibrates + captures FP8 arena
        model.enable_cuda_graph(warmup=1)
        for _ in range(3):
            model.forward_vision(
                hidden_states=hidden1,
                timestep=t1,
                action=act1,
                viewmats=vm1,
                Ks=ks1,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )
        reset_fp8_attention_scales()
        model.set_attention_backend("cudnn_fp8")
        assert model.graph_report()["capture"] == 0  # graphs invalidated on backend switch
        for _ in range(4):
            out_fp8, _ = model.forward_vision(
                hidden_states=hidden1,
                timestep=t1,
                action=act1,
                viewmats=vm1,
                Ks=ks1,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )
        assert torch.isfinite(out_fp8).all() and float(out_fp8.abs().max()) > 0.01
        assert (hidden1.device, ("vision", "rope", 1)) in _FP8_ATTN_SCALES
        assert (hidden1.device, ("vision", "prope", 1)) in _FP8_ATTN_SCALES
        fp8_rep = model.graph_report()
        assert fp8_rep["capture"] >= 1 and fp8_rep["replay"] >= 1 and fp8_rep["capture_failed"] == 0

    # 4. torch.inference_mode() must not fail on tensor._version tracking
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        kv_inf = model.forward_txt(timestep_txt=t_txt, text_states=txt_a, encoder_attention_mask=mask_a, cache_txt=True)
        out_inf, _ = model.forward_vision(
            hidden_states=hidden1,
            timestep=t1,
            action=act1,
            viewmats=vm1,
            Ks=ks1,
            kv_cache=kv_inf,
            rope_temporal_size=4,
            start_rope_start_idx=0,
        )
        assert torch.isfinite(out_inf).all()


@requires_cuda
def test_worldplay_bi_compacted_txt_and_cudnn_fp8_exact_parity() -> None:
    device = torch.device("cuda")
    torch.manual_seed(53)
    cfg = _tiny_worldplay_config()
    cfg["hidden_size"] = 128
    cfg["heads_num"] = 2
    cfg["rope_dim_list"] = [16, 24, 24]
    cfg["mm_double_blocks_depth"] = 3
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model.add_prope_parameters()
    model.add_discrete_action_parameters()
    _init_zero_params(model)
    model = model.to(device=device, dtype=torch.bfloat16).eval()

    hs = torch.randn(1, 8, 2, 2, 2, device=device, dtype=torch.bfloat16)
    ts = torch.full((2,), 500.0, device=device, dtype=torch.bfloat16)
    t_txt = torch.zeros((1,), device=device, dtype=torch.bfloat16)
    tx = torch.randn(1, 10, 32, device=device, dtype=torch.bfloat16)
    mk = torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 0, 0]], device=device, dtype=torch.long)
    vm = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    ks = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    act = torch.zeros((1, 2), device=device, dtype=torch.long)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_compact = model.forward_bi(
            hidden_states=hs,
            timestep=ts,
            timestep_txt=t_txt,
            text_states=tx,
            text_states_2=None,
            encoder_attention_mask=mk,
            viewmats=vm,
            Ks=ks,
            action=act,
        )[0]
        # Non-uniform batch disables compaction; batch item 0 is the exact uncompacted reference.
        out_ref = model.forward_bi(
            hidden_states=torch.cat([hs, hs], dim=0),
            timestep=torch.cat([ts, ts], dim=0),
            timestep_txt=torch.cat([t_txt, t_txt], dim=0),
            text_states=torch.cat([tx, tx], dim=0),
            text_states_2=None,
            encoder_attention_mask=torch.cat([mk, torch.ones_like(mk)], dim=0),
            viewmats=torch.cat([vm, vm], dim=0),
            Ks=torch.cat([ks, ks], dim=0),
            action=torch.cat([act, act], dim=0),
        )[0][:1]
        assert float(out_ref.abs().max()) > 0.01
        assert torch.allclose(out_compact, out_ref, atol=2e-2, rtol=2e-2)

        # cudnn_fp8 on forward_bi works cleanly for both compacted (unmasked) & non-uniform (fallback)
        reset_fp8_attention_scales()
        model.set_attention_backend("cudnn_fp8")
        out_fp8 = model.forward_bi(
            hidden_states=hs,
            timestep=ts,
            timestep_txt=t_txt,
            text_states=tx,
            text_states_2=None,
            encoder_attention_mask=mk,
            viewmats=vm,
            Ks=ks,
            action=act,
        )[0]
        assert torch.isfinite(out_fp8).all()
        assert (hs.device, ("bi", "rope", 1)) in _FP8_ATTN_SCALES
        assert (hs.device, ("bi", "prope", 1)) in _FP8_ATTN_SCALES
        assert (hs.device, ("bi", "rope", 0)) not in _FP8_ATTN_SCALES  # edge-protected
        out_fp8_nonuniform = model.forward_bi(
            hidden_states=torch.cat([hs, hs], dim=0),
            timestep=torch.cat([ts, ts], dim=0),
            timestep_txt=torch.cat([t_txt, t_txt], dim=0),
            text_states=torch.cat([tx, tx], dim=0),
            text_states_2=None,
            encoder_attention_mask=torch.cat([mk, torch.ones_like(mk)], dim=0),
            viewmats=torch.cat([vm, vm], dim=0),
            Ks=torch.cat([ks, ks], dim=0),
            action=torch.cat([act, act], dim=0),
        )[0]
        assert torch.isfinite(out_fp8_nonuniform).all()


@requires_cuda
def test_camera_modulation_non_divisible_seq_len_and_triton_fp8_quant() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_attention import (
        _quantize_static_fp8_into,
    )

    # Non-divisible seq_len=5, latent_length=3 where seq_len // latent_length == 1 must raise ValueError cleanly
    x = torch.randn(1, 5, 16)
    shift = torch.randn(3, 16)
    scale = torch.randn(3, 16)
    gate = torch.randn(3, 16)
    with pytest.raises(ValueError, match="not divisible"):
        modulate(x, shift=shift, scale=scale)
    with pytest.raises(ValueError, match="not divisible"):
        apply_gate(x, gate=gate)

    # Single-pass Triton BF16 -> FP8 static-scale quantization bitwise parity vs FP32 ref
    device = torch.device("cuda")
    torch.manual_seed(61)
    src = torch.randn(1, 4, 19, 128, device=device, dtype=torch.bfloat16) * 5.0
    dst = torch.empty(1, 4, 25, 128, device=device, dtype=torch.float8_e4m3fn)
    descale = 0.075
    _quantize_static_fp8_into(src, dst, descale=descale, dst_offset=3)
    ref = (src.float() * (1.0 / descale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(dst[:, :, 3:22, :].view(torch.uint8), ref.view(torch.uint8))


def test_matrix_game_runtime_guards_and_cpu_stream_overlapped() -> None:
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
        MatrixGame2RealtimeSession,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.worldfoundry_runtime import (
        _matrix_game2_runtime_policy,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _matrix_game2_runtime_policy({"vae_compile": True, "vae_cuda_graph": True}, device="cpu", dtype=torch.bfloat16)

    # CPU stream_overlapped must delegate once per chunk without double-appending actions
    session: Any = object.__new__(MatrixGame2RealtimeSession)
    session.configured = True
    session._noise_generator = torch.Generator()
    session.core = SimpleNamespace(_session=SimpleNamespace(current_start_frame=0))
    session.device = torch.device("cpu")
    session.runtime = SimpleNamespace(mode="universal")
    calls: list[Any] = []
    session.realtime_spec = lambda: SimpleNamespace(to_payload=lambda: {"mode": "universal"})
    session._append_actions = lambda *a, **kw: calls.append(("append", a))
    session.generate = lambda *, interactions: (calls.append(("gen", list(interactions))), {"video": interactions})[1]
    out = list(session.stream_overlapped([["forward"], ["left"]]))
    assert len(out) == 2
    assert calls == [("gen", ["forward"]), ("gen", ["left"])]


def test_longlive_refresh_crossattn_dtype_truncation_and_compile_lora_guard() -> None:
    torch.manual_seed(71)

    class _FusedCross(nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self._kv_fused = True
            self.kv = nn.Linear(dim, 2 * dim, bias=False)
            self.norm_k = nn.Identity()

    class _Block(nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.cross_attn = _FusedCross(dim, heads)

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_len = 6
            self.text_embedding = nn.Linear(16, 32, bias=False)
            self.blocks = nn.ModuleList([_Block(32, 4)])

    model = _Model().to(dtype=torch.bfloat16).eval()
    pipe = object.__new__(NativeLongLivePipeline)
    # fp32 context longer than text_len=6 on bf16 model with fused _kv_fused
    ctx_long_f32 = torch.randn(1, 10, 16, dtype=torch.float32)
    cross_cache = [
        {
            "is_init": True,
            "k": torch.zeros(1, 6, 4, 8, dtype=torch.bfloat16),
            "v": torch.zeros(1, 6, 4, 8, dtype=torch.bfloat16),
        }
    ]
    ptr_before = (cross_cache[0]["k"].data_ptr(), cross_cache[0]["v"].data_ptr())
    ok = pipe._refresh_crossattn_cache_inplace(model, ctx_long_f32, cross_cache)
    assert ok is True
    assert (cross_cache[0]["k"].data_ptr(), cross_cache[0]["v"].data_ptr()) == ptr_before
    assert torch.isfinite(cross_cache[0]["k"]).all() and float(cross_cache[0]["k"].abs().max()) > 0.0

    with pytest.raises(ValueError, match="cannot combine with load-time compile=True"):
        NativeLongLivePipeline.from_pretrained(
            "/nonexistent",
            lora_path="/nonexistent/lora.pt",
            apply_lora=True,
            compile=True,
        )
