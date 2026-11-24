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


@requires_cuda
def test_longlive_wave2_triton_kernels_and_o1_slot_canonicalization() -> None:
    from worldfoundry.pipelines.longlive.fast_kernels import (
        LongLiveFastDenoiserEngine,
        wan_layernorm_adaln,
        wan_qk_rmsnorm_rope_write_kv,
        wan_residual_gate_add,
        wan_rmsnorm,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(101)
    B, F_cur, H_sp, W_sp = 1, 3, 4, 4
    S = F_cur * H_sp * W_sp
    num_heads, head_dim = 4, 64
    C = num_heads * head_dim

    x = torch.randn(B, S, C, device=device, dtype=dtype)
    y = torch.randn_like(x)
    e = torch.randn(B, F_cur, 6, C, device=device, dtype=torch.float32)
    mod = torch.randn(1, 6, C, device=device, dtype=torch.float32)

    with torch.inference_mode():
        # 1. Exact LayerNorm + AdaLN matches WanLayerNorm(x).float() * (1 + scale) + shift bit-for-bit
        e_eff = mod.unsqueeze(1) + e
        shift0 = e_eff[:, :, 0, :]
        scale0 = e_eff[:, :, 1, :]
        gate0 = e_eff[:, :, 2, :]
        ref_ln = (
            torch.nn.functional.layer_norm(x, (C,), eps=1e-6).float().view(B, F_cur, H_sp * W_sp, C)
            * (1.0 + scale0.unsqueeze(2))
            + shift0.unsqueeze(2)
        ).view(B, S, C)
        fast_ln = wan_layernorm_adaln(x, scale0, shift0, num_frames=F_cur, frame_seq=H_sp * W_sp, eps=1e-6, exact=True)
        assert torch.allclose(fast_ln, ref_ln, atol=1e-6, rtol=1e-6)
        assert torch.equal(fast_ln.to(dtype), ref_ln.to(dtype))

        # 2. Exact Residual-Gate add matches PyTorch reference bit-for-bit
        ref_rg = (
            x.float().view(B, F_cur, H_sp * W_sp, C) + y.float().view(B, F_cur, H_sp * W_sp, C) * gate0.unsqueeze(2)
        ).view(B, S, C)
        fast_rg = wan_residual_gate_add(x, y, gate0, num_frames=F_cur, frame_seq=H_sp * W_sp)
        assert torch.equal(fast_rg, ref_rg)

        # 3. Exact WanRMSNorm matches PyTorch reference bit-for-bit
        w = torch.randn(C, device=device, dtype=dtype)
        ref_rms = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(x) * w
        fast_rms = wan_rmsnorm(x, w, eps=1e-6, exact=True)
        assert torch.equal(fast_rms, ref_rms)

        # 4. Exact fused QK-RMSNorm + FP64 3D RoPE + KV write matches complex-FP64 reference bit-for-bit
        q_proj = torch.randn(B, S, C, device=device, dtype=dtype)
        k_proj = torch.randn(B, S, C, device=device, dtype=dtype)
        qw = torch.randn(C, device=device, dtype=dtype)
        kw = torch.randn(C, device=device, dtype=dtype)
        theta = torch.randn(S, head_dim // 2, device=device, dtype=torch.float64)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        k_rot_dst = torch.zeros(B, S + 16, num_heads, head_dim, device=device, dtype=dtype)
        k_raw_dst = torch.zeros_like(k_rot_dst)
        q_fast = wan_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            qw,
            kw,
            cos,
            sin,
            k_rot_dst,
            k_raw_dst,
            dst_token_offset=16,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            exact=True,
        )

        def _ref_rope_fp64(t_bshd: torch.Tensor) -> torch.Tensor:
            xc = torch.view_as_complex(t_bshd.to(torch.float64).reshape(B, S, num_heads, head_dim // 2, 2))
            fc = torch.complex(cos, sin).view(1, S, 1, head_dim // 2)
            return torch.view_as_real(xc * fc).flatten(3).to(torch.float32).to(dtype)

        q_norm_ref = (
            (q_proj.float() * torch.rsqrt(q_proj.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(q_proj) * qw
        ).view(B, S, num_heads, head_dim)
        k_norm_ref = (
            (k_proj.float() * torch.rsqrt(k_proj.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(k_proj) * kw
        ).view(B, S, num_heads, head_dim)
        assert torch.equal(q_fast, _ref_rope_fp64(q_norm_ref))
        assert torch.equal(k_raw_dst[:, 16 : 16 + S], k_norm_ref)
        assert torch.equal(k_rot_dst[:, 16 : 16 + S], _ref_rope_fp64(k_norm_ref))

    # 5. LongLiveFastDenoiserEngine default lossless policy & slot key boundedness
    dummy_model = nn.Module()
    dummy_model.register_parameter("p", nn.Parameter(torch.zeros(1, device=device, dtype=dtype)))
    dummy_denoiser = SimpleNamespace(model=dummy_model, _piecewise_graph=None, enable_cuda_graph=False)
    engine = LongLiveFastDenoiserEngine(dummy_denoiser, attention_backend="torch", enable_cuda_graph=False)
    assert engine.attention_backend == "torch"
    assert engine.exact_norms is True
    assert engine.report()["fallbacks"] == []


@requires_cuda
def test_matrix_game2_wave2_triton_kernels_and_runtime_policy() -> None:
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.fast_kernels import (
        mg2_layernorm_adaln,
        mg2_main_self_attention,
        mg2_qk_rmsnorm_rope_write_kv,
        mg2_residual_gate_add,
        mg2_rmsnorm,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
        VAEDecoderWrapper,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.worldfoundry_runtime import (
        _matrix_game2_runtime_policy,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(102)
    B, F_cur, HW = 1, 3, 16
    S = F_cur * HW
    num_heads, head_dim = 4, 64
    C = num_heads * head_dim

    x = torch.randn(B, S, C, device=device, dtype=dtype)
    y = torch.randn_like(x)
    e = torch.randn(B, F_cur, 6, C, device=device, dtype=torch.float32)
    mod = torch.randn(1, 6, C, device=device, dtype=torch.float32)

    with torch.inference_mode():
        e_eff = mod.unsqueeze(1) + e
        shift0 = e_eff[:, :, 0, :]
        scale0 = e_eff[:, :, 1, :]
        gate0 = e_eff[:, :, 2, :]
        ref_ln = (
            torch.nn.functional.layer_norm(x, (C,), eps=1e-6).float().view(B, F_cur, HW, C)
            * (1.0 + scale0.unsqueeze(2))
            + shift0.unsqueeze(2)
        ).view(B, S, C)
        fast_ln = mg2_layernorm_adaln(x, scale0, shift0, num_frames=F_cur, frame_seq=HW, eps=1e-6)
        assert torch.allclose(fast_ln, ref_ln, atol=1e-6, rtol=1e-6)
        assert torch.equal(fast_ln.to(dtype), ref_ln.to(dtype))

        ref_rg = (x.float().view(B, F_cur, HW, C) + y.float().view(B, F_cur, HW, C) * gate0.unsqueeze(2)).view(B, S, C)
        assert torch.equal(mg2_residual_gate_add(x, y, gate0, num_frames=F_cur, frame_seq=HW), ref_rg)

        w = torch.randn(C, device=device, dtype=dtype)
        ref_rms = (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(x) * w
        assert torch.equal(mg2_rmsnorm(x, w, eps=1e-6, exact=True), ref_rms)

        q_proj = torch.randn(B, S, C, device=device, dtype=dtype)
        k_proj = torch.randn(B, S, C, device=device, dtype=dtype)
        qw = torch.randn(C, device=device, dtype=dtype)
        kw = torch.randn(C, device=device, dtype=dtype)
        theta = torch.randn(S, head_dim // 2, device=device, dtype=torch.float64)
        cos, sin = torch.cos(theta), torch.sin(theta)
        k_dst = torch.zeros(B, S + 8, num_heads, head_dim, device=device, dtype=dtype)
        q_out = mg2_qk_rmsnorm_rope_write_kv(
            q_proj,
            k_proj,
            qw,
            kw,
            cos,
            sin,
            k_dst,
            dst_token_offset=8,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=1e-6,
            exact=True,
        )

        def _ref_rope(t_bshd: torch.Tensor) -> torch.Tensor:
            xc = torch.view_as_complex(t_bshd.to(torch.float64).reshape(B, S, num_heads, head_dim // 2, 2))
            fc = torch.complex(cos, sin).view(1, S, 1, head_dim // 2)
            return torch.view_as_real(xc * fc).flatten(3).to(torch.float32).to(dtype)

        qn = (
            (q_proj.float() * torch.rsqrt(q_proj.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(q_proj) * qw
        ).view(B, S, num_heads, head_dim)
        kn = (
            (k_proj.float() * torch.rsqrt(k_proj.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(k_proj) * kw
        ).view(B, S, num_heads, head_dim)
        assert torch.equal(q_out, _ref_rope(qn))
        assert torch.equal(k_dst[:, 8 : 8 + S], _ref_rope(kn))

        # Main self-attention FA3 vs SDPA parity
        v = torch.randn(B, S + 8, num_heads, head_dim, device=device, dtype=dtype)
        o_torch = mg2_main_self_attention(q_out, k_dst, v, backend="torch")
        o_fa3 = mg2_main_self_attention(q_out, k_dst, v, backend="flash_attention_3")
        assert torch.allclose(o_torch, o_fa3, atol=2e-2, rtol=2e-2)

    # Policy checks: accept flash_attention_3 & vae_compile; reject unsupported sage
    pol_fa3 = _matrix_game2_runtime_policy(
        {"attention_backend": "flash_attention_3", "vae_compile": True}, device=device, dtype=dtype
    )
    assert pol_fa3.options["_resolved_attention_backend"] == "flash_attention_3"
    assert pol_fa3.options["vae_compile"] is True
    with pytest.raises(ValueError, match="attention_backend"):
        _matrix_game2_runtime_policy({"attention_backend": "sage"}, device=device, dtype=dtype)

    # Steady compile method exists on VAEDecoderWrapper
    assert callable(getattr(VAEDecoderWrapper, "enable_steady_compile", None))


@requires_cuda
def test_hy_worldplay_wave2_fused_triton_and_bshd_fa3_parity() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.action_camera import (
        _fused_qkv_rmsnorm_rope_prope,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_attention import (
        prepare_static_kv_arena,
        prepare_static_kv_arena_bshd,
        sequence_parallel_attention_bi_fa3,
        sequence_parallel_attention_vision,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_modulation import (
        fused_layernorm_modulate,
        fused_residual_dual_gate,
        fused_residual_gate,
    )
    from worldfoundry.core.attention import apply_nd_rotary_embedding, get_nd_rotary_pos_embed
    from worldfoundry.core.nn import normalization_layer

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(103)
    B, T, HW, H, D = 1, 2, 16, 4, 64
    S = T * HW
    C = H * D

    with torch.inference_mode():
        # 1. Fused LayerNorm+Modulate, Residual-Gate, Dual-Residual-Gate (chunked strided views!)
        x = torch.randn(B, S, C, device=device, dtype=dtype)
        y1 = torch.randn_like(x)
        y2 = torch.randn_like(x)
        mods = torch.randn(B * T, 6 * C, device=device, dtype=dtype)
        sh1, sc1, g1, _, _, _ = mods.chunk(6, dim=-1)
        assert not sh1.is_contiguous()
        ref_ln = modulate(torch.nn.functional.layer_norm(x, (C,), eps=1e-6), shift=sh1, scale=sc1)
        fast_ln = fused_layernorm_modulate(x, shift=sh1, scale=sc1, eps=1e-6)
        assert torch.allclose(fast_ln, ref_ln, atol=5e-2, rtol=2e-2)

        ref_rg = x + apply_gate(y1, gate=g1)
        fast_rg = fused_residual_gate(x, y1, g1)
        assert torch.allclose(fast_rg, ref_rg, atol=5e-2, rtol=2e-2)

        ref_rdg = x + apply_gate(y1 + y2, gate=g1)
        fast_rdg = fused_residual_dual_gate(x, y1, y2, g1)
        assert torch.allclose(fast_rdg, ref_rdg, atol=5e-2, rtol=2e-2)

        # 2. Single-pass fused QKV-RMSNorm+RoPE+PRoPE kernel parity
        qkv = torch.randn(B, S, 3 * C, device=device, dtype=dtype)
        rms_cls = normalization_layer("rms")
        qn = rms_cls(D, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        kn = rms_cls(D, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
        qn.weight.data.uniform_(0.5, 1.5)
        kn.weight.data.uniform_(0.5, 1.5)

        vm = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, T, 1, 1)
        vm[0, 1, 0, 3] = 0.25
        ks = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(B, T, 1, 1)
        ks[..., 0, 0] = 1.2
        ks[..., 1, 1] = 1.15
        p_mat, p_inv_t, _ = _precompute_prope_mats(vm, ks)
        fcos, fsin = get_nd_rotary_pos_embed([16, 24, 24], [T, 4, 4], theta=256, use_real=True, theta_rescale_factor=1)
        freqs_cis = (fcos.to(device=device), fsin.to(device=device))

        rq, rk, rv = qkv.chunk(3, dim=-1)
        rq = qn(rq.view(B, S, H, D)).to(rv)
        rk = kn(rk.view(B, S, H, D)).to(rv)
        rv = rv.view(B, S, H, D)
        rq_p = _apply_prope_mat(rq, p_mat)
        rk_p = _apply_prope_mat(rk, p_inv_t)
        rv_p = _apply_prope_mat(rv, p_inv_t)
        rq_r, rk_r = apply_nd_rotary_embedding(rq, rk, freqs_cis, head_first=False)

        fused = _fused_qkv_rmsnorm_rope_prope(
            qkv, qn.weight, kn.weight, freqs_cis, p_mat, p_inv_t, heads_num=H, eps=1e-6
        )
        assert fused is not None
        fq_r, fk_r, fv, fq_p, fk_p, fv_p = fused
        assert torch.allclose(fq_r, rq_r, atol=5e-2, rtol=2e-2)
        assert torch.allclose(fk_r, rk_r, atol=5e-2, rtol=2e-2)
        assert torch.equal(fv, rv)
        assert torch.allclose(fq_p, rq_p, atol=5e-2, rtol=2e-2)
        assert torch.allclose(fk_p, rk_p, atol=5e-2, rtol=2e-2)
        assert torch.equal(fv_p, rv_p)

        # 3. Direct BSHD static KV arena + FA3 vision & bi attention parity
        kv_cache = [
            {
                "k_txt": torch.randn(B, H, 16, D, device=device, dtype=dtype),
                "v_txt": torch.randn(B, H, 16, D, device=device, dtype=dtype),
                "k_vision": torch.randn(B, H, 32, D, device=device, dtype=dtype),
                "v_vision": torch.randn(B, H, 32, D, device=device, dtype=dtype),
            }
        ]
        bhsd_bufs, pref, _ = prepare_static_kv_arena(kv_cache, S, dtype=dtype)
        bshd_bufs, pref_bshd, key_bshd = prepare_static_kv_arena_bshd(kv_cache, S, dtype=dtype)
        assert pref == pref_bshd and key_bshd[0] == "bshd"
        o_vis_sdpa, _ = sequence_parallel_attention_vision(
            fq_r, fk_r, fv, block_idx=0, attn_mode="torch", static_kv_buffer=bhsd_bufs[0], static_prefix_len=pref
        )
        o_vis_fa3, _ = sequence_parallel_attention_vision(
            fq_r,
            fk_r,
            fv,
            block_idx=0,
            attn_mode="flash_attention_3",
            static_bshd_kv_buffer=bshd_bufs[0],
            static_prefix_len=pref_bshd,
        )
        assert torch.allclose(o_vis_sdpa, o_vis_fa3, atol=2e-2, rtol=2e-2)

        tq = torch.randn(B, 16, H, D, device=device, dtype=dtype)
        tk = torch.randn(B, 16, H, D, device=device, dtype=dtype)
        tv = torch.randn(B, 16, H, D, device=device, dtype=dtype)
        bi_fa3 = sequence_parallel_attention_bi_fa3((fq_r, tq), (fk_r, tk), (fv, tv))
        assert bi_fa3.shape == (B, S + 16, C) and torch.isfinite(bi_fa3).all()

    # 4. End-to-end ARHunyuanVideo_1_5_DiffusionTransformer FA3+BSHD+CG (vision) & FA3 compacted (bi) parity
    cfg = _tiny_worldplay_config()
    cfg["hidden_size"] = 128
    cfg["heads_num"] = 2
    cfg["rope_dim_list"] = [16, 24, 24]
    cfg["mm_double_blocks_depth"] = 3
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model.add_prope_parameters()
    model.add_discrete_action_parameters()
    _init_zero_params(model)
    model = model.to(device=device, dtype=dtype).eval()
    model.fuse_qkv()

    hs = torch.randn(1, 8, 2, 2, 2, device=device, dtype=dtype)
    ts = torch.full((2,), 500.0, device=device, dtype=dtype)
    t_txt = torch.zeros((1,), device=device, dtype=dtype)
    tx = torch.randn(1, 10, 32, device=device, dtype=dtype)
    mk = torch.tensor([[1, 1, 1, 0, 0, 1, 0, 1, 0, 0]], device=device, dtype=torch.long)
    vm = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    ks = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    act = torch.zeros((1, 2), device=device, dtype=torch.long)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        # forward_bi: torch vs flash_attention_3 (with padded mask compaction!)
        model.set_attention_backend("torch")
        bi_ref = model.forward_bi(
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
        model.set_attention_backend("flash_attention_3")
        bi_fa3_out = model.forward_bi(
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
        assert torch.allclose(bi_fa3_out, bi_ref, atol=3e-2, rtol=3e-2)

        # forward_vision: torch+CG vs flash_attention_3+BSHD+CG
        kv = model.forward_txt(timestep_txt=t_txt, text_states=tx, encoder_attention_mask=mk, cache_txt=True)
        model.enable_cuda_graph(warmup=1)
        model.set_attention_backend("torch")
        for _ in range(3):
            vis_ref = model.forward_vision(
                hidden_states=hs,
                timestep=ts,
                action=act,
                viewmats=vm,
                Ks=ks,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )[0]
        model.set_attention_backend("flash_attention_3")
        for _ in range(3):
            vis_fa3 = model.forward_vision(
                hidden_states=hs,
                timestep=ts,
                action=act,
                viewmats=vm,
                Ks=ks,
                kv_cache=kv,
                rope_temporal_size=4,
                start_rope_start_idx=0,
            )[0]
        rep = model.graph_report()
        assert rep is not None and rep["capture"] >= 1 and rep["replay"] >= 1 and rep["capture_failed"] == 0
        assert torch.allclose(vis_fa3, vis_ref, atol=3e-2, rtol=3e-2)


@requires_cuda
def test_longlive_wave2_o1_rolling_graph_isolation_and_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    import worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing as sf_mod
    from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
    from worldfoundry.base_models.diffusion_model.models.denoisers.wan import SelfForcingWanDenoiser
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing.self_forcing import (
        CausalWanModel,
    )
    from worldfoundry.pipelines.longlive.fast_kernels import (
        LongLiveFastDenoiserEngine,
        fast_longlive_self_attention,
        wan_layernorm_adaln,
        wan_residual_gate_add,
        wan_rmsnorm,
    )

    # Pin the baseline reference denoiser to exact PyTorch FP64 RoPE (matching causal_rope_apply
    # bit-for-bit without Triton FMA drift) so torch.equal asserts true 0-ULP lossless parity.
    def _exact_pytorch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        s = int(cos.shape[1])
        x0, x1 = x[:, :s].to(cos.dtype).reshape(x.shape[0], s, x.shape[2], x.shape[3] // 2, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2).to(x.dtype)

    monkeypatch.setattr(sf_mod, "_apply_real_rope", _exact_pytorch_rope)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(201)

    def _mk(latents: torch.Tensor, t: torch.Tensor, cond: dict[str, Any]) -> DenoiserInput:
        return DenoiserInput(
            latents=latents, timestep=t, next_timestep=t * 0, step_index=0, total_steps=4, conditioning=cond
        )

    model = (
        CausalWanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=4,
            dim=128,
            ffn_dim=256,
            freq_dim=64,
            text_dim=32,
            out_dim=4,
            num_heads=2,
            num_layers=2,
            local_attn_size=12,
            sink_size=5,
            qk_norm=True,
            cross_attn_norm=True,
            kv_cache_relative_rope=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    for p in model.parameters():
        if p.ndim >= 2:
            nn.init.normal_(p, std=0.05)

    den_fast = SelfForcingWanDenoiser(
        model, block_size=3, kv_window="sink5-window7-rerope", compute_dtype=dtype, enable_cuda_graph=True
    )
    engine = LongLiveFastDenoiserEngine(den_fast, attention_backend="torch", enable_cuda_graph=True, exact_norms=True)
    den_ref = SelfForcingWanDenoiser(
        model, block_size=3, kv_window="sink5-window7-rerope", compute_dtype=dtype, enable_cuda_graph=False
    )

    # 1. Full 8-chunk rolling O(1) rollout: bit-exact parity + strictly 4 segments forever
    kv_f = den_fast.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    kv_r = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    ctx1 = torch.randn(1, 8, 32, device=device, dtype=dtype) * 4.0
    with torch.inference_mode():
        for chunk in range(8):
            for t_val in (900.0, 500.0, 0.0):
                xs = torch.randn(1, 4, 3, 8, 8, device=device, dtype=dtype)
                ts = torch.tensor([t_val], device=device)
                of = engine(_mk(xs, ts, {"context": ctx1, "kv_cache": kv_f, "current_start": chunk * 48})).sample
                or_ = den_ref(_mk(xs, ts, {"context": ctx1, "kv_cache": kv_r, "current_start": chunk * 48})).sample
                assert torch.equal(of, or_)
    rep = den_fast._piecewise_graph.report()
    assert rep["segment_count"] == 4 and rep["capture"] == 4 and rep["eager"] == 0

    # 2. Restart at current_start=0 without reset_causal_host_cache must not produce negative slot indices
    x0 = torch.randn(1, 4, 3, 8, 8, device=device, dtype=dtype)
    t500 = torch.tensor([500.0], device=device)
    kv_r0 = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    with torch.inference_mode():
        o_restart = engine(_mk(x0, t500, {"context": ctx1, "kv_cache": kv_f, "current_start": 0})).sample
        o_ref0 = den_ref(_mk(x0, t500, {"context": ctx1, "kv_cache": kv_r0, "current_start": 0})).sample
    assert torch.equal(o_restart, o_ref0)

    # 3. Reusing pooled cache outside inference_mode after inference_mode init + portrait after landscape (same H*W)
    kv_land = den_fast.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=32, device=device, dtype=dtype
    )
    x_land = torch.randn(1, 4, 3, 8, 16, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(3):
            engine(_mk(x_land, t500, {"context": ctx1, "kv_cache": kv_land, "current_start": 0}))
    # Must not fail with "Inplace update to inference tensor outside InferenceMode is not allowed"
    kv_port = den_fast.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=32, device=device, dtype=dtype
    )
    x_port = torch.randn(1, 4, 3, 16, 8, device=device, dtype=dtype)
    kv_port_ref = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=32, device=device, dtype=dtype
    )
    with torch.inference_mode():
        o_port = engine(_mk(x_port, t500, {"context": ctx1, "kv_cache": kv_port, "current_start": 0})).sample
        o_port_ref = den_ref(_mk(x_port, t500, {"context": ctx1, "kv_cache": kv_port_ref, "current_start": 0})).sample
    assert torch.equal(o_port, o_port_ref)

    # 4. B=2 and fresh unpooled cache with a different prompt must not collide with B=1 captured graph
    kv_b2 = den_fast.create_kv_cache(
        batch_size=2, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    ctx_b2 = torch.randn(2, 8, 32, device=device, dtype=dtype) * -3.0
    x_b2 = torch.randn(2, 4, 3, 8, 8, device=device, dtype=dtype)
    kv_b2_ref = den_ref.create_kv_cache(
        batch_size=2, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    kv_fresh = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=32, device=device, dtype=dtype
    )
    ctx2 = torch.randn(1, 8, 32, device=device, dtype=dtype) * -5.0
    kv_fresh_ref = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=32, device=device, dtype=dtype
    )
    with torch.inference_mode():
        o_b2 = engine(_mk(x_b2, t500, {"context": ctx_b2, "kv_cache": kv_b2, "current_start": 0})).sample
        o_b2_ref = den_ref(_mk(x_b2, t500, {"context": ctx_b2, "kv_cache": kv_b2_ref, "current_start": 0})).sample
        o_fresh = engine(_mk(x_land, t500, {"context": ctx2, "kv_cache": kv_fresh, "current_start": 0})).sample
        o_fresh_ref = den_ref(_mk(x_land, t500, {"context": ctx2, "kv_cache": kv_fresh_ref, "current_start": 0})).sample
    assert torch.equal(o_b2, o_b2_ref)
    assert torch.equal(o_fresh, o_fresh_ref)

    # 5. NaN-poisoned roped_k_arena eviction mid-rollout backfills cleanly from raw kv_cache
    den_nocg = SelfForcingWanDenoiser(
        model, block_size=3, kv_window="sink5-window7-rerope", compute_dtype=dtype, enable_cuda_graph=False
    )
    eng_nocg = LongLiveFastDenoiserEngine(
        den_nocg, attention_backend="torch", enable_cuda_graph=False, exact_norms=True
    )
    kv_a = den_nocg.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    kv_b = den_ref.create_kv_cache(
        batch_size=1, n_views=1, latent_frames_per_view=12, frame_sequence_length=16, device=device, dtype=dtype
    )
    t0 = torch.tensor([0.0], device=device)
    x1 = torch.randn(1, 4, 3, 8, 8, device=device, dtype=dtype)
    with torch.inference_mode():
        eng_nocg(_mk(x0, t0, {"context": ctx1, "kv_cache": kv_a, "current_start": 0}))
        den_ref(_mk(x0, t0, {"context": ctx1, "kv_cache": kv_b, "current_start": 0}))
        for arena in eng_nocg._roped_k_arenas.values():
            for buf in arena:
                buf.fill_(float("nan"))
        eng_nocg._roped_k_arenas.clear()
        oa = eng_nocg(_mk(x1, t0, {"context": ctx1, "kv_cache": kv_a, "current_start": 48})).sample
        ob = den_ref(_mk(x1, t0, {"context": ctx1, "kv_cache": kv_b, "current_start": 48})).sample
    assert torch.equal(oa, ob)

    # 6. Broadcasted B>1 modulation, mixed-dtype rounding, FP32 RMSNorm weight, and FP32 FA3 guard
    with torch.inference_mode():
        xm = torch.randn(2, 48, 256, device=device, dtype=dtype)
        ym = torch.randn_like(xm)
        sc = torch.randn(1, 3, 1, 256, device=device, dtype=dtype)
        sh = torch.randn(1, 3, 1, 256, device=device, dtype=torch.float32)
        ref_ln = (
            torch.nn.functional.layer_norm(xm, (256,), eps=1e-6).type_as(xm).unflatten(1, (3, 16)) * (1 + sc) + sh
        ).flatten(1, 2)
        assert torch.equal(wan_layernorm_adaln(xm, sc, sh, num_frames=3, frame_seq=16, eps=1e-6, exact=True), ref_ln)
        with torch.autocast("cuda", dtype=dtype):
            sc_bf = torch.randn(2, 3, 1, 256, device=device, dtype=dtype)
            sh_bf = torch.randn(2, 3, 1, 256, device=device, dtype=dtype)
            ref_ln_ac = (
                torch.nn.functional.layer_norm(xm, (256,), eps=1e-6).type_as(xm).unflatten(1, (3, 16)) * (1 + sc_bf)
                + sh_bf
            ).flatten(1, 2)
            assert torch.equal(
                wan_layernorm_adaln(xm, sc_bf, sh_bf, num_frames=3, frame_seq=16, eps=1e-6, exact=True), ref_ln_ac
            )

        g_b = torch.randn(2, 1, 1, 256, device=device, dtype=dtype)
        assert torch.equal(
            wan_residual_gate_add(xm.float(), ym, g_b, num_frames=3, frame_seq=16),
            xm.float() + (ym.unflatten(1, (3, 16)) * g_b).flatten(1, 2),
        )
        w32 = torch.randn(256, device=device, dtype=torch.float32)
        rms_ref = (xm.float() * torch.rsqrt(xm.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)).type_as(xm) * w32
        rms_got = wan_rmsnorm(xm, w32, eps=1e-6, exact=True)
        assert rms_got.dtype == torch.float32 and torch.equal(rms_got, rms_ref)

        q32 = torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32)
        o32 = fast_longlive_self_attention(q32, q32, q32, backend="flash_attention_3")
        assert o32.dtype == torch.float32 and torch.isfinite(o32).all()


@requires_cuda
def test_matrix_game2_wave2_edge_cases_and_steady_vae_compile() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_conditioning_21 import (
        ActionModule,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.causal_action_21 import (
        CausalWanModel,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.fast_kernels import (
        mg2_fast_action_module_forward,
        mg2_fast_steady_forward,
        mg2_layernorm_adaln,
        mg2_main_self_attention,
        mg2_residual_gate_add,
        mg2_rmsnorm,
        precompute_action_inputs,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.vae_runtime.vae_block3 import (
        VAEDecoderWrapper,
    )
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.utils.wan_wrapper import (
        WanDiffusionWrapper,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(202)

    with torch.inference_mode():
        # 1. FP32 FA3 guard + B=2 broadcasted modulation + mixed-dtype rounding + FP32 RMSNorm weight
        q32 = torch.randn(1, 16, 4, 64, device=device, dtype=torch.float32)
        o32 = mg2_main_self_attention(q32, q32, q32, backend="flash_attention_3")
        assert o32.dtype == torch.float32 and torch.isfinite(o32).all()

        xm = torch.randn(2, 48, 256, device=device, dtype=dtype)
        ym = torch.randn_like(xm)
        sc = torch.randn(1, 3, 1, 256, device=device, dtype=dtype)
        sh = torch.randn(1, 3, 1, 256, device=device, dtype=torch.float32)
        ref_ln = (
            torch.nn.functional.layer_norm(xm, (256,), eps=1e-6).type_as(xm).unflatten(1, (3, 16)) * (1 + sc) + sh
        ).flatten(1, 2)
        assert torch.equal(mg2_layernorm_adaln(xm, sc, sh, num_frames=3, frame_seq=16, eps=1e-6), ref_ln)
        with torch.autocast("cuda", dtype=dtype):
            sc_bf = torch.randn(2, 3, 1, 256, device=device, dtype=dtype)
            sh_bf = torch.randn(2, 3, 1, 256, device=device, dtype=dtype)
            ref_ln_ac = (
                torch.nn.functional.layer_norm(xm, (256,), eps=1e-6).type_as(xm).unflatten(1, (3, 16)) * (1 + sc_bf)
                + sh_bf
            ).flatten(1, 2)
            assert torch.equal(mg2_layernorm_adaln(xm, sc_bf, sh_bf, num_frames=3, frame_seq=16, eps=1e-6), ref_ln_ac)
        g_b = torch.randn(2, 1, 1, 256, device=device, dtype=dtype)
        assert torch.equal(
            mg2_residual_gate_add(xm.float(), ym, g_b, num_frames=3, frame_seq=16),
            xm.float() + (ym.unflatten(1, (3, 16)) * g_b).flatten(1, 2),
        )
        w32 = torch.randn(256, device=device, dtype=torch.float32)
        assert mg2_rmsnorm(xm, w32, eps=1e-6, exact=True).dtype == torch.float32

        # 2. ActionModule qk_norm=False must match exact path when exact_mouse_sdpa=False
        am = (
            ActionModule(
                mouse_dim_in=2,
                keyboard_dim_in=4,
                hidden_size=32,
                img_hidden_size=128,
                keyboard_hidden_dim=128,
                mouse_hidden_dim=128,
                heads_num=2,
                qk_norm=False,
                enable_mouse=True,
                enable_keyboard=True,
                local_attn_size=6,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        B, T, S = 1, 3, 16
        xa = torch.randn(B, T * S, 128, device=device, dtype=dtype)
        mouse = torch.randn(B, 9, 2, device=device, dtype=dtype)
        kb = torch.randn(B, 9, 4, device=device, dtype=dtype)
        gm, kbs = precompute_action_inputs(am, mouse_cond=mouse, keyboard_cond=kb, num_frames=T, spatial_s=S)
        kv_m1 = {
            "k": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
            "v": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
        }
        kv_k1 = {
            "k": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
            "v": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
        }
        kv_m2 = {"k": torch.zeros_like(kv_m1["k"]), "v": torch.zeros_like(kv_m1["v"])}
        kv_k2 = {"k": torch.zeros_like(kv_k1["k"]), "v": torch.zeros_like(kv_k1["v"])}
        o_ex = mg2_fast_action_module_forward(
            am,
            xa,
            num_frames=T,
            spatial_s=S,
            group_mouse_pre=gm,
            kb_cond_slice=kbs,
            kv_cache_mouse=kv_m1,
            kv_cache_keyboard=kv_k1,
            action_local_start=0,
            action_local_end=3,
            action_k_start=0,
            start_frame=0,
            exact_mouse_sdpa=True,
        )
        o_fa = mg2_fast_action_module_forward(
            am,
            xa,
            num_frames=T,
            spatial_s=S,
            group_mouse_pre=gm,
            kb_cond_slice=kbs,
            kv_cache_mouse=kv_m2,
            kv_cache_keyboard=kv_k2,
            action_local_start=0,
            action_local_end=3,
            action_k_start=0,
            start_frame=0,
            exact_mouse_sdpa=False,
        )
        assert torch.equal(o_ex, o_fa)

        # 3. local_attn_size=-1 (global attention) must not slice negatively or fail non-contiguous .view()
        m_global = (
            CausalWanModel(
                model_type="i2v",
                patch_size=(1, 2, 2),
                in_dim=20,
                dim=128,
                ffn_dim=256,
                freq_dim=64,
                text_dim=32,
                out_dim=16,
                num_heads=2,
                num_layers=1,
                local_attn_size=-1,
                sink_size=0,
                qk_norm=True,
                cross_attn_norm=True,
                action_config={
                    "blocks": [0],
                    "mouse_dim_in": 2,
                    "keyboard_dim_in": 4,
                    "hidden_size": 32,
                    "img_hidden_size": 128,
                    "keyboard_hidden_dim": 128,
                    "mouse_hidden_dim": 128,
                    "heads_num": 2,
                    "enable_mouse": True,
                    "enable_keyboard": True,
                },
            )
            .to(device=device, dtype=dtype)
            .eval()
        )
        m_global.freqs = m_global.freqs.to(device)
        xv = torch.randn(B, 16, T, 8, 8, device=device, dtype=dtype)
        cc = torch.randn(B, 4, T, 8, 8, device=device, dtype=dtype)
        t_in = torch.full((B, T), 500.0, device=device, dtype=torch.float32)
        kv_g = [
            {
                "k": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
                "v": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
            }
        ]
        ca_g = [
            {
                "k": torch.randn(B, 8, 2, 64, device=device, dtype=dtype),
                "v": torch.randn(B, 8, 2, 64, device=device, dtype=dtype),
                "is_init": True,
            }
        ]
        s_freqs = torch.pow(10000.0, -torch.arange(32, device=device, dtype=torch.float64).div(32))
        out_g = mg2_fast_steady_forward(
            m_global,
            xv,
            t_in,
            cc,
            kb,
            mouse,
            sinusoid_freqs=s_freqs,
            kv_cache=kv_g,
            kv_cache_mouse=[kv_m1],
            kv_cache_keyboard=[kv_k1],
            crossattn_cache=ca_g,
            local_start_index=0,
            local_end_index=T * S,
            action_local_start=0,
            action_local_end=T,
            start_frame=0,
            attention_backend="torch",
            exact_norms=True,
        )
        assert torch.isfinite(out_g).all()

    # 4. WanDiffusionWrapper fast warmup updates GPU cache indices when host mirror keys are absent
    cfg = {
        "model_type": "i2v",
        "patch_size": (1, 2, 2),
        "in_dim": 20,
        "dim": 128,
        "ffn_dim": 256,
        "freq_dim": 64,
        "text_dim": 32,
        "out_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "local_attn_size": 6,
        "sink_size": 0,
        "qk_norm": True,
        "cross_attn_norm": True,
        "action_config": {
            "blocks": [0],
            "mouse_dim_in": 2,
            "keyboard_dim_in": 4,
            "hidden_size": 32,
            "img_hidden_size": 128,
            "keyboard_hidden_dim": 128,
            "mouse_hidden_dim": 128,
            "heads_num": 2,
            "enable_mouse": True,
            "enable_keyboard": True,
        },
    }
    wrapper = WanDiffusionWrapper(model_config=cfg, is_causal=True).to(device=device, dtype=dtype).eval()
    wrapper.enable_block_cuda_graph(warmup=1)
    kv_w = [
        {
            "k": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
            "v": torch.zeros(B, 6 * S, 2, 64, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
    ]
    kvm_w = [
        {
            "k": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
            "v": torch.zeros(B * S, 6, 2, 64, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
    ]
    kvk_w = [
        {
            "k": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
            "v": torch.zeros(B, 6, 2, 64, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
    ]
    ca_w: list[dict[str, Any]] = [{"is_init": False}]
    cond_w = {
        "cond_concat": cc,
        "visual_context": torch.randn(B, 8, 1280, device=device, dtype=dtype),
        "keyboard_cond": kb,
        "mouse_cond": mouse,
    }
    with torch.inference_mode():
        wrapper(
            xv,
            cond_w,
            t_in,
            kv_cache=kv_w,
            kv_cache_mouse=kvm_w,
            kv_cache_keyboard=kvk_w,
            crossattn_cache=ca_w,
            current_start=0,
        )
    assert kv_w[0]["global_end_index"].item() == T * S and kv_w[0]["local_end_index"].item() == T * S
    ca_w[0]["k"].zero_()

    # 5. VAEDecoderWrapper.enable_steady_compile() numerical parity across chunk 0 -> steady chunks
    vae_ref = VAEDecoderWrapper().to(device=device, dtype=torch.float16).eval()
    vae_comp = VAEDecoderWrapper().to(device=device, dtype=torch.float16).eval()
    vae_comp.load_state_dict(vae_ref.state_dict())
    vae_comp.enable_steady_compile(mode="default")
    z0 = torch.randn(1, 3, 16, 8, 8, device=device, dtype=torch.float16)
    z1 = torch.randn(1, 3, 16, 8, 8, device=device, dtype=torch.float16)
    with torch.inference_mode():
        ce: list[Any] = [None] * 34
        cc_vae: list[Any] = [None] * 34
        _, ce = vae_ref(z0, *ce)
        _, cc_vae = vae_comp(z0, *cc_vae)
        o1_e, ce = vae_ref(z1, *ce)
        o1_c, cc_vae = vae_comp(z1, *cc_vae)
        o2_e, _ = vae_ref(z1, *ce)
        o2_c, _ = vae_comp(z1, *cc_vae)
    assert torch.allclose(o1_c, o1_e, atol=1e-2, rtol=1e-2)
    assert torch.allclose(o2_c, o2_e, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_hy_worldplay_wave2_edge_cases_and_h15_double_block_fa3() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.action_camera import (
        MMDoubleStreamBlock,
        _fused_qkv_rmsnorm_rope_prope,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_attention import (
        prepare_static_kv_arena_bshd,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.camera_modulation import (
        fused_layernorm_modulate,
        fused_residual_gate,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.h15.hunyuanvideo_1_5_transformer import (
        MMDoubleStreamBlock as H15DoubleBlock,
    )
    from worldfoundry.core.attention import get_nd_rotary_pos_embed

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(203)

    # 1. qk_norm_type="layer" must not invoke RMSNorm-only fused QKV kernel after fuse_qkv()
    blk = (
        MMDoubleStreamBlock(hidden_size=128, heads_num=2, mlp_width_ratio=2.0, qk_norm=True, qk_norm_type="layer")
        .to(device=device, dtype=dtype)
        .eval()
    )
    assert isinstance(blk.img_attn_q_norm, nn.LayerNorm) and isinstance(blk.img_attn_k_norm, nn.LayerNorm)
    assert blk.img_attn_q_norm.bias is not None and blk.img_attn_k_norm.bias is not None
    blk.img_attn_q_norm.bias.data.fill_(2.5)
    blk.img_attn_k_norm.bias.data.fill_(-1.5)
    img = torch.randn(1, 32, 128, device=device, dtype=dtype) + 3.0
    vec = torch.randn(2, 128, device=device, dtype=dtype)
    vm = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    ks = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    pm = _precompute_prope_mats(vm, ks)
    fcos, fsin = get_nd_rotary_pos_embed([16, 24, 24], [2, 4, 4], theta=256, use_real=True, theta_rescale_factor=1)
    fc = (fcos.to(device), fsin.to(device))
    with torch.inference_mode():
        ref = blk._img_qkv_rope_prope(vec, img, fc, pm, vm, ks)
        blk.fuse_qkv()
        fused = blk._img_qkv_rope_prope(vec, img, fc, pm, vm, ks)
    assert torch.allclose(fused[0], ref[0], atol=2e-2, rtol=2e-2)

    # 2. _fused_qkv_rmsnorm_rope_prope rejects mismatched batch p_mat (no OOB GPU read)
    qkv = torch.randn(2, 32, 384, device=device, dtype=dtype)
    qw = torch.ones(64, device=device, dtype=dtype)
    with torch.inference_mode():
        assert _fused_qkv_rmsnorm_rope_prope(qkv, qw, qw, fc, pm[0], pm[1], heads_num=2) is None

    # 3. camera_modulation handles B=2 with broadcast row-count=1 without ZeroDivisionError
    x2 = torch.randn(2, 16, 128, device=device, dtype=dtype)
    sc1 = torch.randn(1, 128, device=device, dtype=dtype)
    sh1 = torch.randn(1, 128, device=device, dtype=dtype)
    with torch.inference_mode():
        assert fused_layernorm_modulate(x2, sh1, sc1).shape == x2.shape
        assert fused_residual_gate(x2, x2, sc1).shape == x2.shape

    # 4. prepare_static_kv_arena_bshd allocates non-inference tensors safe across inference_mode -> no_grad
    arena: dict[tuple[Any, ...], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    kv_a = [
        {
            "k_txt": torch.randn(1, 2, 8, 64, device=device, dtype=dtype),
            "v_txt": torch.randn(1, 2, 8, 64, device=device, dtype=dtype),
        }
    ]
    with torch.inference_mode():
        bufs, _, _ = prepare_static_kv_arena_bshd(kv_a, 16, arena=arena, dtype=dtype)
    bufs[0][0].zero_()

    # 5. HunyuanVideo_1_5_DiffusionTransformer MMDoubleStreamBlock FA3 parity (masked & unmasked, B=2)
    h15_blk = (
        H15DoubleBlock(hidden_size=128, heads_num=2, mlp_width_ratio=2.0, attn_mode="torch")
        .to(device=device, dtype=dtype)
        .eval()
    )
    img2 = torch.randn(2, 32, 128, device=device, dtype=dtype)
    txt2 = torch.randn(2, 10, 128, device=device, dtype=dtype)
    vec2 = torch.randn(2, 128, device=device, dtype=dtype)
    mask2 = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], device=device, dtype=torch.bool
    )
    with torch.inference_mode():
        h15_blk.set_attention_backend("torch")
        ri_m, rt_m = h15_blk(img2, txt2, vec2, freqs_cis=fc, text_mask=mask2)
        ri_0, rt_0 = h15_blk(img2, txt2, vec2, freqs_cis=fc, text_mask=None)
        h15_blk.set_attention_backend("flash_attention_3")
        fi_m, ft_m = h15_blk(img2, txt2, vec2, freqs_cis=fc, text_mask=mask2)
        fi_0, ft_0 = h15_blk(img2, txt2, vec2, freqs_cis=fc, text_mask=None)
    assert torch.allclose(fi_m, ri_m, atol=5e-2, rtol=2e-2)
    assert torch.allclose(ft_m, rt_m, atol=5e-2, rtol=2e-2)
    assert torch.allclose(fi_0, ri_0, atol=5e-2, rtol=2e-2)
    assert torch.allclose(ft_0, rt_0, atol=5e-2, rtol=2e-2)
