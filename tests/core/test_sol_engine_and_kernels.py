# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sol-Engine 5-pillar stack integration, scheduled 3D-tiled Sol-Attn,
token pruning, padded Ulysses, and real-valued RoPE / tiled FP8 kernels."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch

from worldfoundry.core.acceleration.profiles import resolve_acceleration_profile
from worldfoundry.core.acceleration.sol_engine import (
    ExecutionPillar,
    ModelHardwareProfile,
    QualityBudget,
    SolEngineStack,
    SparseAttentionPillar,
    compose_sol_stack,
)
from worldfoundry.core.attention.backends.sol import validate_sol_options
from worldfoundry.core.kernels.diffusion import apply_real_rope_bshd


def test_control_plane_modules_do_not_import_torch() -> None:
    """Pure-Python control-plane specs must have zero top-level or transitive torch imports."""
    import ast
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for rel in (
        "worldfoundry/core/acceleration/profiles.py",
        "worldfoundry/core/acceleration/sol_engine.py",
        "worldfoundry/core/nn/transformer/contracts.py",
    ):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "torch" and not alias.name.startswith("torch.") for alias in node.names), rel
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module != "torch" and not node.module.startswith("torch."), rel

    # Live subprocess proof that core/acceleration/__init__.py lazy __getattr__ leaks zero torch
    code = (
        "import sys; "
        "import worldfoundry.core.acceleration.profiles; "
        "import worldfoundry.core.acceleration.sol_engine; "
        "assert 'torch' not in sys.modules, 'torch leaked through core.acceleration'"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_quality_budget_tiers_and_acceptance() -> None:
    lossless = QualityBudget.for_tier("lossless")
    balanced = QualityBudget.for_tier("balanced")

    assert lossless.accepts({"reference": True}) is True
    assert lossless.accepts({"finite": True, "video_psnr_db": "infinity", "video_ssim": 1.0, "latent_relative_l2": 0.0})
    assert not lossless.accepts({})
    assert not lossless.accepts({"finite": False, "video_psnr_db": 60.0, "video_ssim": 1.0, "latent_relative_l2": 0.0})
    assert balanced.accepts({"finite": True, "video_psnr_db": 29.0, "video_ssim": 0.92, "latent_relative_l2": 0.18})
    assert not balanced.accepts({"finite": True, "video_psnr_db": 24.0, "video_ssim": 0.92, "latent_relative_l2": 0.18})
    assert not balanced.accepts({"finite": True, "video_psnr_db": 32.0, "video_ssim": 0.85, "latent_relative_l2": 0.18})
    assert not balanced.accepts({"finite": True, "video_psnr_db": 32.0, "video_ssim": 0.95, "latent_relative_l2": 0.40})

    with pytest.raises(ValueError, match="unsupported quality budget tier"):
        QualityBudget.for_tier("unknown-tier")


def test_compose_sol_stack_conflict_resolution_and_recipe_roundtrip() -> None:
    hw_14b = ModelHardwareProfile(
        model_id="wan2.1-t2v-14b",
        num_heads=40,
        head_dim=128,
        num_layers=40,
        world_size=1,
        sm_major=9,
        available_attention_backends=("torch", "sol_attn"),
    )
    stack = compose_sol_stack(hw_14b, budget="balanced")
    kwargs = stack.to_pipeline_kwargs()

    assert kwargs["cuda_graph"] == "piecewise"
    assert kwargs["inplace_residual"] is False
    assert kwargs["teacache"] == pytest.approx(0.15)
    assert kwargs["quantization"]["mode"] == "fp8"
    assert "cross_attn" in kwargs["quantization"]["exclude"]
    assert kwargs["quantization"]["options"]["fuse_fp8_ffn"] is True
    assert kwargs["quantization"]["options"]["use_fast_accum"] is False
    sol_cfg = kwargs["accelerations"]["attention_policy"]["self"]
    assert sol_cfg["backend"] == "sol_attn"
    assert sol_cfg["dense_steps"] == 10
    assert sol_cfg["dense_blocks"] == (0, 1, -1)
    assert sol_cfg["tile_3d"] == (1, 8, 8)

    # Full JSON recipe round-trip
    recipe_json = json.dumps(stack.to_recipe_dict())
    reconstructed = SolEngineStack.from_recipe_dict(json.loads(recipe_json))
    assert reconstructed.to_pipeline_kwargs() == kwargs

    # Multi-GPU SP > 1 must disable CUDA Graph cleanly and record a composition note
    hw_4gpu = ModelHardwareProfile(
        model_id="wan2.1-t2v-14b",
        num_heads=40,
        head_dim=128,
        num_layers=40,
        world_size=4,
        sm_major=9,
        available_attention_backends=("torch", "sol_attn"),
    )
    sp_stack = compose_sol_stack(
        hw_4gpu,
        budget="balanced",
        execution=ExecutionPillar(cuda_graph="piecewise"),
    )
    assert sp_stack.execution.cuda_graph == "off"
    assert any("cuda_graph" in note.lower() or "ulysses" in note.lower() for note in sp_stack.composition_notes)
    sp_kwargs = sp_stack.to_pipeline_kwargs()
    assert sp_kwargs["cfg_parallel"] == 2
    assert sp_kwargs["sequence_parallel"] == 2
    assert "cuda_graph" not in sp_kwargs

    # Fail-closed on incompatible explicit Sol-Attn head_dim or compile
    hw_hd64 = ModelHardwareProfile(
        model_id="tiny-dit",
        num_heads=12,
        head_dim=64,
        num_layers=12,
        available_attention_backends=("torch", "sol_attn"),
    )
    with pytest.raises(ValueError, match="head_dim=128"):
        compose_sol_stack(
            hw_hd64,
            sparse_attention=SparseAttentionPillar(backend="sol_attn", tau=1.0),
        )
    with pytest.raises(ValueError, match="compile"):
        compose_sol_stack(
            hw_14b,
            sparse_attention=SparseAttentionPillar(backend="sol_attn", tau=1.0),
            execution=ExecutionPillar(compile=True, cuda_graph="piecewise"),
        )


def test_profiles_sol_realtime_and_padded_ulysses_flags() -> None:
    sol = resolve_acceleration_profile("sol", world_size=1, num_heads=12, sm_major=9)
    sol_self = sol.options["accelerations"]["attention_policy"]["self"]
    assert sol_self["backend"] == "sol_attn"
    assert sol_self["dense_steps"] == 10
    assert sol_self["dense_blocks"] == (0, 1, -1)
    assert sol_self["tile_3d"] == (1, 8, 8)
    assert sol.options["quantization"]["options"]["use_fast_accum"] is False
    assert "cross_attn" in sol.options["quantization"]["exclude"]

    rt = resolve_acceleration_profile("realtime", world_size=1)
    assert rt.options["rope_precision"] == "fp64"
    assert rt.options["vae_channels_last_3d"] is True
    assert rt.options["vae_streaming_compile"] is True

    padded = resolve_acceleration_profile("exact", world_size=4, num_heads=7, supports_cfg=False)
    assert padded.options["sequence_parallel"] == 4
    assert padded.options["sequence_parallel_pad_heads"] is True


def test_validate_sol_options_sink_tokens() -> None:
    default = validate_sol_options(tau=1.0)
    assert "sink_tokens" not in default
    assert "sink_start" not in default
    with_sink = validate_sol_options(tau=0.5, sink_tokens=128, sink_start=64)
    assert with_sink["sink_tokens"] == 128
    assert with_sink["sink_start"] == 64
    multi = validate_sol_options(tau=1.0, kv_splits=4, allow_multi_split=True)
    assert multi["kv_splits"] == 4
    with pytest.raises(ValueError, match="kv_splits=1 only"):
        validate_sol_options(kv_splits=2)
    with pytest.raises(ValueError, match="sink_tokens"):
        validate_sol_options(sink_tokens=-1)
    with pytest.raises(ValueError, match="sink_tokens"):
        validate_sol_options(sink_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sink_start"):
        validate_sol_options(sink_start=-1)


def test_allow_approximate_composition_unlocks_multi_pillar_plugins(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
    from worldfoundry.core.attention.backends import probe
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda req, dev=None: req)
    model = WanModel(
        dim=256,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=4,
        has_image_input=False,
    ).eval()
    policy = RuntimePolicy(
        dtype=torch.bfloat16,
        options={"allow_approximate_composition": True, "cuda_graph": "piecewise"},
    )
    session = install_diffusion_accelerations(
        model,
        {
            "easycache": {"threshold": 0.1},
            "token_pruning": {"keep_ratio": 0.75, "start_block": 1, "end_block": -1},
            "attention_policy": {"self": {"backend": "sol_attn", "tau": 1.0}},
        },
        policy=policy,
    )
    assert hasattr(model, "_worldfoundry_easycache_config")
    assert hasattr(model, "_worldfoundry_token_pruning_config")
    assert cast(Any, model.blocks[0]).self_attn.attn.attention_backend == "sol_attn"
    session.uninstall()
    assert not hasattr(model, "_worldfoundry_easycache_config")
    assert not hasattr(model, "_worldfoundry_token_pruning_config")


def test_multimodal_sequence_routes_explicit_backends_through_dispatch(monkeypatch) -> None:
    from worldfoundry.core.attention.backends import dispatch
    from worldfoundry.core.attention.sequence import multimodal_sequence as mm_seq

    seen: list[str | None] = []
    monkeypatch.setattr(mm_seq, "resolve_attention_backend", lambda req, dev=None, **kw: req)

    def fake_forward(q, k, v, **kwargs):
        seen.append(kwargs.get("backend"))
        return q

    monkeypatch.setattr(dispatch, "attention_forward", fake_forward)
    q = torch.randn(1, 8, 2, 128)
    out = mm_seq.attention(q, q, q, attn_mode="sol_attn")
    assert out.shape == (1, 8, 256)
    out_sp = mm_seq.sequence_parallel_attention(
        (q, q[:, :4]), (q, q[:, :4]), (q, q[:, :4]), 8, 8, attn_mode="cudnn_fp8"
    )
    assert out_sp.shape == (1, 12, 256)
    assert seen == ["sol_attn", "cudnn_fp8"]


def test_wan_scheduled_sol_attn_and_whole_stack_3d_tile_equivariance(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import (
        install_diffusion_accelerations,
    )
    from worldfoundry.core.attention.backends import probe

    torch.manual_seed(0)
    model = WanModel(
        dim=256,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,  # head_dim = 128
        num_layers=4,
        has_image_input=False,
    ).eval()

    # 1. Verify plugin installs dense_blocks=(0, -1) as 'torch' and middle blocks as 'sol_attn',
    #    attaches _worldfoundry_sol_schedule, and restores cleanly on undo().
    monkeypatch.setattr(probe, "resolve_attention_backend", lambda req, dev=None: req)
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    session = install_diffusion_accelerations(
        model,
        {
            "attention_policy": {
                "self": {
                    "backend": "sol_attn",
                    "tau": 0.75,
                    "dense_steps": 4,
                    "dense_blocks": (0, -1),
                    "tile_3d": (1, 2, 2),
                }
            }
        },
        policy=RuntimePolicy(dtype=torch.bfloat16),
    )
    blocks_any: Any = model.blocks
    assert blocks_any[0].self_attn.attn.attention_backend == "torch"
    assert blocks_any[1].self_attn.attn.attention_backend == "sol_attn"
    assert blocks_any[2].self_attn.attn.attention_backend == "sol_attn"
    assert blocks_any[3].self_attn.attn.attention_backend == "torch"
    schedule_any: Any = getattr(model, "_worldfoundry_sol_schedule")
    assert schedule_any["dense_steps"] == 4
    assert schedule_any["tile_3d"] == (1, 2, 2)
    session.uninstall()
    assert not hasattr(model, "_worldfoundry_sol_schedule")
    assert all(cast(Any, b).self_attn.attn.attention_backend is None for b in model.blocks)

    # 2. Verify whole-stack 3D tile permutation + RoPE frequency permutation + unpermutation
    #    is mathematically exact (permutation-equivariant) when attention is dense.
    x = torch.randn(1, 4, 2, 6, 10)  # post-patchify grid: F=2, H=3, W=5 (non-divisible by 2x2!)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 8, 64)
    with torch.no_grad():
        ref_out = model(x, t, ctx)
        cast(Any, model)._worldfoundry_sol_schedule = {"dense_steps": 0, "dense_blocks": (), "tile_3d": (1, 2, 2)}
        try:
            tiled_out = model(x, t, ctx, _worldfoundry_sol_step_index=5)
        finally:
            delattr(model, "_worldfoundry_sol_schedule")
    assert torch.allclose(tiled_out, ref_out, atol=1e-5, rtol=1e-4)

    # 3. Verify token_pruning plugin installs, prunes middle blocks, and closes cleanly.
    tp_session = install_diffusion_accelerations(
        model,
        {
            "token_pruning": {
                "keep_ratio": 0.75,
                "method": "feat_norm",
                "start_block": 1,
                "end_block": -1,
                "dense_first_steps": 0,
                "dense_last_steps": 0,
            }
        },
    )
    with torch.no_grad():
        # Call 1 populates previous-step compensation reference; call 2 executes pruned middle blocks.
        _ = model(x, t, ctx, _worldfoundry_token_pruning_step=(2, 20, "req1", "positive"))
        pruned_out = model(x, t, ctx, _worldfoundry_token_pruning_step=(3, 20, "req1", "positive"))
    assert pruned_out.shape == ref_out.shape
    assert torch.isfinite(pruned_out).all()
    assert model._worldfoundry_token_pruning_runtime["dense_calls"] >= 1
    assert model._worldfoundry_token_pruning_runtime["pruned_calls"] == 1
    tp_session.uninstall()
    assert not hasattr(model, "_worldfoundry_token_pruning_config")


def test_apply_real_rope_bshd_cpu_matches_complex_reference() -> None:
    torch.manual_seed(1)
    B, S, H, D = 1, 16, 4, 32
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    angles = torch.randn(S, D // 2, dtype=torch.float64)
    cos = torch.cos(angles).view(1, S, 1, D // 2)
    sin = torch.sin(angles).view(1, S, 1, D // 2)
    out = apply_real_rope_bshd(x, cos, sin)
    # Reference complex multiplication in FP64
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(B, S, H, D // 2, 2))
    freqs_c = torch.complex(cos, sin)
    ref = torch.view_as_real(x_c * freqs_c).flatten(-2).to(x.dtype)
    assert torch.equal(out, ref)


def test_padded_ulysses_requires_explicit_opt_in() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention
    from worldfoundry.base_models.diffusion_model.optimizations.sequence_parallel import enable_sequence_parallel

    class _Wrap(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = SelfAttention(dim=96, num_heads=6)

    with pytest.raises(ValueError, match="not divisible"):
        enable_sequence_parallel(_Wrap(), sp_degree=4)
    state = enable_sequence_parallel(_Wrap(), sp_degree=4, allow_head_padding=True)
    assert state.backend == "native-padded-ulysses"
    assert state.wrapped_blocks == 1


def test_piecewise_cuda_graph_and_off_string_handling() -> None:
    from worldfoundry.base_models.diffusion_model.loaders.module import _cuda_graph_enabled
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import (
        _cuda_graph_blocks_dynamic_plugin,
        install_diffusion_accelerations,
    )
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    for disabled in (None, False, "", "off", "OFF", "none", "false", "disabled"):
        assert not _cuda_graph_enabled(disabled)
        assert not _cuda_graph_blocks_dynamic_plugin(disabled)
    assert _cuda_graph_enabled("piecewise")
    assert not _cuda_graph_blocks_dynamic_plugin("piecewise")
    assert _cuda_graph_enabled("whole")
    assert _cuda_graph_blocks_dynamic_plugin("whole")
    assert _cuda_graph_blocks_dynamic_plugin(True)

    model = WanModel(
        dim=256,
        ffn_dim=512,
        in_dim=4,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=4,
        has_image_input=False,
    ).eval()

    # Piecewise CUDA Graph must allow token_pruning, while whole CUDA Graph must reject it.
    session = install_diffusion_accelerations(
        model,
        {"token_pruning": {"keep_ratio": 0.75, "start_block": 1, "end_block": -1}},
        policy=RuntimePolicy(dtype=torch.bfloat16, options={"cuda_graph": "piecewise"}),
    )
    session.uninstall()
    with pytest.raises(ValueError, match="token_pruning requires cuda_graph='piecewise' or 'off'"):
        install_diffusion_accelerations(
            model,
            {"token_pruning": {"keep_ratio": 0.75, "start_block": 1, "end_block": -1}},
            policy=RuntimePolicy(dtype=torch.bfloat16, options={"cuda_graph": "whole"}),
        )


def test_profiles_use_exact_fp8_accum_and_scheduled_sol_attn() -> None:
    for tier in ("fast", "turbo", "sol"):
        spec = resolve_acceleration_profile(tier, world_size=1, sm_major=9)
        qcfg = spec.options["quantization"]
        assert qcfg["mode"] == "fp8"
        assert "cross_attn" in qcfg["exclude"]
        assert qcfg["options"]["use_fast_accum"] is False
        assert qcfg["options"]["keep_dense_fallback"] is False
        assert qcfg["options"]["fuse_fp8_ffn"] is True
    sol_spec = resolve_acceleration_profile("sol", world_size=1, sm_major=9)
    sol_attn_cfg = sol_spec.options["accelerations"]["attention_policy"]["self"]
    assert sol_attn_cfg["backend"] == "sol_attn"
    assert sol_attn_cfg["dense_steps"] >= 8
    assert sol_attn_cfg["dense_blocks"] == (0, 1, -1)
    assert sol_attn_cfg["tile_3d"] == (1, 8, 8)
    assert sol_spec.options["cuda_graph"] == "piecewise"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_apply_real_rope_and_tiled_fp8_parity() -> None:
    device = torch.device("cuda")
    torch.manual_seed(7)

    # 1. Triton FP64 Real-RoPE custom_op parity
    B, S, H, D = 1, 1024, 12, 128
    x = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    angles = torch.randn(S, D // 2, device=device, dtype=torch.float64)
    cos = torch.cos(angles).view(1, S, 1, D // 2)
    sin = torch.sin(angles).view(1, S, 1, D // 2)
    with torch.inference_mode():
        out = apply_real_rope_bshd(x, cos, sin)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(B, S, H, D // 2, 2))
    ref = torch.view_as_real(x_c * torch.complex(cos, sin)).flatten(-2).to(x.dtype)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2)

    # 2. Tiled 2-pass FP8 quantization kernel parity for cols > 4096 (Wan 1.3B 8960 & 14B 13824)
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton

    for cols in (8960, 13824):
        inp = torch.randn(256, cols, device=device, dtype=torch.bfloat16)
        for act in ("identity", "gelu_tanh"):
            q, s = quantize_rowwise_fp8_triton(inp, torch.float8_e4m3fn, activation=act)
            ref_in = inp.float() if act == "identity" else torch.nn.functional.gelu(inp.float(), approximate="tanh")
            fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
            ref_s = (ref_in.abs().amax(dim=1, keepdim=True) / fp8_max).clamp_min(1e-12)
            assert torch.allclose(s.float(), ref_s, rtol=1e-2, atol=1e-4)
            deq = q.float() * s.float()
            rel_err = (deq - ref_in).norm() / ref_in.norm()
            assert rel_err < 0.03

    # 3. Batched cos/sin (B > 1) and empty sequences in apply_real_rope_bshd must match CPU fallback
    x_batched = torch.randn(2, 16, 4, 64, device=device, dtype=torch.bfloat16)
    angles_b = torch.randn(2, 16, 1, 32, device=device, dtype=torch.float64)
    cos_b, sin_b = torch.cos(angles_b), torch.sin(angles_b)
    with torch.inference_mode():
        out_b = apply_real_rope_bshd(x_batched, cos_b, sin_b)
        out_empty = apply_real_rope_bshd(x_batched[:, :0], cos_b[:, :0], sin_b[:, :0])
    x_bc = torch.view_as_complex(x_batched.to(torch.float64).reshape(2, 16, 4, 32, 2))
    ref_b = torch.view_as_real(x_bc * torch.complex(cos_b, sin_b)).flatten(-2).to(x_batched.dtype)
    assert torch.equal(out_b, ref_b)
    assert out_empty.shape == (2, 0, 4, 64)

    # 4. FP64 Triton hidden_qk_rmsnorm_rope_3d dispatches to Triton and matches complex128 reference
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
        SelfAttention,
        WanModel,
        rope_apply,
    )
    from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_3d
    from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope

    f, h, w = 2, 8, 8
    dim, heads = 5120, 40
    attn = SelfAttention(dim=dim, num_heads=heads, eps=1e-6).to(device=device, dtype=torch.bfloat16).eval()
    wan = (
        WanModel(
            dim=dim,
            in_dim=16,
            ffn_dim=dim * 2,
            out_dim=16,
            text_dim=64,
            freq_dim=32,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=heads,
            num_layers=0,
            has_image_input=False,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    freqs_3d = (
        torch.cat(
            [
                wan.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                wan.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wan.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(device)
    )
    fp64_table = wan.fused_rope_table(device=device, precision="fp64")
    qkv = torch.randn(1, f * h * w, 3 * dim, device=device, dtype=torch.bfloat16)
    q_in, k_in, _ = qkv.chunk(3, dim=-1)
    receipt: dict[str, object] = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
        q_f64, k_f64 = hidden_qk_rmsnorm_rope_3d(
            q_in,
            k_in,
            attn.norm_q.weight,
            attn.norm_k.weight,
            fp64_table,
            num_heads=heads,
            grid_size=(f, h, w),
            eps=1e-6,
        )
    raw_dispatches = receipt.get("dispatches")
    dispatches = [d for d in raw_dispatches if isinstance(d, dict)] if isinstance(raw_dispatches, list) else []
    assert len(dispatches) == 1
    assert dispatches[0]["accelerated"] is True
    assert dispatches[0]["implementation"] == "triton_hidden_qk_rmsnorm_rope_3d"
    with torch.inference_mode():
        q_ref = rope_apply(attn.norm_q(q_in), freqs_3d, heads)
        k_ref = rope_apply(attn.norm_k(k_in), freqs_3d, heads)
    assert (q_f64 - q_ref).float().norm() / q_ref.float().norm() < 1e-4
    assert (k_f64 - k_ref).float().norm() / k_ref.float().norm() < 1e-4


def test_native_pipeline_forwards_sol_engine_and_padded_sp_options() -> None:
    from worldfoundry.pipelines.forcing.pipeline_native_forcing import NativeSelfForcingPipeline
    from worldfoundry.pipelines.native_diffusion import NativeVisualDiffusionPipeline

    for key in (
        "allow_approximate_composition",
        "overlap_vae_decode",
        "ring_degree",
        "sequence_parallel_pad_heads",
    ):
        assert key in NativeVisualDiffusionPipeline.RUNTIME_POLICY_OPTION_KEYS
    assert "overlap_vae_decode" in NativeSelfForcingPipeline.RUNTIME_POLICY_OPTION_KEYS

    resolved = NativeVisualDiffusionPipeline._runtime_options(
        {
            "allow_approximate_composition": True,
            "sequence_parallel_pad_heads": True,
            "ring_degree": 2,
            "overlap_vae_decode": True,
        }
    )
    assert resolved["allow_approximate_composition"] is True
    assert resolved["sequence_parallel_pad_heads"] is True
    assert resolved["ring_degree"] == 2
    assert resolved["overlap_vae_decode"] is True


def test_selective_fp8_composes_with_easycache_and_sol_attn_when_opted_in(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
    from worldfoundry.core.attention.backends import probe
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda req, dev=None: req)
    model = WanModel(
        dim=256,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=4,
        has_image_input=False,
    ).eval()
    accelerations = {
        "easycache": {"threshold": 0.1},
        "attention_policy": {"self": {"backend": "sol_attn", "tau": 1.0}},
        "selective_fp8": {"include": ["blocks.*.ffn.0"], "min_features": 256},
    }
    # Fail closed by default without allow_approximate_composition
    with pytest.raises(ValueError, match="conflicts|seam"):
        install_diffusion_accelerations(
            model,
            accelerations,
            policy=RuntimePolicy(dtype=torch.bfloat16, options={"cuda_graph": "piecewise"}),
        )
    # Succeeds when explicitly opted in via allow_approximate_composition=True
    session = install_diffusion_accelerations(
        model,
        accelerations,
        policy=RuntimePolicy(
            dtype=torch.bfloat16,
            options={"allow_approximate_composition": True, "cuda_graph": "piecewise"},
        ),
    )
    assert hasattr(model, "_worldfoundry_easycache_config")
    assert cast(Any, model.blocks[0]).self_attn.attn.attention_backend == "sol_attn"
    session.uninstall()


def test_multimodal_sequence_routes_fa4_and_2d_mask(monkeypatch) -> None:
    from worldfoundry.core.attention.backends import dispatch
    from worldfoundry.core.attention.sequence import multimodal_sequence as mm_seq

    captured: list[tuple[str | None, tuple[int, ...] | None]] = []
    monkeypatch.setattr(mm_seq, "resolve_attention_backend", lambda req, dev=None, **kw: req)

    def fake_forward(q, k, v, **kwargs):
        mask = kwargs.get("attn_mask")
        captured.append((kwargs.get("backend"), None if mask is None else tuple(mask.shape)))
        return q

    monkeypatch.setattr(dispatch, "attention_forward", fake_forward)
    q = torch.randn(1, 8, 2, 128)
    all_true_mask = torch.ones(1, 8, dtype=torch.bool)
    partial_mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    _ = mm_seq.attention(q, q, q, attn_mask=all_true_mask, attn_mode="flash_attention_4")
    _ = mm_seq.attention(q, q, q, attn_mask=partial_mask, attn_mode="triton_tma")
    _ = mm_seq.sequence_parallel_attention(
        (q, q[:, :4]), (q, q[:, :4]), (q, q[:, :4]), 8, 8, attn_mode="flash_attention_4"
    )
    assert captured == [
        ("flash_attention_4", None),
        ("triton_tma", (1, 1, 8, 8)),
        ("flash_attention_4", None),
    ]


def test_sol_engine_recipe_preserves_vae_parallel_false_and_kv_splits() -> None:
    hw_4gpu = ModelHardwareProfile(
        model_id="wan2.1-t2v-14b",
        num_heads=40,
        head_dim=128,
        num_layers=40,
        world_size=4,
        sm_major=9,
        available_attention_backends=("torch", "sol_attn"),
    )
    stack = compose_sol_stack(
        hw_4gpu,
        budget="balanced",
        sparse_attention=SparseAttentionPillar(
            backend="sol_attn",
            tau=1.0,
            kv_splits=2,
            allow_multi_split=True,
            sink_tokens=64,
            sink_start=0,
        ),
        enable_vae_parallel=False,
    )
    assert stack.topology.vae_parallel == 1
    kwargs = stack.to_pipeline_kwargs()
    assert "vae_parallel" not in kwargs
    assert "vae_spatial_tiling" not in kwargs
    sol_cfg = kwargs["accelerations"]["attention_policy"]["self"]
    assert sol_cfg["kv_splits"] == 2
    assert sol_cfg["allow_multi_split"] is True
    assert sol_cfg["sink_tokens"] == 64
    assert sol_cfg["sink_start"] == 0

    reloaded = SolEngineStack.from_recipe_dict(json.loads(json.dumps(stack.to_recipe_dict())))
    assert reloaded.topology.vae_parallel == 1
    assert reloaded.to_pipeline_kwargs() == kwargs

    # Resolution-aware VAE tiling gate: <=480p skips spatial tiling on 2+ GPUs
    p480 = resolve_acceleration_profile("exact", world_size=2, num_heads=40, height=480)
    p720 = resolve_acceleration_profile("exact", world_size=2, num_heads=40, height=720)
    assert "vae_spatial_tiling" not in p480.options
    assert p720.options["vae_spatial_tiling"] is True


def test_videogen_report_scale_eff_normalizes_baseline_world_size() -> None:
    from benchmarks.videogen.report import format_markdown_leaderboard

    md_4gpu = format_markdown_leaderboard(
        [
            {"label": "4gpu_ref", "world_size": 4, "median_e2e_s": 300.0},
            {"label": "4gpu_sol", "world_size": 4, "median_e2e_s": 200.0},
            {"label": "8gpu_sol", "world_size": 8, "median_e2e_s": 100.0},
        ]
    )
    lines = [ln for ln in md_4gpu.splitlines() if ln.startswith("| 4gpu_") or ln.startswith("| 8gpu_")]
    assert "| 1.00x | - |" in lines[0]
    assert "| 1.50x | - |" in lines[1]
    assert "| 3.00x | 150.0% |" in lines[2]


def test_wan_tile_3d_skips_dense_warmup_steps() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel

    model = WanModel(
        dim=256,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    ).eval()
    cast(Any, model)._worldfoundry_sol_schedule = {"dense_steps": 5, "dense_blocks": (), "tile_3d": (1, 2, 2)}
    x = torch.randn(1, 4, 2, 6, 10)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 8, 64)
    with torch.no_grad():
        _ = model(x, t, ctx, _worldfoundry_sol_step_index=0)
        assert not model.__dict__.get("_worldfoundry_tile_perm_cache")
        _ = model(x, t, ctx, _worldfoundry_sol_step_index=5)
        assert len(model.__dict__.get("_worldfoundry_tile_perm_cache", {})) == 1


def test_wan_sol_tile_permutation_strips_and_teacache_transition() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
        WanModel,
        _build_sol_tile_permutation,
    )

    # Exact bijection + drastically reduced forced-dense sink counts on 480p and 720p grids
    for (f, h, w), expected_sink in (
        ((21, 30, 52), 1848),  # 480p: down from 8,568 (26.2%) to 1,848 (5.6%)
        ((21, 45, 80), 336),  # 720p: down from 8,400 (11.1%) to 336 (0.44%) via 4x16 + 1x64 strips
    ):
        perm, inv, sink = _build_sol_tile_permutation(f, h, w, (1, 8, 8), torch.device("cpu"))
        total = f * h * w
        expected_range = torch.arange(total, dtype=torch.long)
        assert perm.shape == (total,)
        assert torch.equal(inv[perm], expected_range)
        assert torch.equal(perm[inv], expected_range)
        assert sink == expected_sink
        # Every tiled 64-token block before the suffix sink must stay intra-frame
        tiled_count = total - sink
        assert tiled_count % 64 == 0
        frame_ids = (perm[:tiled_count] // (h * w)).view(-1, 64)
        assert torch.equal(frame_ids, frame_ids[:, :1].expand_as(frame_ids))

    # TeaCache HIT across dense-to-tiled boundary must reuse canonical raster residual (zero scrambling)
    torch.manual_seed(7)
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=1,
        has_image_input=False,
    ).eval()
    cast(Any, model)._worldfoundry_sol_schedule = {"dense_steps": 1, "dense_blocks": (), "tile_3d": (1, 2, 2)}

    class _ReplayResidualCache:
        signal_kind = "timestep-modulation"

        def __init__(self) -> None:
            self.saved: torch.Tensor | None = None
            self.dense_calls = 0

        def run(self, step: int, signal: torch.Tensor, compute: Any, *, total_steps: int | None = None) -> torch.Tensor:
            del signal, total_steps
            if step == 0:
                self.dense_calls += 1
                self.saved = compute()
                return self.saved
            assert self.saved is not None
            return self.saved

    x = torch.randn(1, 4, 2, 8, 12)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 4, 32)
    cache = _ReplayResidualCache()
    with torch.no_grad():
        out_dense_miss = model(
            x,
            t,
            ctx,
            feature_cache=cache,
            feature_cache_step=0,
            feature_cache_total_steps=2,
            _worldfoundry_sol_step_index=0,
        )
        out_tiled_hit = model(
            x,
            t,
            ctx,
            feature_cache=cache,
            feature_cache_step=1,
            feature_cache_total_steps=2,
            _worldfoundry_sol_step_index=1,
        )
    assert cache.dense_calls == 1
    assert torch.equal(out_dense_miss, out_tiled_hit)


def test_sol_attn_preserves_dense_fa3_backend_on_edge_blocks_and_warmup(monkeypatch) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.optimizations.plugins import install_diffusion_accelerations
    from worldfoundry.core.attention.backends import probe
    from worldfoundry.core.model_loading.policy import RuntimePolicy

    monkeypatch.setattr(probe, "resolve_attention_backend", lambda req, dev=None: req)
    model = WanModel(
        dim=256,
        in_dim=4,
        ffn_dim=512,
        out_dim=4,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=4,
        has_image_input=False,
    ).eval()
    for block in model.blocks:
        cast(Any, block).self_attn.attn.set_attention_backend("flash_attention_3")
    session = install_diffusion_accelerations(
        model,
        {
            "attention_policy": {
                "self": {
                    "backend": "sol_attn",
                    "tau": 1.0,
                    "dense_steps": 4,
                    "dense_blocks": (0, -1),
                }
            }
        },
        policy=RuntimePolicy(dtype=torch.bfloat16),
    )
    blocks_any: Any = model.blocks
    assert blocks_any[0].self_attn.attn.attention_backend == "flash_attention_3"
    assert blocks_any[3].self_attn.attn.attention_backend == "flash_attention_3"
    assert blocks_any[1].self_attn.attn.attention_backend == "sol_attn"
    assert blocks_any[1].self_attn.attn._worldfoundry_dense_attention_backend == "flash_attention_3"
    assert blocks_any[2].self_attn.attn._worldfoundry_dense_attention_backend == "flash_attention_3"
    session.uninstall()
    assert all(cast(Any, b).self_attn.attn.attention_backend == "flash_attention_3" for b in model.blocks)
    assert all(not hasattr(cast(Any, b).self_attn.attn, "_worldfoundry_dense_attention_backend") for b in model.blocks)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_wave2_fused_fp8_ffn_exact_accum_and_fused_adaln_kernels() -> None:
    import copy

    import torch.nn as nn
    import torch.nn.functional as F

    from worldfoundry.core.acceleration.quantization.fused_ffn import FusedFP8GELUFeedForward
    from worldfoundry.core.acceleration.quantization.linear import Float8Linear
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
    from worldfoundry.core.kernels import (
        fused_layer_norm_scale_shift,
        residual_add_layer_norm_scale_shift,
        residual_gate_layer_norm,
    )
    from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # 1. Single-pass (K=13824) & ultra-wide tiled (K=20480) FP8 GELU quantizer parity
    for cols in (5120, 13824, 20480):
        inp = torch.randn(128, cols, device=device, dtype=dtype)
        q_id, s_id = quantize_rowwise_fp8_triton(inp, torch.float8_e4m3fn, activation="identity")
        q_g, s_g = quantize_rowwise_fp8_triton(inp, torch.float8_e4m3fn, activation="gelu_tanh")
        g_ref = F.gelu(inp, approximate="tanh")
        q_ref, s_ref = quantize_rowwise_fp8_triton(g_ref, torch.float8_e4m3fn, activation="identity")
        assert torch.allclose(s_g, s_ref, rtol=1e-3, atol=1e-6)
        assert (q_g.float() * s_g - q_ref.float() * s_ref).norm() / (q_ref.float() * s_ref).norm() < 5e-3
        assert torch.isfinite(q_id.float()).all() and torch.isfinite(s_id).all()

    # 2. FusedFP8GELUFeedForward with use_fast_accum=False on Wan 14B FFN width (5120 -> 13824 -> 5120)
    if torch.cuda.get_device_capability(device)[0] >= 9:
        dim, ffn_dim = 5120, 13824
        up = nn.Linear(dim, ffn_dim, device=device, dtype=dtype)
        down = nn.Linear(ffn_dim, dim, device=device, dtype=dtype)
        unfused = nn.Sequential(
            Float8Linear.from_linear(up, keep_dense_fallback=False, scaling="rowwise", use_fast_accum=False),
            nn.GELU(approximate="tanh"),
            Float8Linear.from_linear(down, keep_dense_fallback=False, scaling="rowwise", use_fast_accum=False),
        ).eval()
        fused = FusedFP8GELUFeedForward(copy.deepcopy(unfused)).eval()
        assert cast(Float8Linear, fused[0]).use_fast_accum is False
        assert cast(Float8Linear, fused[2]).use_fast_accum is False
        x_ffn = torch.randn(1, 256, dim, device=device, dtype=dtype)
        with torch.inference_mode():
            y_unfused = unfused(x_ffn)
            y_fused = fused(x_ffn)
            y_empty = fused(x_ffn[:, :0])
        rel_ffn = (y_fused.float() - y_unfused.float()).norm() / y_unfused.float().norm()
        assert rel_ffn < 5e-3
        assert y_empty.shape == (1, 0, dim)
        assert fused.fused_calls == 1
        assert fused.fallback_calls == 0

    # 3. Single-pass fused AdaLN, residual_gate_layer_norm, and residual_add_layer_norm_scale_shift
    #    Verify broadcast [B, 1, D] and per-token [B, L, D], functional + inplace storage reuse.
    dim = 5120
    for mod_shape in ((1, 1, dim), (1, 64, dim)):
        x = torch.randn(1, 64, dim, device=device, dtype=dtype)
        u = torch.randn_like(x)
        sc = torch.randn(mod_shape, device=device, dtype=dtype)
        sh = torch.randn(mod_shape, device=device, dtype=dtype)
        gt = torch.randn(mod_shape, device=device, dtype=dtype)
        w = torch.randn(dim, device=device, dtype=dtype)
        b = torch.randn(dim, device=device, dtype=dtype)
        receipt: dict[str, Any] = {}
        with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt):
            ln_out = fused_layer_norm_scale_shift(x, sc, sh, eps=1e-6, backend="triton")
            rg_x, rg_y = residual_gate_layer_norm(x, u, gt, w, b, eps=1e-6, inplace=False, backend="triton")
            x_ip1 = x.clone()
            ptr1 = x_ip1.data_ptr()
            rg_x_ip, rg_y_ip = residual_gate_layer_norm(x_ip1, u, gt, w, b, eps=1e-6, inplace=True, backend="triton")
            add_x, add_y = residual_add_layer_norm_scale_shift(x, u, sc, sh, eps=1e-6, inplace=False, backend="triton")
            x_ip2 = x.clone()
            ptr2 = x_ip2.data_ptr()
            add_x_ip, add_y_ip = residual_add_layer_norm_scale_shift(
                x_ip2, u, sc, sh, eps=1e-6, inplace=True, backend="triton"
            )
        ref_ln = F.layer_norm(x, (dim,), eps=1e-6) * (1 + sc) + sh
        ref_rg_x = x + u * gt
        ref_rg_y = F.layer_norm(ref_rg_x, (dim,), weight=w, bias=b, eps=1e-6)
        ref_add_x = x + u
        ref_add_y = F.layer_norm(ref_add_x, (dim,), eps=1e-6) * (1 + sc) + sh
        assert (ln_out.float() - ref_ln.float()).norm() / ref_ln.float().norm() < 1e-4
        assert torch.equal(rg_x, ref_rg_x)
        assert torch.equal(rg_x_ip, ref_rg_x)
        assert rg_x_ip.data_ptr() == ptr1
        assert (rg_y.float() - ref_rg_y.float()).norm() / ref_rg_y.float().norm() < 1e-4
        assert torch.equal(rg_y_ip, rg_y)
        assert torch.equal(add_x, ref_add_x)
        assert torch.equal(add_x_ip, ref_add_x)
        assert add_x_ip.data_ptr() == ptr2
        assert (add_y.float() - ref_add_y.float()).norm() / ref_add_y.float().norm() < 1e-4
        assert torch.equal(add_y_ip, add_y)
        dispatches = receipt.get("dispatches") or []
        assert len(dispatches) == 5
        assert all(d["accelerated"] and d["backend"] == "triton" for d in dispatches)

    # 5. Single-pass Triton FP64 hidden_qk_rmsnorm_rope_freqs (for 3D-tiled Sol-Attn & pruned tokens)
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import SelfAttention, rope_apply
    from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_freqs

    seq_t, dim_t, heads_t = 128, 5120, 40
    attn_t = SelfAttention(dim=dim_t, num_heads=heads_t, eps=1e-6).to(device=device, dtype=dtype).eval()
    qkv_t = torch.randn(1, seq_t, 3 * dim_t, device=device, dtype=dtype)
    q_s, k_s, _ = qkv_t.chunk(3, dim=-1)
    angles_t = torch.randn(seq_t, 1, (dim_t // heads_t) // 2, device=device, dtype=torch.float64)
    freqs_c = torch.polar(torch.ones_like(angles_t), angles_t)
    receipt_r: dict[str, Any] = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(receipt_r):
        q_fr, k_fr = hidden_qk_rmsnorm_rope_freqs(
            q_s,
            k_s,
            attn_t.norm_q.weight,
            attn_t.norm_k.weight,
            freqs_c,
            num_heads=heads_t,
            eps=1e-6,
        )
        q_ref_fr = rope_apply(attn_t.norm_q(q_s), freqs_c, heads_t, compute_dtype=torch.float64)
        k_ref_fr = rope_apply(attn_t.norm_k(k_s), freqs_c, heads_t, compute_dtype=torch.float64)
    assert (q_fr.float() - q_ref_fr.float()).norm() / q_ref_fr.float().norm() < 1e-4
    assert (k_fr.float() - k_ref_fr.float()).norm() / k_ref_fr.float().norm() < 1e-4
    r_disp = receipt_r.get("dispatches") or []
    assert len(r_disp) == 1
    assert r_disp[0]["implementation"] == "triton_hidden_qk_rmsnorm_rope_freqs"
    assert r_disp[0]["accelerated"] is True
    from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
        FusedRoPERuntimeState,
        fused_rope_runtime_report,
    )

    rope_state = FusedRoPERuntimeState(installed_blocks=1)
    rope_state.record_eager_dispatch(receipt_r)
    rope_report = fused_rope_runtime_report(rope_state)
    assert rope_report["effective"] == "accelerated-provider-executed"
    assert rope_report["malformed_receipts"] == 0
    assert rope_report["provider_calls"] == 1
    assert rope_report["provider_paths"] == ["triton_hidden_qk_rmsnorm_rope_freqs"]

    # 4. Compile-safe Hopper FA3 custom_op for Self-Forcing (when flash_attn_interface is installed)
    try:
        __import__("flash_attn_interface")
    except ImportError:
        return
    if torch.cuda.get_device_capability(device)[0] >= 9:
        from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
            wan_flash_attn_3_bshd,
        )

        q = torch.randn(1, 256, 12, 128, device=device, dtype=dtype)
        k = torch.randn(1, 512, 12, 128, device=device, dtype=dtype)
        v = torch.randn(1, 512, 12, 128, device=device, dtype=dtype)
        compiled_fa3 = torch.compile(wan_flash_attn_3_bshd, fullgraph=True)
        with torch.inference_mode():
            out_fa3 = compiled_fa3(q, k, v)
            out_ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(
                1, 2
            )
        assert (out_fa3.float() - out_ref.float()).norm() / out_ref.float().norm() < 5e-3


def test_wan_tile_3d_guards_phase_block_caches_vace_and_fused_adaln_pop() -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.multiworld import MultiWorldDiTBlock

    torch.manual_seed(11)
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    ).eval()
    cast(Any, model)._worldfoundry_sol_schedule = {"dense_steps": 1, "dense_blocks": (), "tile_3d": (1, 2, 2)}
    cast(Any, model)._worldfoundry_fused_adaln = True

    x = torch.randn(1, 4, 2, 6, 6)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, 6, 32)

    # 1. BlockTaylorSeer (phase_cache_runner) must not scramble tokens across dense_steps -> tiled boundary
    class _PhaseCache:
        def __init__(self) -> None:
            self.saved: dict[int, dict[str, torch.Tensor]] = {}

        def run_phase_blocks(
            self,
            step: int,
            block_input: torch.Tensor,
            run_block: Any,
            run_phase_block: Any,
            *,
            block_count: int,
            total_steps: int | None = None,
        ) -> torch.Tensor:
            del run_block, total_steps
            h = block_input
            for bid in range(block_count):
                if step == 0:
                    h, phases = run_phase_block(bid, h, None)
                    self.saved[bid] = {k: v.clone() for k, v in phases.items()}
                else:
                    h, _ = run_phase_block(bid, h, self.saved[bid])
            return h

    pc = _PhaseCache()
    with torch.no_grad():
        y0 = model(x, t, ctx, feature_cache=pc, feature_cache_step=0, _worldfoundry_sol_step_index=0)
        y1 = model(x, t, ctx, feature_cache=pc, feature_cache_step=1, _worldfoundry_sol_step_index=1)
    assert torch.equal(y0, y1)

    # 2. FirstBlock (block_cache_runner) must not scramble tokens across dense_steps -> tiled boundary
    class _BlockCache:
        def __init__(self) -> None:
            self.saved_res: torch.Tensor | None = None

        def run_blocks(
            self,
            step: int,
            block_input: torch.Tensor,
            run_block: Any,
            *,
            block_count: int,
            total_steps: int | None = None,
        ) -> torch.Tensor:
            del total_steps
            if step == 0:
                h = block_input
                for bid in range(block_count):
                    h = run_block(bid, h)
                self.saved_res = h - block_input
                return h
            assert self.saved_res is not None
            return block_input + self.saved_res

    bc = _BlockCache()
    with torch.no_grad():
        b0 = model(x, t, ctx, feature_cache=bc, feature_cache_step=0, _worldfoundry_sol_step_index=0)
        b1 = model(x, t, ctx, feature_cache=bc, feature_cache_step=1, _worldfoundry_sol_step_index=1)
    assert torch.allclose(b0, b1, rtol=1e-6, atol=1e-6)

    # 3. VACE-style after_transformer_block hint injection must not be scrambled by tile_3d
    hint = torch.randn(1, 2 * 3 * 3, 64)

    def _vace_after(hidden: torch.Tensor, block_id: int, token_state: Any, **kwargs: Any) -> torch.Tensor:
        del token_state, kwargs
        return hidden + hint if block_id == 0 else hidden

    cast(Any, model).after_transformer_block = _vace_after
    with torch.no_grad():
        v_dense = model(x, t, ctx, _worldfoundry_sol_step_index=0)
        v_tiled = model(x, t, ctx, _worldfoundry_sol_step_index=1)
    assert torch.equal(v_dense, v_tiled)

    # 4. forward_with_phase_cache & MultiWorldDiTBlock pop _worldfoundry_fused_adaln cleanly
    seen_cross_kwargs: list[set[str]] = []
    block0 = cast(Any, model.blocks[0])
    orig_cross = block0.cross_attn.forward

    def _recording_cross(hidden: torch.Tensor, context: torch.Tensor, **kw: Any) -> torch.Tensor:
        seen_cross_kwargs.append(set(kw))
        return orig_cross(hidden, context, **kw)

    block0.cross_attn.forward = _recording_cross
    hx = torch.randn(1, 16, 64)
    t_mod = torch.randn(1, 6, 64)
    freqs = torch.polar(torch.ones(16, 1, 16, dtype=torch.float64), torch.zeros(16, 1, 16, dtype=torch.float64))
    with torch.no_grad():
        block0.forward_with_phase_cache(hx, torch.randn(1, 6, 64), t_mod, freqs, _worldfoundry_fused_adaln=True)
    assert seen_cross_kwargs and "_worldfoundry_fused_adaln" not in seen_cross_kwargs[0]

    mblk = MultiWorldDiTBlock(has_image_input=False, dim=64, num_heads=2, ffn_dim=128).eval()
    seen_self_kwargs: list[set[str]] = []
    orig_self = mblk.self_attn.forward

    def _recording_self(hidden: torch.Tensor, fr: torch.Tensor, **kw: Any) -> torch.Tensor:
        seen_self_kwargs.append(set(kw))
        return orig_self(hidden, fr, **kw)

    cast(Any, mblk.self_attn).forward = _recording_self
    with torch.no_grad():
        mblk(
            hx,
            torch.randn(1, 6, 64),
            t_mod,
            freqs,
            action_embeds=torch.randn(1, 6, 64),
            _worldfoundry_fused_adaln=True,
        )
    assert seen_self_kwargs and "_worldfoundry_fused_adaln" not in seen_self_kwargs[0]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_wave2_review_regressions_fp_fusion_rope_precision_cols0_and_fa3_autograd() -> None:
    import torch.nn.functional as F

    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import (
        SelfAttention,
        apply_wan_qk_norm_rope,
    )
    from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
        _can_use_wan_fa3_bshd,
    )
    from worldfoundry.core.acceleration.quantization.triton_fp8 import quantize_rowwise_fp8_triton
    from worldfoundry.core.kernels import (
        fused_layer_norm_scale_shift,
        residual_add_layer_norm_scale_shift,
        residual_gate_layer_norm,
    )
    from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope

    device = torch.device("cuda")
    torch.manual_seed(0)

    # 1. cols == 0 initializes scale to tiny (never uninitialized memory)
    _, s0 = quantize_rowwise_fp8_triton(torch.empty((4, 0), device=device, dtype=torch.bfloat16))
    assert torch.all(s0 == float(torch.finfo(torch.float32).tiny))

    # 2. Mixed-dtype (x=bf16, mod=fp32) & uppercase backend='TRITON' & weight=None inplace custom_op
    x = torch.randn(1, 64, 5120, device=device, dtype=torch.bfloat16)
    u = torch.randn_like(x)
    sc32 = torch.randn(1, 1, 5120, device=device, dtype=torch.float32)
    sh32 = torch.randn(1, 1, 5120, device=device, dtype=torch.float32)
    gt = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
    rec: dict[str, Any] = {}
    with torch.inference_mode(), kernel_dispatch_receipt_scope(rec):
        y_mix = fused_layer_norm_scale_shift(x, sc32, sh32, eps=1e-6, backend="TRITON")
        y_ref = F.layer_norm(x, (5120,), eps=1e-6) * (1 + sc32) + sh32
        x_ip = x.clone()
        rg_x, rg_y = residual_gate_layer_norm(x_ip, u, gt, None, None, eps=1e-6, inplace=True, backend="TRITON")
        add_x, add_y = residual_add_layer_norm_scale_shift(x, u, sc32, sh32, eps=1e-6, inplace=False, backend="TRITON")
    dispatches = rec.get("dispatches") or []
    assert len(dispatches) == 3
    assert [d["op"] for d in dispatches] == [
        "fused_layer_norm_scale_shift",
        "residual_gate_layer_norm",
        "residual_add_layer_norm_scale_shift",
    ]
    # With enable_fp_fusion=False, mixed-dtype mul+add matches PyTorch without FMA drift on >99.99% of elements
    assert (y_mix != y_ref).float().mean().item() < 1e-4
    assert torch.equal(rg_x, x + u * gt)
    assert (rg_y.float() - F.layer_norm(rg_x, (5120,), eps=1e-6).float()).norm() / rg_y.float().norm() < 1e-4
    assert (add_y != F.layer_norm(add_x, (5120,), eps=1e-6) * (1 + sc32) + sh32).float().mean().item() < 1e-4

    # 3. apply_wan_qk_norm_rope(fused_indexed=True, precision="fp32") honors FP32 even with complex128 freqs
    attn32 = SelfAttention(dim=256, num_heads=4, eps=1e-6).to(device, torch.float32).eval()
    q32 = torch.randn(2, 64, 256, device=device, dtype=torch.float32)
    k32 = torch.randn(2, 64, 256, device=device, dtype=torch.float32)
    big_angles = torch.linspace(1e4, 1e5, 64 * 32, device=device, dtype=torch.float64).view(64, 1, 32)
    big_freqs = torch.polar(torch.ones_like(big_angles), big_angles)
    with torch.inference_mode():
        q_idx32, _ = apply_wan_qk_norm_rope(attn32, q32, k32, big_freqs, fused_indexed=True, precision="fp32")
        q_idx64, _ = apply_wan_qk_norm_rope(attn32, q32, k32, big_freqs, fused_indexed=True, precision="fp64")
        q_ref32, _ = apply_wan_qk_norm_rope(attn32, q32, k32, big_freqs, fused_indexed=False, precision="fp32")
        q_ref64, _ = apply_wan_qk_norm_rope(attn32, q32, k32, big_freqs, fused_indexed=False, precision="fp64")
    assert not torch.equal(q_idx32, q_idx64)
    assert (q_idx32 - q_ref32).abs().max().item() < 1e-5
    assert (q_idx64 - q_ref64).abs().max().item() < 1e-5

    # 4. FA3 custom_op guard declines autograd tensors so training/grad-check falls back cleanly
    qg = torch.randn(1, 32, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    kg = torch.randn_like(qg)
    vg = torch.randn_like(qg)
    assert _can_use_wan_fa3_bshd(qg, kg, vg) is False
    with torch.inference_mode():
        assert _can_use_wan_fa3_bshd(qg.detach(), kg.detach(), vg.detach()) is True
