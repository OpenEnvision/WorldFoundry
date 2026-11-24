# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit and CUDA tests for acceleration profiles, protocols, piecewise graphs, and fused kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from worldfoundry.core.acceleration.profiles import (
    plan_parallel_topology,
    resolve_acceleration_profile,
)
from worldfoundry.core.execution.graphs.piecewise_graph import PiecewiseBlockGraphRunner
from worldfoundry.core.nn.transformer.contracts import (
    AttentionBackendConfigurable,
    AttentionProcessorHost,
    QKVProjectionModule,
    TransformerBlockStack,
)


def test_parallel_topology_8gpu_h200_meshes() -> None:
    # 12-head Wan 1.3B on 8 GPUs -> CFG=2 x Ulysses=4 (exact divisor, no padding!)
    plan_13b = plan_parallel_topology(8, num_heads=12, supports_cfg=True)
    assert plan_13b.cfg_parallel == 2
    assert plan_13b.ulysses_degree == 4
    assert plan_13b.ring_degree == 1
    assert plan_13b.vae_parallel == 8
    assert plan_13b.padded_ulysses is False

    # 40-head Wan 14B on 8 GPUs -> CFG=2 x Ulysses=4
    plan_14b = plan_parallel_topology(8, num_heads=40, supports_cfg=True)
    assert plan_14b.cfg_parallel == 2
    assert plan_14b.ulysses_degree == 4
    assert plan_14b.ring_degree == 1
    assert plan_14b.padded_ulysses is False

    # 12-head without CFG on 8 GPUs -> 2D USP Ulysses=4 x Ring=2
    plan_nocfg = plan_parallel_topology(8, num_heads=12, supports_cfg=False)
    assert plan_nocfg.cfg_parallel == 1
    assert plan_nocfg.ulysses_degree == 4
    assert plan_nocfg.ring_degree == 2

    # Prime heads (e.g. 7) on 4 GPUs without CFG -> padded Ulysses
    plan_prime = plan_parallel_topology(4, num_heads=7, supports_cfg=False)
    assert plan_prime.ulysses_degree == 4
    assert plan_prime.padded_ulysses is True

    # Single GPU
    plan_1 = plan_parallel_topology(1, num_heads=40)
    assert plan_1.cfg_parallel == 1
    assert plan_1.sequence_parallel_degree == 1
    assert plan_1.vae_parallel == 1


def test_resolve_acceleration_profiles() -> None:
    exact = resolve_acceleration_profile("exact", world_size=4, num_heads=40)
    assert exact.options["dit_weight_dtype"] == "bfloat16"
    assert exact.options["fuse_qkv"] is True
    assert "fused_rope" not in exact.options
    assert exact.options["cfg_parallel"] == 2
    assert exact.options["sequence_parallel"] == 2
    assert "cuda_graph" not in exact.options
    assert "quantization" not in exact.options

    exact_1gpu = resolve_acceleration_profile("exact", world_size=1, num_heads=40)
    assert exact_1gpu.options["cuda_graph"] is True

    fast = resolve_acceleration_profile("fast", world_size=8, num_heads=40, sm_major=9)
    assert fast.options["fused_rope"] is True
    assert fast.options["quantization"]["mode"] == "fp8"
    assert "time_embedding" in fast.options["quantization"]["exclude"]
    assert fast.options["quantization"]["options"]["fuse_fp8_ffn"] is True
    assert fast.options["quantization"]["options"]["use_fast_accum"] is False
    assert fast.options["cfg_parallel"] == 2
    assert fast.options["sequence_parallel"] == 4

    turbo = resolve_acceleration_profile("turbo", world_size=1, num_heads=40, sm_major=9)
    assert turbo.options["teacache"] == pytest.approx(0.15)
    assert turbo.options["cuda_graph"] == "piecewise"
    assert turbo.options["inplace_residual"] is False

    sol = resolve_acceleration_profile("sol", world_size=1, num_heads=40, sm_major=9)
    assert sol.options["teacache"] == pytest.approx(0.15)
    assert sol.options["cuda_graph"] == "piecewise"
    assert sol.options["accelerations"]["attention_policy"]["self"]["backend"] == "sol_attn"

    realtime = resolve_acceleration_profile("realtime", world_size=1)
    assert realtime.options["fuse_qkv"] is True
    assert realtime.options["rope_precision"] == "fp64"
    assert realtime.options["cuda_graph"] is True

    with pytest.raises(ValueError, match="unsupported acceleration tier"):
        resolve_acceleration_profile("invalid-tier")


def test_piecewise_block_graph_runner_cpu_lifecycle() -> None:
    runner = PiecewiseBlockGraphRunner(warmup=1, max_graphs_per_segment=4)
    calls = 0

    def seg_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return x * 2.0 + 1.0

    runner.register_segment("blocks", seg_fn, extra_key="test:blocks")
    x = torch.ones(2, 4)
    out = runner.run_segment("blocks", x)
    assert torch.allclose(out, torch.full((2, 4), 3.0))
    assert calls == 1

    runner.record_skip("blocks")
    report = runner.report()
    assert report["eager"] == 1
    assert report["cache_skips"] == 1
    assert report["segments"]["blocks"]["cache_skips"] == 1

    runner.begin_request_window()
    report_after = runner.report()
    assert report_after["request_window"]["cache_skips"] == 0
    assert report_after["request_window"]["eager"] == 0

    runner.freeze()
    assert runner.report()["frozen"] is True
    runner.invalidate()
    assert runner.report()["segment_count"] == 1


def test_transformer_protocols_and_hunyuan_seam() -> None:
    class _DummyAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_backend: str | None = None
            self.num_heads = 4
            self.processor: Any = object()
            self.q = nn.Linear(8, 8)
            self.k = nn.Linear(8, 8)
            self.v = nn.Linear(8, 8)
            self.o = nn.Linear(8, 8)

        def set_attention_backend(self, backend: str | None) -> None:
            self.attention_backend = backend

        def get_processor(self) -> Any:
            return self.processor

        def set_processor(self, processor: Any) -> None:
            self.processor = processor

    class _DummyStack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity()])

    attn = _DummyAttn()
    stack = _DummyStack()
    assert isinstance(attn, AttentionBackendConfigurable)
    assert isinstance(attn, AttentionProcessorHost)
    assert isinstance(attn, QKVProjectionModule)
    assert isinstance(stack, TransformerBlockStack)

    from worldfoundry.base_models.diffusion_model.models.networks.hunyuan_video.original import (
        MMDoubleStreamBlock,
        MMSingleStreamBlock,
    )

    dbl = MMDoubleStreamBlock(hidden_size=256, heads_num=4, mlp_width_ratio=2)
    sgl = MMSingleStreamBlock(hidden_size=256, heads_num=4, mlp_width_ratio=2)
    assert isinstance(dbl, AttentionBackendConfigurable)
    assert isinstance(sgl, AttentionBackendConfigurable)
    dbl.set_attention_backend("cudnn_fp8")
    sgl.set_attention_backend("cudnn_fp8")
    assert dbl.attention_backend == "cudnn_fp8"
    assert sgl.attention_backend == "cudnn_fp8"


def test_piecewise_cuda_graph_validation_rules() -> None:
    from worldfoundry.base_models.diffusion_model.models.denoisers.graph_wrapped import (
        resolve_cuda_graph_mode,
        validate_cuda_graph_options,
    )

    @dataclass
    class _Policy:
        options: dict[str, Any]

    @dataclass
    class _Ctx:
        component_options: dict[str, Any]
        policy: _Policy

    assert resolve_cuda_graph_mode(_Ctx({}, _Policy({}))) == "off"
    assert resolve_cuda_graph_mode(_Ctx({"cuda_graph": True}, _Policy({}))) == "whole"
    assert resolve_cuda_graph_mode(_Ctx({"cuda_graph": "piecewise"}, _Policy({}))) == "piecewise"

    # Whole-model CUDA graph still rejects TeaCache
    with pytest.raises(ValueError, match="cannot be combined with stateful"):
        validate_cuda_graph_options(_Ctx({"cuda_graph": True, "teacache": 0.15}, _Policy({})))

    # Piecewise CUDA graph composes cleanly with TeaCache & MagCache & TaylorSeer
    validate_cuda_graph_options(_Ctx({"cuda_graph": "piecewise", "teacache": 0.15}, _Policy({})))
    validate_cuda_graph_options(_Ctx({"cuda_graph": "piecewise", "taylorseer": True}, _Policy({})))

    # Piecewise still fails closed on block-selective / approximate options
    with pytest.raises(ValueError, match="cannot be combined with stateful"):
        validate_cuda_graph_options(_Ctx({"cuda_graph": "piecewise", "firstblock": True}, _Policy({})))


def test_fused_rope_dynamo_token_and_quantized_qkv_receipts() -> None:
    from worldfoundry.base_models.diffusion_model.optimizations.fused_rope import (
        FusedRoPERuntimeState,
        _register_fused_rope_state,
        record_compiled_fused_rope_graph_trace,
    )
    from worldfoundry.base_models.diffusion_model.optimizations.qkv_fusion import (
        QKVFusionState,
        project_fused_qkv,
    )

    rope_state = FusedRoPERuntimeState(installed_blocks=4)
    token = _register_fused_rope_state(rope_state)
    assert isinstance(token, str)
    assert record_compiled_fused_rope_graph_trace(token) == 1
    assert rope_state.compiled_graph_traces == 1

    # Simulate Float8Linear replacing _FusedQKVLinear (no project_qkv method)
    qkv_state = QKVFusionState(fused_blocks=1)
    plain_linear = nn.Linear(16, 48)
    q, k, v = project_fused_qkv(plain_linear, torch.randn(2, 4, 16), fusion_state=qkv_state)
    assert q.shape == (2, 4, 16)
    assert k.shape == (2, 4, 16)
    assert v.shape == (2, 4, 16)
    assert qkv_state.request_eager_projection_calls == 1
    assert qkv_state.request_eager_packed_projection_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fused RoPE test")
def test_triton_strided_qk_rmsnorm_rope_3d_wan14b() -> None:
    from worldfoundry.core.kernels import hidden_qk_rmsnorm_rope_3d
    from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope

    device = torch.device("cuda")
    dtype = torch.bfloat16
    f, h, w = 2, 8, 8
    seq = f * h * w
    dim = 5120
    num_heads = 40
    head_dim = dim // num_heads

    qkv = torch.randn(1, seq, 3 * dim, device=device, dtype=dtype)
    q, k, _ = qkv.chunk(3, dim=-1)  # strided slices (stride(-1) == 1, non-contiguous)
    assert not q.is_contiguous()

    w_q = torch.ones(dim, device=device, dtype=dtype)
    w_k = torch.ones(dim, device=device, dtype=dtype)
    table = torch.randn(1024, head_dim // 2, 2, device=device, dtype=torch.float32)
    table = table / table.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    receipt: dict[str, Any] = {}
    with kernel_dispatch_receipt_scope(receipt):
        q_out, k_out = hidden_qk_rmsnorm_rope_3d(
            q,
            k,
            w_q,
            w_k,
            table,
            num_heads=num_heads,
            grid_size=(f, h, w),
            eps=1e-6,
        )
    dispatches = [d for d in receipt.get("dispatches", []) if d.get("op") == "hidden_qk_rmsnorm_rope_3d"]
    assert len(dispatches) == 1
    assert dispatches[0]["accelerated"] is True
    assert dispatches[0]["fallback"] is False

    # Compare against contiguous copy to verify strided addressing parity
    q_ref, k_ref = hidden_qk_rmsnorm_rope_3d(
        q.contiguous(),
        k.contiguous(),
        w_q,
        w_k,
        table,
        num_heads=num_heads,
        grid_size=(f, h, w),
        eps=1e-6,
    )
    assert torch.allclose(q_out, q_ref, atol=1e-3, rtol=1e-3)
    assert torch.allclose(k_out, k_ref, atol=1e-3, rtol=1e-3)
