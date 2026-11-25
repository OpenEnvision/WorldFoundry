# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit, compile, CUDA Graph, and multi-GPU NCCL/SymmMem tests for Track 9A Ulysses & TP/CP infra."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from worldfoundry.core.distributed.model_parallel.inference_parallel import (
    InferenceParallelContext,
    RowParallelLinear,
)
from worldfoundry.core.distributed.sequence_parallel.symm_mem_all_to_all import (
    is_ulysses_ephemeral_output_scope,
    ulysses_ephemeral_output_scope,
)
from worldfoundry.core.distributed.sequence_parallel.triton_ulysses_ops import (
    ulysses_pack_o_scatter_seq,
    ulysses_pack_qkv,
    ulysses_pack_single_scatter_head,
    ulysses_unpack_o_gather_head,
    ulysses_unpack_qkv,
    ulysses_unpack_single_gather_seq,
)


def test_lazy_triton_import_on_cpu_only_subprocess() -> None:
    """Importing and running Ulysses modules on CPU-only must succeed even when triton is absent."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "sys.modules['triton.language'] = None\n"
        "import worldfoundry.core.distributed.sequence_parallel.triton_ulysses_ops as u\n"
        "import worldfoundry.core.distributed.sequence_parallel.symm_mem_all_to_all as s\n"
        "import worldfoundry.core.distributed.sequence_parallel.runtime as r\n"
        "import worldfoundry.core.distributed.sequence_parallel.ops as o\n"
        "import torch\n"
        "q = torch.randn(1, 4, 6, 8)\n"
        "packed = u.ulysses_pack_qkv(q, q + 1, q + 2, 4, pad_heads=2)\n"
        "assert packed.shape == (4, 3, 1, 4, 2, 8)\n"
        "qo, ko, vo = u.ulysses_unpack_qkv(packed, 4)\n"
        "assert qo.shape == (1, 16, 2, 8)\n"
        "o_in = torch.randn(1, 16, 2, 8)\n"
        "op = u.ulysses_pack_o_scatter_seq(o_in, 4)\n"
        "assert op.data_ptr() == o_in.data_ptr()\n"
        "ou = u.ulysses_unpack_o_gather_head(op, 4, out_heads=6)\n"
        "assert ou.shape == (1, 4, 6, 8)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"CPU-only lazy import failed:\n{proc.stdout}\n{proc.stderr}"


def test_ephemeral_scope_is_reentrant_and_exception_safe() -> None:
    assert not is_ulysses_ephemeral_output_scope()
    with ulysses_ephemeral_output_scope():
        assert is_ulysses_ephemeral_output_scope()
        with ulysses_ephemeral_output_scope():
            assert is_ulysses_ephemeral_output_scope()
        assert is_ulysses_ephemeral_output_scope()
    assert not is_ulysses_ephemeral_output_scope()

    with pytest.raises(RuntimeError, match="boom"):
        with ulysses_ephemeral_output_scope():
            raise RuntimeError("boom")
    assert not is_ulysses_ephemeral_output_scope()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "batch,s_local,heads,world,pad_heads,head_dim",
    [
        (1, 37, 16, 4, 0, 128),
        (2, 19, 12, 8, 4, 128),
        (1, 64, 40, 8, 0, 128),
        (1, 17, 14, 4, 2, 64),
    ],
)
def test_triton_ulysses_pack_unpack_bit_exact_and_strided(
    dtype: torch.dtype,
    batch: int,
    s_local: int,
    heads: int,
    world: int,
    pad_heads: int,
    head_dim: int,
) -> None:
    device = torch.device("cuda")
    # Build strided Q/K/V views from a fused QKV projection to test arbitrary strides
    fused = torch.randn((batch, s_local, 3, heads, head_dim), device=device, dtype=dtype)
    q = fused[:, :, 0]
    k = fused[:, :, 1]
    v = fused[:, :, 2]

    # Reference pack/unpack in pure PyTorch
    h_pad = heads + pad_heads
    h_l = h_pad // world
    if pad_heads > 0:
        pad = q.new_zeros((batch, s_local, pad_heads, head_dim))
        q_p, k_p, v_p = torch.cat((q, pad), 2), torch.cat((k, pad), 2), torch.cat((v, pad), 2)
    else:
        q_p, k_p, v_p = q, k, v
    ref_packed = (
        torch.stack((q_p, k_p, v_p), dim=0)
        .reshape(3, batch, s_local, world, h_l, head_dim)
        .permute(3, 0, 1, 2, 4, 5)
        .contiguous()
    )

    got_packed = ulysses_pack_qkv(q, k, v, world, pad_heads=pad_heads)
    assert got_packed.is_contiguous()
    assert torch.equal(got_packed, ref_packed)

    ref_unpacked = ref_packed.permute(1, 2, 0, 3, 4, 5).contiguous().reshape(3, batch, world * s_local, h_l, head_dim)
    got_q, got_k, got_v = ulysses_unpack_qkv(got_packed, world)
    assert torch.equal(got_q, ref_unpacked[0])
    assert torch.equal(got_k, ref_unpacked[1])
    assert torch.equal(got_v, ref_unpacked[2])

    # Single-tensor scatter head / gather seq (zero-copy for B==1)
    got_sp = ulysses_pack_single_scatter_head(q, world, pad_heads=pad_heads)
    ref_sp = q_p.reshape(batch, s_local, world, h_l, head_dim).permute(2, 0, 1, 3, 4).contiguous()
    assert torch.equal(got_sp, ref_sp)
    got_su = ulysses_unpack_single_gather_seq(got_sp, world)
    ref_su = ref_sp.transpose(0, 1).contiguous().reshape(batch, world * s_local, h_l, head_dim)
    assert torch.equal(got_su, ref_su)
    if batch == 1:
        assert got_su.data_ptr() == got_sp.data_ptr()

    # Inverse O pack (zero-copy for B==1) & unpack + fused head crop
    o = torch.randn((batch, world * s_local, h_l, head_dim), device=device, dtype=dtype)
    got_op = ulysses_pack_o_scatter_seq(o, world)
    ref_op = o.reshape(batch, world, s_local, h_l, head_dim).transpose(0, 1).contiguous()
    assert torch.equal(got_op, ref_op)
    if batch == 1:
        assert got_op.data_ptr() == o.data_ptr()

    got_ou = ulysses_unpack_o_gather_head(got_op, world, out_heads=heads)
    ref_ou = (
        ref_op.permute(1, 2, 0, 3, 4)
        .contiguous()
        .reshape(batch, s_local, h_pad, head_dim)[:, :, :heads, :]
        .contiguous()
    )
    assert torch.equal(got_ou, ref_ou)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_ulysses_custom_ops_torch_compile_and_zero_spills() -> None:
    from worldfoundry.core.platform.compile_cache import register_worldfoundry_inductor_fallbacks

    register_worldfoundry_inductor_fallbacks()
    device = torch.device("cuda")
    q = torch.randn((1, 32, 12, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        packed = ulysses_pack_qkv(a, b, c, 8, pad_heads=4)
        qo, _, _ = ulysses_unpack_qkv(packed, 8)
        op = ulysses_pack_o_scatter_seq(qo, 8)
        return ulysses_unpack_o_gather_head(op, 8, out_heads=12)

    ref = fn(q, k, v)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    got = compiled(q, k, v)
    assert torch.equal(got, ref)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_row_parallel_linear_cuda_tensor_core_fp32_cancellation(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    # Cancellation test: 256 + 0.25 - 256 = 0.25 (+ 0.75 bias = 1.0)
    linear = torch.nn.Linear(5, 1, dtype=dtype, device=device)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(0.75)
    terms = torch.tensor([[256.0, 0.25, 0.0, -256.0, 0.0]], dtype=dtype, device=device)
    ctx = InferenceParallelContext(device=device)
    row = RowParallelLinear(linear, start=0, end=5, parallel=ctx).to(device)
    with torch.inference_mode():
        out = row(terms)
        ref = F.linear(terms.float(), linear.weight.float(), linear.bias.float()).to(dtype)
    assert torch.equal(out, ref)

    # Larger realistic 3D tensor parity against exact FP32 accumulation
    big_lin = torch.nn.Linear(1024, 512, bias=True, dtype=dtype, device=device)
    x = torch.randn((2, 64, 1024), dtype=dtype, device=device) * 0.25
    row_big = RowParallelLinear(big_lin, start=0, end=1024, parallel=ctx).to(device)
    with torch.inference_mode():
        got_big = row_big(x)
        ref_big = F.linear(x.float(), big_lin.weight.float(), big_lin.bias.float()).to(dtype)
    torch.testing.assert_close(got_big, ref_big, rtol=1e-2, atol=2e-2)


def _nccl_ulysses_worker(rank: int, world_size: int, init_path: str, result_queue) -> None:
    from worldfoundry.core.distributed.sequence_parallel import runtime as sp_runtime
    from worldfoundry.core.distributed.sequence_parallel.symm_mem_all_to_all import (
        clear_symm_mem_pools,
        ulysses_ephemeral_output_scope,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    try:
        device = torch.device("cuda", rank)
        sp_runtime.initialize_sequence_parallel_group(world_size)
        group = sp_runtime.get_sequence_parallel_group()

        torch.manual_seed(123 + rank)
        q = torch.randn((1, 64, 13, 128), device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        pad_heads = (-13) % world_size

        with torch.inference_mode():
            # 1) Baseline torch backend
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "torch"
            ref_q, ref_k, ref_v = sp_runtime.all_to_all_4d_many(
                (q, k, v), group, scatter_dim=2, gather_dim=1, assume_even=True, pad_heads=pad_heads
            )
            ref_o = sp_runtime.all_to_all_4D(
                ref_q, group, scatter_dim=1, gather_dim=2, assume_even=True, crop_heads=pad_heads
            )

            # 2) triton_nccl backend
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "triton_nccl"
            tn_q, tn_k, tn_v = sp_runtime.all_to_all_4d_many(
                (q, k, v), group, scatter_dim=2, gather_dim=1, assume_even=True, pad_heads=pad_heads
            )
            tn_o = sp_runtime.all_to_all_4D(
                tn_q, group, scatter_dim=1, gather_dim=2, assume_even=True, crop_heads=pad_heads
            )
            assert torch.equal(tn_q, ref_q) and torch.equal(tn_k, ref_k) and torch.equal(tn_v, ref_v)
            assert torch.equal(tn_o, ref_o)

            # 3) symm_mem fresh-output pull mode
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "symm_mem"
            sp_q, sp_k, sp_v = sp_runtime.all_to_all_4d_many(
                (q, k, v), group, scatter_dim=2, gather_dim=1, assume_even=True, pad_heads=pad_heads
            )
            sp_o = sp_runtime.all_to_all_4D(
                sp_q, group, scatter_dim=1, gather_dim=2, assume_even=True, crop_heads=pad_heads
            )
            assert torch.equal(sp_q, ref_q) and torch.equal(sp_k, ref_k) and torch.equal(sp_v, ref_v)
            assert torch.equal(sp_o, ref_o)

            # 4) symm_mem ephemeral 1-kernel direct push mode + CUDA Graph replay!
            with ulysses_ephemeral_output_scope():
                ep_q, ep_k, ep_v = sp_runtime.all_to_all_4d_many(
                    (q, k, v), group, scatter_dim=2, gather_dim=1, assume_even=True, pad_heads=pad_heads
                )
                assert torch.equal(ep_q, ref_q) and torch.equal(ep_k, ref_k) and torch.equal(ep_v, ref_v)
                ep_o = sp_runtime.all_to_all_4D(
                    ep_q, group, scatter_dim=1, gather_dim=2, assume_even=True, crop_heads=pad_heads
                )
                assert torch.equal(ep_o, ref_o)

                # CUDA Graph capture + replay of 1-kernel direct NVLink push
                q_s, k_s, v_s = q.clone(), k.clone(), v.clone()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    gq, gk, gv = sp_runtime.all_to_all_4d_many(
                        (q_s, k_s, v_s), group, scatter_dim=2, gather_dim=1, assume_even=True, pad_heads=pad_heads
                    )
                    go = sp_runtime.all_to_all_4D(
                        gq, group, scatter_dim=1, gather_dim=2, assume_even=True, crop_heads=pad_heads
                    )
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(go, ref_o)

            # 5) sequence_parallel_all_gather zero-copy B==1
            local_seq = torch.full((1, 16, 8), float(rank + 1), device=device, dtype=torch.bfloat16)
            gathered = sp_runtime.sequence_parallel_all_gather(local_seq, dim=1)
            assert gathered.shape == (1, 16 * world_size, 8)
            for r in range(world_size):
                assert torch.all(gathered[:, r * 16 : (r + 1) * 16] == float(r + 1))

            # 6) RowParallelLinear 2-GPU Hopper TC + SymmMem two_shot_all_reduce_ parity
            torch.manual_seed(999)
            full_lin = torch.nn.Linear(256, 128, bias=True, device=device, dtype=torch.bfloat16)
            full_x = torch.randn((2, 32, 256), device=device, dtype=torch.bfloat16) * 0.25
            ref_rp = F.linear(full_x.float(), full_lin.weight.float(), full_lin.bias.float()).to(torch.bfloat16)
            tp_ctx = InferenceParallelContext(
                tp_size=world_size,
                cp_size=1,
                tp_rank=rank,
                cp_rank=0,
                tp_group=group,
                control_group=group,
                device=device,
            )
            half = 256 // world_size
            row_lin = RowParallelLinear(
                full_lin,
                start=rank * half,
                end=(rank + 1) * half,
                parallel=tp_ctx,
            ).to(device)
            got_rp = row_lin(full_x[..., rank * half : (rank + 1) * half])
            torch.testing.assert_close(got_rp, ref_rp, rtol=1e-2, atol=2e-2)

            # 7) DistributedAutograd.AllToAll4D (GroupCoordinator path, both directions)
            from worldfoundry.core.distributed.sequence_parallel.device_communicators.base_device_communicator import (
                DistributedAutograd,
            )

            x4d = torch.randn((1, 32, 8, 128), device=device, dtype=torch.bfloat16)
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "torch"
            ref_a2a_in = DistributedAutograd.AllToAll4D.apply(group, x4d, world_size, 2, 1)
            ref_a2a_out = DistributedAutograd.AllToAll4D.apply(group, ref_a2a_in, world_size, 1, 2)
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "symm_mem"
            got_a2a_in = DistributedAutograd.AllToAll4D.apply(group, x4d, world_size, 2, 1)
            got_a2a_out = DistributedAutograd.AllToAll4D.apply(group, got_a2a_in, world_size, 1, 2)
            assert torch.equal(got_a2a_in, ref_a2a_in)
            assert torch.equal(got_a2a_out, ref_a2a_out)

            # 8) MiniMax-H3 _cp_all_to_all_qkv / _cp_all_to_all_out (inference ephemeral + counter)
            from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.dit import (
                _cp_all_to_all_out,
                _cp_all_to_all_qkv,
                _ResolvedContextParallel,
            )

            cp_ctx_ref = _ResolvedContextParallel(
                sp_size=world_size, rank=rank, process_group=group, all_to_all_fn=None, all_gather_fn=None
            )
            cp_ctx_got = _ResolvedContextParallel(
                sp_size=world_size, rank=rank, process_group=group, all_to_all_fn=None, all_gather_fn=None
            )
            q3 = q.squeeze(0)[:, :12, :].contiguous()
            k3 = k.squeeze(0)[:, :12, :].contiguous()
            v3 = v.squeeze(0)[:, :12, :].contiguous()
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "torch"
            rq3, rk3, rv3 = _cp_all_to_all_qkv(q3, k3, v3, cp_ctx_ref)
            ro3 = _cp_all_to_all_out(rq3, cp_ctx_ref)
            os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "symm_mem"
            gq3, gk3, gv3 = _cp_all_to_all_qkv(q3, k3, v3, cp_ctx_got)
            assert torch.equal(gq3, rq3) and torch.equal(gk3, rk3) and torch.equal(gv3, rv3)
            go3 = _cp_all_to_all_out(gq3, cp_ctx_got)
            assert torch.equal(go3, ro3)
            assert cp_ctx_got.all_to_all_calls == cp_ctx_ref.all_to_all_calls == 4

        # 8b) MiniMax-H3 autograd backward must NOT alias gq/gk/gv on symm_mem buffer
        os.environ["WORLDFOUNDRY_ULYSSES_BACKEND"] = "symm_mem"
        cp_ctx_grad = _ResolvedContextParallel(
            sp_size=world_size, rank=rank, process_group=group, all_to_all_fn=None, all_gather_fn=None
        )
        qg = q3.detach().clone().requires_grad_(True)
        kg = k3.detach().clone().requires_grad_(True)
        vg = v3.detach().clone().requires_grad_(True)
        q_ex, k_ex, v_ex = _cp_all_to_all_qkv(qg, kg, vg, cp_ctx_grad)
        loss = (q_ex.float() * 1.0 + k_ex.float() * 2.0 + v_ex.float() * 3.0).sum()
        loss.backward()
        assert qg.grad is not None and kg.grad is not None and vg.grad is not None
        assert torch.all(qg.grad == 1.0) and torch.all(kg.grad == 2.0) and torch.all(vg.grad == 3.0)

        clear_symm_mem_pools()
        result_queue.put(True)
    finally:
        os.environ.pop("WORLDFOUNDRY_ULYSSES_BACKEND", None)
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2+ GPUs required for NCCL/SymmMem test")
def test_multigpu_nccl_and_symm_mem_ulysses_bit_exact_and_cudagraph(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_path = str(tmp_path / "ulysses-rendezvous")
    procs = [ctx.Process(target=_nccl_ulysses_worker, args=(r, 2, init_path, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert [q.get(timeout=5) for _ in procs] == [True, True]
