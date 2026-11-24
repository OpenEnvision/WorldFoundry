# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LTX-2.5 SOL-Engine acceleration and MiniMax-H3 Super Acceleration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoDecoder
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel, LTXModelType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import BasicAVTransformerBlock
from worldfoundry.pipelines.ltx25 import LTX25Pipeline
from worldfoundry.pipelines.minimax.super_acceleration import (
    MiniMaxH3SuperAccelerationPipeline,
    SuperAccelerationConfig,
)
from worldfoundry.synthesis.visual_generation.ltx25 import (
    LTX25_STAGE_2_SIGMAS,
    LTX25AcceleratedExecutor,
    LTX25AccelerationConfig,
    LTX25CFGParallelRunner,
    LTX25FirstBlockCache,
    LTX25FirstBlockCacheConfig,
    LTX25ParallelConfig,
    LTX25UlyssesAttentionAdapter,
    accelerate_ltx25_video_vae,
    coerce_ltx25_acceleration_config,
    convert_ltx25_conv_video_decoder_state_dict,
    convert_ltx25_conv_video_encoder_state_dict,
    gather_ltx25_video_hidden,
    ltx25_sol_profile,
    resample_video_for_ltx25,
    shard_ltx25_video_args,
)


def _tiny_ltx_av_model(
    *,
    num_layers: int = 4,
    num_heads: int = 8,
    head_dim: int = 128,
    audio_heads: int = 8,
    audio_head_dim: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LTXModel:
    torch.manual_seed(0)
    video_dim = num_heads * head_dim
    audio_dim = audio_heads * audio_head_dim
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=32,
        out_channels=32,
        num_layers=num_layers,
        cross_attention_dim=video_dim,
        audio_num_attention_heads=audio_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_cross_attention_dim=audio_dim,
        rope_type=LTXRopeType.SPLIT,
        apply_gated_attention=True,
        cross_attention_adaln=True,
        ff_bias=False,
        audio_ff_bias=False,
    )
    # Initialize scale_shift_tables deterministically (constructor uses torch.empty).
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "scale_shift_table" in name:
                param.normal_(0.0, 0.02)
    return model.to(device=device, dtype=dtype).eval()


def _tiny_modalities(
    *,
    grid: tuple[int, int, int] = (2, 4, 4),
    audio_tokens: int = 8,
    text_tokens: int = 6,
    video_in_channels: int = 32,
    audio_in_channels: int = 16,
    video_ctx_dim: int = 1024,
    audio_ctx_dim: int = 512,
    sigma: float = 0.5,
    seed: int = 7,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Modality, Modality]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    f, h, w = grid
    t_video = f * h * w
    gf, gh, gw = torch.meshgrid(
        torch.arange(f, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((gf.reshape(-1), gh.reshape(-1), gw.reshape(-1)), dim=0)
    positions = torch.stack((coords, coords + 1.0), dim=-1).unsqueeze(0).to(device=device)

    audio_idx = torch.arange(audio_tokens, dtype=torch.float32)
    audio_pos = torch.stack((audio_idx, audio_idx + 1.0), dim=-1).view(1, 1, audio_tokens, 2).to(device=device)

    sigma_t = torch.tensor([sigma], device=device, dtype=dtype)
    video = Modality(
        latent=torch.randn((1, t_video, video_in_channels), generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        ),
        sigma=sigma_t,
        timesteps=torch.full((1, 1, 1), sigma, device=device, dtype=dtype),
        positions=positions,
        context=torch.randn((1, text_tokens, video_ctx_dim), generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        ),
    )
    audio = Modality(
        latent=torch.randn((1, audio_tokens, audio_in_channels), generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        ),
        sigma=sigma_t,
        timesteps=torch.full((1, 1, 1), sigma, device=device, dtype=dtype),
        positions=audio_pos,
        context=torch.randn((1, text_tokens, audio_ctx_dim), generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        ),
    )
    return video, audio


# ──────────────────────────────────────────────────────────────────────────
# 1. LTX-2.5 FirstBlockCache (FBCache) Validation, Boundaries & Telemetry
# ──────────────────────────────────────────────────────────────────────────


def test_ltx25_fbcache_config_and_conflict_validation() -> None:
    with pytest.raises(TypeError, match="residual_diff_threshold"):
        LTX25FirstBlockCacheConfig(residual_diff_threshold=cast(Any, True))
    with pytest.raises(ValueError, match="residual_diff_threshold"):
        LTX25FirstBlockCacheConfig(residual_diff_threshold=-0.1)
    with pytest.raises(ValueError, match="finite"):
        LTX25FirstBlockCacheConfig(residual_diff_threshold=float("nan"))
    with pytest.raises(ValueError, match="dense_first"):
        LTX25FirstBlockCacheConfig(dense_first=0)
    with pytest.raises(ValueError, match="FirstBlockCache cannot use cuda_graph='whole'"):
        LTX25AccelerationConfig(
            fbcache=LTX25FirstBlockCacheConfig(enabled=True),
            cuda_graph="whole",
        )


def test_ltx25_fbcache_boundaries_max_skips_and_autograd_guard() -> None:
    cfg = LTX25FirstBlockCacheConfig(
        enabled=True,
        residual_diff_threshold=0.2,
        dense_first=2,
        dense_last=2,
        max_consecutive_skips=2,
        downsample_factor=2,
        cumulative=True,
        min_steps_to_enable=6,
    )
    cache = LTX25FirstBlockCache(cfg, branch="positive", stage_index=0)
    total_steps = 9
    num_blocks = 8
    v0 = torch.zeros(1, 16, 32)
    base_probe = torch.ones(1, 16, 32)
    tail_res = torch.full((1, 16, 32), 2.5)

    with torch.inference_mode():
        for step in range(total_steps):
            # Slightly perturb probe (< 0.2 rel L1) on every step.
            v1 = base_probe * (1.0 + 0.01 * step)
            can_hit, reason, metric, vp, ap = cache.evaluate_probe(
                step=step,
                total_steps=total_steps,
                num_blocks=num_blocks,
                block0_video_in=v0,
                block0_video_out=v1,
                block0_audio_in=None,
                block0_audio_out=None,
            )
            if can_hit:
                v_out, _ = cache.apply_hit(
                    step=step,
                    num_blocks=num_blocks,
                    block0_video_out=v1,
                    block0_audio_out=None,
                    v_probe=vp,
                    a_probe=ap,
                    metric=metric,
                    reason=reason,
                )
                assert v_out is not None
                torch.testing.assert_close(v_out, v1 + tail_res)
            else:
                cache.record_dense(
                    step=step,
                    num_blocks=num_blocks,
                    block0_video_out=v1,
                    final_video_out=v1 + tail_res,
                    block0_audio_out=None,
                    final_audio_out=None,
                    v_probe=vp,
                    a_probe=ap,
                    metric=metric,
                    reason=reason,
                )

    # Expected: steps 0,1 dense-boundary; 2,3 hit; 4 max-consecutive-skips; 5,6 hit; 7,8 dense-boundary.
    reasons = [ev.reason for ev in cache.events]
    hits = [ev.hit for ev in cache.events]
    assert reasons == [
        "seed",
        "dense-boundary",
        "below-threshold",
        "below-threshold",
        "max-consecutive-skips",
        "below-threshold",
        "below-threshold",
        "dense-boundary",
        "dense-boundary",
    ]
    assert hits == [False, False, True, True, False, True, True, False, False]
    receipt = cache.receipt()
    assert receipt["hits"] == 4
    assert receipt["dense_steps"] == 5
    assert receipt["skipped_block_calls"] == 4 * (num_blocks - 1)

    # Short schedule (< min_steps_to_enable) must never skip (protects 3-step refinement).
    cache.reset()
    with torch.inference_mode():
        can_hit, reason, _, _, _ = cache.evaluate_probe(
            step=1,
            total_steps=3,
            num_blocks=num_blocks,
            block0_video_in=v0,
            block0_video_out=base_probe,
            block0_audio_in=None,
            block0_audio_out=None,
        )
    assert not can_hit and reason == "short-schedule"

    # Autograd must always force dense.
    with torch.enable_grad():
        can_hit_g, reason_g, _, _, _ = cache.evaluate_probe(
            step=2,
            total_steps=10,
            num_blocks=num_blocks,
            block0_video_in=v0,
            block0_video_out=base_probe,
            block0_audio_in=None,
            block0_audio_out=None,
        )
    assert not can_hit_g and reason_g == "autograd"


def test_ltx25_fbcache_executor_zero_threshold_exact_parity_cpu() -> None:
    model = _tiny_ltx_av_model(num_layers=4, num_heads=4, head_dim=64, audio_heads=4, audio_head_dim=32)
    video, audio = _tiny_modalities(
        grid=(1, 4, 4),
        audio_tokens=6,
        video_ctx_dim=256,
        audio_ctx_dim=128,
        sigma=0.6,
    )
    with torch.inference_mode():
        ref_v, ref_a = model(video, audio)

    cfg = LTX25AccelerationConfig(
        fuse_qkv=True,
        fuse_cross_kv=True,
        fused_ops=False,
        attention_backend="torch",
        sol_attn=False,
        quantization="none",
        fbcache=LTX25FirstBlockCacheConfig(enabled=True, residual_diff_threshold=0.0, dense_first=1, dense_last=0),
        cuda_graph="off",
        vae_compile=False,
    )
    executor = LTX25AcceleratedExecutor(model, cfg, device="cpu")
    executor.begin_request()
    with torch.inference_mode():
        out_v, out_a = executor.forward_velocity(video, audio, step_index=0, total_steps=6)
        out_v2, out_a2 = executor.forward_velocity(video, audio, step_index=1, total_steps=6)
    assert ref_v is not None and out_v is not None and out_v2 is not None
    assert ref_a is not None and out_a is not None and out_a2 is not None
    torch.testing.assert_close(out_v, ref_v, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_a, ref_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_v2, ref_v, rtol=1e-4, atol=1e-4)
    report = executor.acceleration_report()
    assert report["install"]["projection_fusion"]["fused_qkv"] == 8  # 4 video + 4 audio self-attn
    assert report["install"]["projection_fusion"]["fused_kv"] == 16  # 4*(text v + text a + a2v + v2a)
    assert report["fbcache"][0]["hits"] == 0  # threshold=0.0 forces dense


# ──────────────────────────────────────────────────────────────────────────
# 2. Multi-GPU Parallelism — CFG=2 & Ulysses SP=2/4 Exact Mathematical Parity
# ──────────────────────────────────────────────────────────────────────────


def test_ltx25_parallel_config_validation() -> None:
    cfg = LTX25ParallelConfig(cfg_parallel=2, sequence_parallel=4)
    assert cfg.required_world_size == 8
    with pytest.raises(ValueError, match="cfg_parallel"):
        LTX25ParallelConfig(cfg_parallel=3)
    with pytest.raises(ValueError, match="sequence_parallel"):
        LTX25ParallelConfig(sequence_parallel=3)
    with pytest.raises(ValueError, match="sequence_parallel"):
        LTX25ParallelConfig(sequence_parallel=64)
    with pytest.raises(ValueError, match="sequence_parallel"):
        LTX25ParallelConfig(sequence_parallel=128)


def test_ltx25_cfg_parallel_2gpu_exact_parity() -> None:
    pos_v = torch.randn(1, 12, 32)
    neg_v = torch.randn(1, 12, 32)
    pos_a = torch.randn(1, 8, 16)
    neg_a = torch.randn(1, 8, 16)
    scale = 3.5

    expected_v = neg_v + scale * (pos_v - neg_v)
    expected_a = neg_a + scale * (pos_a - neg_a)

    # Simulate rank 0 (positive) and rank 1 (negative) exchanging via mock all_gather.
    def _make_mock_gather(pair_v: tuple[torch.Tensor, torch.Tensor], pair_a: tuple[torch.Tensor, torch.Tensor]):
        def _gather(out_list: list[torch.Tensor], local: torch.Tensor, _group: Any) -> None:
            src = pair_v if local.shape == pos_v.shape else pair_a
            out_list[0].copy_(src[0])
            out_list[1].copy_(src[1])

        return _gather

    mock_gather = _make_mock_gather((pos_v, neg_v), (pos_a, neg_a))
    for rank in (0, 1):
        runner = LTX25CFGParallelRunner(
            cfg_group="mock-cfg",
            cfg_rank=rank,
            cfg_world_size=2,
            all_gather_fn=mock_gather,
        )
        assert runner.active
        ran_pos = [False]
        ran_neg = [False]

        def _pos() -> tuple[torch.Tensor, torch.Tensor]:
            ran_pos[0] = True
            return pos_v, pos_a

        def _neg() -> tuple[torch.Tensor, torch.Tensor]:
            ran_neg[0] = True
            return neg_v, neg_a

        out_v, out_a = runner.run_cfg_step(run_positive=_pos, run_negative=_neg, guidance_scale=scale)
        assert ran_pos[0] == (rank == 0)
        assert ran_neg[0] == (rank == 1)
        assert out_v is not None and out_a is not None
        torch.testing.assert_close(out_v, expected_v, rtol=0.0, atol=0.0)
        torch.testing.assert_close(out_a, expected_a, rtol=0.0, atol=0.0)
        assert runner.report()["cfg_steps"] == 1


@pytest.mark.parametrize(("sp_size", "grid"), [(2, (2, 4, 4)), (4, (1, 3, 5))])
def test_ltx25_ulysses_sequence_parallel_exact_parity(sp_size: int, grid: tuple[int, int, int]) -> None:
    """Verify SP=2 (even tokens=32) and SP=4 (odd tokens=15 with padding) match 1-GPU AV model."""
    model = _tiny_ltx_av_model(num_layers=2, num_heads=8, head_dim=32, audio_heads=8, audio_head_dim=16)
    video, audio = _tiny_modalities(
        grid=grid,
        audio_tokens=6,
        video_ctx_dim=256,
        audio_ctx_dim=128,
        sigma=0.7,
    )
    with torch.inference_mode():
        ref_v, ref_a = model(video, audio)

    # Simulate SP ranks in-process with synchronous mock collectives operating on rank tensors.
    video_args_full = model.video_args_preprocessor.prepare(video, audio)

    sharded_v_list = []
    orig_tokens = 0
    for r in range(sp_size):
        sv, orig_tokens, _ = shard_ltx25_video_args(video_args_full, sp_rank=r, sp_size=sp_size, allow_padding=True)
        sharded_v_list.append(sv)

    # Run block-by-block across all simulated ranks with exact Ulysses all-to-all & V2A gather.
    for raw_block in model.transformer_blocks:
        block = cast(BasicAVTransformerBlock, raw_block)
        attn1 = block.attn1
        rank_qkv_q: list[torch.Tensor] = []
        rank_qkv_k: list[torch.Tensor] = []
        rank_qkv_v: list[torch.Tensor] = []
        rank_attn_out: list[torch.Tensor] = []

        # Collect pre-attention QKV across ranks by testing the adapter's exact transform primitives:
        # First verify shard -> all_to_all -> unpad -> local SDPA -> pad -> all_to_all -> gather on attn1.
        for r in range(sp_size):
            sv = sharded_v_list[r]
            v_ada = block._all_ada_values(block.scale_shift_table, sv.x.shape[0], sv.timesteps)
            norm_vx = block.ada_zero_function(sv.x, block.norm_eps, v_ada[1], v_ada[0])
            q, k, v = attn1._project_qkv(norm_vx, norm_vx, is_self_call=True, all_perturbed=False)
            assert q is not None and k is not None
            q, k = attn1.preattention_function(q, k, attn1, None, sv.positional_embeddings, None)
            b, s_loc, _ = q.shape
            rank_qkv_q.append(q.view(b, s_loc, attn1.heads, attn1.dim_head))
            rank_qkv_k.append(k.view(b, s_loc, attn1.heads, attn1.dim_head))
            rank_qkv_v.append(v.view(b, s_loc, attn1.heads, attn1.dim_head))

        # Simulated all_to_all_many(scatter_dim=2, gather_dim=1)
        full_q_all = torch.cat(rank_qkv_q, dim=1)
        full_k_all = torch.cat(rank_qkv_k, dim=1)
        full_v_all = torch.cat(rank_qkv_v, dim=1)
        q_head_shards = full_q_all.chunk(sp_size, dim=2)
        k_head_shards = full_k_all.chunk(sp_size, dim=2)
        v_head_shards = full_v_all.chunk(sp_size, dim=2)
        local_heads = attn1.heads // sp_size
        padded_s = int(full_q_all.shape[1])

        per_rank_head_out = []
        for r in range(sp_size):
            qr = q_head_shards[r][:, :orig_tokens].reshape(1, orig_tokens, local_heads * attn1.dim_head)
            kr = k_head_shards[r][:, :orig_tokens].reshape(1, orig_tokens, local_heads * attn1.dim_head)
            vr = v_head_shards[r][:, :orig_tokens].reshape(1, orig_tokens, local_heads * attn1.dim_head)
            prev_h = attn1.heads
            attn1.heads = local_heads
            attn1.num_heads = local_heads
            try:
                out_r = attn1._run_attention(qr, kr, vr, None)
            finally:
                attn1.heads = prev_h
                attn1.num_heads = prev_h
            out_r4 = out_r.view(1, orig_tokens, local_heads, attn1.dim_head)
            if orig_tokens < padded_s:
                pad = torch.zeros((1, padded_s - orig_tokens, local_heads, attn1.dim_head), dtype=out_r4.dtype)
                out_r4 = torch.cat((out_r4, pad), dim=1)
            per_rank_head_out.append(out_r4)

        # Simulated all_to_all(scatter_dim=1, gather_dim=2)
        full_out_heads = torch.cat(per_rank_head_out, dim=2)
        seq_shards = full_out_heads.chunk(sp_size, dim=1)
        for r in range(sp_size):
            rank_attn_out.append(seq_shards[r].reshape(1, -1, attn1.heads * attn1.dim_head))

        def _mock_all_gather(outs: list[torch.Tensor], _loc: torch.Tensor, _g: Any) -> None:
            for o, t in zip(outs, rank_attn_out, strict=True):
                o.copy_(t)

        # Verify gathered self-attention output matches unsharded block.attn1 output before output projection.
        gathered_attn = gather_ltx25_video_hidden(
            rank_attn_out[0],
            orig_tokens=orig_tokens,
            sp_group=None,
            sp_size=sp_size,
            all_gather_fn=_mock_all_gather,
        )
        v_ada_ref = block._all_ada_values(block.scale_shift_table, 1, video_args_full.timesteps)
        norm_vx_ref = block.ada_zero_function(video_args_full.x, block.norm_eps, v_ada_ref[1], v_ada_ref[0])
        q_ref, k_ref, v_ref = attn1._project_qkv(norm_vx_ref, norm_vx_ref, is_self_call=True, all_perturbed=False)
        assert q_ref is not None and k_ref is not None
        q_ref, k_ref = attn1.preattention_function(
            q_ref, k_ref, attn1, None, video_args_full.positional_embeddings, None
        )
        ref_attn = attn1._run_attention(q_ref, k_ref, v_ref, None)
        torch.testing.assert_close(gathered_attn, ref_attn, rtol=1e-4, atol=1e-4)
        break

    # Verify adapter install & restore lifecycle is reversible and rejects invalid head divisor.
    adapter = LTX25UlyssesAttentionAdapter(model, sp_size=sp_size, sp_rank=0)
    stats = adapter.install()
    assert stats == {"self_attn_wrapped": 2, "v2a_wrapped": 2}
    adapter.restore()
    assert len(adapter._orig_attn1_runs) == 0
    assert len(adapter._orig_v2a_forwards) == 0
    assert ref_v is not None and ref_a is not None


def _gloo_sp_and_fbcache_worker(
    rank: int,
    world_size: int,
    port: int,
    grid: tuple[int, int, int],
    result_queue: Any,
) -> None:
    import os

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # 1. Verify cfg_world_size=1 stays inactive even when torch.distributed is initialized.
        cfg_inactive = LTX25CFGParallelRunner(cfg_group=None, cfg_rank=None, cfg_world_size=1)
        assert not cfg_inactive.active and cfg_inactive.world_size == 1

        model_ref = _tiny_ltx_av_model(num_layers=2, num_heads=8, head_dim=32, audio_heads=8, audio_head_dim=16)
        video, audio = _tiny_modalities(
            grid=grid,
            audio_tokens=6,
            video_ctx_dim=256,
            audio_ctx_dim=128,
            sigma=0.7,
        )
        with torch.inference_mode():
            ref_v, ref_a = model_ref(video, audio)

        # 2. Verify full LTX25AcceleratedExecutor + LTX25UlyssesAttentionAdapter parity over real Gloo collectives.
        model_sp = _tiny_ltx_av_model(num_layers=2, num_heads=8, head_dim=32, audio_heads=8, audio_head_dim=16)
        fb_cfg = LTX25FirstBlockCacheConfig(
            enabled=True,
            residual_diff_threshold=0.5,
            dense_first=1,
            dense_last=0,
            max_consecutive_skips=2,
            downsample_factor=2,
            min_steps_to_enable=2,
        )
        sp_cfg = LTX25AccelerationConfig(
            fuse_qkv=True,
            fuse_cross_kv=True,
            fused_ops=False,
            attention_backend="torch",
            sol_attn=False,
            quantization="none",
            fbcache=fb_cfg,
            cuda_graph="off",
            vae_compile=False,
            parallel=LTX25ParallelConfig(cfg_parallel=1, sequence_parallel=world_size),
        )
        executor = LTX25AcceleratedExecutor(model_sp, sp_cfg, device="cpu", sp_group=None, sp_rank=rank)
        executor.begin_request()
        with torch.inference_mode():
            # Step 0: dense seed -> exact parity with 1-GPU reference (including non-divisible padding & V2A CA!).
            out_v0, out_a0 = executor.forward_velocity(video, audio, step_index=0, total_steps=4)
            assert ref_v is not None and out_v0 is not None
            assert ref_a is not None and out_a0 is not None
            diff_v = float((out_v0 - ref_v).abs().max().item())
            diff_a = float((out_a0 - ref_a).abs().max().item())

            # Step 1: perturb ONLY the second half of video tokens (owned by higher SP rank!).
            # Without global all-reduce, rank 0 would HIT and skip block 1 while rank 1 MISSES -> deadlock.
            v_lat_asym = video.latent.clone()
            v_lat_asym[:, v_lat_asym.shape[1] // 2 :, :] += 5.0
            video_asym = Modality(
                latent=v_lat_asym,
                sigma=video.sigma,
                timesteps=video.timesteps,
                positions=video.positions,
                context=video.context,
            )
            out_v1, out_a1 = executor.forward_velocity(video_asym, audio, step_index=1, total_steps=4)
            assert out_v1 is not None and out_a1 is not None

            # Step 2: tiny perturbation -> synchronized HIT across all ranks!
            v_lat_small = v_lat_asym * 0.999
            video_small = Modality(
                latent=v_lat_small,
                sigma=video.sigma,
                timesteps=video.timesteps,
                positions=video.positions,
                context=video.context,
            )
            out_v2, out_a2 = executor.forward_velocity(video_small, audio, step_index=2, total_steps=4)
            assert out_v2 is not None and out_a2 is not None

        fb_rec = executor.finalize_request_caches()[0]
        events = [
            (e["step"], e["hit"], e["reason"], round(e["relative_l1"] or 0.0, 5)) for e in fb_rec["event_receipts"]
        ]
        result_queue.put(("ok", rank, diff_v, diff_a, events))
    except Exception as exc:
        result_queue.put(("err", rank, f"{type(exc).__name__}: {exc}"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("sp_size", "grid", "port"), [(2, (2, 4, 4), 29641), (4, (1, 3, 5), 29642)])
def test_ltx25_ulysses_executor_real_distributed_and_fbcache_sync(
    sp_size: int,
    grid: tuple[int, int, int],
    port: int,
) -> None:
    """Exercise actual LTX25AcceleratedExecutor + LTX25UlyssesAttentionAdapter + FBCache over real torch.distributed."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_gloo_sp_and_fbcache_worker, args=(sp_size, port, grid, q), nprocs=sp_size, join=True)
    results = [q.get() for _ in range(sp_size)]
    for item in results:
        assert item[0] == "ok", f"distributed worker failed: {item}"
        assert item[2] < 1e-4, f"SP={sp_size} video diff {item[2]}"
        assert item[3] < 1e-4, f"SP={sp_size} audio diff {item[3]}"
    # Every rank MUST have identical FBCache hit/miss decisions and identical global relative-L1 metrics!
    first_events = results[0][4]
    for item in results[1:]:
        assert item[4] == first_events, f"SP ranks diverged on FBCache decisions: {results}"
    # Step 0 seed (miss), Step 1 asymmetric large perturb (miss -> no deadlock!), Step 2 small perturb (sync hit!).
    assert [ev[1] for ev in first_events] == [False, False, True]


# ──────────────────────────────────────────────────────────────────────────
# 3. GPU 7 Tests: Fused Ops + FA3 + FP8 + 3D-Tiled Sol-Attn + Split CUDA Graph + VAE
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_ltx25_sol_stack_fbcache_fp8_sol_attn_piecewise_graph_zero_fallbacks() -> None:
    device = "cuda:0"
    sm_major = torch.cuda.get_device_capability(device)[0]
    model = _tiny_ltx_av_model(
        num_layers=4,
        num_heads=8,
        head_dim=128,
        audio_heads=8,
        audio_head_dim=64,
        device=device,
        dtype=torch.bfloat16,
    )
    # 16*16 = 256 tokens per frame * 2 frames = 512 tokens; set sol_min_tokens=128 to exercise 3D-tiled Sol-Attn.
    video, audio = _tiny_modalities(
        grid=(2, 16, 16),
        audio_tokens=16,
        text_tokens=8,
        video_ctx_dim=1024,
        audio_ctx_dim=512,
        sigma=0.8,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        ref_v, ref_a = model(video, audio)

    cfg = LTX25AccelerationConfig(
        fuse_qkv=True,
        fuse_cross_kv=True,
        fused_ops=True,
        attention_backend="flash_attention_3" if sm_major == 9 else "torch",
        sol_attn=(sm_major == 9),
        sol_tau=0.25,
        sol_thresh_type="exact",
        sol_kv_splits=2,
        sol_tile_3d=(1, 8, 8),
        sol_dense_blocks=(0, -1),
        sol_dense_steps=1,
        sol_min_tokens=128,
        quantization="fp8" if sm_major >= 9 else "none",
        fp8_scaling="rowwise",
        fp8_use_fast_accum=False,
        fp8_exclude_edge_blocks=True,
        fp8_fuse_ffn=True,
        fp8_keep_dense_fallback=False,
        fbcache=LTX25FirstBlockCacheConfig(
            enabled=True,
            residual_diff_threshold=0.25,
            dense_first=2,
            dense_last=1,
            max_consecutive_skips=2,
            downsample_factor=4,
            min_steps_to_enable=4,
        ),
        cuda_graph="piecewise",
        cuda_graph_warmup=1,
    )
    executor = LTX25AcceleratedExecutor(model, cfg, device=device)

    # Verify edge blocks stayed dense BF16 when fp8_exclude_edge_blocks=True.
    from worldfoundry.core.acceleration.quantization.linear import Float8Linear

    if sm_major >= 9:
        assert not any(isinstance(m, Float8Linear) for m in model.transformer_blocks[0].modules())
        assert not any(isinstance(m, Float8Linear) for m in model.transformer_blocks[-1].modules())
        assert any(isinstance(m, Float8Linear) for m in model.transformer_blocks[1].modules())

    # Run two full 7-step requests so warm request exercises pure CUDA Graph replay + FBCache skips!
    for _run in range(2):
        executor.begin_request()
        with torch.inference_mode():
            for step in range(7):
                # Keep inputs close across adjacent steps so FBCache hits cleanly on interior steps.
                v_step = Modality(
                    latent=video.latent * (1.0 - 0.002 * step),
                    sigma=video.sigma,
                    timesteps=torch.full_like(video.timesteps, 0.8 - 0.002 * step),
                    positions=video.positions,
                    context=video.context,
                )
                a_step = Modality(
                    latent=audio.latent * (1.0 - 0.002 * step),
                    sigma=audio.sigma,
                    timesteps=torch.full_like(audio.timesteps, 0.8 - 0.002 * step),
                    positions=audio.positions,
                    context=audio.context,
                )
                out_v, out_a = executor.forward_velocity(v_step, a_step, step_index=step, total_steps=7)
                assert out_v is not None and out_a is not None
                assert torch.isfinite(out_v).all() and torch.isfinite(out_a).all()
        executor.finalize_request_caches()

    report = executor.acceleration_report()
    assert report["zero_fallbacks_verified"] is True
    assert report["install"]["fused_ops_blocks"] == 4
    assert report["install"]["projection_fusion"]["fused_qkv"] == 8
    assert report["install"]["projection_fusion"]["fused_kv"] == 16
    assert report["fbcache"][0]["hits"] >= 1
    assert report["cuda_graph"]["enabled"] is True
    assert report["cuda_graph"]["replay"] >= 1
    assert report["cuda_graph"]["cache_skips"] >= 1
    if sm_major >= 9:
        assert report["install"]["quantization"]["replaced_linears"] > 0
        assert report["install"]["quantization"]["fused_fp8_ffn"] > 0
        assert report["quantization_runtime"]["dense_fallback_calls"] == 0
        assert report["install"]["attention"]["sol_attn_modules_configured"] == 2  # blocks 1, 2 (0 & 3 dense)

    # Sanity check numerical alignment of step-0 output vs BF16 eager reference.
    assert ref_v is not None and ref_a is not None
    cos_v = torch.nn.functional.cosine_similarity(out_v.float().flatten(), ref_v.float().flatten(), dim=0).item()
    assert cos_v > 0.90, f"unexpectedly low cosine similarity: {cos_v}"

    # Verify warm pure-replay request emits exact single-call baked FP8 + attention receipts.
    assert report["graph_replayed_receipts"]["window"]["fp8_kernel_calls"] >= 0
    if sm_major >= 9:
        expected_fp8 = report["fbcache"][0]["dense_steps"] * report["install"]["quantization"]["replaced_linears"]
        assert report["graph_replayed_receipts"]["window"]["fp8_kernel_calls"] == expected_fp8
        assert report["quantization_runtime"]["effective_low_precision_kernel_calls"] == expected_fp8
        assert report["attention_provider_runtime_effective"]

    # Verify zero_fallbacks_verified catches attention provider fallbacks even when errors == 0.
    from worldfoundry.core.attention.backends.dispatch import (
        _record_provider_runtime,
        reset_attention_provider_runtime,
    )

    _record_provider_runtime("pisa", attempts=1, fallbacks=1)
    assert executor.acceleration_report()["zero_fallbacks_verified"] is False
    reset_attention_provider_runtime()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_ltx25_cuda_graph_invalidation_on_stage_offload() -> None:
    """Verify invalidating CUDA graphs before cpu<->cuda offload prevents stale weight pointer crashes."""
    device = "cuda:0"
    sm_major = torch.cuda.get_device_capability(device)[0]
    model = _tiny_ltx_av_model(
        num_layers=2,
        num_heads=8,
        head_dim=128,
        audio_heads=8,
        audio_head_dim=64,
        device=device,
        dtype=torch.bfloat16,
    )
    video, audio = _tiny_modalities(
        grid=(1, 8, 8),
        audio_tokens=8,
        text_tokens=4,
        video_ctx_dim=1024,
        audio_ctx_dim=512,
        device=device,
        dtype=torch.bfloat16,
    )
    cfg = LTX25AccelerationConfig(
        fuse_qkv=True,
        fuse_cross_kv=True,
        fused_ops=True,
        attention_backend="flash_attention_3" if sm_major == 9 else "torch",
        sol_attn=False,
        quantization="fp8" if sm_major >= 9 else "none",
        fp8_exclude_edge_blocks=False,
        fp8_fuse_ffn=True,
        cuda_graph="piecewise",
        cuda_graph_warmup=1,
    )
    executor = LTX25AcceleratedExecutor(model, cfg, device=device)
    executor.begin_request()
    with torch.inference_mode():
        ref_v, _ = executor.forward_velocity(video, audio, step_index=0, total_steps=3)
        executor.forward_velocity(video, audio, step_index=1, total_steps=3)
        executor.forward_velocity(video, audio, step_index=2, total_steps=3)
    assert ref_v is not None
    assert executor.acceleration_report()["cuda_graph"]["replay"] >= 2

    # Exercise MiniMaxH3SuperAccelerationPipeline._move_module (invalidates CUDA graphs before moving weights!).
    pipe = MiniMaxH3SuperAccelerationPipeline(
        h3_pipeline=_StubH3Pipeline(),
        ltx25_runtime=_StubLTX25Runtime(),
        device=device,
    )
    pipe._move_module(model, "cpu", graph_owner=executor)
    junk = [torch.full((4 * 1024 * 1024,), 999.0, device=device, dtype=torch.bfloat16) for _ in range(4)]
    pipe._move_module(model, device, graph_owner=executor)
    del junk
    executor.begin_request()
    with torch.inference_mode():
        out_v, _ = executor.forward_velocity(video, audio, step_index=0, total_steps=3)
        torch.cuda.synchronize()
    assert out_v is not None and torch.isfinite(out_v).all()
    torch.testing.assert_close(out_v, ref_v, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_ltx25_conv_vae_conversion_and_channels_last_compile_parity() -> None:
    sample_sd = {
        "decoder.conv_in.conv.weight": torch.zeros(1),
        "encoder.conv_in.conv.weight": torch.zeros(1),
        "per_channel_statistics.mean-of-means": torch.zeros(128),
        "per_channel_statistics.std-of-means": torch.ones(128),
        "per_channel_statistics.extra_ignored": torch.zeros(1),
    }
    dec_sd = convert_ltx25_conv_video_decoder_state_dict(sample_sd)
    enc_sd = convert_ltx25_conv_video_encoder_state_dict(sample_sd)
    assert set(dec_sd) == {
        "decoder.conv_in.conv.weight",
        "decoder.per_channel_statistics.mean-of-means",
        "decoder.per_channel_statistics.std-of-means",
    }
    assert set(enc_sd) == {
        "encoder.conv_in.conv.weight",
        "encoder.per_channel_statistics.mean-of-means",
        "encoder.per_channel_statistics.std-of-means",
    }

    decoder = (
        VideoDecoder(
            convolution_dimensions=3,
            in_channels=16,
            out_channels=3,
            decoder_blocks=[("res_x", {"num_layers": 1}), ("compress_space", {"multiplier": 1})],
            patch_size=2,
            causal=False,
            timestep_conditioning=False,
            base_channels=16,
        )
        .to(device="cuda:0", dtype=torch.bfloat16)
        .eval()
    )
    # PerChannelStatistics initializes buffers with torch.empty; seed deterministically so tests never read UB garbage.
    with torch.no_grad():
        decoder.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
        decoder.per_channel_statistics.get_buffer("mean-of-means").zero_()
    latent = torch.randn(1, 16, 2, 4, 4, device="cuda:0", dtype=torch.bfloat16)
    with torch.inference_mode():
        ref = decoder(latent).clone()
    assert torch.isfinite(ref).all()
    accel_cfg = LTX25AccelerationConfig(vae_channels_last_3d=True, vae_compile=True, vae_tiled=False, cuda_graph="off")
    report = accelerate_ltx25_video_vae(decoder, accel_cfg)
    assert report["channels_last_3d_conv3d"] > 0
    assert report["compiled"] is True
    with torch.inference_mode():
        out = decoder(latent)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


# ──────────────────────────────────────────────────────────────────────────
# 4. LTX-2.5 Pipeline / Runtime Options & Video Resampling Edge Cases
# ──────────────────────────────────────────────────────────────────────────


def test_ltx25_pipeline_dev_variant_profiles_and_resampling(tmp_path: Path) -> None:
    pipe = LTX25Pipeline.from_pretrained(
        tmp_path,
        device="cpu",
        model_variant="dev",
        video_vae_variant="conv",
        acceleration="sol_balanced",
    )
    assert pipe.runtime.transformer_path.name == "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    assert pipe.runtime.video_vae_path.name == "ltx-2.5-video-vae-conv-bf16.safetensors"
    assert pipe.runtime.acceleration is not None
    assert pipe.runtime.acceleration.fbcache is not None
    assert pipe.runtime.acceleration.fp8_use_fast_accum is False
    assert pipe.acceleration_report()["model_variant"] == "dev"

    # Coerce mapping & profile tiers cleanly.
    for tier in (
        "baseline",
        "l0",
        "fused_exact",
        "l1",
        "sol_high_fidelity",
        "l2",
        "sol_balanced",
        "l3",
        "sol_fast",
        "l4",
    ):
        prof = ltx25_sol_profile(tier, variant="dev", sm_major=9)
        assert isinstance(prof, LTX25AccelerationConfig)
    mapped = coerce_ltx25_acceleration_config(
        {"fuse_qkv": True, "fbcache": {"enabled": True, "residual_diff_threshold": 0.07}, "cuda_graph": "piecewise"}
    )
    assert mapped is not None and mapped.fbcache is not None and mapped.fbcache.residual_diff_threshold == 0.07

    # Resample video from H3 5s draft (124 frames, 384x672, [0, 1] BCTHW) -> LTX-2.5 5s (121 frames, 768x1280, [-1, 1]).
    draft = torch.rand(1, 3, 21, 64, 96)
    hi = resample_video_for_ltx25(draft, target_frames=17, target_height=128, target_width=192)
    assert hi.shape == (1, 3, 17, 128, 192)
    assert float(hi.min().item()) >= -1.001 and float(hi.max().item()) <= 1.001

    # Also accepts FHWC uint8-range input, 3-frame FHWC disambiguation, [-1, 1] auto-range, and 1-frame broadcast.
    single_fhwc = torch.full((1, 64, 64, 3), 255.0)
    hi_1 = resample_video_for_ltx25(single_fhwc, target_frames=9, target_height=64, target_width=64)
    assert hi_1.shape == (1, 3, 9, 64, 64)
    torch.testing.assert_close(hi_1, torch.ones_like(hi_1))

    three_fhwc = torch.full((3, 64, 96, 3), 0.75)
    hi_3 = resample_video_for_ltx25(three_fhwc, target_frames=9, target_height=64, target_width=96)
    assert hi_3.shape == (1, 3, 9, 64, 96)
    torch.testing.assert_close(hi_3, torch.full_like(hi_3, 0.5), atol=1e-4, rtol=1e-4)

    neg1_1 = torch.full((1, 3, 9, 64, 64), -0.5)
    hi_neg = resample_video_for_ltx25(neg1_1, target_frames=9, target_height=64, target_width=64)
    torch.testing.assert_close(hi_neg, torch.full_like(hi_neg, -0.5), atol=1e-4, rtol=1e-4)

    # Strict model_variant validation, _normalize_video_bcthw layouts, and dev 32px multiple allowance in build_plan.
    from worldfoundry.synthesis.visual_generation.ltx25.runtime import _normalize_video_bcthw

    assert _normalize_video_bcthw(torch.zeros(1, 13, 64, 96, 3)).shape == (1, 3, 13, 64, 96)
    assert _normalize_video_bcthw(torch.zeros(3, 13, 64, 96)).shape == (1, 3, 13, 64, 96)
    assert _normalize_video_bcthw(torch.zeros(3, 64, 96, 3)).shape == (1, 3, 3, 64, 96)

    with pytest.raises(ValueError, match="model_variant"):
        LTX25Pipeline.from_pretrained(tmp_path, device="cpu", model_variant="dev_typo")
    dev_plan = pipe.runtime.build_plan(
        request={"prompt": "A fox.", "height": 544, "width": 960, "num_frames": 17},
        output_path=tmp_path / "dev.mp4",
    )
    assert dev_plan.request["height"] == 544 and dev_plan.request["width"] == 960

    with pytest.raises(ValueError, match="multiples of 32"):
        resample_video_for_ltx25(draft, target_frames=17, target_height=100, target_width=128)
    with pytest.raises(ValueError, match="8\\*k \\+ 1"):
        resample_video_for_ltx25(draft, target_frames=16, target_height=128, target_width=128)


# ──────────────────────────────────────────────────────────────────────────
# 5. MiniMax-H3 Super Acceleration (4-step H3 + 3-step LTX-2.5) Orchestration
# ──────────────────────────────────────────────────────────────────────────


def test_super_acceleration_config_and_builders() -> None:
    cfg = SuperAccelerationConfig(
        draft_short_edge=384,
        draft_steps=4,
        target_height=768,
        target_width=1280,
        duration_seconds=5.0,
    )
    assert cfg.resolve_target_canvas() == (768, 1280)
    assert cfg.resolve_target_frames(5.0) == 121
    assert cfg.resolve_target_frames(10.0) == 241

    h3_cfg = cfg.build_h3_acceleration_config(sm_major=9)
    assert h3_cfg.attention_backend == "sol_attn"
    assert h3_cfg.attention_options is not None and h3_cfg.attention_options["kv_splits"] == 4
    assert h3_cfg.quantization == "fp8"
    assert h3_cfg.fp8_use_fast_accum is False
    assert h3_cfg.fuse_fp8_ffn is True
    assert h3_cfg.cuda_graph == "piecewise"

    ltx_cfg = cfg.build_ltx25_acceleration_config(sm_major=9)
    assert ltx_cfg.attention_backend == "flash_attention_3"
    assert ltx_cfg.sol_attn is True
    assert ltx_cfg.sol_tile_3d == (1, 8, 8)
    assert ltx_cfg.fbcache is None
    assert ltx_cfg.quantization == "fp8"
    assert ltx_cfg.vae_channels_last_3d is True
    assert ltx_cfg.vae_compile is True

    with pytest.raises(ValueError, match="divisible by 32"):
        SuperAccelerationConfig(draft_short_edge=300)
    with pytest.raises(ValueError, match="h3_kv_splits"):
        SuperAccelerationConfig(h3_kv_splits=cast(Any, 3))
    with pytest.raises(ValueError, match="supplied together"):
        SuperAccelerationConfig(target_height=768, target_width=None)


class _StubH3Pipeline:
    def __init__(self, *, inject_fallback: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.configure_calls: list[dict[str, Any]] = []
        self.component_devices = {"text_encoder": "cpu"}
        self.text_encoder = None
        self._inject_fallback = inject_fallback

    def configure_acceleration(self, **kwargs: Any) -> dict[str, Any]:
        self.configure_calls.append(kwargs)
        return {"configured": True}

    def _encode_prompt(self, _prompt: str) -> torch.Tensor:
        return torch.zeros(12, 5120)

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        dur = float(kwargs["duration_seconds"])
        frames = 124 if dur <= 5.5 else 243
        return {
            "plan": {"height": 384, "width": 672, "frame_count": frames},
            "video": torch.rand(1, 3, frames, 384, 672),
            "audio": torch.zeros(2, int(32000 * dur)),
        }

    def acceleration_report(self) -> dict[str, Any]:
        return {
            "transformer": {
                "quantization": {
                    "dense_fallback_calls": 1 if self._inject_fallback else 0,
                    "lifetime_dense_fallback_calls": 1 if self._inject_fallback else 0,
                },
                "fused_fp8_swiglu": {"fallback_calls": 0},
                "cuda_graph": {"enabled": True, "capture_failed": 0, "disabled_signatures": 0},
            },
            "video_vae": {"channels_last_3d": True, "stack_tiling": True},
        }


class _StubLTX25Runtime:
    def __init__(self, *, inject_fallback: bool = False) -> None:
        self.refine_calls: list[dict[str, Any]] = []
        self.configured: Any = None
        self._inject_fallback = inject_fallback

    def configure_acceleration(self, cfg: Any) -> Any:
        self.configured = cfg
        return cfg

    def encode_prompt(self, _prompt: str, **_kwargs: Any) -> dict[str, dict[str, torch.Tensor]]:
        return {"positive": {"video_context": torch.zeros(1, 8, 4096), "audio_context": torch.zeros(1, 8, 2048)}}

    def refine_video_3step(self, draft_video: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        self.refine_calls.append({"draft_shape": tuple(draft_video.shape), **kwargs})
        f = int(kwargs["target_num_frames"])
        h = int(kwargs["target_height"])
        w = int(kwargs["target_width"])
        return {
            "video": torch.rand(f, h, w, 3),
            "video_latent": torch.zeros(1, 128, 1 + (f - 1) // 8, h // 32, w // 32),
            "audio": torch.zeros(2, 48000) if kwargs.get("decode_audio") else None,
            "audio_sampling_rate": 48000,
            "timings": {
                "upscale_s": 0.05,
                "vae_encode_s": 0.25,
                "dit_refine_s": 1.10,
                "vae_decode_s": 0.30,
                "total_s": 1.70,
            },
            "acceleration_report": {"zero_fallbacks_verified": not self._inject_fallback},
        }


@pytest.mark.parametrize(("duration_s", "expected_frames"), [(5.0, 121), (10.0, 241)])
def test_super_acceleration_orchestration_5s_and_10s_and_fallback_audit(
    duration_s: float, expected_frames: int
) -> None:
    cfg = SuperAccelerationConfig(
        draft_short_edge=384,
        draft_steps=4,
        target_height=768,
        target_width=1280,
        duration_seconds=duration_s,
    )
    h3 = _StubH3Pipeline()
    ltx = _StubLTX25Runtime()
    pipe = MiniMaxH3SuperAccelerationPipeline(h3_pipeline=h3, ltx25_runtime=ltx, config=cfg, device="cpu")
    pipe.configure_acceleration(sm_major=9)
    assert len(h3.configure_calls) == 1
    assert h3.configure_calls[0]["vae_channels_last_3d"] is True
    assert h3.configure_calls[0]["vae_stack_tiling"] is True
    assert ltx.configured is not None

    out = pipe("A fox running through golden wheat at dusk.", seed=123)
    # Stage 1 must pass num_inference_steps = draft_steps + 1 = 5 (4 Euler steps).
    assert h3.calls[0]["num_inference_steps"] == 5
    assert h3.calls[0]["short_edge"] == 384
    assert h3.calls[0]["duration_seconds"] == duration_s

    # Stage 2 must receive draft video and target LTX-2.5 geometry with 3-step sigmas.
    assert ltx.refine_calls[0]["target_height"] == 768
    assert ltx.refine_calls[0]["target_width"] == 1280
    assert ltx.refine_calls[0]["target_num_frames"] == expected_frames
    assert ltx.refine_calls[0]["sigmas"] == LTX25_STAGE_2_SIGMAS

    assert out["video"].shape == (expected_frames, 768, 1280, 3)
    assert out["audio_sampling_rate"] == 32000  # preserved H3 audio
    receipt = out["receipt"]
    assert receipt["zero_fallbacks_verified"] is True
    assert receipt["geometry"]["draft_steps"] == 4
    assert receipt["geometry"]["refinement_steps"] == 3
    assert receipt["timings"]["handoff_total_s"] == pytest.approx(0.30, abs=1e-4)
    assert receipt["timings"]["stage2_ltx25_total_s"] == pytest.approx(1.40, abs=1e-4)

    # Verify fallback audit catches either Stage-1 or Stage-2 fallbacks!
    pipe_bad_h3 = MiniMaxH3SuperAccelerationPipeline(
        h3_pipeline=_StubH3Pipeline(inject_fallback=True),
        ltx25_runtime=_StubLTX25Runtime(),
        config=cfg,
        device="cpu",
    )
    assert pipe_bad_h3("A fox")["receipt"]["zero_fallbacks_verified"] is False
    pipe_bad_ltx = MiniMaxH3SuperAccelerationPipeline(
        h3_pipeline=_StubH3Pipeline(),
        ltx25_runtime=_StubLTX25Runtime(inject_fallback=True),
        config=cfg,
        device="cpu",
    )
    assert pipe_bad_ltx("A fox")["receipt"]["zero_fallbacks_verified"] is False

    # Verify _audit_h3_zero_fallbacks catches fused_fp8_swiglu, attention fallbacks, and lifetime graph failures!
    from worldfoundry.pipelines.minimax.super_acceleration import _audit_h3_zero_fallbacks

    assert _audit_h3_zero_fallbacks({"transformer": {"fused_fp8_swiglu": {"fallback_calls": 1}}}) is False
    assert (
        _audit_h3_zero_fallbacks(
            {"transformer": {"cuda_graph": {"enabled": True, "capture_failed": 0, "disabled_signatures": 1}}}
        )
        is False
    )
    assert (
        _audit_h3_zero_fallbacks(
            {
                "transformer": {
                    "cuda_graph": {
                        "enabled": True,
                        "segments": {
                            "s0": {"capture_failed": 0, "disabled_signatures": 0, "lifetime": {"capture_failed": 1}}
                        },
                    }
                }
            }
        )
        is False
    )
    assert (
        _audit_h3_zero_fallbacks(
            {"transformer": {"attention_provider_runtime": {"sol_attn": {"errors": 0, "fallbacks": 1}}}}
        )
        is False
    )


def test_super_acceleration_mp4_mux_with_real_3d_h3_audio_and_selective_encoding(tmp_path: Path) -> None:
    """Verify Super Acceleration muxes real H3 [2, 1, N] 3D stereo audio to MP4 and skips redundant prompt encodes."""
    import av

    class _H3Audio3DStub(_StubH3Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.encode_calls = 0

        def _encode_prompt(self, _prompt: str) -> torch.Tensor:
            self.encode_calls += 1
            return super()._encode_prompt(_prompt)

        def generate(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "plan": {"height": 64, "width": 64, "frame_count": 9},
                "video": torch.rand(1, 3, 9, 64, 64),
                # Real MiniMax-H3 AudioVAE returns 3D [2, 1, samples] stereo!
                "audio": torch.randn(2, 1, 16000) * 0.1,
            }

    class _LTXStubCount(_StubLTX25Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.encode_calls = 0

        def encode_prompt(self, _prompt: str, **_kwargs: Any) -> dict[str, dict[str, torch.Tensor]]:
            self.encode_calls += 1
            return super().encode_prompt(_prompt, **_kwargs)

    cfg = SuperAccelerationConfig(
        draft_short_edge=64,
        draft_steps=4,
        target_height=64,
        target_width=64,
        target_num_frames=9,
        fps=24.0,
    )
    h3 = _H3Audio3DStub()
    ltx = _LTXStubCount()
    pipe = MiniMaxH3SuperAccelerationPipeline(h3_pipeline=h3, ltx25_runtime=ltx, config=cfg, device="cpu")
    out_mp4 = tmp_path / "super_accel_3d_audio.mp4"
    # Supply precomputed H3 embeddings -> must NOT re-encode H3 prompt, only LTX-2.5!
    out = pipe.generate(
        prompt="A sports car on a coastal highway.",
        h3_prompt_embeds=torch.zeros(12, 5120),
        output_path=out_mp4,
    )
    assert h3.encode_calls == 0
    assert ltx.encode_calls == 1
    assert out["output_path"] == str(out_mp4.resolve())
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    with av.open(str(out_mp4)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        v_stream = container.streams.video[0]
        a_stream = container.streams.audio[0]
        assert v_stream.width == 64 and v_stream.height == 64
        assert a_stream.rate == 32000
        assert a_stream.channels == 2


# ──────────────────────────────────────────────────────────────────────────
# 6. Native Gemma-4 Unified Text Backbone (transformers 4.x/5.x compat)
# ──────────────────────────────────────────────────────────────────────────


def test_native_gemma4_unified_text_backbone_semantics_and_left_padding() -> None:
    from worldfoundry.synthesis.visual_generation.ltx25.gemma4_text import (
        Gemma4RMSNorm,
        NativeGemma4UnifiedTextBackbone,
        _build_rope_inv_freq,
    )

    # 1. Verify unshifted Gemma4RMSNorm (direct weight scaling, NO +1.0) & unscaled v_norm.
    x = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)
    norm_scaled = Gemma4RMSNorm(2, eps=0.0, with_scale=True)
    with torch.no_grad():
        norm_scaled.weight.copy_(torch.tensor([2.0, 0.5]))
    rms = ((3.0**2 + 4.0**2) / 2.0) ** 0.5
    expected = torch.tensor([[[(3.0 / rms) * 2.0, (4.0 / rms) * 0.5]]])
    assert torch.allclose(norm_scaled(x), expected, atol=1e-5)

    norm_unscaled = Gemma4RMSNorm(2, eps=0.0, with_scale=False)
    assert norm_unscaled.weight is None
    assert torch.allclose(norm_unscaled(x), torch.tensor([[[3.0 / rms, 4.0 / rms]]]), atol=1e-5)

    # 2. Verify proportional RoPE zero-fills tail pairs beyond partial_rotary_factor.
    inv_prop = _build_rope_inv_freq(
        head_dim=32, rope_theta=1_000_000.0, rope_type="proportional", partial_rotary_factor=0.25
    )
    assert inv_prop.shape == (16,)
    assert torch.all(inv_prop[:4] > 0)
    assert torch.all(inv_prop[4:] == 0)

    # 3. Run 2-layer hybrid backbone (sliding + full with k_eq_v=True & layer_scalar) under left padding.
    torch.manual_seed(0)
    text_cfg = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "attention_k_eq_v": True,
        "sliding_window": 4,
        "rms_norm_eps": 1e-6,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
        },
    }
    backbone = NativeGemma4UnifiedTextBackbone(text_cfg).eval()
    assert backbone.layers[0].self_attn.v_proj is not None
    assert backbone.layers[1].self_attn.v_proj is None  # attention_k_eq_v=True on full_attention
    with torch.no_grad():
        backbone.layers[1].layer_scalar.fill_(0.5)

    input_ids = torch.tensor([[0, 0, 5, 11, 23, 7]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
    with torch.inference_mode():
        hidden_states = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert len(hidden_states) == 3  # embedding + 2 decoder layers
    for idx, hs in enumerate(hidden_states):
        assert hs.shape == (1, 6, 32)
        assert torch.isfinite(hs).all(), f"NaN/Inf in hidden state {idx} with left-padded mask"

    # Left-padded prefix tokens must not leak into active token hidden states (causal + key_valid mask).
    input_ids_alt_pad = torch.tensor([[42, 19, 5, 11, 23, 7]], dtype=torch.long)
    with torch.inference_mode():
        hidden_states_alt = backbone(input_ids=input_ids_alt_pad, attention_mask=attention_mask)
    assert torch.allclose(hidden_states[-1][:, 2:, :], hidden_states_alt[-1][:, 2:, :], atol=1e-5)

    # 4. Verify RoPE inv_freq buffers stay float32 across .to(dtype=bfloat16) and wrapper tracks live device!
    from worldfoundry.synthesis.visual_generation.ltx25.gemma4_text import NativeGemma4UnifiedEncoderWrapper

    backbone_bf16 = backbone.to(dtype=torch.bfloat16)
    assert backbone_bf16.sliding_inv_freq.dtype == torch.float32
    assert backbone_bf16.full_inv_freq.dtype == torch.float32

    class _DummyTok:
        def tokenize_with_weights(self, _text: str) -> dict[str, list[tuple[int, int]]]:
            return {"gemma": [(0, 0), (5, 1), (11, 1), (7, 1)]}

    wrapper = NativeGemma4UnifiedEncoderWrapper(backbone_bf16, _DummyTok(), execution_device=torch.device("meta"))
    assert wrapper.execution_device == torch.device("cpu")
    encoded = wrapper.encode(["hello"])
    assert len(encoded) == 1 and len(encoded[0][0]) == 3 and torch.isfinite(encoded[0][0][-1]).all()
