"""Tests for GPU-resident multi-GPU parallel VideoVAE decode infrastructure.

Covers Wan2.1/Wan2.2 (WanVideoVAE / WanVideoVAE38), HunyuanVideo-1.5
(AutoencoderKLConv3D), HunyuanVideo-13B (HunyuanVideoCausal3DAutoencoder),
and LTX Conv VideoDecoder across both CPU Gloo and 2-GPU CUDA NCCL collectives.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import socket
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist

from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.causal3d import (
    HunyuanVideoCausal3DAutoencoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.h15 import (
    AutoencoderKLConv3D,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import (
    NormLayerType,
    PaddingModeType,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import (
    VideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    WanVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    VideoVAE38_,
    VideoVAE_,
    WanVideoVAE,
    WanVideoVAE38,
)
from worldfoundry.core.acceleration.convolution_layout import (
    convert_convolution_weight_layouts,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.float().cpu() - b.float().cpu()
    mse = float(diff.square().mean().item())
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(torch.log10(torch.tensor(4.0 / mse)).item())


def _tiny_wan_vae(
    *,
    vae38: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> WanVideoVAE:
    torch.manual_seed(17 if not vae38 else 19)
    if vae38:
        vae = WanVideoVAE38.__new__(WanVideoVAE38)
        torch.nn.Module.__init__(vae)
        vae.z_dim = 12
        vae.upsampling_factor = 16
        vae.mean = torch.zeros(12)
        vae.std = torch.ones(12)
        vae.scale = [vae.mean, vae.std]
        vae.decode_autocast_dtype = None
        vae.model = (
            VideoVAE38_(
                dim=8,
                z_dim=12,
                dec_dim=16,
                dim_mult=[1, 1, 1, 1],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False, True, True],
            )
            .eval()
            .requires_grad_(False)
        )
    else:
        vae = WanVideoVAE.__new__(WanVideoVAE)
        torch.nn.Module.__init__(vae)
        vae.z_dim = 4
        vae.upsampling_factor = 8
        vae.mean = torch.zeros(4)
        vae.std = torch.ones(4)
        vae.scale = [vae.mean, vae.std]
        vae.decode_autocast_dtype = None
        vae.model = (
            VideoVAE_(
                dim=8,
                z_dim=4,
                dim_mult=[1, 1, 1, 1],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False, True, True],
            )
            .eval()
            .requires_grad_(False)
        )
    return vae.to(device=device, dtype=dtype).eval()


def _tiny_h15_vae(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AutoencoderKLConv3D:
    torch.manual_seed(23)
    vae = AutoencoderKLConv3D(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        ffactor_spatial=4,
        ffactor_temporal=2,
        sample_size=16,
        sample_tsize=4,
    )
    vae.tile_overlap_factor = 0.25
    vae.enable_spatial_tiling(True)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _tiny_causal3d_vae(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HunyuanVideoCausal3DAutoencoder:
    torch.manual_seed(29)
    vae = HunyuanVideoCausal3DAutoencoder(
        in_channels=3,
        out_channels=3,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
        sample_tsize=8,
        spatial_compression_ratio=4,
        time_compression_ratio=2,
    )
    vae.tile_overlap_factor = 0.25
    vae.enable_spatial_tiling(True)
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _tiny_ltx_decoder(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> VideoDecoder:
    torch.manual_seed(31)
    blocks: list[tuple[str, int | dict[str, Any]]] = [
        ("res_x", {"num_layers": 1}),
        ("compress_all", {"multiplier": 2}),
        ("res_x", {"num_layers": 1}),
        ("compress_all", {"multiplier": 2}),
        ("res_x", {"num_layers": 1}),
        ("compress_all", {"multiplier": 2}),
    ]
    dec = VideoDecoder(
        convolution_dimensions=3,
        in_channels=16,
        out_channels=3,
        decoder_blocks=blocks,
        patch_size=4,
        norm_layer=NormLayerType.PIXEL_NORM,
        causal=False,
        timestep_conditioning=False,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        base_channels=2,
    )
    dec.per_channel_statistics.get_buffer("std-of-means").fill_(1.0)
    dec.per_channel_statistics.get_buffer("mean-of-means").zero_()
    return dec.to(device=device, dtype=dtype).eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# Unit tests (no distributed process required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (60, 104),  # 480p Wan2.1
        (90, 160),  # 720p Wan2.1
        (135, 240),  # 1080p Wan2.1
        (45, 80),  # 720p Wan2.2 VAE38
        (16, 24),  # compact test shape
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_wan_parallel_tile_config_for_world_size_covers_grid(
    height: int,
    width: int,
    world_size: int,
) -> None:
    tile_size, tile_stride = WanVideoVAE.parallel_tile_config_for_world_size(
        height,
        width,
        world_size,
        min_overlap=4,
    )
    tasks = WanVideoVAE._spatial_tile_tasks(height, width, tile_size, tile_stride)
    assert len(tasks) == world_size
    assert all(stride <= size for size, stride in zip(tile_size, tile_stride, strict=True))
    covered = torch.zeros((height, width), dtype=torch.bool)
    for h, h_, w, w_ in tasks:
        covered[h : min(h_, height), w : min(w_, width)] = True
    assert bool(covered.all())


def test_wan_merge_spatial_decode_tiles_oom_fallback_to_cpu(monkeypatch) -> None:
    vae = WanVideoVAE.__new__(WanVideoVAE)
    torch.nn.Module.__init__(vae)
    vae.upsampling_factor = 1
    hidden = torch.zeros((1, 3, 2, 6, 6), dtype=torch.float32)
    tasks = vae._spatial_tile_tasks(6, 6, (4, 4), (3, 3))
    tiles = [torch.full((1, 3, 5, min(h_, 6) - h, min(w_, 6) - w), 0.25) for h, h_, w, w_ in tasks]

    real_zeros = torch.zeros
    oom_calls = 0

    def flaky_zeros(shape, *args, **kwargs):
        nonlocal oom_calls
        dev = kwargs.get("device")
        if dev is not None and torch.device(dev).type == "cuda":
            oom_calls += 1
            raise torch.OutOfMemoryError("simulated small GPU OOM")
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", flaky_zeros)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    merged = vae._merge_spatial_decode_tiles(
        tiles,
        tasks,
        hidden_states=hidden,
        tile_size=(4, 4),
        tile_stride=(3, 3),
        merge_device=torch.device("cuda:0"),
    )
    assert oom_calls == 1
    assert merged.device.type == "cpu"
    assert merged.shape == (1, 3, 5, 6, 6)
    torch.testing.assert_close(merged, torch.full_like(merged, 0.25))


def test_wan_component_parallel_decode_avoids_cpu_staging_on_cuda(monkeypatch) -> None:
    seen_devices: list[str] = []

    class _FakeParallelVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.z_dim = 4
            self.model = SimpleNamespace(clear_cache=lambda: None)
            self.decode_autocast_dtype = None

        _spatial_tile_tasks = staticmethod(WanVideoVAE._spatial_tile_tasks)

        def _decode_autocast(self, input_dtype=None):
            del input_dtype
            return contextlib.nullcontext()

        def parallel_tiled_decode(self, hidden_states, device, *args, **kwargs):
            del device, args, kwargs
            seen_devices.append(hidden_states.device.type)
            return hidden_states[:, :3]

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    vae = _FakeParallelVAE()
    decoder = WanVideoDecoder(
        cast(WanVideoVAE, vae),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        parallel_degree=2,
        tile_size=(4, 4),
        tile_stride=(3, 3),
        temporal_chunk_size=0,
    )
    if torch.cuda.is_available():
        latents = torch.randn(1, 4, 2, 6, 6, device="cuda:0")
        out = decoder.decode(latents)
        assert seen_devices == ["cuda"]
        assert out.device.type == "cuda"

        decoder_chunked = WanVideoDecoder(
            cast(WanVideoVAE, vae),
            device=torch.device("cuda:0"),
            parallel_degree=2,
            tile_size=(4, 4),
            tile_stride=(3, 3),
            temporal_chunk_size=1,
        )
        seen_devices.clear()
        decoder_chunked.decode(latents)
        assert seen_devices == ["cpu"]
    else:
        latents = torch.randn(1, 4, 2, 6, 6)
        decoder.decode(latents)
        assert seen_devices == ["cpu"]


def test_ltx_spatial_only_tiling_without_temporal_config() -> None:
    dec = _tiny_ltx_decoder(device="cpu", dtype=torch.float32)
    latent = torch.randn(1, 16, 3, 6, 8)
    cfg = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=32),
        temporal_config=None,
    )
    with torch.inference_mode():
        chunks = list(dec.tiled_decode(latent, cfg))
    assert len(chunks) == 1
    assert chunks[0].shape == (1, 3, 17, 192, 256)


# ---------------------------------------------------------------------------
# 2-rank CPU Gloo distributed workers & tests
# ---------------------------------------------------------------------------


def _gloo_wan_worker(rank: int, world_size: int, init_path: str, queue) -> None:
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        results: dict[str, bool] = {}
        with torch.inference_mode():
            vae = _tiny_wan_vae(vae38=False, device="cpu", dtype=torch.float32)
            torch.manual_seed(101)
            hidden = torch.randn(1, 4, 2, 6, 8, dtype=torch.float32)
            ref = vae.tiled_decode(hidden, torch.device("cpu"), (4, 5), (3, 4))
            par = vae.parallel_tiled_decode(hidden, torch.device("cpu"), (4, 5), (3, 4))
            results["standard_bit_exact"] = bool(torch.equal(par, ref))

            # Edge case: world_size > len(tasks) (1 tile across 2 ranks)
            ref_single = vae.tiled_decode(hidden, torch.device("cpu"), (6, 8), (6, 8))
            par_single = vae.parallel_tiled_decode(hidden, torch.device("cpu"), (6, 8), (6, 8))
            results["single_tile_bit_exact"] = bool(torch.equal(par_single, ref_single))

            # Edge case: rank-1 error propagates cleanly without deadlock
            class _FailingBackbone:
                def decode(self, value, scale):
                    del scale
                    if rank == 1:
                        raise ValueError("synthetic tile error on rank 1")
                    return value

            err_vae = WanVideoVAE.__new__(WanVideoVAE)
            torch.nn.Module.__init__(err_vae)
            err_vae.model = cast(Any, _FailingBackbone())
            err_vae.scale = [torch.zeros(3), torch.ones(3)]
            err_vae.z_dim = 3
            err_vae.upsampling_factor = 1
            err_in = torch.randn(1, 3, 1, 4, 4)
            caught = False
            try:
                err_vae.parallel_tiled_decode(err_in, torch.device("cpu"), (3, 3), (2, 2))
            except RuntimeError as exc:
                caught = "synthetic tile error on rank 1" in str(exc)
            results["error_propagated"] = caught

        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


def _gloo_hunyuan_and_ltx_worker(rank: int, world_size: int, init_path: str, queue) -> None:
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size)
    try:
        results: dict[str, bool] = {}
        with torch.inference_mode():
            # 1. HunyuanVideo-1.5 AutoencoderKLConv3D (batch=2 and world_size > tiles)
            h15 = _tiny_h15_vae(device="cpu", dtype=torch.float32)
            torch.manual_seed(201)
            z_h15 = torch.randn(2, 4, 3, 6, 8, dtype=torch.float32)
            h15.disable_tile_parallelism()
            ref_h15 = h15.spatial_tiled_decode(z_h15)
            h15.enable_tile_parallelism()
            par_h15 = h15.tile_parallel_spatial_tiled_decode(z_h15)
            results["h15_bit_exact"] = bool(torch.equal(par_h15, ref_h15)) if rank == 0 else (par_h15.numel() == 0)

            # world_size > total_tiles (1 tile across 2 ranks -> previously UnboundLocalError!)
            z_h15_small = torch.randn(1, 4, 2, 3, 3, dtype=torch.float32)
            h15.disable_tile_parallelism()
            ref_h15_small = h15.spatial_tiled_decode(z_h15_small)
            h15.enable_tile_parallelism()
            par_h15_small = h15.tile_parallel_spatial_tiled_decode(z_h15_small)
            results["h15_single_tile_no_unbound"] = (
                bool(torch.equal(par_h15_small, ref_h15_small)) if rank == 0 else (par_h15_small.numel() == 0)
            )

            # 2. HunyuanVideo-13B Causal3D (direct + enable_tile_parallelism via decode())
            c3d = _tiny_causal3d_vae(device="cpu", dtype=torch.float32)
            torch.manual_seed(202)
            z_c3d = torch.randn(2, 4, 3, 6, 8, dtype=torch.float32)
            c3d.disable_tile_parallelism()
            ref_c3d = c3d.decode(z_c3d).sample
            par_c3d = c3d.parallel_spatial_tiled_decode(z_c3d).sample
            results["causal3d_direct_bit_exact"] = bool(torch.equal(par_c3d, ref_c3d))

            c3d.enable_tile_parallelism(True)
            dispatched_c3d = c3d.decode(z_c3d).sample
            results["causal3d_dispatch_bit_exact"] = bool(torch.equal(dispatched_c3d, ref_c3d))

            # world_size > total_tiles on Causal3D
            z_c3d_small = torch.randn(1, 4, 2, 3, 3, dtype=torch.float32)
            c3d.disable_tile_parallelism()
            ref_c3d_small = c3d._spatial_tiled_decode(z_c3d_small).sample
            par_c3d_small = c3d.parallel_spatial_tiled_decode(z_c3d_small).sample
            results["causal3d_single_tile_bit_exact"] = bool(torch.equal(par_c3d_small, ref_c3d_small))

            # 3. LTX Conv VideoDecoder (spatial-only + spatio-temporal tiling)
            ltx = _tiny_ltx_decoder(device="cpu", dtype=torch.float32)
            torch.manual_seed(203)
            z_ltx = torch.randn(1, 16, 4, 6, 8, dtype=torch.float32)
            cfg_spatial = TilingConfig(
                spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=32),
                temporal_config=None,
            )
            ref_ltx_sp = torch.cat(list(ltx.tiled_decode(z_ltx, cfg_spatial)), dim=2)
            par_ltx_sp = torch.cat(list(ltx.parallel_tiled_decode(z_ltx, cfg_spatial)), dim=2)
            results["ltx_spatial_lossless"] = _psnr(par_ltx_sp, ref_ltx_sp) >= 120.0

            cfg_st = TilingConfig(
                spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=32),
                temporal_config=TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8),
            )
            ref_ltx_st = torch.cat(list(ltx.tiled_decode(z_ltx, cfg_st)), dim=2)
            ltx.enable_tile_parallelism(True)
            par_ltx_st = torch.cat(list(ltx.tiled_decode(z_ltx, cfg_st)), dim=2)
            results["ltx_spatiotemporal_lossless"] = _psnr(par_ltx_st, ref_ltx_st) >= 120.0

        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


def _run_mp_test(worker, tmp_path, *, world_size: int = 2) -> list[dict[str, bool]]:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    init_path = str(tmp_path / "rendezvous")
    procs = [ctx.Process(target=worker, args=(r, world_size, init_path, queue)) for r in range(world_size)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker exited with {proc.exitcode}"
    collected = dict(queue.get(timeout=5) for _ in procs)
    return [collected[r] for r in range(world_size)]


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_wan_parallel_tiled_decode_gloo_2rank(tmp_path) -> None:
    per_rank = _run_mp_test(_gloo_wan_worker, tmp_path, world_size=2)
    for rank_results in per_rank:
        assert rank_results == {
            "standard_bit_exact": True,
            "single_tile_bit_exact": True,
            "error_propagated": True,
        }


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_hunyuan_and_ltx_parallel_decode_gloo_2rank(tmp_path) -> None:
    per_rank = _run_mp_test(_gloo_hunyuan_and_ltx_worker, tmp_path, world_size=2)
    for rank_results in per_rank:
        assert all(rank_results.values()), f"failed checks: {rank_results}"


# ---------------------------------------------------------------------------
# 2-GPU CUDA NCCL distributed tests (zero CPU round-trips, CL3D + BF16)
# ---------------------------------------------------------------------------


def _nccl_2gpu_worker(rank: int, world_size: int, port: int, queue) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        results: dict[str, object] = {}
        with torch.inference_mode():
            # 1. Wan2.1 BF16 CL3D GPU-resident bit-exact
            wan = _tiny_wan_vae(vae38=False, device=device, dtype=torch.bfloat16)
            convert_convolution_weight_layouts(wan.model, conv2d=True, conv3d=True)
            torch.manual_seed(301)
            z_wan = torch.randn(1, 4, 3, 12, 16, device=device, dtype=torch.bfloat16)
            tile_size, tile_stride = WanVideoVAE.parallel_tile_config_for_world_size(12, 16, 4, min_overlap=2)
            ref_wan = wan.tiled_decode(z_wan, device, tile_size, tile_stride)
            par_wan = wan.parallel_tiled_decode(z_wan, device, tile_size, tile_stride)
            results["wan21_cuda_resident"] = bool(ref_wan.is_cuda and par_wan.is_cuda)
            results["wan21_bit_exact"] = bool(torch.equal(par_wan, ref_wan))

            # 2. Wan2.2 VAE38 BF16 CL3D GPU-resident bit-exact
            wan38 = _tiny_wan_vae(vae38=True, device=device, dtype=torch.bfloat16)
            convert_convolution_weight_layouts(wan38.model, conv2d=True, conv3d=True)
            torch.manual_seed(302)
            z_wan38 = torch.randn(1, 12, 3, 12, 16, device=device, dtype=torch.bfloat16)
            ref_wan38 = wan38.tiled_decode(z_wan38, device, tile_size, tile_stride)
            par_wan38 = wan38.parallel_tiled_decode(z_wan38, device, tile_size, tile_stride)
            results["wan22_vae38_cuda_resident"] = bool(ref_wan38.is_cuda and par_wan38.is_cuda)
            results["wan22_vae38_bit_exact"] = bool(torch.equal(par_wan38, ref_wan38))

            # 3. HunyuanVideo-1.5 BF16 CL3D (tests contiguous gather fix!)
            h15 = _tiny_h15_vae(device=device, dtype=torch.bfloat16)
            convert_convolution_weight_layouts(h15.decoder, conv3d=True)
            torch.manual_seed(303)
            z_h15 = torch.randn(1, 4, 3, 8, 12, device=device, dtype=torch.bfloat16)
            h15.disable_tile_parallelism()
            ref_h15 = h15.spatial_tiled_decode(z_h15)
            h15.enable_tile_parallelism()
            par_h15 = h15.tile_parallel_spatial_tiled_decode(z_h15)
            if rank == 0:
                results["h15_cuda_bit_exact"] = bool(par_h15.is_cuda and torch.equal(par_h15, ref_h15))
            else:
                results["h15_cuda_bit_exact"] = bool(par_h15.is_cuda and par_h15.numel() == 0)

            # 4. HunyuanVideo-13B Causal3D BF16 CL3D bit-exact
            c3d = _tiny_causal3d_vae(device=device, dtype=torch.bfloat16)
            convert_convolution_weight_layouts(c3d.decoder, conv3d=True)
            torch.manual_seed(304)
            z_c3d = torch.randn(1, 4, 3, 8, 12, device=device, dtype=torch.bfloat16)
            c3d.disable_tile_parallelism()
            ref_c3d = c3d.decode(z_c3d).sample
            c3d.enable_tile_parallelism(True)
            par_c3d = c3d.decode(z_c3d).sample
            results["causal3d_cuda_bit_exact"] = bool(par_c3d.is_cuda and torch.equal(par_c3d, ref_c3d))

            # 5. LTX Conv VideoDecoder BF16 CL3D lossless parity
            ltx = _tiny_ltx_decoder(device=device, dtype=torch.bfloat16)
            ltx.enable_channels_last_3d()
            torch.manual_seed(305)
            z_ltx = torch.randn(1, 16, 3, 6, 8, device=device, dtype=torch.bfloat16)
            cfg = TilingConfig(
                spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=32),
                temporal_config=None,
            )
            ref_ltx = torch.cat(list(ltx.tiled_decode(z_ltx, cfg)), dim=2)
            par_ltx = torch.cat(list(ltx.parallel_tiled_decode(z_ltx, cfg)), dim=2)
            psnr_ltx = _psnr(par_ltx, ref_ltx)
            results["ltx_cuda_resident"] = bool(par_ltx.is_cuda)
            results["ltx_cuda_psnr_ge_70"] = bool(psnr_ltx >= 70.0)

        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(
    not (dist.is_available() and torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="requires at least 2 CUDA GPUs and torch.distributed",
)
def test_all_vaes_nccl_2gpu_gpu_resident_and_lossless() -> None:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_nccl_2gpu_worker, args=(r, 2, port, queue)) for r in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0, f"NCCL worker exited with {proc.exitcode}"
    collected = dict(queue.get(timeout=5) for _ in procs)
    for rank in range(2):
        assert all(collected[rank].values()), f"rank {rank} failed: {collected[rank]}"
