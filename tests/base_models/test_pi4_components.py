# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free unit tests for Pi4 architecture, state_dict schema, sparse 3D fallback, and upstream parity."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.geometry import depth_normal_edge
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.decoder import (
    GeometryDecoder,
    MotionDecoder,
)
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.encoder import Encoder
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.heads import (
    CameraHead,
    MetricHead,
    MotionMask,
    PointHead,
    Tracker,
)
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.layers import patch_positions
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.pi4 import Pi4
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.models.refiner import (
    DepthRefiner,
    SparsePool3d,
    SparseUpsample3d,
    SubmanifoldConv3d,
)
from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.postprocess import geometry_confidence

pytestmark = pytest.mark.unit

UPSTREAM_PI4_ROOT = Path(__file__).resolve().parents[2] / "tmp" / "upstream_refs" / "Pi4"


def _install_mock_flex_gemm(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install a lightweight ``flex_gemm.nn`` stub in ``sys.modules`` for upstream import / delegation tests."""
    calls = {"conv": 0, "pool": 0, "upsample": 0}

    class _MockSubmanifoldConv3d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True, **kwargs):
            super().__init__()
            del kwargs
            k = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
            self.weight = nn.Parameter(torch.empty(out_channels, *k, in_channels))
            self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        def forward(self, features, coordinates, shape, neighbor_cache=None):
            calls["conv"] += 1
            return features, neighbor_cache or {"mock_conv": True}

    class _MockSparsePool3d(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            del kwargs

        def forward(self, features, coordinates, shape, output_shape=None):
            calls["pool"] += 1
            return features, coordinates, output_shape or shape, types.SimpleNamespace(T="mock_T")

    class _MockSparseUpsample3d(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            del kwargs

        def forward(self, features, coordinates, shape, output_coords=None, output_shape=None, neighbor_cache=None):
            calls["upsample"] += 1
            return features, output_coords, output_shape, neighbor_cache

    fg = types.ModuleType("flex_gemm")
    fg_nn = types.ModuleType("flex_gemm.nn")
    setattr(fg_nn, "SubmanifoldConv3d", _MockSubmanifoldConv3d)
    setattr(fg_nn, "SparsePool3d", _MockSparsePool3d)
    setattr(fg_nn, "SparseUpsample3d", _MockSparseUpsample3d)
    setattr(fg, "nn", fg_nn)
    monkeypatch.setitem(sys.modules, "flex_gemm", fg)
    monkeypatch.setitem(sys.modules, "flex_gemm.nn", fg_nn)
    return calls


def _import_upstream_pi4(monkeypatch: pytest.MonkeyPatch):
    """Import the pinned local upstream ``pi4`` reference package."""
    _install_mock_flex_gemm(monkeypatch)
    monkeypatch.syspath_prepend(str(UPSTREAM_PI4_ROOT))
    for name in list(sys.modules):
        if name == "pi4" or name.startswith("pi4."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("pi4")


def _randomize_module(module: nn.Module, seed: int = 0) -> None:
    """Populate all parameters and buffers deterministically with finite values."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "weight" in name and param.ndim >= 2:
                param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype) * 0.02)
            elif "residual_scales" in name:
                param.fill_(0.1)
            elif "norm" in name and "weight" in name:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype) * 0.01)
        for name, buf in module.named_buffers():
            if name.endswith("periods"):
                continue
            buf.copy_(torch.randn(buf.shape, generator=gen, dtype=buf.dtype) * 0.05)


def test_pi4_state_dict_has_exact_1319_keys_and_matches_upstream_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Pi4 state_dict matches the exact 1,319-tensor yyfz233/Pi4 safetensors schema."""
    with torch.device("meta"):
        model = Pi4()
    sd = model.state_dict()
    assert len(sd) == 1319
    assert sum(p.numel() for p in model.parameters()) == 498_896_411
    assert sum(b.numel() for b in model.buffers()) == 7_184

    # Verify channels-last sparse 3D conv weights [C_out, 3, 3, 3, C_in] and bias [C_out]
    expected_refiner_channels = [
        ("depth_refiner.network.down_stages.0.0", 32),
        ("depth_refiner.network.down_stages.1.0", 64),
        ("depth_refiner.network.down_stages.2.0", 128),
        ("depth_refiner.network.down_stages.3.0", 256),
        ("depth_refiner.network.down_stages.4.0", 512),
        ("depth_refiner.network.bottleneck_stage.0", 512),
        ("depth_refiner.network.up_stages.0.0", 256),
        ("depth_refiner.network.up_stages.1.0", 128),
        ("depth_refiner.network.up_stages.2.0", 64),
        ("depth_refiner.network.up_stages.3.0", 32),
    ]
    for prefix, ch in expected_refiner_channels:
        for conv in ("conv1", "conv2"):
            assert tuple(sd[f"{prefix}.{conv}.weight"].shape) == (ch, 3, 3, 3, ch)
            assert sd[f"{prefix}.{conv}.weight"].dtype == torch.float32
            assert tuple(sd[f"{prefix}.{conv}.bias"].shape) == (ch,)
            assert sd[f"{prefix}.{conv}.bias"].dtype == torch.float32

    if UPSTREAM_PI4_ROOT.is_dir():
        upstream_pkg = _import_upstream_pi4(monkeypatch)
        with torch.device("meta"):
            upstream_model = upstream_pkg.Pi4()
        upstream_sd = upstream_model.state_dict()
        assert list(sd.keys()) == list(upstream_sd.keys())
        for key in sd:
            assert sd[key].shape == upstream_sd[key].shape, f"Shape mismatch for {key}"
            assert sd[key].dtype == upstream_sd[key].dtype, f"Dtype mismatch for {key}"


def test_submanifold_conv3d_matches_dense_conv3d_exact() -> None:
    """Verify pure-PyTorch SubmanifoldConv3d matches dense F.conv3d at all active voxels."""
    torch.manual_seed(123)
    batch_size, sx, sy, sz = 2, 5, 6, 7
    in_ch, out_ch = 8, 12

    # Build sparse active voxel set including corners, edges, interior, and disconnected clusters
    coords_list = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [0, 2, 2, 3],
        [0, 4, 5, 6],
        [1, 0, 0, 0],
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 3, 3, 4],
        [1, 4, 5, 6],
    ]
    coords = torch.tensor(coords_list, dtype=torch.int32)
    feats = torch.randn(len(coords_list), in_ch)
    shape = torch.Size((batch_size, sx, sy, sz, in_ch))

    conv1 = SubmanifoldConv3d(in_ch, out_ch, kernel_size=3)
    conv2 = SubmanifoldConv3d(out_ch, out_ch, kernel_size=3)

    # First conv builds neighbor_cache; second conv reuses it
    out1, cache = conv1(feats, coords, shape, neighbor_cache=None)
    out2, cache2 = conv2(out1, coords, shape, neighbor_cache=cache)
    assert cache2 is cache

    # Dense reference for conv1 and conv2 (masking output back to active coordinates between layers)
    dense_in = torch.zeros(batch_size, in_ch, sx, sy, sz)
    for idx, (b, x, y, z) in enumerate(coords_list):
        dense_in[b, :, x, y, z] = feats[idx]
    assert conv1.bias is not None and conv2.bias is not None
    w1_dense = conv1.weight.detach().permute(0, 4, 1, 2, 3)  # [Co, Ci, 3, 3, 3]
    dense_out1 = F.conv3d(dense_in, w1_dense, bias=conv1.bias.detach(), padding=1)
    expected_out1 = torch.stack([dense_out1[b, :, x, y, z] for b, x, y, z in coords_list], dim=0)
    torch.testing.assert_close(out1, expected_out1, atol=1e-5, rtol=1e-5)

    dense_mid = torch.zeros(batch_size, out_ch, sx, sy, sz)
    for idx, (b, x, y, z) in enumerate(coords_list):
        dense_mid[b, :, x, y, z] = expected_out1[idx]
    w2_dense = conv2.weight.detach().permute(0, 4, 1, 2, 3)
    dense_out2 = F.conv3d(dense_mid, w2_dense, bias=conv2.bias.detach(), padding=1)
    expected_out2 = torch.stack([dense_out2[b, :, x, y, z] for b, x, y, z in coords_list], dim=0)
    torch.testing.assert_close(out2, expected_out2, atol=1e-5, rtol=1e-5)

    # Empty voxel input edge case
    empty_feats = torch.empty(0, in_ch)
    empty_coords = torch.empty(0, 4, dtype=torch.int32)
    empty_out, _ = conv1(empty_feats, empty_coords, shape)
    assert empty_out.shape == (0, out_ch)


def test_sparse_pool3d_and_upsample3d_exact() -> None:
    """Verify SparsePool3d stride-2 mean pooling and SparseUpsample3d (with cache.T and without cache)."""
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 2, 3, 4],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.int32,
    )
    feats = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [10.0, 20.0],
            [2.0, 8.0],
            [6.0, 12.0],
        ],
        dtype=torch.float32,
    )
    fine_shape = torch.Size((2, 4, 4, 6, 2))
    pool = SparsePool3d(kernel_size=2, stride=2, reduce="mean")
    upsample = SparseUpsample3d(scale_factor=2, mode="nearest")

    coarse_feats, coarse_coords, coarse_shape, cache = pool(
        feats,
        coords,
        fine_shape,
        output_shape=torch.Size((2, 2, 2, 3, 2)),
    )
    assert coarse_shape == torch.Size((2, 2, 2, 3, 2))
    assert coarse_coords.shape == (3, 4)

    # Verify manual mean per coarse voxel
    expected_map = {
        (0, 0, 0, 0): torch.tensor([3.0, 4.0]),  # mean of (1,2), (3,4), (5,6)
        (0, 1, 1, 2): torch.tensor([10.0, 20.0]),
        (1, 0, 0, 0): torch.tensor([4.0, 10.0]),  # mean of (2,8), (6,12)
    }
    for row_idx in range(coarse_coords.shape[0]):
        b, x, y, z = (int(v) for v in coarse_coords[row_idx].tolist())
        torch.testing.assert_close(coarse_feats[row_idx], expected_map[(b, x, y, z)])

    # Upsample with cache.T
    up_cached, up_coords, up_shape, _ = upsample(
        coarse_feats,
        coarse_coords,
        coarse_shape,
        output_coords=coords,
        output_shape=fine_shape,
        neighbor_cache=cache.T,
    )
    assert up_shape == fine_shape
    assert torch.equal(up_coords, coords)

    # Upsample without cache must match cached result identically
    up_lookup, _, _, _ = upsample(
        coarse_feats,
        coarse_coords,
        coarse_shape,
        output_coords=coords,
        output_shape=fine_shape,
        neighbor_cache=None,
    )
    torch.testing.assert_close(up_cached, up_lookup)
    for i, row in enumerate(coords.tolist()):
        parent_key = (row[0], row[1] // 2, row[2] // 2, row[3] // 2)
        torch.testing.assert_close(up_cached[i], expected_map[parent_key])


def test_submanifold_conv3d_flex_delegate_does_not_pollute_state_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure creating a flex_gemm delegate never duplicates parameters in state_dict."""
    _install_mock_flex_gemm(monkeypatch)
    conv = SubmanifoldConv3d(4, 4, kernel_size=3)
    assert set(conv.state_dict().keys()) == {"weight", "bias"}
    delegate = conv._get_flex_delegate(sys.modules["flex_gemm.nn"])
    assert delegate._parameters["weight"] is conv.weight
    assert delegate._parameters["bias"] is conv.bias
    assert set(conv.state_dict().keys()) == {"weight", "bias"}


def test_pi4_components_forward_parity_against_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Pi4 submodules on CPU and compare numerically against upstream when present."""
    has_upstream = UPSTREAM_PI4_ROOT.is_dir()
    upstream_pkg = _import_upstream_pi4(monkeypatch) if has_upstream else None
    up_depth_normal_edge = importlib.import_module("pi4.geometry").depth_normal_edge if has_upstream else None
    up_decoder_mod = importlib.import_module("pi4.models.decoder") if has_upstream else None
    UpGeometryDecoder = up_decoder_mod.GeometryDecoder if up_decoder_mod is not None else None
    UpMotionDecoder = up_decoder_mod.MotionDecoder if up_decoder_mod is not None else None
    UpEncoder = importlib.import_module("pi4.models.encoder").Encoder if has_upstream else None
    up_heads_mod = importlib.import_module("pi4.models.heads") if has_upstream else None
    UpCameraHead = up_heads_mod.CameraHead if up_heads_mod is not None else None
    UpMetricHead = up_heads_mod.MetricHead if up_heads_mod is not None else None
    UpMotionMask = up_heads_mod.MotionMask if up_heads_mod is not None else None
    UpPointHead = up_heads_mod.PointHead if up_heads_mod is not None else None
    UpTracker = up_heads_mod.Tracker if up_heads_mod is not None else None
    up_geometry_confidence = importlib.import_module("pi4.postprocess").geometry_confidence if has_upstream else None

    torch.manual_seed(7)
    batch_size, frames, height, width = 1, 2, 32, 32
    images = torch.rand(batch_size, frames, 3, height, width)
    positions = patch_positions(batch_size * frames, height // 16, width // 16, torch.device("cpu"))

    with torch.inference_mode():
        # 1. Encoder
        enc = Encoder().eval()
        _randomize_module(enc, seed=1)
        tokens = enc(images.flatten(0, 1))
        assert tokens.shape == (frames, (height // 16) * (width // 16), 1024)
        assert torch.isfinite(tokens).all()
        up_tokens = None
        if UpEncoder is not None:
            up_enc = UpEncoder().eval()
            up_enc.load_state_dict(enc.state_dict(), strict=True)
            up_tokens = up_enc(images.flatten(0, 1))
            torch.testing.assert_close(tokens, up_tokens, atol=1e-5, rtol=1e-5)

        # 2. GeometryDecoder
        gdec = GeometryDecoder().eval()
        _randomize_module(gdec, seed=2)
        geom, initial = gdec(tokens, positions, batch_size, frames)
        assert geom.shape == (frames, 5 + (height // 16) * (width // 16), 1024)
        assert torch.isfinite(geom).all() and torch.isfinite(initial).all()
        up_geom, up_initial = None, None
        if UpGeometryDecoder is not None and up_tokens is not None:
            up_gdec = UpGeometryDecoder().eval()
            up_gdec.load_state_dict(gdec.state_dict(), strict=True)
            up_geom, up_initial = up_gdec(up_tokens, positions, batch_size, frames)
            torch.testing.assert_close(geom, up_geom, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(initial, up_initial, atol=1e-5, rtol=1e-5)

        # 3. PointHead
        phead = PointHead().eval()
        _randomize_module(phead, seed=3)
        rays, log_depth, conf_logits, feat = phead(geom[:1, 5:], height, width)
        assert rays.shape == (1, height, width, 2)
        assert log_depth.shape == (1, height, width)
        assert conf_logits.shape == (1, height, width)
        assert feat.shape == (1, 1026, height // 16, width // 16)
        if UpPointHead is not None and up_geom is not None:
            up_phead = UpPointHead().eval()
            up_phead.load_state_dict(phead.state_dict(), strict=True)
            up_rays, up_log_depth, up_conf_logits, up_feat = up_phead(up_geom[:1, 5:], height, width)
            torch.testing.assert_close(rays, up_rays, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(log_depth, up_log_depth, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(conf_logits, up_conf_logits, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(feat, up_feat, atol=1e-5, rtol=1e-5)

        # 4. DepthRefiner (pure-PyTorch sparse 3D U-Net)
        refiner = DepthRefiner().eval()
        _randomize_module(refiner, seed=4)
        coarse_depth = log_depth.exp().clamp_min(1e-6).log()
        refined_pts = refiner(rays, coarse_depth, feat)
        assert refined_pts.shape == (1, height, width, 3)
        assert torch.isfinite(refined_pts).all()
        assert (refined_pts[..., 2] > 0).all()

        # 5. CameraHead
        chead = CameraHead().eval()
        _randomize_module(chead, seed=5)
        poses = chead(geom, positions)
        assert poses.shape == (frames, 4, 4)
        rot = poses[:, :3, :3]
        torch.testing.assert_close(
            rot @ rot.transpose(-1, -2),
            torch.eye(3).expand(frames, 3, 3),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(torch.det(rot), torch.ones(frames), atol=1e-5, rtol=1e-5)
        if UpCameraHead is not None and up_geom is not None:
            up_chead = UpCameraHead().eval()
            up_chead.load_state_dict(chead.state_dict(), strict=True)
            up_poses = up_chead(up_geom, positions)
            torch.testing.assert_close(poses, up_poses, atol=1e-5, rtol=1e-5)

        # 6. MetricHead
        mhead = MetricHead().eval()
        _randomize_module(mhead, seed=6)
        metric = mhead(geom[:, 5:], positions[:, 5:], batch_size)
        assert metric.shape == (batch_size,)
        assert (metric > 0).all()
        if UpMetricHead is not None and up_geom is not None:
            up_mhead = UpMetricHead().eval()
            up_mhead.load_state_dict(mhead.state_dict(), strict=True)
            up_metric = up_mhead(up_geom[:, 5:], positions[:, 5:], batch_size)
            torch.testing.assert_close(metric, up_metric, atol=1e-5, rtol=1e-5)

        # 7. MotionDecoder
        mdec = MotionDecoder().eval()
        _randomize_module(mdec, seed=7)
        motion = mdec(initial, geom, positions, batch_size, frames)[:, 5:]
        assert motion.shape == (frames, (height // 16) * (width // 16), 1024)
        assert torch.isfinite(motion).all()
        up_motion = None
        if UpMotionDecoder is not None and up_initial is not None and up_geom is not None:
            up_mdec = UpMotionDecoder().eval()
            up_mdec.load_state_dict(mdec.state_dict(), strict=True)
            up_motion = up_mdec(up_initial, up_geom, positions, batch_size, frames)[:, 5:]
            torch.testing.assert_close(motion, up_motion, atol=1e-5, rtol=1e-5)

        # 8. Tracker
        tracker = Tracker().eval()
        _randomize_module(tracker, seed=8)
        queries = torch.tensor([0], dtype=torch.long)
        tracks_2d, vis, tconf = tracker(motion, images, queries)
        assert tracks_2d.shape == (batch_size, frames, height, width, 2)
        assert vis.shape == (batch_size, frames, height, width)
        assert tconf.shape == (batch_size, frames, height, width)
        if UpTracker is not None and up_motion is not None:
            up_tracker = UpTracker().eval()
            up_tracker.load_state_dict(tracker.state_dict(), strict=True)
            up_tracks_2d, up_vis, up_tconf = up_tracker(up_motion, images, queries)
            torch.testing.assert_close(tracks_2d, up_tracks_2d, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(vis, up_vis, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(tconf, up_tconf, atol=1e-5, rtol=1e-5)

        # 9. MotionMask (single frame for fast CPU parity check)
        mmask = MotionMask().eval()
        _randomize_module(mmask, seed=9)
        mask_out = mmask(motion[:1], images.flatten(0, 1)[:1])
        assert mask_out.shape == (1, height, width)
        assert torch.isfinite(mask_out).all()
        if UpMotionMask is not None and up_motion is not None:
            up_mmask = UpMotionMask().eval()
            up_mmask.load_state_dict(mmask.state_dict(), strict=True)
            up_mask_out = up_mmask(up_motion[:1], images.flatten(0, 1)[:1])
            torch.testing.assert_close(mask_out, up_mask_out, atol=1e-5, rtol=1e-5)

        # 10. Pi4.lift_tracks parity + out-of-bounds NaN masking
        local_pts = torch.rand(batch_size, frames, height, width, 3) + 0.5
        test_tracks = tracks_2d.clone()
        test_tracks[0, 0, 0, 0] = torch.tensor([-1.0, 5.0])  # out of bounds -> NaN
        lifted = Pi4.lift_tracks(test_tracks, local_pts, poses.reshape(batch_size, frames, 4, 4), metric)
        assert torch.isnan(lifted[0, 0, 0, 0]).all()
        if upstream_pkg is not None:
            up_lifted = upstream_pkg.Pi4.lift_tracks(
                test_tracks, local_pts, poses.reshape(batch_size, frames, 4, 4), metric
            )
            torch.testing.assert_close(lifted, up_lifted, atol=1e-5, rtol=1e-5, equal_nan=True)

        # 11. geometry.depth_normal_edge & postprocess.geometry_confidence parity
        pts_hw3 = local_pts[0, 0]
        conf_hw = torch.rand(height, width)
        mask_hw = torch.ones(height, width, dtype=torch.bool)
        edge_mask = depth_normal_edge(pts_hw3, mask_hw)
        geom_conf = geometry_confidence(pts_hw3, conf_hw)
        assert edge_mask.shape == (height, width) and edge_mask.dtype == torch.bool
        assert geom_conf.shape == (height, width) and torch.isfinite(geom_conf).all()
        if up_depth_normal_edge is not None and up_geometry_confidence is not None:
            assert torch.equal(edge_mask, up_depth_normal_edge(pts_hw3, mask_hw))
            torch.testing.assert_close(geom_conf, up_geometry_confidence(pts_hw3, conf_hw))


def test_pi4_validation_and_cpu_geometry_forward(tmp_path: Path) -> None:
    """Verify Pi4 input validation, strict safetensors loading, and CPU geometry-only execution."""
    model = Pi4().eval()
    _randomize_module(model, seed=11)

    # Validation checks
    with pytest.raises(ValueError, match="floating-point RGB"):
        model(torch.zeros(1, 3, 16, 16))
    with pytest.raises(ValueError, match="multiples of 16"):
        model(torch.rand(1, 1, 3, 18, 16))
    with pytest.raises(ValueError, match="at least two images"):
        model(torch.rand(1, 1, 3, 16, 16), query_frame=0)
    with pytest.raises(ValueError, match="integer frame indices"):
        model(torch.rand(1, 2, 3, 16, 16), query_frame=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid index"):
        model(torch.rand(1, 2, 3, 16, 16), query_frame=2)

    # Real CPU geometry-only forward pass on 16x16 input
    images = torch.rand(1, 1, 3, 16, 16)
    pred = model(images, query_frame=None)
    assert set(pred.keys()) == {"local_points", "camera_poses", "metric_scale", "confidence"}
    assert pred["local_points"].shape == (1, 1, 16, 16, 3)
    assert pred["camera_poses"].shape == (1, 1, 4, 4)
    assert pred["metric_scale"].shape == (1,)
    assert pred["confidence"].shape == (1, 1, 16, 16)
    assert all(torch.isfinite(val).all() for val in pred.values())
    assert (pred["confidence"] >= 0).all() and (pred["confidence"] <= 1).all()

    # Verify Pi4.from_pretrained strict loading from a local directory containing model.safetensors
    ckpt_dir = tmp_path / "pi4_ckpt"
    ckpt_dir.mkdir()
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    reloaded = Pi4.from_pretrained(ckpt_dir)
    reloaded_pred = reloaded(images, query_frame=None)
    for key in pred:
        torch.testing.assert_close(reloaded_pred[key], pred[key], atol=1e-5, rtol=1e-5)
