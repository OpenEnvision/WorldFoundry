"""Weight-free checks for the Depth Anything 3 Gaussian-branch options."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("addict")
pytest.importorskip("plyfile")

from addict import Dict as AddictDict
from plyfile import PlyData

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import DepthAnything3
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.model.da3 import (
    DepthAnything3Net,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.specs import (
    Gaussians,
    Prediction,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.gsply_helpers import (
    save_gaussian_ply,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.pose_align import (
    align_poses_umeyama,
)


def _rotation_z(angle: float) -> torch.Tensor:
    """Return a rotation of ``angle`` radians about the z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _trajectory_w2c(num_views: int) -> torch.Tensor:
    """Return non-degenerate world-to-camera extrinsics ``(1, N, 4, 4)``."""
    c2w = torch.eye(4).repeat(num_views, 1, 1)
    for idx in range(num_views):
        c2w[idx, :3, :3] = _rotation_z(0.2 * idx)
        c2w[idx, :3, 3] = torch.tensor([float(idx), 0.5 * idx**2, 0.1 * idx])
    return torch.linalg.inv(c2w)[None]


class _RecordingGSHead:
    """GS head stand-in returning raw Gaussians at ``H/down_ratio`` resolution."""

    def __init__(self, num_views: int, channels: int) -> None:
        self.down_ratio = 1
        self.gs_ds_feature_mode = False
        self.num_views = num_views
        self.channels = channels
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, *, feats: Any, H: int, W: int, patch_start_idx: int, images: torch.Tensor) -> AddictDict:
        self.calls.append((self.down_ratio, self.gs_ds_feature_mode))
        h, w = H // self.down_ratio, W // self.down_ratio
        return AddictDict(
            raw_gs=torch.zeros(1, self.num_views, h, w, self.channels),
            raw_gs_conf=torch.zeros(1, self.num_views, h, w),
        )


class _RecordingAdapter:
    """GS adapter stand-in recording its keyword arguments."""

    pred_offset_depth = False

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "gaussians"


def _gs_net(num_views: int) -> tuple[DepthAnything3Net, _RecordingGSHead, _RecordingAdapter]:
    """Build a ``DepthAnything3Net`` shell with recording GS head and adapter."""
    net = object.__new__(DepthAnything3Net)
    torch.nn.Module.__init__(net)
    head, adapter = _RecordingGSHead(num_views, channels=8), _RecordingAdapter()
    object.__setattr__(net, "gs_head", head)
    object.__setattr__(net, "gs_adapter", adapter)
    return net, head, adapter


def _gs_output(pred_w2c: torch.Tensor, height: int, width: int) -> AddictDict:
    """Return a depth/camera output dict like the DA3 depth and camera heads."""
    num_views = pred_w2c.shape[1]
    intrinsics = torch.tensor([[10.0, 0.0, width / 2], [0.0, 10.0, height / 2], [0.0, 0.0, 1.0]])
    return AddictDict(
        depth=torch.full((1, num_views, height, width), 2.0),
        extrinsics=pred_w2c[..., :3, :],
        intrinsics=intrinsics.expand(1, num_views, 3, 3).clone(),
    )


def test_gs_head_defaults_unproject_with_predicted_cameras() -> None:
    num_views, height, width = 4, 8, 12
    net, head, adapter = _gs_net(num_views)
    pred_w2c = _trajectory_w2c(num_views)
    output = _gs_output(pred_w2c, height, width)
    depth, intrinsics = output.depth, output.intrinsics
    gt_w2c = _trajectory_w2c(num_views)

    result = net._process_gs_head([], height, width, output, torch.zeros(1), gt_w2c, None)

    assert result.gaussians == "gaussians"
    assert head.calls == [(1, False)]
    assert adapter.kwargs["image_shape"] == (height, width)
    assert adapter.kwargs["depths"] is depth
    assert adapter.kwargs["intrinsics"] is intrinsics
    torch.testing.assert_close(adapter.kwargs["extrinsics"], pred_w2c)
    assert adapter.kwargs["gt_extrinsics"] is gt_w2c
    assert adapter.kwargs["gs_scale_extra_multiplier"] == 1.0


def test_gs_head_aligned_cameras_map_predictions_into_input_frame() -> None:
    num_views, height, width, scale = 4, 8, 12, 3.0
    net, head, adapter = _gs_net(num_views)
    pred_w2c = _trajectory_w2c(num_views)
    # The input frame is the predicted frame rotated, translated and scaled by 1 / scale.
    world_to_input = torch.eye(4)
    world_to_input[:3, :3] = _rotation_z(0.7)
    world_to_input[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
    input_c2w = world_to_input @ torch.linalg.inv(pred_w2c[0])
    input_c2w[:, :3, 3] /= scale
    input_w2c = torch.linalg.inv(input_c2w)[None]
    input_intrinsics = torch.eye(3).expand(1, num_views, 3, 3).clone() * 20.0

    net._process_gs_head(
        [],
        height,
        width,
        _gs_output(pred_w2c, height, width),
        torch.zeros(1),
        torch.zeros(1, num_views, 4, 4),
        None,
        input_extrinsics=input_w2c,
        input_intrinsics=input_intrinsics,
        use_aligned_pred_cam=True,
        gs_down_ratio=2,
        gs_scale_extra_multiplier=1.5,
        gs_ds_feature_mode=True,
    )

    assert head.calls == [(2, True)]
    assert (head.down_ratio, head.gs_ds_feature_mode) == (1, False)
    assert adapter.kwargs["image_shape"] == (height // 2, width // 2)
    assert adapter.kwargs["gt_extrinsics"] is None
    assert adapter.kwargs["gs_scale_extra_multiplier"] == 1.5
    torch.testing.assert_close(adapter.kwargs["extrinsics"], input_w2c, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        adapter.kwargs["depths"], torch.full((1, num_views, height // 2, width // 2), 2.0 / scale)
    )
    expected_intrinsics = input_intrinsics.clone()
    expected_intrinsics[..., :2, :] *= 0.5
    torch.testing.assert_close(adapter.kwargs["intrinsics"], expected_intrinsics)


def test_inference_forwards_gs_option_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(DepthAnything3)
    torch.nn.Module.__init__(model)
    calls: list[dict[str, Any]] = []
    ex_t = torch.eye(4)[None, None]
    in_t = torch.eye(3)[None, None]

    def run_model_forward(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        calls.append(kwargs)
        return {}

    stubs = {
        "_preprocess_inputs": lambda *args: (torch.zeros(1, 3, 2, 2), None, None),
        "_prepare_model_inputs": lambda *args: (torch.zeros(1, 1, 3, 2, 2), ex_t, in_t),
        "_normalize_extrinsics": lambda value: value,
        "_run_model_forward": run_model_forward,
        "_convert_to_prediction": lambda raw: Prediction(depth=np.zeros((1, 2, 2)), is_metric=0),
        "_add_processed_images": lambda prediction, imgs: prediction,
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(model, name, stub)

    model.inference([np.zeros((2, 2, 3), dtype=np.uint8)])
    model.inference(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        use_aligned_pred_cam=True,
        gs_down_ratio=2,
        gs_scale_extra_multiplier=1.5,
        gs_ds_feature_mode=True,
    )

    assert calls[0] == {
        "input_extrinsics": ex_t,
        "input_intrinsics": in_t,
        "use_aligned_pred_cam": False,
        "gs_down_ratio": 1,
        "gs_scale_extra_multiplier": 1.0,
        "gs_ds_feature_mode": False,
    }
    assert calls[1] | {"input_extrinsics": None, "input_intrinsics": None} == {
        "input_extrinsics": None,
        "input_intrinsics": None,
        "use_aligned_pred_cam": True,
        "gs_down_ratio": 2,
        "gs_scale_extra_multiplier": 1.5,
        "gs_ds_feature_mode": True,
    }


def _gaussians(num: int) -> Gaussians:
    """Return ``num`` Gaussians whose opacity increases with their index."""
    return Gaussians(
        means=torch.randn(1, num, 3),
        scales=torch.full((1, num, 3), 0.1),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, num, 4).clone(),
        harmonics=torch.zeros(1, num, 3, 1),
        opacities=torch.linspace(0.05, 0.95, num)[None],
    )


def test_save_gaussian_ply_defaults_keep_full_resolution_layout(tmp_path: Path) -> None:
    path = tmp_path / "full.ply"
    save_gaussian_ply(_gaussians(64), str(path), ctx_depth=torch.ones(1, 8, 8, 1), prune_border_gs=False)
    assert PlyData.read(str(path))["vertex"].count == 64


def test_save_gaussian_ply_handles_downsampled_gs_and_opacity_pruning(tmp_path: Path) -> None:
    path = tmp_path / "pruned.ply"
    save_gaussian_ply(
        _gaussians(16),
        str(path),
        ctx_depth=torch.ones(1, 8, 8, 1),
        prune_border_gs=False,
        prune_by_opacity_percentile=0.25,
    )
    vertices = PlyData.read(str(path))["vertex"].data
    assert len(vertices) == 13
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"]))
    assert opacity.min() >= np.linspace(0.05, 0.95, 16)[3] - 1e-6


def test_pose_alignment_retries_degenerate_trajectories_only_on_request() -> None:
    static = np.repeat(np.eye(4)[None], 4, axis=0)
    with pytest.raises(ValueError, match="Degenerate"):
        align_poses_umeyama(static, static)
    r, t, s = align_poses_umeyama(static, static, retry_degenerate=True)
    assert np.isfinite(r).all() and np.isfinite(t).all() and np.isfinite(s)


def test_alignment_default_path_is_unchanged_for_regular_trajectories() -> None:
    ext = _trajectory_w2c(5)[0].numpy().astype(np.float64)
    default = align_poses_umeyama(ext, ext)
    retried = align_poses_umeyama(ext, ext, retry_degenerate=True)
    for left, right in zip(default, retried, strict=True):
        np.testing.assert_array_equal(left, right)
    assert default[2] == pytest.approx(1.0)
