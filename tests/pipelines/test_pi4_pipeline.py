# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Weight-free unit tests for Pi4Operator, Pi4Representation, Pi4Pipeline, and catalog/binding resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData

from worldfoundry.base_models.three_dimensions.point_clouds.pi4_inference.runtime import Pi4Representation
from worldfoundry.evals.api import GenerationRequest
from worldfoundry.evals.models.pipelines.invocation import build_pipeline_invocation, invoke_pipeline
from worldfoundry.operators.pi4_operator import NAVIGATION_TEMPLATE, Pi4Operator
from worldfoundry.pipelines.pi4 import Pi4Pipeline, Pi4Result
from worldfoundry.runtime.inference_specs.pi4 import PI4_INFERENCE_SPEC
from worldfoundry.zoo.bindings import resolve_pipeline_route
from worldfoundry.zoo.catalog import load_model_zoo_registry
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

pytestmark = pytest.mark.unit


class _StubPi4Model(torch.nn.Module):
    """Fast deterministic stub mimicking Pi4.forward() geometry and 4D tracking outputs."""

    def __init__(self, scale: float = 2.5) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.scale = scale

    def forward(self, images: torch.Tensor, query_frame: Any = None) -> dict[str, torch.Tensor]:
        batch_size, frames, _, height, width = images.shape
        device = images.device
        # Smooth planar pointmap with positive depth so depth_normal_edge marks no discontinuities
        ys, xs = torch.meshgrid(
            torch.linspace(-0.5, 0.5, height, device=device),
            torch.linspace(-0.5, 0.5, width, device=device),
            indexing="ij",
        )
        z = torch.full((height, width), 2.0, device=device)
        frame_pts = torch.stack((xs, ys, z), dim=-1)
        local_points = frame_pts[None, None].expand(batch_size, frames, height, width, 3).clone()

        camera_poses = torch.eye(4, device=device)[None, None].repeat(batch_size, frames, 1, 1)
        for t in range(frames):
            camera_poses[:, t, 0, 3] = float(t) * 0.25

        metric_scale = torch.full((batch_size,), self.scale, device=device)
        confidence = torch.full((batch_size, frames, height, width), 0.9, device=device)
        confidence[:, :, 0, 0] = 0.01  # below default 0.05 threshold -> masked out

        prediction: dict[str, torch.Tensor] = {
            "local_points": local_points,
            "camera_poses": camera_poses,
            "metric_scale": metric_scale,
            "confidence": confidence,
        }
        if query_frame is not None:
            if frames < 2:
                raise ValueError("tracking requires at least two images")
            queries = torch.as_tensor(query_frame, device=device, dtype=torch.long)
            if queries.ndim == 0:
                queries = queries.expand(batch_size)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32),
                torch.arange(width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            tracks_2d = torch.stack((grid_x, grid_y), dim=-1)[None, None].repeat(batch_size, frames, 1, 1, 1)
            tracks_3d = torch.einsum("btij,bthwj->bthwi", camera_poses[..., :3, :3], local_points)
            tracks_3d = (tracks_3d + camera_poses[..., :3, 3][:, :, None, None]) * metric_scale[
                :, None, None, None, None
            ]
            prediction.update(
                tracks_2d=tracks_2d,
                tracks_3d=tracks_3d,
                visibility=torch.full((batch_size, frames, height, width), 0.85, device=device),
                tracking_confidence=torch.full((batch_size, frames, height, width), 0.8, device=device),
                dynamic_mask=torch.full((batch_size, frames, height, width), 0.6, device=device),
                query_frame=queries,
            )
        return prediction


def _write_test_frames_dir(root: Path, count: int = 5, width: int = 50, height: int = 35) -> Path:
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        arr = np.full((height, width, 3), fill_value=(idx * 40) % 256, dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(frame_dir / f"frame_{idx:03d}.png")
    return frame_dir


def _write_test_video(path: Path, count: int = 6, width: int = 64, height: int = 48) -> Path:
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    assert writer.isOpened()
    for idx in range(count):
        frame = np.full((height, width, 3), fill_value=(idx * 35) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_pi4_operator_patch16_resize_and_sampling(tmp_path: Path) -> None:
    """Verify Pi4Operator resizes to multiples of 16 and honors stride/max_frames on dirs and videos."""
    op = Pi4Operator()
    assert op.PATCH_SIZE == 16

    for orig_w, orig_h, max_sz in [(1920, 1080, 512), (50, 35, 64), (10, 10, 32), (288, 512, 512)]:
        tw, th = op._compute_target_size(orig_w, orig_h, max_size=max_sz)
        assert tw >= 16 and th >= 16
        assert tw % 16 == 0 and th % 16 == 0

    frame_dir = _write_test_frames_dir(tmp_path, count=5, width=50, height=35)
    frames = op.process_perception(frame_dir, stride=2, max_frames=2)
    assert len(frames) == 2
    tensor = op.images_to_tensor(frames, device="cpu", max_size=64)
    assert tensor.shape == (1, 2, 3, 48, 64)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

    # Video loading with stride and max_frames
    video_path = _write_test_video(tmp_path / "sample.mp4", count=6)
    video_frames = op.process_perception(video_path, stride=2, max_frames=2)
    assert len(video_frames) == 2

    # Navigation interactions
    assert "forward" in NAVIGATION_TEMPLATE
    op.get_interaction(["forward", "camera_r"])
    delta = op.process_interaction()
    assert len(delta) == 5 and delta[2] < 0 and delta[4] > 0
    with pytest.raises(ValueError, match="not in interaction_template"):
        op.check_interaction("invalid_move")


def test_pi4_representation_geometry_and_4d_tracking() -> None:
    """Verify Pi4Representation metric world transform, geometry_confidence masks, and 4D tracking keys."""
    rep = Pi4Representation(model=_StubPi4Model(scale=2.5), device="cpu")
    imgs = torch.rand(1, 2, 3, 16, 16)

    # Geometry-only
    geom = rep.get_representation({"images": imgs})
    assert set(geom.keys()) == {
        "points",
        "local_points",
        "camera_poses",
        "metric_scale",
        "confidence",
        "depth_map",
        "masks",
    }
    assert geom["points"].shape == (1, 2, 16, 16, 3)
    assert geom["masks"].dtype == np.bool_
    assert not geom["masks"][0, 0, 0, 0]  # low-confidence pixel filtered
    assert geom["masks"][0, 0, 8, 8]  # high-confidence planar pixel retained

    # Check metric scaling consistency: world_point = scale * (R @ local_point + t_native)
    np.testing.assert_allclose(geom["camera_poses"][0, 1, 0, 3], 0.25 * 2.5, rtol=1e-5)
    expected_world = (geom["local_points"][0, 1] * 2.5) + np.array([0.25 * 2.5, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(geom["points"][0, 1], expected_world, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(geom["depth_map"], geom["local_points"][..., 2])

    # Full 4D tracking
    tracked = rep.get_representation({"images": imgs, "query_frame": 0})
    for key in ("tracks_2d", "tracks_3d", "visibility", "tracking_confidence", "dynamic_mask", "query_frame"):
        assert key in tracked
    assert tracked["tracks_2d"].shape == (1, 2, 16, 16, 2)
    assert tracked["tracks_3d"].shape == (1, 2, 16, 16, 3)
    assert tracked["dynamic_mask"].shape == (1, 2, 16, 16)
    assert tracked["query_frame"].dtype == np.int64
    assert int(tracked["query_frame"][0]) == 0


def test_pi4_pipeline_geometry_tracking_save_render_and_runner_invocation(tmp_path: Path) -> None:
    """Verify Pi4Pipeline geometry, 4D tracking, Pi4Result.save, render_view, stream, and return_dict contract."""
    rep = Pi4Representation(model=_StubPi4Model(scale=2.0), device="cpu")
    pipeline = Pi4Pipeline(representation_model=rep)
    frame_dir = _write_test_frames_dir(tmp_path, count=3, width=32, height=32)

    # 1. Geometry-only direct call returns Pi4Result
    geom_res = pipeline(images=str(frame_dir), max_size=32)
    assert isinstance(geom_res, Pi4Result)
    assert not geom_res.has_tracking
    assert len(geom_res) == 3
    assert geom_res[0]["depth_image"] is not None

    # 2. 4D tracking direct call + save
    track_res = pipeline(images=str(frame_dir), query_frame=0, max_size=32)
    assert isinstance(track_res, Pi4Result)
    assert track_res.has_tracking

    export_dir = tmp_path / "export_4d"
    saved_files = track_res.save(export_dir)
    assert len(saved_files) == len(set(saved_files))
    assert {Path(p) for p in saved_files} == {p for p in export_dir.rglob("*") if p.is_file()}

    # Verify PLY vertices match masked points exactly
    ply = PlyData.read(export_dir / "point_cloud" / "result.ply")["vertex"].data
    expected_pts = track_res.numpy_data["points"][0][track_res.numpy_data["masks"][0]]
    actual_pts = np.column_stack([ply["x"], ply["y"], ply["z"]])
    np.testing.assert_allclose(actual_pts, expected_pts, rtol=1e-5, atol=1e-5)

    # Verify 4D tracking raw arrays and metadata
    for key in ("points", "camera_poses", "metric_scale", "tracks_2d", "tracks_3d", "visibility", "dynamic_mask"):
        loaded = np.load(export_dir / "raw_data" / f"{key}.npy", allow_pickle=False)
        np.testing.assert_array_equal(loaded, track_res.numpy_data[key])
    meta = json.loads((export_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["has_tracking"] is True
    assert meta["query_frame"] == 0
    assert meta["camera_range"]["num_views"] == 3

    # 3. render_view, render_trajectory, and interactive stream
    rendered = pipeline.render_view(camera_view=0)
    assert isinstance(rendered, Image.Image) and rendered.size == (32, 32)
    traj_frames = pipeline(task_type="render_trajectory", n_interp=2)
    assert isinstance(traj_frames, list) and len(traj_frames) == 5
    stream_img = pipeline.stream("forward")
    assert isinstance(stream_img, Image.Image) and stream_img.size == (32, 32)

    # 4. Structured dict return when output_dir or return_dict=True (WorldFoundryPipelineRunner contract)
    out_dir = tmp_path / "runner_out"
    payload = pipeline(images=str(frame_dir), task_type="tracking", max_size=32, output_dir=out_dir)
    assert isinstance(payload, dict)
    assert payload["status"] == "succeeded"
    assert payload["artifact_kind"] == "generated_4d_scene"
    assert Path(payload["point_cloud_path"]).is_file()
    assert Path(payload["metadata_path"]).is_file()

    # 5. Standard WorldFoundryPipelineLifecycle invoke_pipeline contract
    req = GenerationRequest(
        sample_id="s0",
        task_name="4d_tracking",
        inputs={"input_path": str(frame_dir)},
        generation_kwargs={"task_type": "tracking", "max_size": 32},
    )
    invocation = build_pipeline_invocation(
        request=req, output_dir=tmp_path / "lifecycle_out", artifact_filename="meta.json"
    )
    lifecycle_result = invoke_pipeline(pipeline, invocation)
    assert lifecycle_result["status"] == "succeeded"
    assert lifecycle_result["model_id"] == "pi4"

    # 6. reset_for_evaluation clears cached session state
    pipeline.reset_for_evaluation()
    with pytest.raises(RuntimeError, match="No result available"):
        pipeline.render_view()


def test_pi4_catalog_binding_profile_and_inference_spec_resolution() -> None:
    """Verify Pi4 catalog entry, pipeline binding, runtime profile, conda env, and inference spec."""
    registry = load_model_zoo_registry()
    entry = registry.get("pi4")
    assert entry.model_id == "pi4"
    assert registry.get("pi-4").model_id == "pi4"
    assert registry.get("yyfz233/Pi4").model_id == "pi4"
    assert entry.source.license == "cc-by-nc-4.0"
    assert entry.pipeline_binding == "pi4"
    assert "yyfz233/Pi4" in str(entry.to_dict())

    route = resolve_pipeline_route(model_id="pi4")
    assert route is not None
    pipeline_target, binding_id, _ = route
    assert pipeline_target == "worldfoundry.pipelines.pi4.pipeline_pi4:Pi4Pipeline"
    assert binding_id == "pi4"

    profile = load_runtime_profile("pi4")
    assert profile.model_id == "pi4"
    assert profile.task_family == "four_dimension"
    assert profile.conda_env.get("env_name") == "worldfoundry-unified-cu128"
    assert profile.checkpoints and profile.checkpoints[0].get("repo_id") == "yyfz233/Pi4"

    spec_dict = PI4_INFERENCE_SPEC.to_dict()
    assert spec_dict["model_family_id"] == "pi4"
    assert PI4_INFERENCE_SPEC.variant("default").primary_checkpoint_uri == "yyfz233/Pi4"
    assert PI4_INFERENCE_SPEC.task("reconstruction").task_id == "reconstruction"
    assert PI4_INFERENCE_SPEC.task("tracking").task_id == "4d-tracking"
