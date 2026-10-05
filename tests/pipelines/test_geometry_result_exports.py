"""Check real result serializers with known geometry, masks and camera data."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from worldfoundry.pipelines.pi3.pipeline_pi3 import Pi3Result
from worldfoundry.pipelines.vggt.pipeline_vggt import VGGTResult


def _geometry(views):
    points = np.arange(views * 2 * 3 * 3, dtype=np.float32).reshape(1, views, 2, 3, 3) / 4
    masks = np.ones((1, views, 2, 3), dtype=bool)
    masks[..., 0, 1] = False
    colors = [np.full((2, 3, 3), [0.0, 0.5, 1.0], dtype=np.float32) for _ in range(views)]
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None, None], views, axis=1)
    cameras[0, :, 0, 3] = np.arange(views)
    return points, masks, colors, cameras


def _assert_raw_arrays(directory, expected):
    for name, array in expected.items():
        loaded = np.load(directory / f"{name}.npy", allow_pickle=False)
        assert loaded.dtype == array.dtype
        assert loaded.shape == array.shape
        np.testing.assert_array_equal(loaded, array)


@pytest.mark.parametrize("views", [1, 3])
@pytest.mark.parametrize("mask_mode", ["some", "none", "all"])
def test_pi3_save_preserves_masked_coordinates_colors_cameras_and_raw_precision(tmp_path, views, mask_mode):
    points, masks, colors, cameras = _geometry(views)
    if mask_mode != "some":
        masks[:] = mask_mode == "all"
    geometry = {
        "points": points,
        "masks": masks,
        "camera_poses": cameras,
        "depth_map": points[..., 2].astype(np.float64),
        "confidence": np.array(0.875),
    }
    original = {key: value.copy() for key, value in geometry.items()}
    camera_params = [{"camera_to_world": pose.tolist()} for pose in cameras[0]]
    depths = [Image.fromarray(np.full((2, 3), index + 7, dtype=np.uint8)) for index in range(views)]
    camera_range = {"num_views": views, "available_view_indices": list(range(views))}
    result = Pi3Result(depths, geometry, camera_params, camera_range, input_images=colors)

    for destination in [tmp_path / "first", tmp_path / "second"]:
        files = result.save(str(destination))
        assert len(files) == len(set(files))
        assert {Path(path) for path in files} == {path for path in destination.rglob("*") if path.is_file()}
        _assert_raw_arrays(destination / "raw_data", original)
        vertices = PlyData.read(destination / "point_cloud/result.ply")["vertex"].data
        expected_points = points[0][masks[0]]
        actual_points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
        np.testing.assert_array_equal(actual_points, expected_points)
        actual_colors = np.column_stack([vertices[channel] for channel in ("red", "green", "blue")])
        expected_colors = np.tile(np.array([0, 127, 255], dtype=np.uint8), (len(expected_points), 1))
        np.testing.assert_array_equal(actual_colors, expected_colors)
        for index, camera in enumerate(camera_params):
            assert json.loads((destination / f"camera_poses/pose_{index:04d}.json").read_text()) == camera
            with Image.open(destination / f"depth/depth_{index:04d}.png") as image:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(depths[index]))
        assert json.loads((destination / "meta.json").read_text()) == {"camera_range": camera_range}
    for key, before in original.items():
        np.testing.assert_array_equal(geometry[key], before)


@pytest.mark.parametrize("views", [1, 3])
def test_vggt_result_indexes_per_view_arrays_without_slicing_global_metadata(tmp_path, views):
    images = [Image.fromarray(np.full((2, 3, 3), index + 11, dtype=np.uint8)) for index in range(views)]
    arrays = {
        "depth_map": np.arange(views * 6, dtype=np.float32).reshape(views, 2, 3),
        "extrinsic": np.repeat(np.eye(4, dtype=np.float64)[None], views, axis=0),
        "global_scale": np.array(1.25, dtype=np.float64),
        "global_statistics": np.arange(views + 2, dtype=np.int64),
    }
    cameras = [{"extrinsic": pose.tolist(), "intrinsic": np.eye(3).tolist()} for pose in arrays["extrinsic"]]
    result = VGGTResult(images, arrays, cameras)
    assert len(result) == views
    for index in range(views):
        selected = result[index]
        np.testing.assert_array_equal(selected["numpy_data"]["depth_map"], arrays["depth_map"][index])
        np.testing.assert_array_equal(selected["numpy_data"]["extrinsic"], arrays["extrinsic"][index])
        assert selected["numpy_data"]["global_scale"] is arrays["global_scale"]
        assert selected["numpy_data"]["global_statistics"] is arrays["global_statistics"]
        assert selected["camera_params"] == cameras[index]
    files = result.save(str(tmp_path / "export"))
    assert len(files) == views * 2 + len(arrays)
    assert all(Path(path).is_file() for path in files)
    _assert_raw_arrays(tmp_path / "export/numpy", arrays)
    for index in range(views):
        assert json.loads((tmp_path / f"export/json/camera_{index:04d}.json").read_text()) == cameras[index]
        with Image.open(tmp_path / f"export/visualizations/result_{index:04d}.png") as image:
            np.testing.assert_array_equal(np.asarray(image), np.asarray(images[index]))


@pytest.mark.parametrize("family", ["pi3", "vggt"])
def test_result_can_cross_a_fresh_worker_process_without_losing_geometry(tmp_path, family):
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    image = Image.fromarray(np.full((2, 3), 77, dtype=np.uint8))
    camera = {"camera_to_world": np.eye(4).tolist()}
    if family == "pi3":
        result = Pi3Result([image], {"depth_map": values}, [camera], {"num_views": 1})
        raw_dir = "raw_data"
        camera_file = "camera_poses/pose_0000.json"
    else:
        result = VGGTResult([image], {"depth_map": values}, [camera])
        raw_dir = "numpy"
        camera_file = "json/camera_0000.json"
    payload = tmp_path / "owned-result.pkl"
    payload.write_bytes(pickle.dumps(result))
    destination = tmp_path / "worker-export"
    source_root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    code = """
import pickle, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
with Path(sys.argv[2]).open('rb') as stream:
    result = pickle.load(stream)
files = result.save(sys.argv[3])
assert files and all(Path(path).is_file() for path in files)
"""
    subprocess.run(
        [sys.executable, "-c", code, source_root, str(payload), str(destination)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    _assert_raw_arrays(destination / raw_dir, {"depth_map": values})
    assert json.loads((destination / camera_file).read_text()) == camera
