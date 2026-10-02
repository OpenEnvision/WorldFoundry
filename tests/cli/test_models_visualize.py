from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "imageio" dependency at import time; skip when it is unavailable.
pytest.importorskip("imageio")

import json

import imageio.v2 as iio
import numpy as np
from PIL import Image

from worldfoundry.cli.main import main


def _run_visualize(*args: str) -> None:
    assert main(["models", "visualize", *args]) == 0


def test_models_visualize_renders_supported_perception_artifacts(tmp_path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 1] = np.arange(64, dtype=np.uint8)
    image_path = tmp_path / "input.png"
    Image.fromarray(image).save(image_path)

    detections = tmp_path / "detections.json"
    detections.write_text(
        json.dumps([{"box": [8, 6, 45, 36], "label": "object", "score": 0.95}]),
        encoding="utf-8",
    )
    masks = tmp_path / "masks.npz"
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:38, 15:50] = 1
    np.savez(masks, masks=mask)
    flow = tmp_path / "flow.npy"
    flow_array = np.zeros((48, 64, 2), dtype=np.float32)
    flow_array[..., 0] = np.linspace(-3, 3, 64)
    np.save(flow, flow_array)
    features = tmp_path / "features.npz"
    np.savez(features, features=np.random.default_rng(5).normal(size=(4, 6, 12)).astype(np.float32))
    depth = tmp_path / "depth.npy"
    np.save(depth, np.linspace(0.2, 4.0, 48 * 64, dtype=np.float32).reshape(48, 64))
    normals = tmp_path / "normals.npy"
    normal_array = np.zeros((48, 64, 3), dtype=np.float32)
    normal_array[..., 2] = np.linspace(-1, 1, 64)
    np.save(normals, normal_array)
    keypoints = tmp_path / "keypoints.npz"
    np.savez(
        keypoints,
        keypoints=np.asarray([[8, 12, 0.9], [30, 20, 0.8], [50, 36, 0.7]], dtype=np.float32),
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
    )
    caption = tmp_path / "caption.json"
    caption.write_text(json.dumps({"caption": "running through the scene"}), encoding="utf-8")

    outputs = {
        "detection": tmp_path / "detection.png",
        "mask": tmp_path / "mask.png",
        "flow": tmp_path / "flow.png",
        "feature-pca": tmp_path / "features.png",
        "depth": tmp_path / "depth.png",
        "normal": tmp_path / "normals.png",
        "keypoints": tmp_path / "keypoints.png",
        "text": tmp_path / "caption.png",
    }
    _run_visualize(
        "--kind",
        "detection",
        "--artifact",
        str(detections),
        "--media",
        str(image_path),
        "--output",
        str(outputs["detection"]),
    )
    _run_visualize(
        "--kind", "mask", "--artifact", str(masks), "--media", str(image_path), "--output", str(outputs["mask"])
    )
    _run_visualize("--kind", "flow", "--artifact", str(flow), "--output", str(outputs["flow"]))
    _run_visualize(
        "--kind",
        "feature-pca",
        "--artifact",
        str(features),
        "--media",
        str(image_path),
        "--output",
        str(outputs["feature-pca"]),
    )
    _run_visualize("--kind", "depth", "--artifact", str(depth), "--output", str(outputs["depth"]))
    _run_visualize("--kind", "normal", "--artifact", str(normals), "--output", str(outputs["normal"]))
    _run_visualize(
        "--kind",
        "keypoints",
        "--artifact",
        str(keypoints),
        "--media",
        str(image_path),
        "--output",
        str(outputs["keypoints"]),
    )
    _run_visualize(
        "--kind",
        "text",
        "--artifact",
        str(caption),
        "--media",
        str(image_path),
        "--output",
        str(outputs["text"]),
    )

    for output in outputs.values():
        rendered = np.asarray(Image.open(output))
        assert output.stat().st_size > 100
        assert rendered.std() > 0


def test_models_visualize_renders_track_video(tmp_path):
    source = tmp_path / "source.mp4"
    frames = np.zeros((3, 48, 64, 3), dtype=np.uint8)
    with iio.get_writer(source, fps=5, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)

    artifact = tmp_path / "tracks.npz"
    tracks = np.asarray([[[8, 12]], [[24, 18]], [[42, 26]]], dtype=np.float32)
    np.savez(artifact, tracks=tracks, visibility=np.ones((3, 1), dtype=bool))
    output = tmp_path / "tracks.mp4"

    _run_visualize(
        "--kind", "tracks", "--artifact", str(artifact), "--media", str(source), "--output", str(output), "--fps", "5"
    )

    reader = iio.get_reader(output)
    rendered = [reader.get_data(index) for index in range(3)]
    reader.close()
    assert output.stat().st_size > 100
    assert len(rendered) == 3
    assert np.asarray(rendered[-1]).std() > 0
