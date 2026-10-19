from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from worldfoundry.operators.worldfm_operator import WorldFMOperator

pytestmark = pytest.mark.unit


def test_navigation_preserves_world_translation_and_local_rx_rz() -> None:
    operator = WorldFMOperator()
    operator.get_interaction(["forward", "camera_ur", "right"])
    poses = operator.process_interaction()

    rx = np.eye(4)
    rx[1:3, 1:3] = [[math.cos(-0.1), -math.sin(-0.1)], [math.sin(-0.1), math.cos(-0.1)]]
    rz = np.eye(4)
    rz[:2, :2] = [[math.cos(0.1), -math.sin(0.1)], [math.sin(0.1), math.cos(0.1)]]
    expected = rx @ rz
    expected[:3, 3] = [0.5, 0.0, -0.5]

    assert len(poses) == 4
    np.testing.assert_array_equal(poses[0], np.eye(4))
    np.testing.assert_array_equal(poses[1][:3, 3], [0.0, 0.0, -0.5])
    np.testing.assert_array_equal(poses[-1], expected)


@pytest.mark.parametrize(
    "token,translation",
    [
        ("backward", [0.0, 0.0, 0.5]),
        ("left", [-0.5, 0.0, 0.0]),
        ("forward_left", [-0.35, 0.0, -0.35]),
        ("camera_zoom_in", [0.0, 0.0, -0.3]),
        ("camera_zoom_out", [0.0, 0.0, 0.3]),
    ],
)
def test_navigation_deltas(token: str, translation: list[float]) -> None:
    operator = WorldFMOperator()
    operator.get_interaction([token])
    np.testing.assert_array_equal(operator.process_interaction()[-1][:3, 3], translation)


def test_unsupported_navigation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported token: jump"):
        WorldFMOperator().get_interaction(["jump"])


@pytest.mark.parametrize("as_mapping", [False, True])
def test_explicit_poses_preserve_values_and_do_not_alias_inputs(as_mapping: bool) -> None:
    pose = np.eye(4)
    pose[0, 3] = 2.0
    interaction = [{"c2w": pose[:3]}] if as_mapping else [pose[:3]]
    operator = WorldFMOperator()
    operator.get_interaction(interaction)
    result = operator.process_interaction()
    np.testing.assert_array_equal(result[0], pose)
    result[0][0, 3] = 10.0
    assert pose[0, 3] == 2.0


@pytest.mark.parametrize("from_path", [False, True])
def test_image_intrinsics_are_available_without_studio(tmp_path: Path, from_path: bool) -> None:
    image = Image.new("RGB", (24, 12))
    path = tmp_path / "image.png"
    image.save(path)
    perception = WorldFMOperator().process_perception(images=path if from_path else image)
    np.testing.assert_array_equal(perception["K"], [[24.0, 0.0, 12.0], [0.0, 24.0, 6.0], [0.0, 0.0, 1.0]])


def test_explicit_and_meta_intrinsics_take_precedence_over_image_defaults(tmp_path: Path) -> None:
    meta_k = [[100.0, 0.0, 11.0], [0.0, 90.0, 7.0], [0.0, 0.0, 1.0]]
    explicit_k = np.eye(3) * 2.0
    meta_path = tmp_path / "scene.json"
    meta_path.write_text(
        json.dumps({"name": "scene", "image": "image.png", "K": meta_k, "c2w": [np.eye(4).tolist()]}),
        encoding="utf-8",
    )
    operator = WorldFMOperator()
    image = Image.new("RGB", (24, 12))
    from_meta = operator.process_perception(images=image, meta_path=str(meta_path))
    explicit = operator.process_perception(images=image, K=explicit_k, meta_path=str(meta_path))
    np.testing.assert_array_equal(from_meta["K"], meta_k)
    np.testing.assert_array_equal(explicit["K"], explicit_k)


def test_missing_calibration_without_source_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires `K`"):
        WorldFMOperator().process_perception(panorama_path="panorama.png")
