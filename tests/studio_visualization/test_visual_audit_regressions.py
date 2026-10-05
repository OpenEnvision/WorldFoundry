from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from worldfoundry.studio.inference.catalog import CatalogEntry


def _entry() -> CatalogEntry:
    return CatalogEntry(
        model_id="visual-audit-model",
        display_name="Visual Audit Model",
        module_path="worldfoundry.pipelines.visual_audit",
        class_name="VisualAuditPipeline",
        family="visual-audit",
        category="Video Generation",
        summary="Regression fixture",
    )


def test_auto_routing_prefers_artifact_and_distinguishes_ply(tmp_path: Path) -> None:
    from worldfoundry.studio.visualization.backends.frontends import (
        STUDIO_VISUALIZATIONS,
        resolve_frontend_mode,
    )
    from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact

    point_ply = tmp_path / "points.ply"
    point_ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
    splat_ply = tmp_path / "splat.ply"
    splat_ply.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 0",
                "property float opacity",
                "property float scale_0",
                "property float f_dc_0",
                "property float rot_0",
                "end_header",
            )
        ),
        encoding="ascii",
    )
    entry = _entry()
    expected_modes = {
        point_ply: "points",
        splat_ply: "spark",
        Path("scene.glb"): "points",
        Path("preview.mp4"): "media",
        Path("preview.png"): "media",
        Path("run.rrd"): "rerun",
    }
    for path, expected in expected_modes.items():
        artifact = infer_visualization_artifact(path)
        assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "auto", artifact) == expected

    assert resolve_frontend_mode(entry, "auto", "scene.glb") == "points"
    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "auto") == "media"
    assert STUDIO_VISUALIZATIONS.resolve_mode(
        entry,
        "world",
        infer_visualization_artifact("preview.mp4"),
    ) == "world"


def test_rerun_scene_extraction_applies_dumped_node_transforms() -> None:
    from worldfoundry.studio.visualization.studio import _scene_points_and_colors

    class Geometry:
        def __init__(self, x: float) -> None:
            self.vertices = np.array([[x, 2.0, 3.0]], dtype=np.float32)

    class Scene:
        geometry = {"raw": Geometry(0.0)}

        def dump(self, *, concatenate: bool):
            assert concatenate is False
            return [Geometry(10.0)]

    extracted = _scene_points_and_colors(Scene())
    assert extracted is not None
    points, _ = extracted
    np.testing.assert_allclose(points, [[10.0, 2.0, 3.0]])


def test_npz_colors_follow_finite_mask_and_decode_nchw(tmp_path: Path, monkeypatch) -> None:
    from worldfoundry.studio.visualization.backends.viser import _colors_from_npz, _load_xyz_rgb

    path = tmp_path / "cloud.npz"
    np.savez(
        path,
        world_points=np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "trimesh", types.ModuleType("trimesh"))
    points, colors = _load_xyz_rgb(path)
    np.testing.assert_allclose(points, [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(colors, [[255, 0, 0], [0, 0, 255]])

    nchw = np.array(
        [[[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]], [[21, 22, 23], [24, 25, 26]]]],
        dtype=np.uint8,
    )
    nchw_colors = _colors_from_npz({"images": nchw}, row_count=6)
    np.testing.assert_array_equal(
        nchw_colors,
        [[1, 11, 21], [2, 12, 22], [3, 13, 23], [4, 14, 24], [5, 15, 25], [6, 16, 26]],
    )


def test_lingbot_depth_fallback_inverts_world_to_camera_extrinsic(monkeypatch) -> None:
    stubbed_cv2 = False
    try:
        __import__("cv2")
    except ModuleNotFoundError:
        cv2 = types.ModuleType("cv2")
        for index, name in enumerate(
            (
                "COLORMAP_VIRIDIS",
                "COLORMAP_INFERNO",
                "COLORMAP_PLASMA",
                "COLORMAP_MAGMA",
                "COLORMAP_TURBO",
                "COLORMAP_JET",
            )
        ):
            setattr(cv2, name, index)
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        stubbed_cv2 = True

    from worldfoundry.pipelines.lingbot_map.pipeline_lingbot_map import LingBotMapResult

    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, 3] = [-1.0, -2.0, -3.0]
    result = LingBotMapResult(
        {
            "depth": np.array([[[1.0]]], dtype=np.float32),
            "intrinsic": np.eye(3, dtype=np.float32)[None],
            "extrinsic": world_to_camera[None],
        }
    )
    projected = result._project_depth_to_points()
    assert projected is not None
    points, _ = projected
    np.testing.assert_allclose(points, [[1.0, 2.0, 4.0]])

    if stubbed_cv2:
        sys.modules.pop("worldfoundry.pipelines.lingbot_map.pipeline_lingbot_map", None)
        sys.modules.pop("worldfoundry.core.media.artifacts", None)
