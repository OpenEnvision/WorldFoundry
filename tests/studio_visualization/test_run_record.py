"""Tests for run-record artifact selection helpers."""

import tempfile
from pathlib import Path

from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

# ---------------------------------------------------------------------------
# Import paths – must use the public package namespace
# ---------------------------------------------------------------------------
from worldfoundry.studio.visualization.run_record import (
    first_geometry_point_candidate,
    first_splat_asset,
    normalize_output_relative,
)


class TestNormalizeOutputRelative:
    def test_simple_relative_path(self):
        result = normalize_output_relative("subdir/file.ply", "/tmp/output")
        assert result == "subdir/file.ply"

    def test_dot_slash_stripped(self):
        result = normalize_output_relative("./file.ply", "/tmp/output")
        assert result == "file.ply"

    def test_dots_stripped(self):
        result = normalize_output_relative("../file.ply", "/tmp/output")
        # "../file.ply" is not absolute, so strip ./ prefix
        # lstrip("./") removes leading ./ characters
        # "../file.ply".lstrip("./") → "file.ply" (strips both . and /)
        assert result == "file.ply"

    def test_absolute_path_under_output_dir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            subdir = Path(output_dir) / "sub"
            subdir.mkdir()
            file_path = subdir / "test.ply"
            file_path.write_text("x")
            result = normalize_output_relative(str(file_path.resolve()), output_dir)
            assert result == "sub/test.ply"

    def test_absolute_path_not_under_output_dir(self):
        result = normalize_output_relative("/completely/different/path.ply", "/tmp/output")
        assert result == "/completely/different/path.ply"


class TestFirstGeometryPointCandidate:
    def test_empty_paths_returns_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate([], output_dir, gs_ply_predicate=lambda p: False)
            # Falls back to glob scan of output_dir
            assert result is None

    def test_ply_not_gaussian_splat(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "points.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate([str(ply_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "points.ply" in result

    def test_ply_is_gaussian_splat_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "splat.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate([str(ply_path)], output_dir, gs_ply_predicate=lambda p: True)
            # The ply is a gs splat, so it's skipped; no other candidates → None
            assert result is None

    def test_pcd_candidate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            pcd_path = Path(output_dir) / "cloud.pcd"
            pcd_path.write_text("x")
            result = first_geometry_point_candidate([str(pcd_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "cloud.pcd" in result

    def test_xyz_candidate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            xyz_path = Path(output_dir) / "scan.xyz"
            xyz_path.write_text("x")
            result = first_geometry_point_candidate([str(xyz_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "scan.xyz" in result

    def test_nonexistent_path_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate(
                ["nonexistent.ply"], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is None

    def test_duplicate_paths_deduped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "points.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate(
                [str(ply_path), str(ply_path)], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is not None

    def test_fallback_glob_scan(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "scan.ply"
            ply_path.write_text("x")
            # Pass no explicit paths → glob scan finds it
            result = first_geometry_point_candidate([], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "scan.ply" in result

    def test_empty_strings_filtered(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate(
                ["", "", None], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is None


class TestFirstSplatAsset:
    def test_empty_iterable_returns_none(self):
        path, fmt = first_splat_asset([])
        assert path is None
        assert fmt is None

    def test_spz_highest_priority(self):
        with tempfile.TemporaryDirectory() as d:
            spz = Path(d) / "model.spz"
            splat = Path(d) / "model.splat"
            spz.write_text("x")
            splat.write_text("x")
            path, fmt = first_splat_asset([str(spz), str(splat)])
            assert path is not None
            assert fmt == "spz"

    def test_ksplat_priority(self):
        with tempfile.TemporaryDirectory() as d:
            ksplat = Path(d) / "model.ksplat"
            splat = Path(d) / "model.splat"
            ksplat.write_text("x")
            splat.write_text("x")
            path, fmt = first_splat_asset([str(ksplat), str(splat)])
            assert fmt == "ksplat"

    def test_splat_priority_after_ksplat(self):
        with tempfile.TemporaryDirectory() as d:
            splat = Path(d) / "model.splat"
            splat.write_text("x")
            path, fmt = first_splat_asset([str(splat)])
            assert fmt == "splat"

    def test_sog_priority(self):
        with tempfile.TemporaryDirectory() as d:
            sog = Path(d) / "model.sog"
            ply = Path(d) / "model.ply"
            sog.write_text("x")
            ply.write_text("x")
            path, fmt = first_splat_asset([str(sog), str(ply)])
            assert fmt == "sog"

    def test_ply_gaussian_format_hint(self):
        with tempfile.TemporaryDirectory() as d:
            ply = Path(d) / "model.ply"
            ply.write_text("x")
            path, fmt = first_splat_asset([str(ply)], gs_ply_predicate=lambda _: True)
            assert fmt == "ply_gaussian"

    def test_nonexistent_files_skipped(self):
        path, fmt = first_splat_asset(["/nonexistent/model.spz", "/nonexistent/model.splat"])
        assert path is None
        assert fmt is None

    def test_dirs_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "subdir.spz"
            sub.mkdir()
            path, fmt = first_splat_asset([str(sub)])
            assert path is None
            assert fmt is None

    def test_empty_strings_skipped(self):
        with tempfile.TemporaryDirectory():
            path, fmt = first_splat_asset(["", "", None])
            assert path is None

    def test_resolution_deduplication(self):
        with tempfile.TemporaryDirectory() as d:
            ply = Path(d) / "model.ply"
            ply.write_text("x")
            path, fmt = first_splat_asset(
                [str(ply), str(ply.resolve())],
                gs_ply_predicate=lambda _: True,
            )
            # Both resolve to same file; deduped
            assert path is not None
            assert fmt == "ply_gaussian"


# ===================================================================
# infer_visualization_artifact (core dependency)
# ===================================================================

class TestInferVisualizationArtifact:
    def test_ply_is_point_cloud(self):
        art = infer_visualization_artifact(Path("cloud.ply"))
        assert art.kind == "point_cloud"
        assert art.format_hint == "ply"

    def test_glb_is_mesh(self):
        art = infer_visualization_artifact(Path("mesh.glb"))
        assert art.kind == "mesh"

    def test_spz_is_gaussian_splat(self):
        art = infer_visualization_artifact(Path("model.spz"))
        assert art.kind == "gaussian_splat"

    def test_png_is_image(self):
        art = infer_visualization_artifact(Path("photo.png"))
        assert art.kind == "image"

    def test_mp4_is_video(self):
        art = infer_visualization_artifact(Path("clip.mp4"))
        assert art.kind == "video"

    def test_wav_is_audio(self):
        art = infer_visualization_artifact(Path("sound.wav"))
        assert art.kind == "audio"

    def test_unknown_suffix_is_artifact(self):
        art = infer_visualization_artifact(Path("data.xyz"))
        assert art.kind == "artifact"

    def test_metadata_passed_through(self):
        art = infer_visualization_artifact(Path("cloud.ply"), metadata={"custom": True})
        assert art.metadata == {"custom": True}

    def test_string_path_works(self):
        art = infer_visualization_artifact("cloud.ply")
        assert art.kind == "point_cloud"


def test_first_geometry_point_candidate_resolves_plain_ply(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ply = root / "plain.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")
    assert first_geometry_point_candidate([], str(root), gs_ply_predicate=lambda _p: False) == "plain.ply"


def test_first_geometry_point_candidate_skips_gs_classified_ply(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ply = root / "maybe_gs.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")
    assert first_geometry_point_candidate([str(ply)], str(root), gs_ply_predicate=lambda _p: True) is None
