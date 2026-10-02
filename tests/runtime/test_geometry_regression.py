"""The numerical replay gate must reject drift and incomparable evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "geometry_regression", Path(__file__).resolve().parents[1] / "manual" / "geometry_regression.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def write_run(root, arrays, **metadata):
    root.mkdir()
    np.savez_compressed(root / "arrays.npz", **arrays)
    manifest = {
        "status": "passed",
        "case": {"id": "fixed-multiview", "seed": 42},
        "assets": {"weights": "pinned-sha", "input": "input-sha"},
        "runtime": {"torch": "same", "dtype": "float32"},
        "arrays_sha256": replay.sha256(root / "arrays.npz"),
        **metadata,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_float_drift_is_reported_and_tolerance_is_explicit(tmp_path):
    reference = write_run(tmp_path / "reference", {"depth": np.array([1.0, 2.0])})
    candidate = write_run(tmp_path / "candidate", {"depth": np.array([1.0, 2.001])})
    report = replay.compare_runs(reference, candidate)
    assert report["status"] == "failed"
    assert report["outputs"]["depth"]["max_abs_error"] == pytest.approx(0.001)
    assert replay.compare_runs(reference, candidate, atol=0.002)["status"] == "passed"


def test_masks_stay_exact_with_float_tolerance(tmp_path):
    reference = write_run(tmp_path / "reference", {"mask": np.array([True, False])})
    candidate = write_run(tmp_path / "candidate", {"mask": np.array([True, True])})
    assert replay.compare_runs(reference, candidate, atol=100)["status"] == "failed"


@pytest.mark.parametrize(
    "change", [{"camera": np.eye(4)}, {"depth": np.ones((2, 1))}, {"depth": np.ones(2, dtype=np.float32)}]
)
def test_missing_outputs_shape_and_precision_changes_fail(tmp_path, change):
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    candidate = write_run(tmp_path / "candidate", change)
    with pytest.raises(ValueError, match="keys differ|shape or dtype"):
        replay.compare_runs(reference, candidate)


@pytest.mark.parametrize("field", ["case", "assets", "runtime"])
def test_changed_inputs_weights_seed_or_environment_are_incomparable(tmp_path, field):
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    candidate = write_run(tmp_path / "candidate", {"depth": np.ones(2)}, **{field: {"different": True}})
    with pytest.raises(ValueError, match="conditions differ"):
        replay.compare_runs(reference, candidate)


@pytest.mark.parametrize("data", [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_identical_invalid_outputs_cannot_pass(tmp_path, data):
    reference = write_run(tmp_path / "reference", {"depth": data})
    candidate = write_run(tmp_path / "candidate", {"depth": data})
    with pytest.raises(ValueError, match="Empty or non-finite"):
        replay.compare_runs(reference, candidate)


def test_failed_inference_and_modified_evidence_cannot_pass(tmp_path):
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    failed = write_run(tmp_path / "failed", {"depth": np.ones(2)}, status="failed")
    with pytest.raises(ValueError, match="did not pass"):
        replay.compare_runs(reference, failed)
    candidate = write_run(tmp_path / "candidate", {"depth": np.ones(2)})
    np.savez_compressed(candidate / "arrays.npz", depth=np.zeros(2))
    with pytest.raises(ValueError, match="evidence was modified"):
        replay.compare_runs(reference, candidate)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_cannot_disable_comparison(tmp_path, tolerance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replay.compare_runs(tmp_path, tmp_path, atol=tolerance)


def test_required_geometry_is_checked_even_if_previews_exist():
    with pytest.raises(ValueError, match="Missing required output"):
        replay.validate_arrays({"preview": np.ones((2, 2, 3))}, ["depth"])


def test_mapping_with_dynamic_attributes_does_not_fake_tensor_methods():
    class AttrDict(dict):
        def __getattr__(self, name):
            raise AssertionError(f"Mapping should not be probed for {name}")

    arrays = {}
    replay.collect_arrays({"prediction": AttrDict(depth=np.ones(2), config=AttrDict(size=518))}, arrays)
    assert set(arrays) == {"result.prediction.depth"}
    np.testing.assert_array_equal(arrays["result.prediction.depth"], np.ones(2))


def test_missing_assets_and_unset_variables_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.hash_assets({"weights": str(tmp_path / "absent")})
    with pytest.raises(ValueError, match="Unset case variable"):
        replay.expand("${WF_REGRESSION_MISSING_VARIABLE}/model.pt", tmp_path)


def test_changed_export_and_missing_export_cannot_pass(tmp_path):
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    candidate = write_run(tmp_path / "candidate", {"depth": np.ones(2)})
    for root, depth in ((reference, [1.0, 2.0]), (candidate, [1.0, 3.0])):
        path = root / "depth.npy"
        np.save(path, depth)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["exported_files"] = [str(path)]
        (root / "manifest.json").write_text(json.dumps(manifest))
    assert replay.compare_runs(reference, candidate)["status"] == "failed"
    (candidate / "depth.npy").unlink()
    with pytest.raises(ValueError, match="Missing or empty exported"):
        replay.compare_runs(reference, candidate)


def test_matrix_gate_rejects_missing_cases_changed_cases_and_stale_code(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = source / "model.py"
    model.write_text("accepted_source = True\n")
    case = {"id": "fixed-multiview", "seed": 42}
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for root in (reference, candidate):
        write_run(root / case["id"], {"depth": np.ones(2)}, source_hashes={"model.py": replay.sha256(model)})
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({case["id"]: case}))
    assert replay.audit_matrix(matrix, reference, candidate, source, [])["status"] == "passed"
    matrix.write_text(json.dumps({case["id"]: case, "unexecuted-model": {"id": "unexecuted-model"}}))
    assert replay.audit_matrix(matrix, reference, candidate, source, [])["status"] == "failed"
    matrix.write_text(json.dumps({case["id"]: {**case, "seed": 99}}))
    assert replay.audit_matrix(matrix, reference, candidate, source, [])["status"] == "failed"
    matrix.write_text(json.dumps({case["id"]: case}))
    model.write_text("accepted_source = False\n")
    report = replay.audit_matrix(matrix, reference, candidate, source, [])
    assert report["status"] == "failed"
    assert "stale" in report["cases"][case["id"]]["error"]
