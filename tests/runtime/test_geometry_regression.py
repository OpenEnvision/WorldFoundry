"""The numerical replay gate must reject drift and incomparable evidence."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

spec = importlib.util.spec_from_file_location(
    "geometry_regression", Path(__file__).resolve().parents[1] / "manual" / "geometry_regression.py"
)
assert spec is not None and spec.loader is not None
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


@pytest.fixture
def artifact_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    module = ModuleType("artifact_replay_fixture")
    module.__file__ = str(source / "artifact_replay_fixture.py")
    Path(module.__file__).write_text("# Artifact-only pipeline fixture.\n")

    class Pipeline:
        pixel = 19
        mode = "numeric"

        @classmethod
        def from_pretrained(cls):
            return cls()

        def __call__(self, output):
            path = Path(output)
            path.parent.mkdir(parents=True)
            if self.mode == "empty":
                path.touch()
            elif self.mode == "nonnumeric":
                np.save(path, np.array(["preview"]))
            elif self.mode == "wrong_shape":
                np.save(path, np.full((1, 3), self.pixel, dtype=np.uint8))
            elif self.mode == "numeric":
                np.save(path, np.full((2, 3), self.pixel, dtype=np.uint8))
            return {"artifact": str(path)}

    setattr(module, "Pipeline", Pipeline)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(replay, "runtime_metadata", lambda torch: {"torch": "fixture"})
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    asset = source / "input.txt"
    asset.write_text("fixed input")
    case_file = tmp_path / "case.json"
    case_file.write_text(
        json.dumps(
            {
                "id": "artifact-only",
                "seed": 42,
                "assets": {"input": str(asset)},
                "target": "artifact_replay_fixture:Pipeline",
                "load": {},
                "call": {"output": "${OUTPUT_DIR}/export/frames.npy"},
                "export_artifacts": ["export/frames.npy"],
                "required_outputs": ["export.export/frames.npy"],
                "array_contracts": {"export.export/frames.npy": {"shape": [2, 3], "dtype": "uint8"}},
            }
        )
    )
    return case_file, source, Pipeline


@pytest.mark.parametrize("changed", [False, True])
def test_artifact_only_inference_is_validated_and_compared(tmp_path, artifact_pipeline, changed):
    case_file, source, pipeline = artifact_pipeline
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    assert replay.run_case(case_file, source, reference)["status"] == "passed"
    pipeline.pixel += int(changed)
    assert replay.run_case(case_file, source, candidate)["status"] == "passed"
    with np.load(candidate / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays.files == []
    report = replay.compare_runs(reference, candidate, atol=100)
    assert report["status"] == ("failed" if changed else "passed")
    assert report["outputs"]["export.export/frames.npy"]["passed"] is (not changed)


def test_changed_asset_contents_fail_before_checkpoint_loading(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, type], monkeypatch: pytest.MonkeyPatch
) -> None:
    case_file, source, pipeline = artifact_pipeline
    reference = tmp_path / "reference"
    assert replay.run_case(case_file, source, reference)["status"] == "passed"
    (source / "input.txt").write_text("different local input")

    def load_checkpoint() -> None:
        raise AssertionError("Changed asset reached checkpoint loading")

    monkeypatch.setattr(pipeline, "from_pretrained", load_checkpoint)

    report = replay.run_case(case_file, source, tmp_path / "candidate", accepted_manifest=reference / "manifest.json")

    assert report["status"] == "failed"
    assert "Replay asset SHA256 differs from accepted reference: input" in report["error"]


@pytest.mark.parametrize(
    "mode, error",
    [
        ("missing", "Missing exported artifacts"),
        ("empty", "Missing or empty exported artifact"),
        ("nonnumeric", "Inference returned no numerical outputs"),
        ("wrong_shape", "Contracted output shape changed"),
    ],
)
def test_artifact_only_inference_rejects_invalid_exports(tmp_path, artifact_pipeline, mode, error):
    case_file, source, pipeline = artifact_pipeline
    pipeline.mode = mode
    result = replay.run_case(case_file, source, tmp_path / "invalid")
    assert result["status"] == "failed"
    assert error in result["error"]


def test_artifact_only_inference_still_requires_declared_outputs(tmp_path, artifact_pipeline):
    case_file, source, _ = artifact_pipeline
    case = json.loads(case_file.read_text())
    case["required_outputs"].append("result.depth")
    case_file.write_text(json.dumps(case))
    result = replay.run_case(case_file, source, tmp_path / "missing-output")
    assert result["status"] == "failed"
    assert "Missing required output: result.depth" in result["error"]


def test_comparison_without_in_memory_or_exported_values_cannot_pass(tmp_path):
    reference = write_run(tmp_path / "reference", {})
    candidate = write_run(tmp_path / "candidate", {})
    with pytest.raises(ValueError, match="Numerical output keys differ or are empty"):
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


@pytest.mark.parametrize("extension", ["npy", "npz", "png", "json", "splat"])
@pytest.mark.parametrize("changed", [False, True])
def test_export_values_are_compared_even_when_in_memory_outputs_match(tmp_path, extension, changed):
    """A serializer regression must fail independently of the model's raw arrays."""
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    candidate = write_run(tmp_path / "candidate", {"depth": np.ones(2)})
    for root, pixel in [(reference, 19), (candidate, 20 if changed else 19)]:
        path = root / f"artifact.{extension}"
        value = np.full((2, 3), pixel, dtype=np.uint8)
        if extension == "npy":
            np.save(path, value)
        elif extension == "npz":
            np.savez(path, confidence=value)
        elif extension == "png":
            from PIL import Image

            Image.fromarray(value).save(path)
        elif extension == "splat":
            np.full((2, 32), pixel, dtype=np.uint8).tofile(path)
        else:
            path.write_text(json.dumps({"intrinsics": value.tolist()}))
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["exported_files"] = [str(path)]
        (root / "manifest.json").write_text(json.dumps(manifest))
    report = replay.compare_runs(reference, candidate, atol=10)
    # Integer pixels and camera IDs must remain exact despite float tolerance.
    assert report["status"] == ("failed" if changed else "passed")
    exported = [result for key, result in report["outputs"].items() if key.startswith("export.")]
    assert exported and all(result["passed"] is (not changed) for result in exported)


def test_sequence_reads_file_outputs_and_preserves_each_response(tmp_path: Path) -> None:
    from PIL import Image

    artifact = tmp_path / "scene.splat"
    preview = tmp_path / "preview.png"

    class Pipeline:
        def __call__(self, value: int) -> dict[str, str]:
            np.full(32, value, dtype=np.uint8).tofile(artifact)
            Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(preview)
            return {"artifact_path": str(artifact), "preview_path": str(preview)}

    sequence = [
        {"name": name, "call": {"value": value}, "outputs": ["artifact_path", "preview_path"]}
        for name, value in (("a", 1), ("b", 2), ("repeat", 1))
    ]
    arrays: dict[str, np.ndarray] = {}
    replay.run_sequence(Pipeline(), sequence, arrays)
    replay.validate_arrays(arrays, [f"{name}.artifact_path" for name in ("a", "b", "repeat")])
    for name, value in (("a", 1), ("b", 2), ("repeat", 1)):
        np.testing.assert_array_equal(arrays[f"{name}.artifact_path.bytes"], np.full(32, value, dtype=np.uint8))
        np.testing.assert_array_equal(arrays[f"{name}.preview_path"], np.full((2, 3, 3), value, dtype=np.uint8))


def test_sequence_rejects_a_missing_selected_file_output(tmp_path: Path) -> None:
    pipeline = SimpleNamespace(generate=lambda: {"artifact_path": str(tmp_path / "missing.splat")})
    sequence = [{"name": "a", "method": "generate", "outputs": ["artifact_path"]}]
    with pytest.raises(ValueError, match="Missing or empty exported artifact"):
        replay.run_sequence(pipeline, sequence, {})


def test_export_manifest_cannot_silently_drop_an_artifact(tmp_path):
    reference = write_run(tmp_path / "reference", {"depth": np.ones(2)})
    candidate = write_run(tmp_path / "candidate", {"depth": np.ones(2)})
    path = reference / "camera.npy"
    np.save(path, np.eye(4))
    metadata = json.loads((reference / "manifest.json").read_text())
    metadata["exported_files"] = [str(path)]
    (reference / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="artifact names differ"):
        replay.compare_runs(reference, candidate)


@pytest.mark.parametrize("path", ["../model.py", "/outside/model.py"])
def test_matrix_gate_rejects_source_hashes_outside_the_checkout(tmp_path, path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    source = tmp_path / "source"
    for root in [reference, candidate, source]:
        root.mkdir()
    case = {"id": "fixed-multiview", "seed": 42}
    for root in [reference, candidate]:
        write_run(root / case["id"], {"depth": np.ones(2)}, source_hashes={path: "invalid"})
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({case["id"]: case}))
    report = replay.audit_matrix(matrix, reference, candidate, source, [])
    assert report["status"] == "failed"
    assert "Invalid imported-source path" in report["cases"][case["id"]]["error"]


def test_empty_matrix_cannot_produce_a_successful_release_gate(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{}")
    with pytest.raises(ValueError, match="matrix is empty"):
        replay.audit_matrix(matrix, tmp_path / "reference", tmp_path / "candidate", tmp_path, [])


@pytest.mark.parametrize("changed", [False, True])
def test_selected_generated_text_is_utf8_and_exact(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    case_file, source, pipeline = artifact_pipeline
    pipeline.text = "你好"

    def generate(self: Any, output: str) -> dict[str, str]:
        Path(output).write_text(json.dumps({"text": self.text, "image_path": str(Path(output).parent)}))
        return {"artifact": output}

    monkeypatch.setattr(pipeline, "__call__", generate)
    case = json.loads(case_file.read_text())
    case.update(
        call={"output": "${OUTPUT_DIR}/prediction.json"},
        export_artifacts=["prediction.json"],
        text_outputs={"prediction.json": ["text"]},
        required_outputs=["text.prediction.json.text"],
        array_contracts={},
    )
    case_file.write_text(json.dumps(case))
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    assert replay.run_case(case_file, source, reference)["status"] == "passed"
    pipeline.text = "您好" if changed else "你好"
    assert replay.run_case(case_file, source, candidate)["status"] == "passed"
    with np.load(candidate / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays.files == ["text.prediction.json.text"]
        assert arrays[arrays.files[0]].dtype == np.uint8
        assert arrays[arrays.files[0]].tobytes().decode("utf-8") == pipeline.text
    assert replay.compare_runs(reference, candidate, atol=100, rtol=100)["status"] == (
        "failed" if changed else "passed"
    )
    (candidate / "prediction.json").write_text(json.dumps({"text": "再见"}))
    assert replay.compare_runs(reference, candidate, atol=100)["status"] == "failed"


@pytest.mark.parametrize("text", [None, 3, []])
def test_selected_text_fields_reject_non_text(tmp_path: Path, text: object) -> None:
    (tmp_path / "prediction.json").write_text(json.dumps({"text": text}))
    with pytest.raises(ValueError, match="not a string"):
        replay.exported_text_arrays(tmp_path, {"prediction.json": ["text"]})


def test_gpu_required_case_fails_before_loading_on_cpu(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any]
) -> None:
    case_file, source, _ = artifact_pipeline
    case = json.loads(case_file.read_text())
    case["requires_gpu"] = True
    case_file.write_text(json.dumps(case))
    result = replay.run_case(case_file, source, tmp_path / "gpu-required")
    assert result["status"] == "failed"
    assert "requires CUDA" in result["error"]


def test_single_call_selects_generated_actions_without_runtime_metadata(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case_file, source, pipeline = artifact_pipeline
    monkeypatch.setattr(
        pipeline,
        "__call__",
        lambda self, **kwargs: {"actions": np.array([[0.1, 0.2]]), "max_cuda_memory_mib": np.array([300.0])},
    )
    case = json.loads(case_file.read_text())
    case.update(call={}, result_outputs=["actions"], required_outputs=["result.actions"], array_contracts={})
    case.pop("export_artifacts", None)
    case_file.write_text(json.dumps(case))
    output = tmp_path / "selected-actions"
    assert replay.run_case(case_file, source, output)["status"] == "passed"
    with np.load(output / "arrays.npz") as arrays:
        assert arrays.files == ["result.actions"]
        np.testing.assert_array_equal(arrays["result.actions"], [[0.1, 0.2]])
    with pytest.raises(KeyError):
        replay.collect_output_fields({"max_cuda_memory_mib": 300.0}, {}, "result", ["actions"])


def test_pinned_input_file_keeps_native_payload_and_path(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    payload = b'{"action_format": "raw_language_table_delta_yx"}\n'
    path.write_bytes(payload)
    case = {"assets": {"trajectory": str(path)}, "asset_sha256": {"trajectory": replay.sha256(path)}}
    replay.verify_asset_pins(case)
    assert path.read_bytes() == payload
    assert case["assets"]["trajectory"] == str(path)


@pytest.mark.parametrize("accepted_matches", [True, False])
def test_pinned_file_digest_is_reused_for_the_accepted_inventory(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch, accepted_matches: bool
) -> None:
    case_file, source, _ = artifact_pipeline
    asset = source / "input.txt"
    case = json.loads(case_file.read_text())
    case["assets"]["input_alias"] = str(asset)
    case["asset_sha256"] = {"input": replay.sha256(asset)}
    case_file.write_text(json.dumps(case))
    reference = tmp_path / "reference"
    expected = replay.run_case(case_file, source, reference)["assets"]
    accepted_manifest = reference / "manifest.json"
    if not accepted_matches:
        manifest = json.loads(accepted_manifest.read_text())
        manifest["assets"]["input_alias"] = "0" * 64
        accepted_manifest.write_text(json.dumps(manifest))
    calls = []
    sha256 = replay.sha256

    def count(path: Path) -> str:
        calls.append(path)
        return sha256(path)

    monkeypatch.setattr(replay, "sha256", count)
    result = replay.run_case(case_file, source, tmp_path / "candidate", accepted_manifest=accepted_manifest)

    if accepted_matches:
        assert result["status"] == "passed"
        assert result["assets"] == expected
    else:
        assert result["status"] == "failed"
        assert "Replay asset SHA256 differs from accepted reference: input_alias" in result["error"]
    assert calls.count(asset) == 1


def test_pinned_asset_replacement_fails_before_model_loading(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case_file, source, pipeline = artifact_pipeline
    asset = source / "input.txt"
    case = json.loads(case_file.read_text())
    case["asset_sha256"] = {"input": replay.sha256(asset)}
    case_file.write_text(json.dumps(case))
    verify_asset_pins = replay.verify_asset_pins

    def replace_after_pinning(resolved: dict) -> dict[str, Any]:
        pinned = verify_asset_pins(resolved)
        replacement = asset.with_suffix(".replacement")
        replacement.write_bytes(asset.read_bytes())
        replacement.replace(asset)
        return pinned

    def load_checkpoint() -> None:
        raise AssertionError("Replaced asset reached checkpoint loading")

    monkeypatch.setattr(replay, "verify_asset_pins", replace_after_pinning)
    monkeypatch.setattr(pipeline, "from_pretrained", load_checkpoint)
    result = replay.run_case(case_file, source, tmp_path / "replaced")

    assert result["status"] == "failed"
    assert "Pinned asset changed during replay validation" in result["error"]


@pytest.mark.parametrize("fault", ["undeclared", "nonhex", "mismatch", "unset"])
def test_asset_pin_failures_precede_model_loading(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    case_file, source, pipeline = artifact_pipeline
    path = source / "trajectory.json"
    path.write_text('{"actions": [[0.25, 0.5]]}')
    case = json.loads(case_file.read_text())
    case["assets"]["trajectory"] = str(path)
    case["asset_sha256"] = {"trajectory": replay.sha256(path)}
    if fault == "undeclared":
        del case["assets"]["trajectory"]
    elif fault == "nonhex":
        case["asset_sha256"]["trajectory"] = "not-a-pin"
    elif fault == "mismatch":
        case["asset_sha256"]["trajectory"] = "0" * 64
    else:
        case["asset_sha256"]["trajectory"] = "${WORLDFOUNDRY_TEST_OBSERVATION_SHA256}"
        monkeypatch.delenv("WORLDFOUNDRY_TEST_OBSERVATION_SHA256", raising=False)
    case_file.write_text(json.dumps(case))
    calls = []
    monkeypatch.setattr(pipeline, "from_pretrained", lambda: calls.append("loaded"))

    def bulk_hash() -> None:
        raise AssertionError("Invalid asset reached bulk checkpoint hashing")

    monkeypatch.setattr(replay, "hash_assets", bulk_hash)
    result = replay.run_case(case_file, source, tmp_path / "invalid-asset")
    assert result["status"] == "failed"
    assert "SHA256" in result["error"] or "Unset case variable" in result["error"]
    assert calls == []


def test_pinned_json_observation_is_materialized_as_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "observation.json"
    path.write_text(json.dumps({"state": [0.25, 0.5], "image": "${OBSERVATION_IMAGES}/front.png"}))
    monkeypatch.setenv("OBSERVATION_IMAGES", "/accepted/images")
    case = {
        "call": {"prompt": "Lift the block", "seed": 42},
        "assets": {"observation": str(path)},
        "input_json": {"observation": {"path": str(path), "sha256": replay.sha256(path)}},
    }
    result = replay.materialize_input_json(case, tmp_path)
    assert result == {
        "prompt": "Lift the block",
        "seed": 42,
        "observation": {"state": [0.25, 0.5], "image": "/accepted/images/front.png"},
    }
    assert case["call"] == {"prompt": "Lift the block", "seed": 42}


def test_archived_observation_paths_resolve_without_changing_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "inputs"
    bundle = local / "episode"
    bundle.mkdir(parents=True)
    camera = bundle / "front.png"
    camera.write_bytes(b"accepted camera data")
    recorded = {"views": [{"path": "/archive/inputs/episode/front.png"}], "state": [0.25]}
    observation = bundle / "observation.json"
    observation.write_text(json.dumps(recorded), encoding="utf-8")
    case = {
        "call": {},
        "assets": {"bundle": str(bundle), "observation": str(observation)},
        "input_json": {"observation": {"path": str(observation), "sha256": replay.sha256(observation)}},
    }
    original_inventory = replay.hash_assets(case["assets"])
    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_INPUT_ROOT", str(local))
    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_ORIGINAL_INPUT_ROOT", "/archive/inputs")

    call = replay.materialize_input_json(case, tmp_path)

    assert call == {"observation": {"views": [{"path": str(camera)}], "state": [0.25]}}
    assert replay.hash_assets(case["assets"]) == original_inventory
    del case["assets"]["bundle"]
    with pytest.raises(ValueError, match="outside the declared input assets"):
        replay.materialize_input_json(case, tmp_path)


@pytest.mark.parametrize("fault", ["undeclared", "collision", "nonhex", "mismatch", "unset"])
def test_json_input_pin_failures_precede_model_loading(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    case_file, source, pipeline = artifact_pipeline
    path = source / "observation.json"
    path.write_text(json.dumps({"state": [0.25, 0.5]}))
    case = json.loads(case_file.read_text())
    case["assets"]["observation"] = str(path)
    case["input_json"] = {"observation": {"path": str(path), "sha256": replay.sha256(path)}}
    if fault == "undeclared":
        del case["assets"]["observation"]
    elif fault == "collision":
        case["call"]["observation"] = {}
    elif fault == "nonhex":
        case["input_json"]["observation"]["sha256"] = "not-a-pin"
    elif fault == "mismatch":
        case["input_json"]["observation"]["sha256"] = "0" * 64
    else:
        case["input_json"]["observation"]["sha256"] = "${WORLDFOUNDRY_TEST_OBSERVATION_SHA256}"
        monkeypatch.delenv("WORLDFOUNDRY_TEST_OBSERVATION_SHA256", raising=False)
    case_file.write_text(json.dumps(case))
    calls = []
    monkeypatch.setattr(pipeline, "from_pretrained", lambda: calls.append("loaded"))
    result = replay.run_case(case_file, source, tmp_path / "invalid-input")
    assert result["status"] == "failed"
    assert any(
        message in result["error"] for message in ("not a declared asset", "collides", "SHA256", "Unset case variable")
    )
    assert calls == []


def test_pinned_observation_replay_requires_exported_actions(
    tmp_path: Path, artifact_pipeline: tuple[Path, Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case_file, source, pipeline = artifact_pipeline
    observation = source / "observation.json"
    observation.write_text(json.dumps({"state": [0.25, 0.5]}))
    observed = []

    def generate(self: Any, output: str, observation: dict[str, object]) -> dict[str, str]:
        observed.append(observation)
        Path(output).write_text(json.dumps({"actions": [[0.1, 0.2]], "metadata": {"input": observation}}))
        return {"artifact_path": output}

    monkeypatch.setattr(pipeline, "__call__", generate)
    case = json.loads(case_file.read_text())
    case["assets"]["observation"] = str(observation)
    case.update(
        call={"output": "${OUTPUT_DIR}/prediction.json"},
        input_json={"observation": {"path": str(observation), "sha256": replay.sha256(observation)}},
        export_artifacts=["prediction.json"],
        required_outputs=["export.prediction.json.actions"],
        array_contracts={},
    )
    case_file.write_text(json.dumps(case))
    output = tmp_path / "actions"
    assert replay.run_case(case_file, source, output)["status"] == "passed"
    assert observed == [{"state": [0.25, 0.5]}]
    (output / "prediction.json").write_text(json.dumps({"metadata": {"input": observed[0]}}))
    with pytest.raises(ValueError, match="Missing required output"):
        replay.validate_arrays(
            replay.exported_arrays(output, [str(output / "prediction.json")]), case["required_outputs"]
        )


def write_emu35_story(
    path: Path,
    *,
    text: str | None = "一只狐狸",
    pixel: int = 19,
    question: str = "Describe the scene",
    image: bool = True,
) -> None:
    from io import BytesIO

    from PIL import Image

    pytest.importorskip("google.protobuf")
    from worldfoundry.synthesis.visual_generation.emu35.emu35_runtime.src.proto import emu_pb

    story = getattr(emu_pb, "Story")()
    story.question = question
    if text is not None or image:
        segment = story.clips.add().segments.add()
        segment.asr = text or ""
        if image:
            encoded = BytesIO()
            Image.fromarray(np.full((2, 3, 3), pixel, dtype=np.uint8)).save(encoded, format="PNG")
            segment.images.add().image.image_data = encoded.getvalue()
    path.write_bytes(story.SerializeToString())


@pytest.mark.parametrize("change", ["none", "question", "text", "image"])
def test_emu35_story_compares_generated_text_and_pixels_exactly(tmp_path: Path, change: str) -> None:
    roots = [write_run(tmp_path / name, {}) for name in ("reference", "candidate")]
    for index, root in enumerate(roots):
        path = root / "story.pb"
        write_emu35_story(
            path,
            text="一只白狼" if index and change == "text" else "一只狐狸",
            pixel=20 if index and change == "image" else 19,
            question="A different input label" if index and change == "question" else "Describe the scene",
        )
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["exported_files"] = [str(path)]
        (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = replay.exported_arrays(roots[0], [str(roots[0] / "story.pb")])
    assert set(arrays) == {"export.story.pb.text.0.0", "export.story.pb.images.0.0.0"}
    assert arrays["export.story.pb.text.0.0"].tobytes().decode("utf-8") == "一只狐狸"
    assert arrays["export.story.pb.images.0.0.0"].shape == (2, 3, 3)
    result = replay.compare_runs(*roots, atol=100, rtol=100)
    assert result["status"] == ("passed" if change in {"none", "question"} else "failed")


@pytest.mark.parametrize("text,image", [(None, False), ("Only generated text", False), (None, True)])
def test_emu35_story_rejects_missing_generated_modalities(tmp_path: Path, text: str | None, image: bool) -> None:
    path = tmp_path / "story.pb"
    write_emu35_story(path, text=text, image=image)
    with pytest.raises(ValueError, match="generated clip text and images"):
        replay.exported_arrays(tmp_path, [str(path)])


@pytest.mark.parametrize("change", ["metadata", "samples", "rate", "channels"])
def test_flac_compares_pcm_and_rate_instead_of_container_bytes(tmp_path: Path, change: str) -> None:
    soundfile = pytest.importorskip("soundfile")

    roots = [write_run(tmp_path / name, {}) for name in ("reference", "candidate")]
    for index, root in enumerate(roots):
        path = root / "audio.flac"
        samples = np.array([[1, -2], [300, -400], [123, 456]], dtype=np.int16)
        if index and change == "samples":
            samples[0, 0] += 1
        if index and change == "channels":
            samples = samples[:, :1]
        with soundfile.SoundFile(
            path,
            "w",
            samplerate=44100 if index and change == "rate" else 16000,
            channels=samples.shape[1],
            subtype="PCM_16",
        ) as audio:
            audio.title = "candidate" if index else "reference"
            audio.write(samples)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["exported_files"] = [str(path)]
        (root / "manifest.json").write_text(json.dumps(manifest))
    assert (roots[0] / "audio.flac").read_bytes() != (roots[1] / "audio.flac").read_bytes()
    arrays = replay.exported_arrays(roots[0], [str(roots[0] / "audio.flac")])
    assert arrays["export.audio.flac.samples"].dtype == np.int32
    assert arrays["export.audio.flac.samples"].shape == (3, 2)
    assert arrays["export.audio.flac.sample_rate"].tolist() == [16000]
    assert arrays["export.audio.flac.channels"].tolist() == [2]
    if change == "channels":
        with pytest.raises(ValueError, match="shape or dtype"):
            replay.compare_runs(*roots, atol=1e10, rtol=1e10)
    else:
        result = replay.compare_runs(*roots, atol=1e10, rtol=1e10)
        assert result["status"] == ("passed" if change == "metadata" else "failed")


def test_flac_rejects_empty_audio_payload(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")

    path = tmp_path / "empty.flac"
    soundfile.write(path, np.empty((0, 2), dtype=np.int16), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="[Ee]mpty"):
        arrays = replay.exported_arrays(tmp_path, [str(path)])
        replay.validate_arrays(arrays, ["export.empty.flac.samples"])


@pytest.mark.parametrize("change", [False, True])
def test_nwm_prediction_pngs_require_exact_pixels(tmp_path: Path, change: bool) -> None:
    from PIL import Image

    roots = [write_run(tmp_path / name, {}) for name in ("reference", "candidate")]
    for index, root in enumerate(roots):
        path = root / "native/id_0/1.png"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (3, 2), color=(20 if index and change else 19, 20, 21)).save(path)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["exported_files"] = [str(path)]
        (root / "manifest.json").write_text(json.dumps(manifest))
    result = replay.compare_runs(*roots, atol=100, rtol=100)
    assert result["status"] == ("failed" if change else "passed")


def test_png_replay_rejects_jpeg_payload(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "prediction.png"
    Image.new("RGB", (3, 2)).save(image, format="JPEG")
    with pytest.raises(ValueError, match="Expected PNG image, got JPEG"):
        replay.exported_arrays(tmp_path, [str(image)])


def write_aac_video(
    path: Path, *, rate: int = 16000, channels: int = 2, frequency: int = 440, audio: bool = True
) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), mode="w") as container:
        container.metadata["title"] = path.parent.name
        video = container.add_stream("mpeg4", rate=8)
        video.width = video.height = 16
        video.pix_fmt = "yuv420p"
        stream = container.add_stream("aac", rate=rate) if audio else None
        layout = "mono" if channels == 1 else "stereo"
        if stream is not None:
            stream.layout = layout
        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 30, dtype=np.uint8), format="rgb24")
        for packet in (*video.encode(frame), *video.encode(None)):
            container.mux(packet)
        if stream is not None:
            waveform = (0.25 * np.sin(np.arange(2048) * 2 * np.pi * frequency / rate)).astype(np.float32)
            samples = np.stack([waveform] * channels)
            sound = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            sound.sample_rate = rate
            for packet in (*stream.encode(sound), *stream.encode(None)):
                container.mux(packet)


@pytest.mark.parametrize("change", ["metadata", "samples", "rate", "channels"])
def test_selected_video_soundtrack_compares_exact_decoded_pcm(tmp_path: Path, change: str) -> None:
    roots = [write_run(tmp_path / name, {}) for name in ("reference", "candidate")]
    for index, root in enumerate(roots):
        write_aac_video(
            root / "video.mp4",
            rate=32000 if index and change == "rate" else 16000,
            channels=1 if index and change == "channels" else 2,
            frequency=660 if index and change == "samples" else 440,
        )
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["case"]["audio_outputs"] = ["video.mp4"]
        (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = replay.exported_audio_arrays(roots[0], ["video.mp4"])
    assert arrays["audio.video.mp4.samples"].dtype == np.float32
    assert arrays["audio.video.mp4.samples"].shape[0] == 2
    assert arrays["audio.video.mp4.sample_rate"].tolist() == [16000]
    assert arrays["audio.video.mp4.layout"].tobytes().decode() == "stereo"
    if change == "channels":
        with pytest.raises(ValueError, match="shape or dtype"):
            replay.compare_runs(*roots, atol=1e6, rtol=1e6)
    else:
        result = replay.compare_runs(*roots, atol=1e6, rtol=1e6)
        assert result["status"] == ("passed" if change == "metadata" else "failed")


def test_selected_video_soundtrack_rejects_video_only_artifact(tmp_path: Path) -> None:
    write_aac_video(tmp_path / "video.mp4", audio=False)
    assert replay.exported_audio_arrays(tmp_path, []) == {}
    with pytest.raises(ValueError, match="soundtrack is missing"):
        replay.exported_audio_arrays(tmp_path, ["video.mp4"])


def test_adaworld_replay_supplies_native_loader_input_and_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.synthesis.visual_generation.adaworld import worldfoundry_runtime

    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_INPUT_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT", str(tmp_path / "weights"))
    matrix = Path(__file__).resolve().parents[1] / "manual/geometry_regression_cases.json"
    case = replay.expand(json.loads(matrix.read_text())["adaworld-context-short"], tmp_path)
    command = worldfoundry_runtime.build_command(
        {"python": "/env/python", "output_path": str(tmp_path / "video.mp4"), "options": case["load"]}
    )
    flags = dict(zip(command[3::2], command[4::2], strict=True))
    assert flags["--source-video"] == case["call"]["video"]
    assert flags["--seed"] == str(case["seed"])
    assert flags["--video-len"] == "20"


def test_ltx25_replay_reaches_checkpoint_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.pipelines.ltx25.pipeline_ltx25 import LTX25Pipeline

    monkeypatch.setenv("WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT", str(tmp_path / "weights"))
    matrix = Path(__file__).resolve().parents[1] / "manual/geometry_regression_cases.json"
    case = replay.expand(json.loads(matrix.read_text())["ltx25-distilled-video-short"], tmp_path)
    load: dict[str, Any] = {**case["load"], "device": "cpu"}
    pipeline = LTX25Pipeline.from_pretrained(**load)
    monkeypatch.setattr(
        pipeline, "preflight", lambda: {"status": "blocked", "blocked_reasons": ["replay checkpoint preflight"]}
    )
    with pytest.raises(RuntimeError, match="LTX-2.5 preflight failed; missing: replay checkpoint preflight"):
        pipeline(**case["call"])
    assert not Path(case["call"]["output_path"]).exists()


def test_neoverse_replay_uses_native_camera_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = root / "tests/manual/geometry_regression_cases.json"
    case = json.loads(matrix.read_text())["neoverse-navigation-short"]
    call = case["call"]
    source = root / "worldfoundry/synthesis/visual_generation/neoverse"
    tree = ast.parse((source / "worldfoundry_runtime.py").read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert {"WanModel", "NeoVerseControlBranch", "NeoVerseWorldMirror", "DepthAnything3Reconstructor"} <= imports
    assert "canonical Wan 2.1 denoising" in case["scope"]
    assert "checkpoint-selected WorldMirror/DepthAnything3" in case["scope"]
    predict = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "predict")
    parameters = {argument.arg for argument in predict.args.kwonlyargs}
    assert {"predefined_trajectory", "distance", "num_frames"} <= parameters & call.keys()
    auxiliary = ast.parse((source / "native/auxiliary.py").read_text())
    choices = next(
        node.value
        for node in ast.walk(auxiliary)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "VALID_TRAJECTORY_TYPES"
    )
    assert choices is not None
    assert call["predefined_trajectory"] in ast.literal_eval(choices)
