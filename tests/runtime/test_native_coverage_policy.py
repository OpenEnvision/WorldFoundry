"""Coverage policy rejects new gaps and regressions while preserving old gaps."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _module("native_coverage_policy_contract", _ROOT / "tests/manual/native_coverage_policy.py")
inventory = _module("native_coverage_inventory_contract", _ROOT / "tests/manual/inference_regression_coverage.py")


def _recipe(name="small"):
    return {"id": name + "-short", "model_id": name,
            "target": "worldfoundry.pipelines.example:Pipeline", "seed": 42, "deterministic": True,
            "load": {"model_path": "${CHECKPOINT_ROOT}/" + name}, "assets": {"input": "${INPUT_IMAGE}"},
            "call": {"prompt": "A slowly moving camera", "seed": 42}, "required_outputs": ["result.video"]}


@pytest.fixture
def project(tmp_path):
    catalog = tmp_path / "worldfoundry/data/models/catalog"
    bindings = tmp_path / "worldfoundry/data/models/bindings/pipelines"
    for directory in (catalog / "video", catalog / "world_models", bindings):
        directory.mkdir(parents=True)
    pipeline = tmp_path / "worldfoundry/pipelines/example.py"
    pipeline.parent.mkdir()
    pipeline.write_text("from worldfoundry.operators.helper import Operation\nclass Pipeline: pass\n")
    helper = tmp_path / "worldfoundry/operators/helper.py"
    helper.parent.mkdir()
    helper.write_text("class Operation: pass\n")
    (catalog / "video/example.yaml").write_text(yaml.safe_dump(
        {"id": "small", "pipeline_binding": "small", "variants": [{"id": "large", "pipeline_binding": "large"}]}
    ))
    for name in ("small", "large"):
        (bindings / (name + ".yaml")).write_text(yaml.safe_dump(
            {"binding_id": name, "model_id": name, "pipeline": {"target": _recipe()["target"]}}
        ))
    matrix = tmp_path / "cases.json"
    matrix.write_text(json.dumps({"small-short": _recipe()}))
    report = inventory.coverage(tmp_path, matrix)
    cases = json.loads(matrix.read_text())
    baseline = policy.make_baseline(report, cases, tmp_path, "a" * 40)
    return tmp_path, matrix, cases, report, baseline


def test_historical_gap_stays_visible_and_does_not_fail_unaffected_changes(project):
    root, _, cases, report, baseline = project
    result = policy.evaluate_policy(report, cases, baseline, root, ["docs/guide.md", "tests/test_contract.py"])
    assert result["status"] == "passed"
    assert result["existing_uncovered_variants"] == ["large@large"]
    assert report["status"] == "partial"
    assert result["accepted_reference_verified"] is result["gpu_inference_verified"] is False


@pytest.mark.parametrize("mutation", ["remove", "identity_only", "wrong_variant", "missing_checkpoint", "no_outputs"])
def test_existing_coverage_cannot_regress_to_a_placeholder(project, mutation):
    root, _, cases, report, baseline = project
    cases = copy.deepcopy(cases)
    if mutation == "remove":
        cases.clear()
    elif mutation == "identity_only":
        cases["small-short"] = {key: cases["small-short"][key] for key in ("id", "model_id", "target")}
    elif mutation == "wrong_variant":
        cases["small-short"]["model_id"] = "large"
    elif mutation == "missing_checkpoint":
        cases["small-short"]["load"] = {"device": "cuda"}
    else:
        cases["small-short"]["required_outputs"] = []
    result = policy.evaluate_policy(report, cases, baseline, root)
    assert result["status"] == "failed"
    assert any(item["reason"] == "required_case_coverage_regressed" for item in result["violations"])


def test_new_variant_needs_its_own_recipe_not_a_shared_pipeline_case(project):
    root, _, cases, report, baseline = project
    added = {**report["models"][0], "model_id": "new", "binding_id": "new", "case_ids": []}
    report["models"].append(added)
    result = policy.evaluate_policy(report, cases, baseline, root)
    assert any(item["reason"] == "new_native_variant_without_replay_recipe" for item in result["violations"])
    cases["new-short"] = _recipe("new")
    assert policy.evaluate_policy(report, cases, baseline, root)["status"] == "passed"


@pytest.mark.parametrize("changes", [["worldfoundry/operators/helper.py"], ["worldfoundry/core/clock.py"],
                                     ["worldfoundry/data/models/bindings/pipelines/large.yaml"]])
def test_affected_uncovered_variant_cannot_use_the_frozen_gap_exception(project, changes):
    root, _, cases, report, baseline = project
    result = policy.evaluate_policy(report, cases, baseline, root, changes)
    assert "large@large" in result["affected_variants"]
    assert any(item == {"variant": "large@large", "reason": "affected_native_variant_without_replay_recipe"}
               for item in result["violations"])


def test_unknown_inference_dependency_expands_conservatively(project):
    root, _, cases, report, baseline = project
    path = "worldfoundry/synthesis/unclassified_runtime.py"
    result = policy.evaluate_policy(report, cases, baseline, root, [path])
    assert result["unknown_inference_paths"] == [path]
    assert result["status"] == "failed"


def test_explicit_affected_model_without_case_fails_and_unknown_identity_is_rejected(project):
    root, _, cases, report, baseline = project
    assert policy.evaluate_policy(report, cases, baseline, root, affected_models=["large"])["status"] == "failed"
    with pytest.raises(ValueError, match="absent"):
        policy.evaluate_policy(report, cases, baseline, root, affected_models=["invented"])


def test_removing_an_uncovered_catalog_variant_cannot_hide_the_inventory_gap(project):
    root, _, cases, report, baseline = project
    report["models"] = [row for row in report["models"] if row["model_id"] != "large"]
    result = policy.evaluate_policy(report, cases, baseline, root)
    assert any(item["reason"] == "frozen_native_variant_disappeared" for item in result["violations"])


def test_new_binding_outside_catalog_cannot_evade_inventory_requirements(project):
    root, _, cases, report, baseline = project
    path = root / "worldfoundry/data/models/bindings/pipelines/unknown.yaml"
    path.write_text(yaml.safe_dump({"binding_id": "unknown", "model_id": "unknown",
                                   "pipeline": {"target": _recipe()["target"]}}))
    result = policy.evaluate_policy(report, cases, baseline, root)
    assert any(item["reason"] == "new_or_changed_unclassified_binding_without_recipe" for item in result["violations"])


def test_successful_sequence_outputs_are_required_not_only_error_or_reset_steps():
    case = _recipe()
    case.pop("call")
    case["required_outputs"] = ["first.video"]
    case["sequence"] = [{"name": "first", "call": {"prompt": "a"}, "outputs": ["video"]}]
    assert policy.has_replay_recipe(case)
    case["sequence"][0]["expect_error"] = {"type": "ValueError"}
    assert not policy.has_replay_recipe(case)


def test_committed_baseline_preserves_real_repository_gaps():
    matrix = _ROOT / "tests/manual/geometry_regression_cases.json"
    baseline = policy.load_frozen_baseline(_ROOT / "tests/manual/native_coverage_baseline.json")
    result = policy.evaluate_policy(inventory.coverage(_ROOT, matrix), json.loads(matrix.read_text()), baseline, _ROOT)
    assert result["status"] == "passed"
    assert result["existing_uncovered_variants"]
    assert result["native_variants_with_replay_recipes"] >= sum(bool(row["required_case_ids"]) for row in baseline["models"])


def test_baseline_json_cannot_be_refreshed_to_grandfather_a_new_gap(tmp_path):
    baseline = policy.load_frozen_baseline(_ROOT / "tests/manual/native_coverage_baseline.json")
    baseline["models"].append({"model_id": "new", "binding_id": "new", "required_case_ids": []})
    path = tmp_path / "refreshed-baseline.json"
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="Frozen coverage baseline changed"):
        policy.load_frozen_baseline(path)
