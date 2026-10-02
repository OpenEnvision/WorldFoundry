"""Missing variant cases, stale GPU results and planning cannot pass as inference."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml


def tool(name):
    path = Path(__file__).resolve().parents[1] / "manual" / (name + ".py")
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = tool("inference_regression_gate")
inventory = tool("inference_regression_coverage")


def documents():
    identities = {"source_revision": "1" * 40, "source_tree": "2" * 40,
                  "matrix_sha256": "3" * 64, "dependencies_sha256": "4" * 64}
    plan = {**identities, "status": "planned", "selected_cases": ["video-a", "world-b"]}
    evidence = {"status": "passed", "atol": 0, "rtol": 0, "outputs": {
        "latents": {"shape": [1, 2, 3, 2, 2], "passed": True, "exact": True, "max_abs_error": 0.0},
        "frames": {"shape": [9, 16, 32, 3], "passed": True, "exact": True, "max_abs_error": 0.0},
    }}
    report = {**identities, "schema_version": 1, "mode": "replay", "status": "passed",
              "reference_index_sha256": "5" * 64, "selected_cases": ["video-a", "world-b"],
              "cases": {"video-a": copy.deepcopy(evidence), "world-b": copy.deepcopy(evidence)}}
    return plan, report


def test_same_commit_complete_exact_replay_is_verified_with_cpu_gates_still_required():
    plan, report = documents()
    result = gate.verify_replay(plan, report, "5" * 64)
    assert result["status"] == "gpu_replay_verified"
    assert result["exact_numeric_fields"] == 4
    assert result["required_checks"] == ["public-cpu", "inference-tensors"]


@pytest.mark.parametrize("key", ["source_revision", "source_tree", "matrix_sha256", "dependencies_sha256"])
def test_gpu_result_from_other_commit_or_definitions_cannot_authorize_merge(key):
    plan, report = documents()
    report[key] = "6" * len(report[key])
    with pytest.raises(ValueError, match="stale or incompatible"):
        gate.verify_replay(plan, report, "5" * 64)


@pytest.mark.parametrize("status,mode", [("preflight_passed", "preflight"), ("running", "replay"),
                                        ("failed", "replay"), ("passed", "preflight")])
def test_preflight_planning_failed_or_unfinished_runs_cannot_pass(status, mode):
    plan, report = documents()
    report.update(status=status, mode=mode)
    with pytest.raises(ValueError, match="real inference"):
        gate.verify_replay(plan, report, "5" * 64)


@pytest.mark.parametrize("change", ["missing_case", "selected_case_missing", "case_record_missing", "duplicate_case",
                                   "empty_outputs", "failed_case", "relaxed_tolerance", "drift", "not_exact",
                                   "nonfinite_error", "missing_shape", "empty_shape_dimension", "different_reference"])
def test_incomplete_or_relaxed_evidence_never_passes(change):
    plan, report = documents()
    case = report["cases"]["world-b"]
    if change == "missing_case":
        report["selected_cases"].remove("world-b")
        report["cases"].pop("world-b")
    elif change == "selected_case_missing":
        report["selected_cases"].remove("world-b")
    elif change == "case_record_missing":
        report["cases"].pop("world-b")
    elif change == "duplicate_case":
        report["selected_cases"].append("world-b")
    elif change == "empty_outputs":
        case["outputs"] = {}
    elif change == "failed_case":
        case["status"] = "failed"
    elif change == "relaxed_tolerance":
        case["atol"] = 0.001
    elif change == "drift":
        case["outputs"]["latents"]["max_abs_error"] = 0.000001
    elif change == "not_exact":
        case["outputs"]["frames"]["exact"] = False
    elif change == "nonfinite_error":
        case["outputs"]["latents"]["max_abs_error"] = float("nan")
    elif change == "missing_shape":
        case["outputs"]["latents"].pop("shape")
    elif change == "empty_shape_dimension":
        case["outputs"]["latents"]["shape"] = [1, 0, 3]
    else:
        report["reference_index_sha256"] = "6" * 64
    with pytest.raises(ValueError):
        gate.verify_replay(plan, report, "5" * 64)


@pytest.mark.parametrize("status", ["uncovered", "failed", "planned"])
def test_uncovered_or_contradictory_plan_is_rejected(status):
    plan, report = documents()
    plan.update(status=status, selected_cases=[])
    with pytest.raises(ValueError):
        gate.verify_replay(plan, report, "5" * 64)


def test_documentation_only_plan_does_not_claim_numerical_inference_passed():
    result = gate.verify_replay({"status": "no_inference_changes", "selected_cases": []}, {}, "5" * 64)
    assert result["status"] == "no_inference_changes"
    assert result["required_checks"] == ["public-cpu", "inference-tensors"]


@pytest.fixture
def committed_plan(project):
    root, _, _, matrix = project
    dependencies = root / "dependencies.json"
    dependencies.write_text(json.dumps({
        "schema_version": 1, "shared_paths": ["cases.json", "dependencies.json"],
        "ignored_paths": ["docs/**", "tests/**"],
        "components": {"video": ["worldfoundry/pipelines/example.py"]},
        "cases": {"small-short": ["video"]},
    }))

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Regression test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "original")
    base = git("rev-parse", "HEAD")
    (root / "worldfoundry/pipelines/example.py").write_text("class Pipeline: changed = True\n")
    git("add", ".")
    git("commit", "-qm", "change pipeline")
    head = git("rev-parse", "HEAD")
    plan = gate.impact.select_cases(matrix, dependencies, root, gate.impact.changed_paths(root, base, head))
    plan.update(base_revision=base, head_revision=head)
    return root, matrix, dependencies, plan


def test_commit_diff_plan_is_recomputed_before_gpu_verification(committed_plan):
    root, matrix, dependencies, plan = committed_plan
    assert plan["selected_cases"] == ["small-short"]
    gate.verify_checkout_plan(plan, root, matrix, dependencies)


@pytest.mark.parametrize("change", ["omitted_path", "manual_paths", "other_head", "stale_definitions", "omitted_case"])
def test_missing_diff_paths_or_cases_cannot_bypass_commit_gpu_verification(committed_plan, change):
    root, matrix, dependencies, plan = committed_plan
    if change == "omitted_path":
        plan.update(changed_paths=[], status="no_inference_changes", selected_cases=[])
    elif change == "manual_paths":
        plan.pop("base_revision")
    elif change == "other_head":
        plan["head_revision"] = plan["base_revision"]
    elif change == "stale_definitions":
        plan["matrix_sha256"] = "0" * 64
    else:
        plan.update(status="no_inference_changes", selected_cases=[])
    with pytest.raises(ValueError):
        gate.verify_checkout_plan(plan, root, matrix, dependencies)


@pytest.fixture
def project(tmp_path):
    catalog = tmp_path / "worldfoundry/data/models/catalog"
    bindings = tmp_path / "worldfoundry/data/models/bindings/pipelines"
    for path in [catalog / "video", catalog / "world_models", bindings]:
        path.mkdir(parents=True)
    module = tmp_path / "worldfoundry/pipelines/example.py"
    module.parent.mkdir()
    module.write_text("class Pipeline: pass\n")
    target = "worldfoundry.pipelines.example:Pipeline"
    entry = {"id": "family", "pipeline_binding": "family", "runner_parity": {"status": "verified"},
             "variants": [{"id": "small", "pipeline_binding": "small"},
                          {"id": "large", "pipeline_binding": "large"}]}
    (catalog / "video/family.yaml").write_text(yaml.safe_dump(entry))
    for name in ["family", "small", "large"]:
        (bindings / (name + ".yaml")).write_text(yaml.safe_dump(
            {"model_id": name, "pipeline": {"target": target}}
        ))
    matrix = tmp_path / "cases.json"
    matrix.write_text(json.dumps({"small-short": {"id": "small-short", "model_id": "small", "target": target}}))
    return tmp_path, catalog, bindings, matrix


def test_shared_pipeline_and_historical_verified_status_do_not_cover_other_variants(project):
    root, _, _, matrix = project
    report = inventory.coverage(root, matrix)
    assert report["status"] == "partial"
    assert report["models_with_cases"] == 1
    assert report["missing_short_cases"] == ["family", "large"]
    assert report["accepted_reference_verified"] is False
    assert report["gpu_inference_verified"] is False


def test_binding_target_mismatch_is_a_broken_case_definition(project):
    root, _, _, matrix = project
    case = json.loads(matrix.read_text())
    case["small-short"]["target"] = "worldfoundry.pipelines.other:Pipeline"
    matrix.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="targets differ"):
        inventory.coverage(root, matrix)


def test_dangling_declared_binding_remains_visible_in_coverage_gaps(project):
    root, _, bindings, matrix = project
    (bindings / "large.yaml").unlink()
    report = inventory.coverage(root, matrix)
    large = next(row for row in report["models"] if row["model_id"] == "large")
    assert large["status"] == "missing_binding"
    assert "large" in report["missing_short_cases"]


@pytest.mark.parametrize("declared_identity", ["small", "small-alias"])
def test_binding_identity_is_independent_of_yaml_filename(project, declared_identity):
    root, catalog, bindings, matrix = project
    original = bindings / "small.yaml"
    config = yaml.safe_load(original.read_text())
    config.update(binding_id="small", aliases=["small-alias"])
    original.unlink()
    (bindings / "unrelated-filename.yml").write_text(yaml.safe_dump(config))
    path = catalog / "video/family.yaml"
    entry = yaml.safe_load(path.read_text())
    entry["variants"][0]["pipeline_binding"] = declared_identity
    path.write_text(yaml.safe_dump(entry))
    report = inventory.coverage(root, matrix)
    small = [row for row in report["models"] if row["model_id"] == "small"]
    assert len(small) == 1
    assert small[0]["binding_id"] == "small"
    assert small[0]["binding_path"].endswith("unrelated-filename.yml")
    assert small[0]["status"] == "case_defined"


def test_ambiguous_binding_identities_fail_instead_of_hiding_a_variant(project):
    root, _, bindings, matrix = project
    config = yaml.safe_load((bindings / "large.yaml").read_text())
    config["aliases"] = ["small"]
    (bindings / "large.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Ambiguous native binding identity"):
        inventory.coverage(root, matrix)


def test_missing_native_implementation_cannot_be_marked_covered(project):
    root, _, _, matrix = project
    (root / "worldfoundry/pipelines/example.py").unlink()
    report = inventory.coverage(root, matrix)
    assert any(row["status"] == "missing_implementation" for row in report["models"])
    assert report["status"] == "partial"


def test_catalog_only_checkpoint_variants_are_listed_without_invented_native_identity(project):
    root, catalog, _, matrix = project
    path = catalog / "video/family.yaml"
    entry = yaml.safe_load(path.read_text())
    entry["variants"].append({"name": "Upstream checkpoint", "checkpoint": "publisher/model"})
    path.write_text(yaml.safe_dump(entry))
    report = inventory.coverage(root, matrix)
    assert report["catalog_without_native_binding"][0]["status"] == "variant_without_native_identity"


@pytest.mark.parametrize("payload", [[], {"models": "bad"}, {"models": ["not a mapping"]}])
def test_invalid_catalog_cannot_silently_reduce_inventory(tmp_path, payload):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        inventory.catalog_entries(path)
