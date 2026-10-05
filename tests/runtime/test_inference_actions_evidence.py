"""Only official, current and complete GPU artifacts can satisfy the CPU gate."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
import yaml

PATH = Path(__file__).resolve().parents[1] / "manual/inference_actions_evidence.py"
SPEC = importlib.util.spec_from_file_location("test_actions_evidence", PATH)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)

PRIMARY_CHECK_CONTEXTS = (
    "geometry-impact", "inference-replay", "cpu-tests", "inference-tensors",
    "public-surface", "packaging-license-gate",
)


def actions_job_expression(value, *, event_name, evidence_json):
    """Evaluate the workflow's string, comparison and short-circuit expressions."""
    if isinstance(value, bool) or not value.startswith("${{"):
        return value
    assert value.endswith("}}"), value
    expression = ast.parse(value[3:-2].strip().replace("&&", " and ").replace("||", " or "), mode="eval")
    contexts = {"github.event_name": event_name, "inputs.evidence_json": evidence_json}

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
            return node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return contexts[f"{node.value.id}.{node.attr}"]
        if isinstance(node, ast.Compare):
            assert len(node.ops) == len(node.comparators) == 1
            left, right = evaluate(node.left), evaluate(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        if isinstance(node, ast.BoolOp):
            result = evaluate(node.values[0])
            for operand in node.values[1:]:
                if isinstance(node.op, ast.And) and not result:
                    return result
                if isinstance(node.op, ast.Or) and result:
                    return result
                result = evaluate(operand)
            return result
        raise AssertionError(f"Unsupported workflow expression: {ast.dump(node)}")

    return evaluate(expression.body)


@pytest.mark.parametrize("event_name", ["push", "pull_request", "workflow_dispatch"])
@pytest.mark.parametrize("report_input", ["", "{}", "invalid-json", " ", "0", "false"])
def test_evidence_dispatch_skips_cannot_replace_required_check_contexts(event_name, report_input):
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    jobs = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    assert set(PRIMARY_CHECK_CONTEXTS) <= jobs.keys()
    attesting = event_name == "workflow_dispatch" and report_input != ""
    emitted_names, skipped_names = {}, []
    for job_id, job in jobs.items():
        context = {"event_name": event_name, "evidence_json": report_input}
        name = actions_job_expression(job.get("name", job_id), **context)
        runs = actions_job_expression(job.get("if", True), **context)
        assert isinstance(name, str) and name, (job_id, name)
        emitted_names[job_id] = name
        if not runs:
            skipped_names.append(name)
        if job_id in PRIMARY_CHECK_CONTEXTS:
            assert runs == (not attesting), (job_id, event_name, report_input)
            if attesting:
                assert name not in PRIMARY_CHECK_CONTEXTS, (job_id, name)
            else:
                assert name == job_id, (job_id, name)
    assert len(set(emitted_names.values())) == len(emitted_names)
    assert set(skipped_names).isdisjoint(PRIMARY_CHECK_CONTEXTS)
    attestation = jobs["attest-inference-evidence"]
    assert actions_job_expression(attestation["if"], event_name=event_name, evidence_json=report_input) == attesting
    if attesting:
        assert len(skipped_names) == len(PRIMARY_CHECK_CONTEXTS)
        assert set(emitted_names.values()).isdisjoint(PRIMARY_CHECK_CONTEXTS)


def documents():
    identities = {"source_revision": "1" * 40, "source_tree": "2" * 40,
                  "matrix_sha256": "3" * 64, "dependencies_sha256": "4" * 64}
    plan = {**identities, "base_revision": "6" * 40, "head_revision": "1" * 40,
            "status": "planned", "selected_cases": ["video-a"]}
    report = {**identities, "schema_version": 1, "mode": "replay", "status": "passed",
              "reference_index_sha256": "5" * 64, "selected_cases": ["video-a"],
              "cases": {"video-a": {"status": "passed", "atol": 0, "rtol": 0, "outputs": {
                  "frames": {"shape": [9, 16, 32, 3], "passed": True, "exact": True, "max_abs_error": 0},
              }}}}
    receipt = evidence.attest_report(plan, report, reference="5" * 64, repository="owner/repo", run_id=73,
                                     control_revision="7" * 40, control_ref="main", expected_control_ref="main")
    run = {"id": 73, "event": "workflow_dispatch", "status": "completed", "conclusion": "success",
           "path": evidence.WORKFLOW, "head_branch": "main", "head_sha": "7" * 40,
           "repository": {"full_name": "owner/repo"}, "head_repository": {"full_name": "owner/repo"}}
    artifact = {"id": 87, "name": evidence.ARTIFACT, "expired": False, "workflow_run": {"id": 73}}
    return plan, run, artifact, receipt


def verify(plan, run, artifact, receipt):
    return evidence.verify_origin(run, artifact, receipt, repository="owner/repo", control_ref="main",
                                  plan=plan, reference="5" * 64)


def test_official_successful_current_dispatch_retains_exact_replay_requirements():
    plan, run, artifact, receipt = documents()
    result = verify(plan, run, artifact, receipt)
    assert result["status"] == "gpu_replay_verified"
    assert result["evidence_run_id"] == 73
    assert result["evidence_artifact_id"] == 87
    assert result["exact_numeric_fields"] == 1
    assert result["required_checks"] == ["public-cpu", "inference-tensors"]


@pytest.mark.parametrize("field,value", [
    ("event", "pull_request"), ("event", "push"), ("status", "in_progress"), ("conclusion", "failure"),
    ("path", ".github/workflows/other.yml"), ("head_branch", "candidate"),
    ("repository", {"full_name": "fork/repo"}), ("head_repository", {"full_name": "fork/repo"}),
])
def test_failed_untrusted_fork_or_different_workflow_run_is_rejected(field, value):
    plan, run, artifact, receipt = documents()
    run[field] = value
    with pytest.raises(ValueError, match="trusted successful dispatch"):
        verify(plan, run, artifact, receipt)


@pytest.mark.parametrize("field,value", [("expired", True), ("expired", None), ("name", "user-report"),
                                         ("workflow_run", {"id": 99})])
def test_expired_unassociated_or_unofficial_artifact_is_rejected(field, value):
    plan, run, artifact, receipt = documents()
    artifact[field] = value
    with pytest.raises(ValueError):
        verify(plan, run, artifact, receipt)


@pytest.mark.parametrize("field,value", [
    ("source_revision", "9" * 40), ("source_tree", "9" * 40), ("base_revision", "9" * 40),
    ("control_revision", "9" * 40), ("run_id", 74), ("repository", "fork/repo"),
    ("workflow_path", ".github/workflows/other.yml"), ("schema_version", 2),
])
def test_reused_receipt_or_changed_identity_is_rejected(field, value):
    plan, run, artifact, receipt = documents()
    receipt[field] = value
    with pytest.raises(ValueError, match="commit identities"):
        verify(plan, run, artifact, receipt)


@pytest.mark.parametrize("change", ["preflight", "missing_case", "skip", "drift", "reference", "source"])
def test_official_provenance_does_not_replace_exact_numeric_validation(change):
    plan, run, artifact, receipt = documents()
    report = receipt["report"]
    if change == "preflight":
        report["mode"] = "preflight"
    elif change == "missing_case":
        report["cases"] = {}
    elif change == "skip":
        report["cases"]["video-a"]["status"] = "skipped"
    elif change == "drift":
        report["cases"]["video-a"]["outputs"]["frames"]["max_abs_error"] = 0.001
    elif change == "reference":
        report["reference_index_sha256"] = "9" * 64
    else:
        report["source_revision"] = "9" * 40
    with pytest.raises(ValueError):
        verify(plan, run, artifact, receipt)


@pytest.mark.parametrize("ref", ["", "candidate"])
def test_maintainer_dispatch_requires_configured_control_branch(ref):
    plan, _, _, receipt = documents()
    with pytest.raises(ValueError, match="control branch"):
        evidence.attest_report(plan, receipt["report"], reference="5" * 64, repository="owner/repo", run_id=73,
                               control_revision="7" * 40, control_ref=ref, expected_control_ref="main")


def archive(value, *, members=None):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in members or [evidence.RECEIPT]:
            bundle.writestr(name, json.dumps(value))
    return target.getvalue()


def test_official_artifact_bytes_are_bound_to_actions_digest():
    *_, receipt = documents()
    body = archive(receipt)
    assert evidence.decode_artifact(body, "sha256:" + hashlib.sha256(body).hexdigest()) == receipt
    with pytest.raises(ValueError, match="SHA256"):
        evidence.decode_artifact(body, "sha256:" + "0" * 64)


@pytest.mark.parametrize("members", [["../receipt.json"], ["receipt.json", "other.json"],
                                      ["receipt.json", "receipt.json"], ["nested/receipt.json"]])
def test_archive_cannot_select_arbitrary_or_duplicate_receipts(members):
    *_, receipt = documents()
    if len(members) != len(set(members)):
        with pytest.warns(UserWarning, match="Duplicate name"):
            body = archive(receipt, members=members)
    else:
        body = archive(receipt, members=members)
    with pytest.raises(ValueError, match="only the bounded"):
        evidence.decode_artifact(body, None)


def test_compressed_artifact_bomb_is_rejected_before_extraction():
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(evidence.RECEIPT, " " * (evidence.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        evidence.decode_artifact(target.getvalue(), None)


class FakeAPI:
    repository = "owner/repo"

    def __init__(self, runs, artifacts, receipts):
        self.runs, self.artifacts, self.receipts = runs, artifacts, receipts
        self.calls = []

    def request(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if "/artifacts?" in path:
            return {"artifacts": self.artifacts}
        return {"workflow_runs": self.runs}

    def artifact(self, value):
        return copy.deepcopy(self.receipts[value["id"]])


def test_ci_downloads_official_artifact_and_does_not_read_repository_report():
    plan, run, artifact, receipt = documents()
    api = FakeAPI([run], [artifact], {artifact["id"]: receipt})
    assert evidence.find_evidence(api, plan, control_ref="main", reference="5" * 64)["status"] == "gpu_replay_verified"
    assert "/actions/workflows/ci.yml/runs?" in api.calls[0][0]
    assert "event=workflow_dispatch" in api.calls[0][0]
    assert all("/contents/" not in path for path, _ in api.calls)


@pytest.mark.parametrize("change", ["missing", "expired", "different_head", "failed", "duplicate"])
def test_missing_stale_failed_or_duplicate_official_evidence_fails_ci(change):
    plan, run, artifact, receipt = documents()
    artifacts = [artifact]
    if change == "missing":
        artifacts = []
    elif change == "expired":
        artifact["expired"] = True
    elif change == "different_head":
        receipt["source_revision"] = "9" * 40
    elif change == "failed":
        run["conclusion"] = "failure"
    else:
        artifacts.append(copy.deepcopy(artifact))
    api = FakeAPI([run], artifacts, {artifact["id"]: receipt})
    with pytest.raises(ValueError, match="No official complete"):
        evidence.find_evidence(api, plan, control_ref="main", reference="5" * 64)


def test_missing_config_cannot_silently_disable_required_gpu_gate():
    plan, run, artifact, receipt = documents()
    api = FakeAPI([run], [artifact], {artifact["id"]: receipt})
    with pytest.raises(ValueError, match="control branch"):
        evidence.find_evidence(api, plan, control_ref="", reference="5" * 64)
    with pytest.raises(ValueError, match="accepted reference"):
        evidence.find_evidence(api, plan, control_ref="main", reference="")
    assert not api.calls


def test_rerun_only_failed_gate_for_exact_commit():
    class API:
        def __init__(self):
            self.calls = []

        def request(self, path, **kwargs):
            self.calls.append((path, kwargs))
            if "/workflows/ci.yml/runs?" in path:
                return {"workflow_runs": [{"id": 1, "head_sha": "1" * 40}, {"id": 2, "head_sha": "2" * 40}]}
            if "/jobs?" in path:
                return {"jobs": [{"id": 10, "name": "inference-replay", "conclusion": "failure"},
                                 {"id": 11, "name": "inference-replay", "conclusion": "success"},
                                 {"id": 12, "name": "cpu-tests", "conclusion": "failure"}]}
            return None

    api = API()
    assert evidence.rerun_gate_jobs(api, "1" * 40) == [10]
    assert [path for path, kwargs in api.calls if kwargs.get("method") == "POST"] == ["/actions/jobs/10/rerun"]
    assert not any("/runs/2/" in path for path, _ in api.calls)


@pytest.fixture
def historical_contract(tmp_path):
    tool = evidence.historical_tool()
    accepted = tool.MG2HistoricalReference()
    paths = set(accepted.required_source_paths) | {evidence.MG2_HELPER, evidence.MG2_TEST, evidence.MG2_FROZEN_HELPER}
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Candidate files are inspected as Git blobs and are never executed.
        target.write_text("# Independent committed source fixture: " + relative + "\n")
    for args in [("init", "-q"), ("config", "user.name", "Regression test"),
                 ("config", "user.email", "test@example.invalid"), ("add", "."), ("commit", "-qm", "sources")]:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    plan, _, _, receipt = documents()
    plan.update(source_revision=evidence.gate.impact.revision(tmp_path),
                source_tree=evidence.gate.impact.revision(tmp_path, "HEAD^{tree}"))
    generic = receipt["report"]
    generic.update(source_revision=plan["source_revision"], source_tree=plan["source_tree"])
    contract = {"schema_version": 1, "contract_id": evidence.MG2_CONTRACT, "status": "passed",
                "source_revision": plan["source_revision"], "source_tree": plan["source_tree"],
                "historical_reference_sha256": tool.REFERENCE_SHA256,
                "historical_reference_revision": tool.REFERENCE_REVISION,
                "numeric_signatures": accepted.expected_signatures,
                "source_sha256": {path: evidence.checkout_digest(tmp_path, path)
                                  for path in set(accepted.required_source_paths) | {evidence.MG2_HELPER}},
                "test_source_sha256": evidence.checkout_digest(tmp_path, evidence.MG2_TEST),
                "frozen_helper_sha256": evidence.checkout_digest(tmp_path, evidence.MG2_FROZEN_HELPER),
                "junit": {"status": "passed", "executed_tests": 6, "skipped": 0,
                          "complete_selection_verified": True, "source_files_verified": 1, "sha256": "8" * 64},
                "raw_report_sha256": "9" * 64}
    bundle = {"schema_version": 2, "mode": "trusted_contract_bundle", "status": "passed",
              "source_revision": plan["source_revision"], "source_tree": plan["source_tree"],
              "generic_replay": generic, "contracts": {evidence.MG2_CONTRACT: contract}}
    return tmp_path, plan, bundle


def test_independent_historical_contract_retains_separate_identity(historical_contract):
    root, plan, bundle = historical_contract
    result = evidence.verify_bundle(plan, bundle, "5" * 64, root)
    assert result["executed_cases"] == ["video-a"]
    assert result["independent_contracts"][evidence.MG2_CONTRACT]["executed_tests"] == 6
    assert evidence.MG2_CONTRACT not in result["executed_cases"]


@pytest.mark.parametrize("change", ["historical_reference", "historical_signature", "missing_historical_case",
                                    "source_commit", "source_tree", "missing_source", "source_blob", "test_blob",
                                    "junit_skip", "junit_missing", "junit_selection", "missing_raw_hash"])
def test_historical_contract_cannot_pass_with_changed_output_source_or_incomplete_execution(historical_contract, change):
    root, plan, bundle = historical_contract
    contract = bundle["contracts"][evidence.MG2_CONTRACT]
    if change == "historical_reference":
        contract["historical_reference_sha256"] = "0" * 64
    elif change == "historical_signature":
        contract["numeric_signatures"]["conditioning"] = "0" * 64
    elif change == "missing_historical_case":
        contract["numeric_signatures"]["contracts"].pop("split_seed7")
    elif change == "source_commit":
        contract["source_revision"] = "0" * 40
    elif change == "source_tree":
        contract["source_tree"] = "0" * 40
    elif change == "missing_source":
        contract["source_sha256"].pop(evidence.MG2_HELPER)
    elif change == "source_blob":
        (root / evidence.MG2_HELPER).write_text("changed after replay\n")
    elif change == "test_blob":
        contract["test_source_sha256"] = "0" * 64
    elif change == "junit_skip":
        contract["junit"]["skipped"] = 1
    elif change == "junit_missing":
        contract["junit"]["executed_tests"] = 5
    elif change == "junit_selection":
        contract["junit"]["complete_selection_verified"] = False
    else:
        contract.pop("raw_report_sha256")
    with pytest.raises(ValueError):
        evidence.verify_bundle(plan, bundle, "5" * 64, root)


def test_gpu_historical_contract_cannot_replace_generic_matrix_replay(historical_contract):
    root, plan, bundle = historical_contract
    generic = bundle["generic_replay"]
    generic["cases"]["video-a"]["status"] = "failed"
    with pytest.raises(ValueError, match="Case failed"):
        evidence.verify_bundle(plan, bundle, "5" * 64, root)


def test_mg2_change_requires_both_matrix_replay_and_historical_contract(historical_contract):
    root, plan, bundle = historical_contract
    generic = bundle["generic_replay"]
    name = "matrix-game2-controls-short"
    plan["selected_cases"] = [name]
    generic["selected_cases"] = [name]
    generic["cases"] = {name: generic["cases"]["video-a"]}
    assert evidence.verify_bundle(plan, bundle, "5" * 64, root)["status"] == "gpu_replay_verified"
    with pytest.raises(ValueError, match="both replay"):
        evidence.verify_bundle(plan, generic, "5" * 64, root)
    bundle["contracts"] = {}
    with pytest.raises(ValueError, match="independent historical"):
        evidence.verify_bundle(plan, bundle, "5" * 64, root)


def test_uncovered_shared_changes_cannot_be_excused_by_one_historical_model_contract(historical_contract):
    root, plan, bundle = historical_contract
    plan.update(status="uncovered", selected_cases=["video-a", "world-b"])
    with pytest.raises(ValueError, match="Uncovered"):
        evidence.verify_bundle(plan, bundle, "5" * 64, root)


def test_independent_contract_needs_current_committed_checkout(historical_contract):
    _, plan, bundle = historical_contract
    with pytest.raises(ValueError, match="Committed source checkout"):
        evidence.verify_bundle(plan, bundle, "5" * 64, None)


def test_optional_diagnostic_gpu_evidence_does_not_mislabel_no_inference_change(historical_contract):
    root, plan, bundle = historical_contract
    plan.update(status="no_inference_changes", selected_cases=[])
    result = evidence.verify_bundle(plan, bundle, "5" * 64, root)
    assert result["status"] == "no_inference_changes"
    assert result["diagnostic_replay"]["status"] == "gpu_replay_verified"
    assert result["independent_contracts"][evidence.MG2_CONTRACT]["status"] == "passed"


def raw_documents(root, plan, contract):
    accepted = evidence.historical_tool().MG2HistoricalReference()
    raw = copy.deepcopy(accepted._reference)
    raw.update(source_revision=plan["source_revision"], source_tree=plan["source_tree"],
               source_sha256={str(root / key): value for key, value in contract["source_sha256"].items()},
               test_source_sha256=contract["test_source_sha256"],
               frozen_helper_sha256=contract["frozen_helper_sha256"],
               historical_reference={**accepted.receipt, "conditioning_passed": True,
                                     "contracts_passed": list(accepted.expected_signatures["contracts"])})
    document = ET.Element("testsuite", tests="6", failures="0", errors="0", skipped="0")
    for key, record in raw["contracts"].items():
        record["historical_reference_passed"] = True
        strategy, seed = key.split("_seed")
        ET.SubElement(document, "testcase", classname=evidence.MG2_TEST[:-3].replace("/", "."),
                      name="test_real_conditioned_public_rollout_preserves_12_blocks_exactly[" + strategy + "-" + seed + "]")
    junit = root / "results.xml"
    ET.ElementTree(document).write(junit, encoding="utf-8")
    return raw, junit


def test_full_raw_report_is_compacted_only_after_complete_junit_and_committed_source_verification(historical_contract):
    root, plan, bundle = historical_contract
    raw, junit = raw_documents(root, plan, bundle["contracts"][evidence.MG2_CONTRACT])
    contract = evidence.compact_conditioned_report(raw, root=root, junit=junit)
    assert contract["numeric_signatures"] == bundle["contracts"][evidence.MG2_CONTRACT]["numeric_signatures"]
    assert contract["junit"]["executed_tests"] == 6
    assert all(not Path(key).is_absolute() for key in contract["source_sha256"])
    assert evidence.verify_conditioned_contract(contract, root=root, plan=plan)["status"] == "passed"


@pytest.mark.parametrize("change", ["old_commit", "missing_historical_case", "partial_junit", "different_test_selection"])
def test_compact_command_cannot_promote_incomplete_or_stale_raw_evidence(historical_contract, change):
    root, plan, bundle = historical_contract
    raw, junit = raw_documents(root, plan, bundle["contracts"][evidence.MG2_CONTRACT])
    if change == "old_commit":
        raw["source_revision"] = "0" * 40
    elif change == "missing_historical_case":
        raw["historical_reference"]["contracts_passed"].pop()
    else:
        document = ET.parse(junit)
        if change == "partial_junit":
            document.getroot().remove(document.getroot().find("testcase"))
            document.getroot().set("tests", "5")
        else:
            document.getroot().find("testcase").set("name", "unrelated_test_passes")
        document.write(junit, encoding="utf-8")
    with pytest.raises(ValueError):
        evidence.compact_conditioned_report(raw, root=root, junit=junit)


def test_compact_generic_report_omits_host_paths_and_preserves_all_numeric_fields():
    _, _, _, receipt = documents()
    report = receipt["report"]
    report.update(run_root="/private/model/host", environments=[{"python": "/private/venv/bin/python"}])
    compact = evidence.compact_generic_replay(report)
    assert "run_root" not in compact and "environments" not in compact
    assert compact["cases"] == report["cases"]
