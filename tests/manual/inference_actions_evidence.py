"""Bind numerical replay evidence to a maintainer-dispatched Actions artifact.

GPU replay runs on a prepared trusted model host. Only a successful dispatch of
the configured control workflow may publish evidence; repository JSON files,
stale runs, incomplete cases and expired artifacts do not authorize inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "actions_inference_gate", Path(__file__).with_name("inference_regression_gate.py")
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

WORKFLOW = ".github/workflows/ci.yml"
ALLOWED_WORKFLOWS = {WORKFLOW}
ARTIFACT = "trusted-inference-evidence"
RECEIPT = "receipt.json"
MAX_BYTES = 2 * 1024 * 1024
MG2_CONTRACT = "mg2-real-conditioned-historical"
MG2_TEST = "tests/synthesis/test_matrix_game_2_checkpoint_conditioned_trajectory.py"
MG2_HELPER = "tests/synthesis/mg2_historical_reference.py"
MG2_FROZEN_HELPER = "tests/synthesis/test_matrix_game_2_checkpoint_trajectory.py"


def load_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_tool():
    return load_tool(Path(__file__).resolve().parents[2] / MG2_HELPER, "actions_mg2_historical")


def checkout_digest(root: Path, relative: str) -> str:
    if not isinstance(relative, str) or gate.impact.relative_path(relative) != relative:
        raise ValueError("Source evidence requires canonical repository-relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Source evidence escapes or is absent from the candidate checkout")
    blob = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:" + relative], stderr=subprocess.PIPE)
    digest = hashlib.sha256(blob).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Candidate source file differs from its committed blob: " + relative)
    return digest


def normalized_sources(report: dict, root: Path) -> dict:
    sources = report.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("GPU report has no source file hashes")
    result = {}
    for value, digest in sources.items():
        path = Path(value)
        if path.is_absolute():
            try:
                value = str(path.resolve().relative_to(root.resolve()))
            except ValueError as error:
                raise ValueError("GPU source file is outside its exact candidate checkout") from error
        if value in result:
            raise ValueError("GPU report repeats a source file")
        valid_sha(digest, 64, "executed source file hash")
        if checkout_digest(root, value) != digest:
            raise ValueError("GPU report did not execute the current committed source: " + value)
        result[value] = digest
    return result


def compact_conditioned_report(report: dict, *, root: Path, junit: Path) -> dict:
    """Validate full raw evidence before emitting a bounded maintainer receipt."""
    tool = historical_tool()
    accepted = tool.MG2HistoricalReference()
    accepted.assert_complete_report(report)
    head, tree = gate.impact.revision(root), gate.impact.revision(root, "HEAD^{tree}")
    if report.get("source_revision") != head or report.get("source_tree") != tree:
        raise ValueError("GPU report does not identify this exact source commit and tree")
    historical = report.get("historical_reference", {})
    expected_cases = set(accepted.expected_signatures["contracts"])
    if (historical.get("sha256") != tool.REFERENCE_SHA256 or historical.get("conditioning_passed") is not True
            or set(historical.get("contracts_passed", [])) != expected_cases
            or len(historical.get("contracts_passed", [])) != len(expected_cases)
            or any(report["contracts"][key].get("historical_reference_passed") is not True for key in expected_cases)):
        raise ValueError("GPU run did not complete every historical reference comparison")
    sources = normalized_sources(report, root)
    if set(sources) != set(accepted.required_source_paths) | {MG2_HELPER}:
        raise ValueError("GPU report omits or adds executed MG2 source files")
    if report.get("test_source_sha256") != checkout_digest(root, MG2_TEST):
        raise ValueError("GPU report test differs from the committed test")
    if report.get("frozen_helper_sha256") != checkout_digest(root, MG2_FROZEN_HELPER):
        raise ValueError("GPU report changed its frozen test helper")
    validator = load_tool(Path(__file__).resolve().parents[2] / "tests/manual/validate_junit_contract.py",
                          "actions_junit_gate")
    nodes = [MG2_TEST + "::test_real_conditioned_public_rollout_preserves_12_blocks_exactly["
             + strategy + "-" + str(seed) + "]" for strategy in ("split", "packed") for seed in (7, 42, 1730)]
    manifest = validator.build_manifest(nodes, root, [MG2_TEST])
    junit_result = validator.validate_junit(manifest, junit, root)
    return {"schema_version": 1, "contract_id": MG2_CONTRACT, "status": "passed",
            "source_revision": head, "source_tree": tree,
            "historical_reference_sha256": tool.REFERENCE_SHA256,
            "historical_reference_revision": tool.REFERENCE_REVISION,
            "numeric_signatures": accepted.numeric_signatures(report), "source_sha256": sources,
            "test_source_sha256": report["test_source_sha256"],
            "frozen_helper_sha256": report["frozen_helper_sha256"],
            "junit": {**junit_result, "sha256": hashlib.sha256(junit.read_bytes()).hexdigest()},
            "raw_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def compact_generic_replay(report: dict) -> dict:
    """Preserve numerical evidence while omitting private model-host paths."""
    fields = ("schema_version", "mode", "status", "source_revision", "source_tree", "matrix_sha256",
              "dependencies_sha256", "reference_index_sha256", "selected_cases", "cases")
    return {key: report[key] for key in fields if key in report}


def verify_conditioned_contract(contract: dict, *, root: Path, plan: dict) -> dict:
    tool = historical_tool()
    accepted = tool.MG2HistoricalReference()
    if (contract.get("schema_version") != 1 or contract.get("contract_id") != MG2_CONTRACT
            or contract.get("status") != "passed"
            or contract.get("source_revision") != plan["source_revision"]
            or contract.get("source_tree") != plan["source_tree"]
            or contract.get("historical_reference_sha256") != tool.REFERENCE_SHA256
            or contract.get("historical_reference_revision") != tool.REFERENCE_REVISION):
        raise ValueError("Historical MG2 contract is incomplete, stale or uses a changed reference")
    if contract.get("numeric_signatures") != accepted.expected_signatures:
        raise ValueError("Historical MG2 numerical signatures changed")
    sources = contract.get("source_sha256", {})
    if not isinstance(sources, dict) or set(sources) != set(accepted.required_source_paths) | {MG2_HELPER}:
        raise ValueError("Historical MG2 contract has incomplete source coverage")
    for relative, digest in sources.items():
        if digest != checkout_digest(root, relative):
            raise ValueError("Historical MG2 evidence source differs from this committed checkout: " + relative)
    for key, relative in (("test_source_sha256", MG2_TEST), ("frozen_helper_sha256", MG2_FROZEN_HELPER)):
        if contract.get(key) != checkout_digest(root, relative):
            raise ValueError("Historical MG2 evidence used a different committed test: " + relative)
    junit = contract.get("junit", {})
    if (junit.get("status") != "passed" or junit.get("executed_tests") != 6 or junit.get("skipped") != 0
            or junit.get("complete_selection_verified") is not True or junit.get("source_files_verified") != 1):
        raise ValueError("Historical MG2 contract did not execute all six tests without skips")
    valid_sha(junit.get("sha256"), 64, "complete GPU JUnit hash")
    valid_sha(contract.get("raw_report_sha256"), 64, "full GPU report hash")
    return {"status": "passed", "contract_id": MG2_CONTRACT, "historical_reference_sha256": tool.REFERENCE_SHA256,
            "executed_tests": 6, "exact_historical_signatures": 7}


def verify_bundle(plan: dict, report: dict, reference: str, root: Path | None) -> dict:
    required = "matrix-game2-controls-short" in plan.get("selected_cases", [])
    if report.get("schema_version") == 2 and report.get("mode") == "trusted_contract_bundle":
        if (report.get("status") != "passed" or report.get("source_revision") != plan["source_revision"]
                or report.get("source_tree") != plan["source_tree"]):
            raise ValueError("GPU contract bundle does not match this exact source commit and tree")
        replay = report["generic_replay"]
        if not plan.get("selected_cases") and replay.get("selected_cases"):
            diagnostic_plan = {**plan, "status": "planned", "selected_cases": replay["selected_cases"]}
            diagnostic = gate.verify_replay(diagnostic_plan, replay, reference)
            result = {**gate.verify_replay(plan, {}, reference), "diagnostic_replay": diagnostic}
        else:
            result = gate.verify_replay(plan, replay, reference)
        contracts = report.get("contracts")
        if not isinstance(contracts, dict) or any(key != MG2_CONTRACT for key in contracts):
            raise ValueError("GPU contract bundle contains an unrecognized independent contract")
        if required and MG2_CONTRACT not in contracts:
            raise ValueError("Affected MG2 changes require the independent historical contract as well as the generic replay")
        verified = {}
        for key, contract in contracts.items():
            if root is None:
                raise ValueError("Committed source checkout is required to verify independent contracts")
            verified[key] = verify_conditioned_contract(contract, root=root, plan=plan)
        return {**result, "independent_contracts": verified}
    if required:
        raise ValueError("Affected MG2 changes require a bundle with both replay and independent historical evidence")
    return gate.verify_replay(plan, report, reference)


def valid_sha(value: str, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value):
        raise ValueError(f"A pinned {label} is required")
    return value


def definitions(root: Path) -> tuple[Path, Path]:
    return root / "tests/manual/geometry_regression_cases.json", root / "tests/manual/geometry_regression_dependencies.json"


def plan_for_checkout(root: Path, base: str, head: str) -> dict:
    valid_sha(base, 40, "base commit")
    valid_sha(head, 40, "source commit")
    if gate.impact.revision(root) != head or gate.impact.revision(root, base + "^{commit}") != base:
        raise ValueError("Evidence checkout does not contain the pinned source and base commits")
    matrix, dependencies = definitions(root)
    plan = gate.impact.select_cases(
        matrix, dependencies, root, gate.impact.changed_paths(root, base, head), base=base, head=head
    )
    plan.update(base_revision=base, head_revision=head)
    gate.verify_checkout_plan(plan, root, matrix, dependencies)
    return plan


def attest_report(plan: dict, report: dict, *, reference: str, repository: str, run_id: int,
                  control_revision: str, control_ref: str, expected_control_ref: str,
                  workflow_path: str = WORKFLOW, root: Path | None = None) -> dict:
    if not expected_control_ref or control_ref != expected_control_ref:
        raise ValueError("Dispatch must use the configured trusted control branch")
    valid_sha(control_revision, 40, "control workflow commit")
    valid_sha(reference, 64, "accepted reference index")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("An exact repository identity is required")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("A positive Actions run ID is required")
    if workflow_path not in ALLOWED_WORKFLOWS:
        raise ValueError("An official evidence workflow path is required")
    if not plan.get("selected_cases") and not report.get("contracts"):
        raise ValueError("Only completed affected inference replays need attestation")
    result = verify_bundle(plan, report, reference, root)
    return {"schema_version": 1, "repository": repository, "workflow_path": workflow_path,
            "run_id": run_id, "control_revision": control_revision, "control_ref": control_ref,
            "base_revision": plan["base_revision"], "source_revision": plan["source_revision"],
            "source_tree": plan["source_tree"], "report": report, "verification": result}


def verify_provenance(run: dict, artifact: dict, receipt: dict, *, repository: str, control_ref: str,
                      workflow_path: str = WORKFLOW) -> dict:
    if not control_ref:
        raise ValueError("Trusted evidence control branch is not configured")
    if workflow_path not in ALLOWED_WORKFLOWS:
        raise ValueError("An official evidence workflow path is required")
    if (run.get("event") != "workflow_dispatch" or run.get("status") != "completed"
            or run.get("conclusion") != "success" or run.get("path") != workflow_path
            or run.get("head_branch") != control_ref
            or run.get("repository", {}).get("full_name") != repository
            or run.get("head_repository", {}).get("full_name") != repository):
        raise ValueError("GPU evidence did not originate from the trusted successful dispatch")
    if artifact.get("name") != ARTIFACT or artifact.get("expired") is not False:
        raise ValueError("Trusted GPU evidence artifact is absent or expired")
    artifact_run = artifact.get("workflow_run", {})
    if artifact_run.get("id") != run.get("id"):
        raise ValueError("Artifact belongs to a different workflow run")
    valid_sha(run.get("head_sha"), 40, "control workflow commit")
    expected = {"schema_version": 1, "repository": repository, "workflow_path": workflow_path,
                "run_id": run["id"], "control_revision": run["head_sha"], "control_ref": control_ref}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Official GPU artifact is stale or its commit identities do not match")
    metadata = {key: valid_sha(receipt.get(key), 40, key)
                for key in ("base_revision", "source_revision", "source_tree")}
    return metadata


def verify_origin(run: dict, artifact: dict, receipt: dict, *, repository: str, control_ref: str,
                  plan: dict, reference: str, workflow_path: str = WORKFLOW, root: Path | None = None) -> dict:
    metadata = verify_provenance(run, artifact, receipt, repository=repository, control_ref=control_ref,
                                 workflow_path=workflow_path)
    if any(value != plan[key] for key, value in metadata.items()):
        raise ValueError("Official GPU artifact is stale or its commit identities do not match")
    result = verify_bundle(plan, receipt["report"], reference, root)
    return {**result, "evidence_run_id": run["id"], "evidence_artifact_id": artifact["id"],
            "evidence_control_revision": run["head_sha"]}


def inspect_receipt(api, *, run_id: int, control_ref: str, workflow_path: str) -> dict:
    """Read only bounded immutable identities before checking out a candidate."""
    run = api.request("/actions/runs/" + str(run_id))
    artifacts = api.request("/actions/runs/" + str(run_id) + "/artifacts?per_page=100")["artifacts"]
    matched = [artifact for artifact in artifacts if artifact.get("name") == ARTIFACT]
    if len(matched) != 1:
        raise ValueError("Official dispatch has no unique trusted GPU evidence artifact")
    artifact = matched[0]
    receipt = api.artifact(artifact)
    metadata = verify_provenance(run, artifact, receipt, repository=api.repository, control_ref=control_ref,
                                 workflow_path=workflow_path)
    return {"status": "identities_verified", **metadata}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, repository: str, token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("An exact repository identity is required")
        if not token:
            raise ValueError("An Actions API token is required to verify official evidence")
        self.repository = repository
        self.base = "https://api.github.com/repos/" + repository
        self.headers = {"Accept": "application/vnd.github+json", "Authorization": "Bearer " + token,
                        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "WorldFoundry-inference-gate"}

    def request(self, path: str, *, data: dict | None = None, method: str = "GET"):
        request = urllib.request.Request(self.base + path, headers=self.headers, method=method,
                                         data=None if data is None else json.dumps(data).encode())
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("GitHub API response exceeds the evidence size limit")
        return json.loads(body) if body else None

    def artifact(self, artifact: dict) -> dict:
        path = "/actions/artifacts/" + str(artifact["id"]) + "/zip"
        request = urllib.request.Request(self.base + path, headers=self.headers)
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(request, timeout=30) as response:
                body = response.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers["Location"]
            if urllib.parse.urlparse(location).scheme != "https":
                raise ValueError("Official artifact redirect must use HTTPS") from error
            # Signed storage redirects receive no repository authorization header.
            with urllib.request.urlopen(location, timeout=30) as response:
                body = response.read(MAX_BYTES + 1)
        return decode_artifact(body, artifact.get("digest"))


def decode_artifact(body: bytes, digest: str | None) -> dict:
    if len(body) > MAX_BYTES:
        raise ValueError("Official artifact exceeds the evidence size limit")
    if digest and digest != "sha256:" + hashlib.sha256(body).hexdigest():
        raise ValueError("Official artifact SHA256 does not match its Actions metadata")
    with zipfile.ZipFile(io.BytesIO(body)) as bundle:
        members = bundle.infolist()
        if len(members) != 1 or members[0].filename != RECEIPT or members[0].file_size > MAX_BYTES:
            raise ValueError("Official evidence must contain only the bounded receipt.json")
        receipt = bundle.read(RECEIPT)
    value = json.loads(receipt)
    if not isinstance(value, dict):
        raise ValueError("Official evidence is not a receipt object")
    return value


def find_evidence(api: GitHub, plan: dict, *, control_ref: str, reference: str, run_id: int | None = None,
                  workflow_path: str = WORKFLOW, root: Path | None = None) -> dict:
    valid_sha(reference, 64, "accepted reference index")
    if not control_ref:
        raise ValueError("Trusted evidence control branch is not configured")
    if workflow_path not in ALLOWED_WORKFLOWS:
        raise ValueError("An official evidence workflow path is required")
    workflow = urllib.parse.quote(workflow_path.rsplit("/", 1)[1], safe="")
    if run_id is not None:
        runs = [api.request("/actions/runs/" + str(run_id))]
    else:
        query = urllib.parse.urlencode({"event": "workflow_dispatch", "status": "success",
                                        "branch": control_ref, "per_page": 100})
        runs = api.request("/actions/workflows/" + workflow + "/runs?" + query)["workflow_runs"]
    errors = []
    for run in runs:
        artifacts = api.request("/actions/runs/" + str(run["id"]) + "/artifacts?per_page=100")["artifacts"]
        matched = [artifact for artifact in artifacts if artifact.get("name") == ARTIFACT]
        if len(matched) != 1:
            continue
        artifact = matched[0]
        if artifact.get("expired") is not False:
            continue
        try:
            receipt = api.artifact(artifact)
            if receipt.get("source_revision") != plan["source_revision"]:
                continue
            return verify_origin(run, artifact, receipt, repository=api.repository, control_ref=control_ref,
                                 plan=plan, reference=reference, workflow_path=workflow_path, root=root)
        except (ValueError, KeyError, TypeError, AssertionError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
            errors.append(str(error))
    raise ValueError("No official complete GPU replay matches this source commit: " + "; ".join(errors[:3]))


def rerun_gate_jobs(api: GitHub, source_revision: str) -> list[int]:
    """Retry only the receipt-consuming CPU gate for this exact candidate commit."""
    query = urllib.parse.urlencode({"head_sha": source_revision, "per_page": 100})
    runs = api.request("/actions/workflows/ci.yml/runs?" + query)["workflow_runs"]
    restarted = []
    for run in runs:
        if run.get("head_sha") != source_revision:
            continue
        jobs = api.request("/actions/runs/" + str(run["id"]) + "/jobs?filter=latest&per_page=100")["jobs"]
        for job in jobs:
            if job.get("name") == "inference-replay" and job.get("conclusion") in {"failure", "cancelled"}:
                api.request("/actions/jobs/" + str(job["id"]) + "/rerun", method="POST")
                restarted.append(job["id"])
    return restarted


def write_check(api: GitHub, source_revision: str, result: dict, *, success: bool) -> None:
    api.request("/check-runs", method="POST", data={
        "name": "trusted-inference-evidence", "head_sha": source_revision, "status": "completed",
        "conclusion": "success" if success else "failure",
        "output": {"title": "Exact trusted GPU replay" if success else "Trusted GPU replay rejected",
                   "summary": json.dumps(result, indent=2)},
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compact = sub.add_parser("compact")
    compact.add_argument("--conditioned-report", type=Path, required=True)
    compact.add_argument("--junit", type=Path, required=True)
    compact.add_argument("--generic-replay", type=Path, required=True)
    compact.add_argument("--base", required=True)
    compact.add_argument("--head", required=True)
    attest = sub.add_parser("attest")
    attest.add_argument("--base", required=True)
    attest.add_argument("--head", required=True)
    attest.add_argument("--control-revision", required=True)
    attest.add_argument("--actual-control-ref", required=True)
    attest.add_argument("--run-id", type=int, required=True)
    verify = sub.add_parser("verify-actions")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--run-id", type=int)
    verify.add_argument("--publish-check", action="store_true")
    inspect = sub.add_parser("inspect-actions")
    inspect.add_argument("--run-id", type=int, required=True)
    inspect.add_argument("--github-output", type=Path)
    for command in [attest, verify, compact, inspect]:
        command.add_argument("--source-root", type=Path, default=Path.cwd())
        command.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
        command.add_argument("--control-ref", default=os.environ.get("INFERENCE_EVIDENCE_CONTROL_REF", ""))
        command.add_argument("--reference-index", default=os.environ.get("INFERENCE_REFERENCE_INDEX_SHA256", ""))
        command.add_argument("--workflow-path", default=os.environ.get("INFERENCE_EVIDENCE_WORKFLOW_PATH", WORKFLOW))
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = None
    plan = None
    try:
        root = args.source_root.resolve()
        if args.command == "inspect-actions":
            api = GitHub(args.repository, os.environ.get("GITHUB_TOKEN", ""))
            result = inspect_receipt(api, run_id=args.run_id, control_ref=args.control_ref,
                                     workflow_path=args.workflow_path)
            if args.github_output:
                with args.github_output.open("a") as output:
                    for key in ("source_revision", "source_tree", "base_revision"):
                        output.write(key + "=" + result[key] + "\n")
        elif args.command == "compact":
            plan = plan_for_checkout(root, args.base, args.head)
            raw = json.loads(args.conditioned_report.read_text())
            contract = compact_conditioned_report(raw, root=root, junit=args.junit)
            result = {"schema_version": 2, "mode": "trusted_contract_bundle", "status": "passed",
                      "source_revision": plan["source_revision"], "source_tree": plan["source_tree"],
                      "generic_replay": compact_generic_replay(json.loads(args.generic_replay.read_text())),
                      "contracts": {MG2_CONTRACT: contract}}
            verify_bundle(plan, result, args.reference_index, root)
            if len(json.dumps(result, separators=(",", ":")).encode()) > 60000:
                raise ValueError("Compact report exceeds the workflow dispatch input limit")
        elif args.command == "attest":
            plan = plan_for_checkout(root, args.base, args.head)
            report = json.loads(os.environ["INFERENCE_REPORT_JSON"])
            result = attest_report(plan, report, reference=args.reference_index, repository=args.repository,
                                   run_id=args.run_id, control_revision=args.control_revision,
                                   control_ref=args.actual_control_ref, expected_control_ref=args.control_ref,
                                   workflow_path=args.workflow_path, root=root)
        else:
            plan = json.loads(args.plan.read_text())
            gate.verify_checkout_plan(plan, root, *definitions(root))
            if plan.get("status") == "no_inference_changes" and args.run_id is None:
                result = gate.verify_replay(plan, {}, args.reference_index)
            else:
                api = GitHub(args.repository, os.environ.get("GITHUB_TOKEN", ""))
                result = find_evidence(api, plan, control_ref=args.control_ref, reference=args.reference_index,
                                       run_id=args.run_id, workflow_path=args.workflow_path, root=root)
            if args.publish_check:
                api = api or GitHub(args.repository, os.environ.get("GITHUB_TOKEN", ""))
                write_check(api, plan["source_revision"], result, success=True)
                result["rerun_gate_jobs"] = rerun_gate_jobs(api, plan["source_revision"])
        code = 0
    except (ValueError, OSError, KeyError, TypeError, AssertionError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        result, code = {"status": "failed", "error": str(error)}, 1
        if args.command == "verify-actions" and args.publish_check and api is not None and plan is not None:
            write_check(api, plan["source_revision"], result, success=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, sort_keys=True, separators=(",", ":")) if args.command == "compact" else json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    # Receipts may contain private model-host paths; log only their verification.
    logged = {"status": "passed", "mode": result["mode"], "source_revision": result["source_revision"]} if args.command == "compact" and code == 0 else result.get("verification", result)
    print(json.dumps(logged, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
