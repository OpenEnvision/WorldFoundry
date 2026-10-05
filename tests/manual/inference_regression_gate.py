"""Verify that a trusted GPU replay report satisfies this commit's affected plan.

CPU CI produces a plan, and a prepared trusted model host produces the replay.
This command checks their immutable identities and exact numerical results;
planning, preflight, a different commit or incomplete evidence cannot pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location("inference_gate_impact", Path(__file__).with_name("geometry_regression_impact.py"))
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)


def verify_replay(plan: dict, report: dict, reference_index_sha256: str) -> dict:
    if plan.get("status") not in {"planned", "no_inference_changes"}:
        raise ValueError("Uncovered or failed plans cannot authorize a merge")
    selected = plan.get("selected_cases")
    if (not isinstance(selected, list) or not all(isinstance(name, str) for name in selected)
            or len(selected) != len(set(selected))
            or bool(selected) != (plan["status"] == "planned")):
        raise ValueError("Invalid or contradictory affected-case selection")
    if not selected:
        return {"status": "no_inference_changes", "required_checks": ["public-cpu", "inference-tensors"]}
    if not re.fullmatch(r"[0-9a-f]{64}", reference_index_sha256):
        raise ValueError("A pinned accepted-reference index is required")
    if report.get("status") != "passed" or report.get("mode") != "replay" or report.get("schema_version") != 1:
        raise ValueError("A completed real inference replay is required; preflight is insufficient")
    for key, size in (("source_revision", 40), ("source_tree", 40), ("matrix_sha256", 64), ("dependencies_sha256", 64)):
        expected = plan.get(key)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + "}", expected):
            raise ValueError(f"Affected plan has no pinned {key}")
        if report.get(key) != expected:
            raise ValueError(f"GPU evidence is stale or incompatible: {key}")
    if report.get("reference_index_sha256") != reference_index_sha256:
        raise ValueError("GPU report used a different accepted reference")
    executed = report.get("selected_cases")
    if (not isinstance(executed, list) or not all(isinstance(name, str) for name in executed)
            or len(executed) != len(set(executed)) or not set(selected).issubset(executed)
            or not isinstance(report.get("cases"), dict) or set(report["cases"]) != set(executed)):
        raise ValueError("GPU evidence omits required cases or contains inconsistent execution records")
    fields = 0
    for name in executed:
        case = report["cases"][name]
        if case.get("status") != "passed" or case.get("atol") != 0 or case.get("rtol") != 0:
            raise ValueError(f"Case failed or numerical tolerance was relaxed: {name}")
        outputs = case.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"Case has no numerical output evidence: {name}")
        for key, output in outputs.items():
            error = output.get("max_abs_error")
            shape = output.get("shape")
            if (output.get("passed") is not True or output.get("exact") is not True
                    or isinstance(error, bool) or not isinstance(error, (int, float))
                    or not math.isfinite(error) or error != 0
                    or not isinstance(shape, list)
                    or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)):
                raise ValueError(f"Case numerical output is incomplete or changed: {name}: {key}")
            fields += 1
    return {"status": "gpu_replay_verified", "source_revision": report["source_revision"],
            "required_cases": selected, "executed_cases": executed, "exact_numeric_fields": fields,
            "required_checks": ["public-cpu", "inference-tensors"]}


def verify_checkout_plan(plan: dict, root: Path, matrix: Path, dependencies: Path) -> None:
    """Recompute the real commit diff so omitted paths cannot bypass replay."""
    if (plan.get("source_revision") != impact.revision(root)
            or plan.get("source_tree") != impact.revision(root, "HEAD^{tree}")):
        raise ValueError("Affected plan is stale for the current commit")
    base, head = plan.get("base_revision"), plan.get("head_revision")
    if (not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base)
            or head != plan["source_revision"] or impact.revision(root, base + "^{commit}") != base):
        raise ValueError("Verification requires a commit-diff plan with pinned base and head revisions")
    if plan.get("changed_paths") != impact.changed_paths(root, base, head):
        raise ValueError("Affected plan omits or changes paths from the actual commit diff")
    for key, path in (("matrix_sha256", matrix), ("dependencies_sha256", dependencies)):
        if plan.get(key) != impact.digest(path):
            raise ValueError(f"Affected plan definitions have changed: {key}")
    minimum = impact.select_cases(matrix, dependencies, root, plan["changed_paths"], base=base, head=head)
    for key in ("proven_relocations", "cpu_only_changes", "required_cpu_contracts"):
        if plan.get(key, []) != minimum[key]:
            raise ValueError(f"Affected plan changes the required CPU proof: {key}")
    selected = plan.get("selected_cases")
    if (not isinstance(selected, list) or minimum["status"] == "uncovered"
            or not set(minimum["selected_cases"]).issubset(selected)):
        raise ValueError("Affected plan omits required cases or contains uncovered changes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        profile = json.loads(args.profile.read_text())
        root = args.source_root.resolve()
        matrix, dependencies = root / profile["matrix"], root / profile["dependencies"]
        verify_checkout_plan(plan, root, matrix, dependencies)
        index = Path(profile["reference"]) / "accepted-index.json"
        digest = hashlib.sha256(index.read_bytes()).hexdigest()
        if digest != profile["reference_index_sha256"]:
            raise ValueError("Pinned accepted-reference index was modified")
        result = verify_replay(plan, json.loads(args.report.read_text()), digest)
        code = 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        result, code = {"status": "failed", "error": str(error)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
