"""Require genuine replay recipes without concealing historical coverage gaps."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import yaml

_IDENTITY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
_BASELINE_SHA256 = "b4aaf03f6698aa451c67d608b5d100b4373462f8e4259cb6f4de4c8449d5e146"


def load_frozen_baseline(path: Path) -> dict:
    """Reject a JSON-only refresh that would grandfather new coverage gaps."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != _BASELINE_SHA256:
        raise ValueError("Frozen coverage baseline changed; an explicit policy revision is required")
    return json.loads(data)


def variant_key(row: dict) -> str:
    return row["model_id"] + "@" + row["binding_id"]


def has_replay_recipe(case: dict) -> bool:
    """A recipe loads a checkpoint, calls inference and requires real outputs.

    This is declaration validation. It never certifies accepted references,
    successful GPU execution, output quality or coverage of another variant.
    """
    if not isinstance(case, dict) or case.get("deterministic") is not True:
        return False
    if isinstance(case.get("seed"), bool) or not isinstance(case.get("seed"), int):
        return False
    load, assets, outputs = case.get("load"), case.get("assets"), case.get("required_outputs")
    if (not isinstance(load, dict) or not isinstance(assets, dict) or not assets
            or not isinstance(outputs, list) or not outputs
            or not all(isinstance(value, str) and value.strip() for value in outputs)
            or len(outputs) != len(set(outputs))):
        return False
    if not any(isinstance(load.get(name), str) and load[name].strip()
               for name in ("model_path", "pretrained_model_path", "checkpoint_path")):
        return False
    runtime_manifest = str(case.get("target", "")).startswith(
        "worldfoundry.pipelines.world_model.pipeline_runtime_manifest:"
    )
    if runtime_manifest and load.get("plan_only"):
        return False
    if isinstance(case.get("call"), dict) and case["call"]:
        return not (runtime_manifest and case["call"].get("plan_only"))
    sequence = case.get("sequence")
    if not isinstance(sequence, list) or not sequence:
        return False
    emitted, names = set(), set()
    for step in sequence:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str) or step["name"] in names:
            return False
        names.add(step["name"])
        fields = step.get("outputs", [])
        if not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields):
            return False
        if fields and not isinstance(step.get("call"), dict):
            return False
        if fields and runtime_manifest and step["call"].get("plan_only"):
            return False
        if step.get("expect_error") and fields:
            return False
        emitted.update(step["name"] + "." + field for field in fields)
    return set(outputs).issubset(emitted)


def recipe_cases(row: dict, cases: dict) -> list[str]:
    if row.get("binding_status", "ready") != "ready":
        return []
    return sorted(name for name, case in cases.items()
                  if case.get("model_id", name) == row["model_id"] and case.get("target") == row["target"]
                  and case.get("id") == name and has_replay_recipe(case))


def outside_bindings(root: Path, rows: list[dict]) -> list[dict]:
    native = {row["binding_id"] for row in rows}
    result = []
    for path in sorted((root / "worldfoundry/data/models/bindings/pipelines").glob("*.y*ml")):
        config = yaml.safe_load(path.read_text())
        binding_id = config.get("binding_id", path.stem)
        if binding_id not in native:
            target = config.get("pipeline", {}).get("target")
            valid_target = isinstance(target, str) and re.fullmatch(r"worldfoundry(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*", target)
            module = root / (target.partition(":")[0].replace(".", "/") + ".py") if valid_target else None
            result.append({"binding_id": binding_id, "model_id": config.get("model_id", binding_id),
                           "target": target, "binding_status": "ready" if module and module.is_file() else "missing_implementation",
                           "binding_path": path.relative_to(root).as_posix()})
    return result


def make_baseline(report: dict, cases: dict, root: Path, source_revision: str) -> dict:
    return {"schema_version": 1, "source_revision": source_revision,
            "scope": "Frozen native identities and replay definitions; existing uncovered variants remain uncovered.",
            "models": [{**{name: row.get(name) for name in
                            ("model_id", "binding_id", "target", "binding_status", "catalog_path", "binding_path")},
                        "required_case_ids": recipe_cases(row, cases)} for row in report["models"]],
            "outside_scope_bindings": outside_bindings(root, report["models"])}


def _load_impact_module():
    spec = importlib.util.spec_from_file_location("coverage_policy_import_graph",
                                                Path(__file__).with_name("geometry_regression_impact.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_dependencies(rows: list[dict], root: Path) -> dict[str, set[str]]:
    """Include runtime declarations and subprocess source roots in ownership.

    A subprocess does not appear in Python's import graph. Its checked-in
    runtime profile explicitly identifies both its binding and source roots.
    This ownership supplements static imports; it never certifies a replay.
    """
    dependencies = {variant_key(row): set() for row in rows}
    profiles = root / "worldfoundry/data/models/runtime/profiles"
    for path in sorted(profiles.glob("*.y*ml")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Runtime profile escapes the inspected source tree")
        profile = yaml.safe_load(path.read_text())
        if not isinstance(profile, dict):
            raise ValueError("Runtime profile must contain a mapping")
        binding = profile.get("execution", {}).get("pipeline_binding")
        owners = [row for row in rows if row["model_id"] == profile.get("model_id")
                  or row["binding_id"] == binding]
        if not owners:
            continue
        paths = {path.relative_to(root).as_posix()}
        for source in profile.get("source_repos", []):
            declared = source.get("in_tree_path")
            if declared is None:
                continue
            if not isinstance(declared, str):
                raise ValueError("Runtime source root must be a canonical in-tree path")
            pure = Path(declared)
            if (pure.is_absolute() or ".." in pure.parts
                    or pure.as_posix() != declared or not declared.startswith("worldfoundry/")):
                raise ValueError("Runtime source root must be a canonical in-tree path")
            target = root / declared
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("Runtime source root escapes the inspected source tree")
            paths.add(declared.rstrip("/") + "/" if target.is_dir() else declared)
        for row in owners:
            dependencies[variant_key(row)].update(paths)
    return dependencies


def affected_variants(rows: list[dict], root: Path, changed_paths: list[str], requested: list[str]) -> tuple[set[str], list[str]]:
    affected, unknown = set(), []
    for identity in requested:
        matched = [row for row in rows if identity in {row["model_id"], variant_key(row)}]
        if not matched:
            raise ValueError(f"Affected native model is absent from the inventory: {identity}")
        affected.update(variant_key(row) for row in matched)
    declared_dependencies = runtime_dependencies(rows, root) if changed_paths else {}
    source_changes = []
    for path in changed_paths:
        pure = Path(path)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
            raise ValueError(f"Invalid changed repository path: {path}")
        direct = [row for row in rows if path in {row.get("catalog_path"), row.get("binding_path")}]
        if direct:
            affected.update(variant_key(row) for row in direct)
        elif path.startswith(("worldfoundry/core/", "worldfoundry/runtime/")) or path in {
            "worldfoundry/pipelines/pipeline_utils.py", "worldfoundry/operators/base_operator.py",
            "worldfoundry/synthesis/base_synthesis.py",
        }:
            affected.update(variant_key(row) for row in rows)
        elif path.startswith(("worldfoundry/pipelines/", "worldfoundry/operators/", "worldfoundry/base_models/",
                              "worldfoundry/synthesis/", "worldfoundry/data/models/runtime/")):
            source_changes.append(path)
    if source_changes:
        graph = _load_impact_module().ImportGraph(root)
        closures = {}
        for row in rows:
            target = row.get("target")
            if target not in closures:
                try:
                    closures[target] = graph.closure(target) if isinstance(target, str) else set()
                except (ValueError, OSError, SyntaxError, UnicodeError):
                    closures[target] = None
        for path in source_changes:
            matched = [row for row in rows if closures[row.get("target")] is not None
                       and path in closures[row.get("target")]]
            matched.extend(row for row in rows if any(
                path.startswith(dependency) if dependency.endswith("/") else path == dependency
                for dependency in declared_dependencies.get(variant_key(row), ())
            ))
            if matched:
                affected.update(variant_key(row) for row in matched)
            else:
                unknown.append(path)
            # Incomplete static analysis cannot rule out a changed dependency.
            affected.update(variant_key(row) for row in rows if closures[row.get("target")] is None)
        if unknown:
            affected.update(variant_key(row) for row in rows)
    return affected, unknown


def evaluate_policy(report: dict, cases: dict, baseline: dict, root: Path,
                    changed_paths: list[str] = (), affected_models: list[str] = ()) -> dict:
    if baseline.get("schema_version") != 1 or not isinstance(baseline.get("models"), list) or not baseline["models"]:
        raise ValueError("A nonempty frozen native coverage baseline is required")
    if not isinstance(baseline.get("source_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", baseline["source_revision"]):
        raise ValueError("Frozen native coverage baseline must name its source revision")
    before = {}
    for row in baseline["models"]:
        if (not isinstance(row, dict) or any(not isinstance(row.get(name), str) or not _IDENTITY.fullmatch(row[name])
                                            for name in ("model_id", "binding_id"))
                or not isinstance(row.get("required_case_ids"), list)
                or not all(isinstance(case, str) for case in row["required_case_ids"])):
            raise ValueError("Frozen native coverage baseline contains an invalid variant")
        key = variant_key(row)
        if key in before:
            raise ValueError("Frozen native coverage baseline repeats a variant")
        before[key] = row
    current = {variant_key(row): row for row in report["models"]}
    if len(current) != len(report["models"]):
        raise ValueError("Native coverage inventory repeats a variant")
    affected, unknown = affected_variants(report["models"], root, list(changed_paths), list(affected_models))
    violations, recipes = [], {key: recipe_cases(row, cases) for key, row in current.items()}
    for key, row in current.items():
        previous = before.get(key)
        if previous is None and not recipes[key]:
            violations.append({"variant": key, "reason": "new_native_variant_without_replay_recipe"})
        if previous is not None:
            missing = set(previous["required_case_ids"]) - set(recipes[key])
            if missing:
                violations.append({"variant": key, "reason": "required_case_coverage_regressed", "case_ids": sorted(missing)})
            if previous.get("target") != row.get("target") or previous.get("binding_status") != row.get("binding_status"):
                affected.add(key)
        if key in affected and not recipes[key]:
            violations.append({"variant": key, "reason": "affected_native_variant_without_replay_recipe"})
    for key in sorted(set(before) - set(current)):
        violations.append({"variant": key, "reason": "frozen_native_variant_disappeared"})
    outside_before = baseline.get("outside_scope_bindings", [])
    if not isinstance(outside_before, list):
        raise ValueError("Frozen coverage baseline has invalid outside-scope bindings")
    outside_index = {row["binding_id"]: row for row in outside_before}
    for row in outside_bindings(root, report["models"]):
        previous = outside_index.get(row["binding_id"])
        if previous is None or any(previous.get(field) != row.get(field) for field in ("model_id", "target")):
            if not recipe_cases(row, cases):
                violations.append({"variant": variant_key(row), "reason": "new_or_changed_unclassified_binding_without_recipe"})
    return {"status": "failed" if violations else "passed", "violations": violations,
            "affected_variants": sorted(affected), "unknown_inference_paths": sorted(unknown),
            "frozen_native_variants": len(before), "native_variants_with_replay_recipes": sum(bool(value) for value in recipes.values()),
            "existing_uncovered_variants": sorted(key for key in set(before) & set(current) if not recipes[key]),
            "accepted_reference_verified": False, "gpu_inference_verified": False}
