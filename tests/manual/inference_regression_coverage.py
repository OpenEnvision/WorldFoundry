"""Report short-case coverage of catalogued native video/world models and variants.

Reads declarations only: a defined case is not an accepted reference, successful
GPU replay or quality benchmark. Aliases and shared pipeline classes do not
automatically cover different checkpoints, modes or runtime profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import yaml


def _coverage_policy():
    spec = importlib.util.spec_from_file_location("native_coverage_policy", Path(__file__).with_name("native_coverage_policy.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog_entries(path: Path) -> list[dict]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Catalog must contain a mapping: {path}")
    entries = next((payload[key] for key in ("models", "entries", "manifests") if key in payload), [payload])
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"Catalog entries must be mappings: {path}")
    return entries


def native_models(root: Path) -> tuple[list[dict], list[dict]]:
    catalog = root / "worldfoundry/data/models/catalog"
    bindings = root / "worldfoundry/data/models/bindings/pipelines"
    if not catalog.is_dir() or not bindings.is_dir():
        raise ValueError("Missing model catalog or pipeline bindings")
    binding_records, binding_lookup = [], {}
    for binding in sorted(bindings.glob("*.y*ml")):
        config = yaml.safe_load(binding.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Invalid native pipeline binding: {binding}")
        binding_id = config.get("binding_id", binding.stem)
        aliases = config.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if (not isinstance(binding_id, str) or not isinstance(aliases, list)
                or not all(isinstance(alias, str) for alias in aliases)):
            raise ValueError(f"Invalid native binding identities: {binding}")
        record = (binding, config)
        binding_records.append(record)
        for identity in [binding_id, config.get("model_id", binding_id), *aliases]:
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"Invalid native binding identity: {binding}")
            identity = identity.strip().lower()
            if identity in binding_lookup and binding_lookup[identity] != record:
                raise ValueError(f"Ambiguous native binding identity: {identity}")
            binding_lookup[identity] = record
    rows, inactive = {}, []
    for category in ("video", "world_models"):
        directory = catalog / category
        if not directory.is_dir():
            raise ValueError(f"Missing catalog category: {category}")
        for path in sorted(directory.glob("*.y*ml")):
            for entry in catalog_entries(path):
                parent = entry.get("model_id", entry.get("id"))
                if not isinstance(parent, str) or not parent:
                    raise ValueError(f"Catalog entry has no model id: {path}")
                variants = entry.get("variants", [])
                if not isinstance(variants, list) or not all(isinstance(variant, dict) for variant in variants):
                    raise ValueError(f"Invalid catalog variants: {parent}")
                for variant in [entry, *variants]:
                    model_id = variant.get("model_id", variant.get("id", variant.get("pipeline_binding")))
                    if model_id is None and isinstance(variant.get("runtime_profile"), str):
                        model_id = variant["runtime_profile"].removeprefix("runtime-profile:")
                    if model_id is None:
                        # Some catalog variants describe only an upstream
                        # checkpoint, with no separately exposed native route.
                        inactive.append({"catalog_id": parent, "category": category,
                                         "variant_name": variant.get("name"),
                                         "checkpoint": variant.get("checkpoint"),
                                         "status": "variant_without_native_identity"})
                        continue
                    if not isinstance(model_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", model_id):
                        raise ValueError(f"Invalid catalog model/variant id: {parent}")
                    binding_id = variant.get("pipeline_binding", model_id)
                    if not isinstance(binding_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", binding_id):
                        raise ValueError(f"Invalid pipeline binding id: {model_id}")
                    record = binding_lookup.get(binding_id.strip().lower())
                    info = {"model_id": model_id, "catalog_id": parent, "category": category,
                            "catalog_path": path.relative_to(root).as_posix()}
                    if record is None:
                        if "pipeline_binding" in variant:
                            rows[(model_id, binding_id)] = {
                                **info, "binding_id": binding_id, "binding_status": "missing_binding", "target": None
                            }
                            continue
                        inactive.append({**info, "status": "no_native_binding"})
                        continue
                    binding, config = record
                    binding_id = config.get("binding_id", binding.stem)
                    target = config.get("pipeline", {}).get("target", "")
                    if not re.fullmatch(r"worldfoundry(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*", target):
                        raise ValueError(f"Invalid native binding target: {binding_id}")
                    module = root / (target.partition(":")[0].replace(".", "/") + ".py")
                    info.update(binding_id=binding_id, binding_model_id=config.get("model_id"), target=target,
                                binding_path=binding.relative_to(root).as_posix(),
                                binding_status="ready" if module.is_file() else "missing_implementation")
                    key = (model_id, binding_id)
                    if key in rows and rows[key]["target"] != target:
                        raise ValueError(f"Contradictory catalog bindings: {model_id}")
                    rows[key] = info
    # Some runtime variants are declared only in bindings. Associate them with
    # a catalog family by the exact class for inventory purposes; coverage
    # below still requires each variant's own model id and recipe.
    families = {}
    for row in rows.values():
        if row.get("target"):
            families.setdefault(row["target"], row)
    for binding, config in binding_records:
        target = config.get("pipeline", {}).get("target")
        if target not in families:
            continue
        model_id, binding_id = config.get("model_id"), config.get("binding_id", binding.stem)
        if not isinstance(model_id, str) or not isinstance(binding_id, str):
            raise ValueError(f"Native pipeline binding has no identity: {binding}")
        key = (model_id, binding_id)
        if key in rows:
            continue
        parent = families[target]
        rows[key] = {"model_id": model_id, "catalog_id": parent["catalog_id"], "category": parent["category"],
                     "catalog_path": parent["catalog_path"], "binding_id": binding_id,
                     "binding_model_id": model_id, "binding_path": binding.relative_to(root).as_posix(),
                     "target": target, "binding_status": parent["binding_status"],
                     "catalog_match": "registered_variant_with_same_pipeline_class"}
    return list(rows.values()), inactive


def coverage(root: Path, matrix: Path) -> dict:
    cases = json.loads(matrix.read_text())
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Inference case matrix is empty")
    rows, inactive = native_models(root)
    covered = set()
    for row in rows:
        matching = []
        for name, case in cases.items():
            if not isinstance(case, dict) or case.get("id") != name:
                raise ValueError(f"Invalid inference case: {name}")
            model_id = case.get("model_id", name)
            if model_id != row["model_id"]:
                continue
            if case.get("target") != row["target"]:
                raise ValueError(f"Case and catalog pipeline targets differ: {name}")
            matching.append(name)
        row.update(case_ids=matching if row["binding_status"] == "ready" else [], status=(
            row["binding_status"] if row["binding_status"] != "ready" else
            ("case_defined" if matching else "missing_short_case")
        ))
        covered.update(row["case_ids"])
    gaps = [row["model_id"] for row in rows if not row["case_ids"]]
    return {
        "schema_version": 1,
        "status": "partial" if gaps else "definitions_complete",
        "scope": "Catalogued video/world native bindings and registered variants of their pipeline classes; definitions only.",
        "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "native_models_and_variants": len(rows),
        "models_with_cases": sum(bool(row["case_ids"]) for row in rows),
        "missing_short_cases": sorted(set(gaps)),
        "case_ids": sorted(covered),
        "models": sorted(rows, key=lambda row: (row["category"], row["model_id"])),
        "catalog_without_native_binding": inactive,
        "accepted_reference_verified": False,
        "gpu_inference_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, default=Path("tests/manual/geometry_regression_cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--baseline", type=Path, default=Path("tests/manual/native_coverage_baseline.json"))
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--affected-model", action="append", default=[])
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        root = args.source_root.resolve()
        matrix = args.matrix if args.matrix.is_absolute() else root / args.matrix
        baseline = args.baseline if args.baseline.is_absolute() else root / args.baseline
        policy = _coverage_policy()
        if args.base and args.changed_file:
            raise ValueError("Use a pinned commit diff or explicit changed paths, not both")
        changed = policy._load_impact_module().changed_paths(root, args.base, args.head) if args.base else args.changed_file
        report = coverage(root, matrix.resolve())
        report["policy"] = policy.evaluate_policy(report, json.loads(matrix.read_text()),
                                                 policy.load_frozen_baseline(baseline), root, changed, args.affected_model)
        report["policy"]["baseline_sha256"] = policy._BASELINE_SHA256
        report["policy"]["change_scope"] = "commit_diff" if args.base else ("explicit_paths" if changed else "inventory_only")
        code = 2 if report["policy"]["status"] != "passed" or (args.require_complete and report["missing_short_cases"]) else 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        report, code = {"status": "failed", "error": str(error)}, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"models", "catalog_without_native_binding"}},
                     indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
