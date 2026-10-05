#!/usr/bin/env python3
"""Prepare a local asset plan for WorldFoundry benchmark evaluation.

The script reads the in-tree benchmark catalog, task manifest, runtime profile,
and local asset template, then emits a checklist for data roots, checkpoints,
API credentials, generated outputs, and real evaluation commands.

It does not download assets or write secrets. Use the generated env template as a
local file outside git, fill paths/tokens, then run the real runtime or
official-result import command shown in the plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path, worldfoundry_path_tokens
from worldfoundry.core.io.formats.serialization import load_serialized

ENV_RE = re.compile(r"\b(?:WORLDFOUNDRY_[A-Z0-9_]+|OPENAI_API_KEY|DASHSCOPE_API_KEY|HF_TOKEN|HUGGINGFACE_HUB_TOKEN|GOOGLE_API_KEY|GEMINI_API_KEY)\b")
SECRET_RE = re.compile(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
DATA_KINDS = {"dataset", "dataset_split", "simulator_asset", "simulator_runtime", "result_dump", "upstream_rollout_execution"}
CKPT_KINDS = {"checkpoint", "checkpoint_cache", "model_checkpoint", "weights", "policy_checkpoint"}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = load_serialized(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _catalog_path(benchmark_id: str, repo_root: Path) -> Path | None:
    for family in ("video", "embodied"):
        path = repo_root / "worldfoundry" / "data" / "benchmarks" / "catalog" / family / f"{benchmark_id}.yaml"
        if path.is_file():
            return path
    return None


def _task_path(benchmark_id: str, repo_root: Path) -> Path:
    return repo_root / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external" / f"{benchmark_id}.yaml"


def _runtime_profile_path(benchmark_id: str, repo_root: Path) -> Path:
    return repo_root / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official" / f"{benchmark_id}.yaml"


def _local_assets_path(repo_root: Path) -> Path:
    return repo_root / "worldfoundry" / "data" / "benchmarks" / "local_assets.example.yaml"


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _walk(item)


def _find_env_names(*payloads: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for payload in payloads:
        for item in _walk(payload):
            if isinstance(item, str):
                names.update(ENV_RE.findall(item))
            elif isinstance(item, Mapping):
                for key, value in item.items():
                    key_text = str(key)
                    if key_text == "env" or key_text.endswith("_env") or key_text in {"required_env", "optional_env"}:
                        if isinstance(value, str):
                            names.update(ENV_RE.findall(value))
                        elif isinstance(value, (list, tuple)):
                            for env_name in value:
                                names.update(ENV_RE.findall(str(env_name)))
    return sorted(names)


def _group_local_assets(benchmark_id: str, local_assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    aliases = {benchmark_id}
    if benchmark_id in {"vbench", "vbench-2.0", "vbench-plus-plus"}:
        aliases.add("vchitect")
    rows: list[dict[str, Any]] = []
    for benchmark in local_assets.get("benchmarks") or ():
        if not isinstance(benchmark, Mapping) or str(benchmark.get("id")) not in aliases:
            continue
        for asset in benchmark.get("assets") or ():
            if isinstance(asset, Mapping):
                row = dict(asset)
                row.setdefault("benchmark_group", benchmark.get("id"))
                rows.append(row)
    return rows


def _runtime_assets(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    required_assets = profile.get("required_assets")
    if isinstance(required_assets, Mapping):
        for key, value in required_assets.items():
            if isinstance(value, list):
                for index, item in enumerate(value, start=1):
                    if isinstance(item, Mapping):
                        assets.append({"id": item.get("id") or f"{key}_{index}", "kind": key, **dict(item)})
                    else:
                        row: dict[str, Any] = {"id": f"{key}_{index}", "kind": key, "value": item}
                        if key in {"hf_dataset_ids", "hf_datasets"}:
                            row["hf_dataset_id"] = item
                            row["kind"] = "dataset"
                        elif key in {"hf_model_ids", "hf_models"}:
                            row["hf_model_id"] = item
                            row["kind"] = "checkpoint"
                        elif key in {"local_dataset_paths", "dataset_paths"}:
                            row["path"] = item
                            row["kind"] = "dataset"
                        elif key in {"local_checkpoint_paths", "checkpoint_paths"}:
                            row["path"] = item
                            row["kind"] = "checkpoint"
                        assets.append(row)
            elif isinstance(value, Mapping):
                assets.append({"id": key, "kind": key, **dict(value)})
            else:
                row: dict[str, Any] = {"id": key, "kind": key, "value": value}
                if key in {"hf_dataset_ids", "hf_datasets"}:
                    row["hf_dataset_id"] = value
                    row["kind"] = "dataset"
                elif key in {"hf_model_ids", "hf_models"}:
                    row["hf_model_id"] = value
                    row["kind"] = "checkpoint"
                elif key in {"local_dataset_paths", "dataset_paths"}:
                    row["path"] = value
                    row["kind"] = "dataset"
                elif key in {"local_checkpoint_paths", "checkpoint_paths", "scorecard_paths"}:
                    row["path"] = value
                assets.append(row)
    for item in profile.get("required_paths") or ():
        if isinstance(item, Mapping):
            assets.append({"kind": "required_path", **dict(item)})
    return assets


def _catalog_assets(catalog: Mapping[str, Any], task: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for source_name, payload in (("catalog", catalog), ("task", task)):
        for item in _walk(payload):
            if not isinstance(item, Mapping):
                continue
            if item.get("repo_id") and ("huggingface" in source_name or item.get("repo_type") or item.get("path")):
                assets.append({"kind": "hf_reference", "source": source_name, **dict(item)})
            if item.get("hf_dataset_id") or item.get("hf_model_id"):
                kind = "dataset" if item.get("hf_dataset_id") else "checkpoint"
                assets.append({"kind": kind, "source": source_name, **dict(item)})
    return assets


def _classify_asset(asset: Mapping[str, Any]) -> str:
    kind = str(asset.get("kind") or "").lower()
    text = json.dumps(asset, ensure_ascii=False, sort_keys=True).lower()
    asset_id = str(asset.get("id") or "").lower()
    repo_type = str(asset.get("repo_type") or "").lower()
    if asset.get("hf_dataset_id") or repo_type == "dataset":
        return "data"
    if asset.get("hf_model_id") or repo_type == "model":
        return "checkpoint"
    if kind in DATA_KINDS or "dataset" in kind or "result" in kind or "simulator" in kind:
        return "data"
    if kind in CKPT_KINDS or "checkpoint" in kind or "weight" in kind or "ckpt" in text or "checkpoint" in asset_id:
        return "checkpoint"
    if "api_key" in text or "token" in text:
        return "api"
    return "other"


def _asset_label(asset: Mapping[str, Any]) -> str:
    for key in ("id", "kind", "repo_id", "hf_dataset_id", "hf_model_id", "path", "value"):
        value = asset.get(key)
        if value not in (None, ""):
            return str(value)
    return "asset"


def _asset_identity(asset: Mapping[str, Any]) -> str:
    for key in ("hf_dataset_id", "hf_model_id"):
        value = asset.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    repo_id = asset.get("repo_id")
    if repo_id not in (None, ""):
        repo_type = str(asset.get("repo_type") or "model").lower()
        if repo_type == "dataset":
            return f"hf_dataset_id:{repo_id}"
        if repo_type == "model":
            return f"hf_model_id:{repo_id}"
        return f"repo:{repo_type}:{repo_id}"
    path = asset.get("path")
    if path not in (None, ""):
        return f"path:{_classify_asset(asset)}:{path}"
    value = asset.get("value")
    if value not in (None, ""):
        return f"value:{_classify_asset(asset)}:{value}"
    return json.dumps(dict(asset), ensure_ascii=False, sort_keys=True, default=str)


def _dedupe_assets(assets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset in assets:
        row = {str(key): value for key, value in dict(asset).items()}
        key = _asset_identity(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _env_template_lines(plan: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Source this file locally after editing paths and secrets.",
        "# Do not commit a filled copy.",
        f"export WORLDFOUNDRY_REPO_ROOT={plan['repo_root']!r}",
        "export WORLDFOUNDRY_HOME=${WORLDFOUNDRY_HOME:-${HOME}/.cache/worldfoundry}",
        "export WORLDFOUNDRY_CACHE_DIR=${WORLDFOUNDRY_CACHE_DIR:-${WORLDFOUNDRY_HOME}/cache}",
        "export WORLDFOUNDRY_DATA_DIR=${WORLDFOUNDRY_DATA_DIR:-${WORLDFOUNDRY_CACHE_DIR}/data}",
        "export WORLDFOUNDRY_CKPT_DIR=${WORLDFOUNDRY_CKPT_DIR:-${WORLDFOUNDRY_CACHE_DIR}/checkpoints}",
        "export WORLDFOUNDRY_HFD_ROOT=${WORLDFOUNDRY_HFD_ROOT:-${WORLDFOUNDRY_CKPT_DIR}/hfd}",
        "export WORLDFOUNDRY_HFD_DATASET_ROOT=${WORLDFOUNDRY_HFD_DATASET_ROOT:-${WORLDFOUNDRY_DATA_DIR}/datasets}",
        "export WORLDFOUNDRY_ARTIFACT_DIR=${WORLDFOUNDRY_ARTIFACT_DIR:-${WORLDFOUNDRY_CACHE_DIR}/artifacts}",
        "export HF_HOME=${HF_HOME:-${WORLDFOUNDRY_HOME}/huggingface}",
        "export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}",
        f"export WORLDFOUNDRY_LOCAL_ASSET_MANIFEST=${{WORLDFOUNDRY_LOCAL_ASSET_MANIFEST:-${{WORLDFOUNDRY_CACHE_DIR}}/local_assets.yaml}}",
        "",
    ]
    for env_name in plan["env"]["all"]:
        if env_name in {
            "WORLDFOUNDRY_REPO_ROOT",
            "WORLDFOUNDRY_HOME",
            "WORLDFOUNDRY_CACHE_DIR",
            "WORLDFOUNDRY_DATA_DIR",
            "WORLDFOUNDRY_CKPT_DIR",
            "WORLDFOUNDRY_HFD_ROOT",
            "WORLDFOUNDRY_HFD_DATASET_ROOT",
            "WORLDFOUNDRY_ARTIFACT_DIR",
        }:
            continue
        default = "<set-locally>" if SECRET_RE.search(env_name) else ""
        lines.append(f"export {env_name}=${{{env_name}:-{default}}}")
    return lines


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _hf_commands(assets: Iterable[Mapping[str, Any]]) -> list[str]:
    commands: list[str] = []
    for asset in assets:
        repo_id = asset.get("hf_dataset_id") or asset.get("hf_model_id") or asset.get("repo_id")
        if not repo_id:
            continue
        repo_type = "dataset" if asset.get("hf_dataset_id") else str(asset.get("repo_type") or "model")
        local_dir = asset.get("path")
        if not local_dir:
            local_dir = "${WORLDFOUNDRY_HFD_DATASET_ROOT}/" + str(repo_id).replace("/", "__") if repo_type == "dataset" else "${WORLDFOUNDRY_HFD_ROOT}/" + str(repo_id).replace("/", "--")
        command = ["hf", "download", str(repo_id)]
        if repo_type == "dataset":
            command.extend(["--repo-type", "dataset"])
        if asset.get("revision"):
            command.extend(["--revision", str(asset["revision"])])
        command.extend(["--local-dir", str(local_dir)])
        commands.append(" ".join(_shell_quote(part) for part in command))
    return sorted(set(commands))


def _format_command(command: Any) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        return " ".join(_shell_quote(str(part)) for part in command)
    return ""


def build_plan(benchmark_id: str, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = (repo_root or project_root()).resolve()
    catalog_path = _catalog_path(benchmark_id, repo_root)
    task_path = _task_path(benchmark_id, repo_root)
    profile_path = _runtime_profile_path(benchmark_id, repo_root)
    local_assets_path = _local_assets_path(repo_root)

    catalog = _load(catalog_path) if catalog_path else {}
    task = _load(task_path)
    profile = _load(profile_path)
    local_assets = _load(local_assets_path)

    assets = _dedupe_assets(
        [
            *_group_local_assets(benchmark_id, local_assets),
            *_runtime_assets(profile),
            *_catalog_assets(catalog, task),
        ]
    )
    env_names = set(_find_env_names(catalog, task, profile, {"assets": assets}))
    if any(asset.get("hf_dataset_id") or asset.get("hf_model_id") or asset.get("repo_id") for asset in assets):
        env_names.add("HF_TOKEN")
    env_names.update(str(item) for item in profile.get("required_env") or ())
    env_names.update(str(item) for item in profile.get("optional_env") or ())
    api_env = sorted(name for name in env_names if SECRET_RE.search(name) or name in {"OPENAI_API_KEY", "DASHSCOPE_API_KEY"})

    grouped_assets = {"data": [], "checkpoints": [], "api": [], "other": []}
    for asset in assets:
        group = _classify_asset(asset)
        key = "checkpoints" if group == "checkpoint" else group
        grouped_assets.setdefault(key, []).append(asset)

    normalize = [
        "worldfoundry-eval",
        "zoo",
        "benchmark-run",
        "--benchmark-id",
        benchmark_id,
        "--mode",
        "official-validation",
        "--official-results-path",
        "<official-results-file-or-dir>",
        "--generated-artifact-dir",
        "<generated-artifact-dir>",
        "--output-dir",
        f"tmp/benchmark_zoo/{benchmark_id}_official_validation",
        "--json",
    ]
    commands = {
        "bootstrap": "bash scripts/setup/bootstrap_worldfoundry.sh && source tmp/worldfoundry_unified_env.sh",
        "asset_plan": f"python scripts/setup/prepare_benchmark_assets.py --benchmark-id {benchmark_id} --json",
        "official_result_import": " ".join(_shell_quote(part) for part in normalize),
    }
    official_runtime = _format_command(profile.get("full_runtime_command") or profile.get("validation_command"))
    if official_runtime:
        commands["official_runtime"] = official_runtime

    return {
        "schema_version": "worldfoundry-benchmark-asset-plan-v1",
        "benchmark_id": benchmark_id,
        "repo_root": str(repo_root),
        "sources": {
            "catalog": None if catalog_path is None else str(catalog_path),
            "task": str(task_path) if task_path.is_file() else None,
            "runtime_profile": str(profile_path) if profile_path.is_file() else None,
            "local_assets_template": str(local_assets_path) if local_assets_path.is_file() else None,
        },
        "environment": {
            "id": profile.get("environment_id") or profile.get("python_env") or "worldfoundry-unified-cu128",
            "needs_new_env": bool(profile.get("needs_new_env")),
            "required_imports": list(profile.get("required_imports") or ()),
            "required_packages": list(profile.get("required_packages") or ()),
            "base_model_dependencies": list(profile.get("base_model_dependencies") or ()),
            "optional_base_model_dependencies": list(profile.get("optional_base_model_dependencies") or ()),
        },
        "env": {
            "all": sorted(env_names),
            "api_or_secret": api_env,
            "required": sorted(str(item) for item in profile.get("required_env") or ()),
            "optional": sorted(str(item) for item in profile.get("optional_env") or ()),
        },
        "assets": {
            "data": grouped_assets.get("data", []),
            "checkpoints": grouped_assets.get("checkpoints", []),
            "api": grouped_assets.get("api", []),
            "other": grouped_assets.get("other", []),
        },
        "download_hints": {
            "hf": _hf_commands(assets),
            "manual": [
                _asset_label(asset)
                for asset in assets
                if asset.get("access") or str(asset.get("kind") or "") in {"result_dump", "simulator_asset", "simulator_runtime"}
            ],
        },
        "commands": commands,
        "notes": list(profile.get("notes") or catalog.get("notes") or ()),
    }


def _write_env_template(path: Path, plan: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_env_template_lines(plan)) + "\n", encoding="utf-8")
    return path


def _create_dirs(plan: Mapping[str, Any], env: Mapping[str, str]) -> list[str]:
    created: list[str] = []
    for group in ("data", "checkpoints", "other"):
        for asset in plan["assets"].get(group, []):
            raw_path = asset.get("path")
            if not raw_path or str(raw_path).startswith("<"):
                continue
            try:
                path = resolve_worldfoundry_path(str(raw_path), env)
            except Exception:
                continue
            if path.suffix and group != "checkpoints":
                path.parent.mkdir(parents=True, exist_ok=True)
                created.append(str(path.parent))
            else:
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))
    return sorted(set(created))


def _print_text(plan: Mapping[str, Any]) -> None:
    print(f"Benchmark: {plan['benchmark_id']}")
    print(f"Environment: {plan['environment']['id']} (needs_new_env={plan['environment']['needs_new_env']})")
    print("\nData / result assets:")
    for item in plan["assets"]["data"][:50]:
        print(f"- {_asset_label(item)}")
    print("\nCheckpoint / metric assets:")
    for item in plan["assets"]["checkpoints"][:50]:
        print(f"- {_asset_label(item)}")
    print("\nAPI / secret env vars:")
    for name in plan["env"]["api_or_secret"] or ["<none declared>"]:
        print(f"- {name}")
    print("\nRequired/optional env vars:")
    for name in plan["env"]["all"]:
        print(f"- {name}")
    print("\nReal run / import commands:")
    for name, command in plan["commands"].items():
        print(f"{name}: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True, help="Benchmark id, for example vbench or wbench.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable asset plan.")
    parser.add_argument("--write-env", type=Path, help="Write a local shell env template for this benchmark.")
    parser.add_argument("--create-dirs", action="store_true", help="Create declared local staging directories.")
    args = parser.parse_args(argv)

    plan = build_plan(args.benchmark_id)
    if args.write_env is not None:
        _write_env_template(args.write_env, plan)
        plan["env_template_path"] = str(args.write_env)
    if args.create_dirs:
        env = {**worldfoundry_path_tokens(os.environ), **os.environ}
        plan["created_directories"] = _create_dirs(plan, env)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(plan)
        if args.write_env is not None:
            print(f"\nEnv template: {args.write_env}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
