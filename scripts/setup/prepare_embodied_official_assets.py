#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldfoundry.evaluation.tasks.execution.framework.io import utc_now_iso, write_json  # noqa: E402
from worldfoundry.runtime.assets import expand_worldfoundry_path  # noqa: E402


DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "local_assets.example.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "benchmark_zoo" / "embodied_official_assets"
EMBODIED_BENCHMARK_IDS: tuple[str, ...] = (
    "behavior1k",
    "libero",
    "libero-mem",
    "libero-plus",
    "libero-pro",
    "simpler-env",
    "robocasa",
    "calvin",
    "maniskill",
    "maniskill2",
    "kinetix",
    "mikasa",
    "molmospaces",
    "rlbench",
    "metaworld",
    "bridgedata-v2",
    "robocerebra",
    "robomme",
    "robotwin",
    "vlabench",
)
BASE_ENV_DEFAULTS = {
    "WORLDFOUNDRY_CACHE_DIR": "cache/worldfoundry",
    "WORLDFOUNDRY_DATA_DIR": "cache/worldfoundry/data",
    "WORLDFOUNDRY_MODEL_DIR": "cache/worldfoundry/models",
    "WORLDFOUNDRY_CKPT_DIR": "${WORLDFOUNDRY_MODEL_DIR}/checkpoints",
    "WORLDFOUNDRY_HFD_DATASET_ROOT": "${WORLDFOUNDRY_DATA_DIR}/datasets",
}

ADDITIONAL_EMBODIED_ASSET_TEMPLATES: dict[str, dict[str, Any]] = {
    "behavior1k": {
        "repo_url": "https://github.com/StanfordVL/OmniGibson",
        "root_env": "WORLDFOUNDRY_BEHAVIOR1K_ROOT",
        "data_env": "WORLDFOUNDRY_BEHAVIOR1K_DATASET_ROOT",
        "split_env": "WORLDFOUNDRY_BEHAVIOR1K_TASK_SPLIT",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_BEHAVIOR1K_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_BEHAVIOR1K_RESULTS_PATH",
        "note": "Stage OmniGibson/BEHAVIOR-1K assets after accepting upstream simulator and dataset terms.",
    },
    "libero-mem": {
        "repo_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "root_env": "WORLDFOUNDRY_LIBERO_MEM_ROOT",
        "data_env": "WORLDFOUNDRY_LIBERO_MEM_DATASET_ROOT",
        "split_env": "WORLDFOUNDRY_LIBERO_MEM_SUITE",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_LIBERO_MEM_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_LIBERO_MEM_RESULTS_PATH",
        "note": "Use the LIBERO-Mem task assets and memory task split that matches the selected policy.",
    },
    "libero-plus": {
        "repo_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "root_env": "WORLDFOUNDRY_LIBERO_PLUS_ROOT",
        "data_env": "WORLDFOUNDRY_LIBERO_PLUS_DATASET_ROOT",
        "split_env": "WORLDFOUNDRY_LIBERO_PLUS_SUITE",
        "default_split": "all",
        "allowed_splits": ("spatial", "all"),
        "policy_env": "WORLDFOUNDRY_LIBERO_PLUS_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_LIBERO_PLUS_RESULTS_PATH",
        "note": "LIBERO-Plus has separate package constraints from base LIBERO; keep the exact upstream package set recorded.",
    },
    "libero-pro": {
        "repo_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "root_env": "WORLDFOUNDRY_LIBERO_PRO_ROOT",
        "data_env": "WORLDFOUNDRY_LIBERO_PRO_DATASET_ROOT",
        "split_env": "WORLDFOUNDRY_LIBERO_PRO_SUITE",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_LIBERO_PRO_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_LIBERO_PRO_RESULTS_PATH",
        "note": "Record the LIBERO-Pro task asset revision and selected policy checkpoint with each imported rollout.",
    },
    "maniskill2": {
        "repo_url": "https://github.com/haosulab/ManiSkill",
        "root_env": "WORLDFOUNDRY_MANISKILL2_ROOT",
        "data_env": "WORLDFOUNDRY_MANISKILL2_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_MANISKILL2_SUITE",
        "default_split": "eval",
        "allowed_splits": ("eval", "custom_env_ids"),
        "policy_env": "WORLDFOUNDRY_MANISKILL2_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_MANISKILL2_RESULTS_PATH",
        "note": "Use this id for the harness-compatible ManiSkill2 runtime; the older maniskill id remains supported.",
    },
    "kinetix": {
        "repo_url": "https://github.com/FLAIROx/Kinetix",
        "root_env": "WORLDFOUNDRY_KINETIX_ROOT",
        "data_env": "WORLDFOUNDRY_KINETIX_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_KINETIX_TASK_SPLIT",
        "default_split": "eval",
        "allowed_splits": ("eval", "realtime"),
        "policy_env": "WORLDFOUNDRY_KINETIX_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_KINETIX_RESULTS_PATH",
        "note": "Kinetix requires a JAX-compatible simulator environment for full rollout execution.",
    },
    "mikasa": {
        "repo_url": "https://github.com/hesic73/MiKASA-Robo",
        "root_env": "WORLDFOUNDRY_MIKASA_ROOT",
        "data_env": "WORLDFOUNDRY_MIKASA_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_MIKASA_TASK_SPLIT",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_MIKASA_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_MIKASA_RESULTS_PATH",
        "note": "Stage MiKASA-Robo task assets and record the simulator package revision.",
    },
    "molmospaces": {
        "repo_url": "https://github.com/allenai/MolmoBot",
        "root_env": "WORLDFOUNDRY_MOLMOSPACES_ROOT",
        "data_env": "WORLDFOUNDRY_MOLMOSPACES_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_MOLMOSPACES_SUITE",
        "default_split": "pick_and_place",
        "allowed_splits": ("pick_and_place",),
        "policy_env": "WORLDFOUNDRY_MOLMOSPACES_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_MOLMOSPACES_RESULTS_PATH",
        "note": "Stage the MolmoSpaces-Bench subset required by the selected suite instead of the full asset corpus.",
    },
    "robocerebra": {
        "repo_url": "https://github.com/FranBesq/robocerebra",
        "root_env": "WORLDFOUNDRY_ROBOCEREBRA_ROOT",
        "data_env": "WORLDFOUNDRY_ROBOCEREBRA_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_ROBOCEREBRA_TASK_SPLIT",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_ROBOCEREBRA_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_ROBOCEREBRA_RESULTS_PATH",
        "note": "Stage RoboCerebra benchmark assets and the official result export for import.",
    },
    "robomme": {
        "repo_url": "https://github.com/RoboMME/robomme_policy_learning",
        "root_env": "WORLDFOUNDRY_ROBOMME_ROOT",
        "data_env": "WORLDFOUNDRY_ROBOMME_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_ROBOMME_SUITE",
        "default_split": "eval",
        "allowed_splits": ("eval", "counting", "imitation", "permanence", "reference"),
        "policy_env": "WORLDFOUNDRY_ROBOMME_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_ROBOMME_RESULTS_PATH",
        "note": "Keep the selected RoboMME memory suite and policy variant in the run manifest.",
    },
    "vlabench": {
        "repo_url": "https://github.com/OpenMOSS/VLABench",
        "root_env": "WORLDFOUNDRY_VLABENCH_ROOT",
        "data_env": "WORLDFOUNDRY_VLABENCH_ASSET_ROOT",
        "split_env": "WORLDFOUNDRY_VLABENCH_TASK_SPLIT",
        "default_split": "eval",
        "allowed_splits": ("eval",),
        "policy_env": "WORLDFOUNDRY_VLABENCH_POLICY_CHECKPOINT",
        "results_env": "WORLDFOUNDRY_VLABENCH_RESULTS_PATH",
        "note": "VLABench full execution requires its SAPIEN-compatible simulator runtime and task assets.",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to prepare embodied asset manifests.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a YAML mapping: {path}")
    return payload


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write embodied asset manifests.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _repo_path(repo_url: str) -> str:
    stem = repo_url.rstrip("/").replace("https://github.com/", "").replace("/", "--")
    return f"${{WORLDFOUNDRY_CACHE_DIR}}/repos/{stem}"


def _generated_benchmark_asset_entry(benchmark_id: str) -> dict[str, Any] | None:
    template = ADDITIONAL_EMBODIED_ASSET_TEMPLATES.get(benchmark_id)
    if template is None:
        return None
    dataset_path = f"${{WORLDFOUNDRY_DATA_DIR}}/datasets/{benchmark_id}"
    checkpoint_path = f"${{WORLDFOUNDRY_MODEL_DIR}}/checkpoints/{benchmark_id}/policy"
    results_path = f"${{WORLDFOUNDRY_ARTIFACT_DIR}}/runs/{benchmark_id}/official_results"
    return {
        "id": benchmark_id,
        "assets": [
            {
                "id": "official_repo",
                "kind": "repo",
                "repo_url": template["repo_url"],
                "path": _repo_path(str(template["repo_url"])),
                "env": template["root_env"],
            },
            {
                "id": "simulator_assets",
                "kind": "simulator_asset",
                "path": dataset_path,
                "env": template["data_env"],
                "note": template["note"],
            },
            {
                "id": "task_split",
                "kind": "dataset_split",
                "path": dataset_path,
                "split_env": template["split_env"],
                "default_split": template["default_split"],
                "allowed_splits": list(template["allowed_splits"]),
                "required_for": "official_rollout_execution",
            },
            {
                "id": "policy_checkpoint",
                "kind": "checkpoint",
                "env": template["policy_env"],
                "path": checkpoint_path,
                "required_for": "official_rollout_execution",
            },
            {
                "id": "official_results_dump",
                "kind": "result_dump",
                "path": results_path,
                "env": template["results_env"],
                "prepare": {
                    "command": f"mkdir -p {results_path}",
                    "note": "Place upstream official rollout result files here; do not commit result dumps.",
                },
            },
            {
                "id": "upstream_rollout_execution",
                "kind": "upstream_rollout_execution",
                "depends_on": [
                    "official_repo",
                    "simulator_assets",
                    "task_split",
                    "policy_checkpoint",
                    "official_results_dump",
                ],
                "required_for": "leaderboard_evidence",
                "note": "Run the official benchmark evaluator with the selected split and policy, then import the result dump with benchmark-run --mode official-validation.",
            },
        ],
    }


def _selected_benchmarks(payload: Mapping[str, Any], benchmark_ids: Iterable[str]) -> list[dict[str, Any]]:
    requested = list(dict.fromkeys(benchmark_ids))
    by_id: dict[str, dict[str, Any]] = {}
    for item in payload.get("benchmarks") or []:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            by_id[item["id"]] = item
    for benchmark_id in requested:
        if benchmark_id not in by_id:
            generated = _generated_benchmark_asset_entry(benchmark_id)
            if generated is not None:
                by_id[benchmark_id] = generated
    missing = [benchmark_id for benchmark_id in requested if benchmark_id not in by_id]
    if missing:
        raise ValueError(f"local asset manifest is missing benchmark ids: {', '.join(missing)}")
    return [dict(by_id[benchmark_id]) for benchmark_id in requested]


def _path_env(output_root: Path) -> dict[str, str]:
    cache_dir = os.environ.get("WORLDFOUNDRY_CACHE_DIR") or str(REPO_ROOT / "cache" / "worldfoundry")
    data_dir = os.environ.get("WORLDFOUNDRY_DATA_DIR") or str(REPO_ROOT / "cache" / "worldfoundry" / "data")
    model_dir = os.environ.get("WORLDFOUNDRY_MODEL_DIR") or str(REPO_ROOT / "cache" / "worldfoundry" / "models")
    ckpt_dir = os.environ.get("WORLDFOUNDRY_CKPT_DIR") or str(Path(model_dir) / "checkpoints")
    hfd_dataset_root = os.environ.get("WORLDFOUNDRY_HFD_DATASET_ROOT") or str(
        Path(data_dir) / "datasets"
    )
    return {
        "WORLDFOUNDRY_CACHE_DIR": cache_dir,
        "WORLDFOUNDRY_DATA_DIR": data_dir,
        "WORLDFOUNDRY_MODEL_DIR": model_dir,
        "WORLDFOUNDRY_CKPT_DIR": ckpt_dir,
        "WORLDFOUNDRY_ARTIFACT_DIR": str(output_root / "artifacts"),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": hfd_dataset_root,
    }


def _asset_path(asset: Mapping[str, Any], env: Mapping[str, str]) -> Path | None:
    raw = asset.get("path") or asset.get("local_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return expand_worldfoundry_path(raw, env)


def _is_file_like(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml"}


def _mkdir_target_for_asset(asset: Mapping[str, Any], env: Mapping[str, str]) -> Path | None:
    path = _asset_path(asset, env)
    if path is None:
        return None
    kind = str(asset.get("kind") or "")
    if kind in {"repo", "source_repo", "upstream_rollout_execution"}:
        return None
    if _is_file_like(path):
        return path.parent
    return path


def _env_lines(
    *,
    output_root: Path,
    manifest_path: Path,
    benchmarks: list[Mapping[str, Any]],
) -> list[str]:
    lines = [
        "# Source this file before running embodied official-runtime checks.",
        "# Edit paths after downloading upstream repos, simulator assets, datasets, and checkpoints.",
        "set -a",
        f"export WORLDFOUNDRY_LOCAL_ASSET_MANIFEST={shlex.quote(str(manifest_path))}",
    ]
    base_defaults = dict(BASE_ENV_DEFAULTS)
    base_defaults["WORLDFOUNDRY_ARTIFACT_DIR"] = str(output_root / "artifacts")
    for name, default in base_defaults.items():
        lines.append(f"export {name}=\"${{{name}:-{default}}}\"")

    seen: set[str] = set()
    for benchmark in benchmarks:
        benchmark_id = str(benchmark["id"])
        lines.extend(["", f"# {benchmark_id}"])
        for asset in benchmark.get("assets") or []:
            if not isinstance(asset, Mapping):
                continue
            env_name = asset.get("env")
            if isinstance(env_name, str) and env_name and env_name not in seen:
                seen.add(env_name)
                path = asset.get("path") or ""
                if str(asset.get("kind")) == "result_dump":
                    fallback = path or str(output_root / "artifacts" / "runs" / benchmark_id / "official_results")
                    if benchmark_id == "robotwin" and not path:
                        fallback = str(output_root / "artifacts" / "runs" / "robotwin" / "eval_result")
                    if "prediction" in str(asset.get("id") or ""):
                        fallback = path or str(output_root / "artifacts" / "runs" / benchmark_id / "offline_predictions")
                    lines.append(f"export {env_name}=\"${{{env_name}:-{fallback}}}\"")
                elif path:
                    lines.append(f"export {env_name}=\"${{{env_name}:-{path}}}\"")
                else:
                    lines.append(f"export {env_name}=\"${{{env_name}:-}}\"")
            split_env = asset.get("split_env")
            default_split = asset.get("default_split")
            if isinstance(split_env, str) and split_env and split_env not in seen:
                seen.add(split_env)
                fallback = default_split if default_split is not None else ""
                lines.append(f"export {split_env}=\"${{{split_env}:-{fallback}}}\"")
    lines.append("set +a")
    return lines


def _shell_assign(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'


def _repo_check_commands(benchmarks: list[Mapping[str, Any]], env_path: Path) -> list[str]:
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source \"${{1:-{env_path}}}\"",
        "missing=0",
        "",
    ]
    for benchmark in benchmarks:
        for asset in benchmark.get("assets") or []:
            if not isinstance(asset, Mapping) or str(asset.get("kind")) != "repo":
                continue
            repo_url = asset.get("repo_url")
            path = asset.get("path")
            revision = asset.get("revision")
            if not isinstance(repo_url, str) or not isinstance(path, str):
                continue
            commands.append(f"# {benchmark['id']}: {asset.get('id')}")
            commands.append(_shell_assign("target", path))
            commands.append('if [ ! -d "$target" ]; then')
            commands.append(
                "  echo \"missing repo/source asset: $target "
                f"(source: {repo_url}"
                + (f", revision: {revision}" if isinstance(revision, str) and revision else "")
                + ")\" >&2"
            )
            commands.append("  missing=1")
            commands.append("fi")
            commands.append("")
    commands.append('exit "$missing"')
    return commands


def _download_commands(benchmarks: list[Mapping[str, Any]], env_path: Path) -> list[str]:
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"source \"${{1:-{env_path}}}\"",
        "",
    ]
    for benchmark in benchmarks:
        for asset in benchmark.get("assets") or []:
            if not isinstance(asset, Mapping):
                continue
            dataset_id = asset.get("hf_dataset_id")
            model_id = asset.get("hf_model_id")
            if isinstance(dataset_id, str) and dataset_id:
                path = asset.get("path")
                commands.append(f"# {benchmark['id']}: {asset.get('id')}")
                if isinstance(path, str) and path:
                    commands.append(_shell_assign("target", path))
                    commands.append('mkdir -p "$target"')
                    commands.append(f"hf download {shlex.quote(dataset_id)} --repo-type dataset --local-dir \"$target\"")
                else:
                    commands.append(f"hf download {shlex.quote(dataset_id)} --repo-type dataset")
                commands.append("")
            if isinstance(model_id, str) and model_id:
                path = asset.get("path")
                commands.append(f"# {benchmark['id']}: {asset.get('id')}")
                if isinstance(path, str) and path:
                    commands.append(_shell_assign("target", path))
                    commands.append('mkdir -p "$target"')
                    commands.append(f"hf download {shlex.quote(model_id)} --repo-type model --local-dir \"$target\"")
                else:
                    commands.append(f"hf download {shlex.quote(model_id)} --repo-type model")
                commands.append("")
    return commands


def _validation_commands(benchmarks: list[Mapping[str, Any]], output_root: Path) -> list[str]:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'source "${1:-%s}"' % (output_root / "embodied_official_env.sh"),
        "",
    ]
    for benchmark in benchmarks:
        benchmark_id = str(benchmark["id"])
        result_env = None
        for asset in benchmark.get("assets") or []:
            if isinstance(asset, Mapping) and str(asset.get("kind")) == "result_dump":
                env_name = asset.get("env")
                if isinstance(env_name, str) and env_name and asset.get("id") == "official_results_dump":
                    result_env = env_name
                    break
                if result_env is None and isinstance(env_name, str) and env_name:
                    result_env = env_name
        if result_env is None:
            continue
        commands.extend(
            [
                f"# {benchmark_id}",
                "worldfoundry-eval zoo benchmark-run \\",
                f"  --benchmark-id {shlex.quote(benchmark_id)} \\",
                "  --mode official-validation \\",
                f"  --official-results-path \"${{{result_env}}}\" \\",
                f"  --output-dir {shlex.quote(str(output_root / 'validation' / benchmark_id))} \\",
                "  --json",
                "",
            ]
        )
    return commands


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_yaml(args.source_manifest)
    benchmark_ids = args.benchmark_id or list(EMBODIED_BENCHMARK_IDS)
    benchmarks = _selected_benchmarks(source, benchmark_ids)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "local_assets_manifest.yaml"
    env_path = output_root / "embodied_official_env.sh"
    repo_check_path = output_root / "check_repo_assets.sh"
    download_plan_path = output_root / "download_public_assets.sh"
    validation_plan_path = output_root / "validate_official_results.sh"

    manifest_payload = {
        "schema_version": source.get("schema_version", "worldfoundry-local-assets-v1"),
        "description": "Local embodied official-runtime asset manifest generated from WorldFoundry template.",
        "generated_at": utc_now_iso(),
        "source_manifest": str(args.source_manifest),
        "policy": source.get("policy", {}),
        "layout": source.get("layout", {}),
        "benchmarks": benchmarks,
    }
    _dump_yaml(manifest_path, manifest_payload)
    env_path.write_text("\n".join(_env_lines(output_root=output_root, manifest_path=manifest_path, benchmarks=benchmarks)) + "\n", encoding="utf-8")
    repo_check_path.write_text("\n".join(_repo_check_commands(benchmarks, env_path)) + "\n", encoding="utf-8")
    download_plan_path.write_text("\n".join(_download_commands(benchmarks, env_path)) + "\n", encoding="utf-8")
    validation_plan_path.write_text("\n".join(_validation_commands(benchmarks, output_root)) + "\n", encoding="utf-8")
    for path in (repo_check_path, download_plan_path, validation_plan_path):
        path.chmod(0o755)

    created_dirs: list[str] = []
    planned_dirs: list[str] = []
    path_env = _path_env(output_root)
    for benchmark in benchmarks:
        for asset in benchmark.get("assets") or []:
            if not isinstance(asset, Mapping):
                continue
            target = _mkdir_target_for_asset(asset, path_env)
            if target is None:
                continue
            planned_dirs.append(str(target))
            if args.create_dirs:
                target.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(target))

    report = {
        "schema_version": "worldfoundry-embodied-official-assets-prepare",
        "generated_at": utc_now_iso(),
        "benchmark_ids": benchmark_ids,
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "env_path": str(env_path),
        "repo_check_path": str(repo_check_path),
        "download_plan_path": str(download_plan_path),
        "validation_plan_path": str(validation_plan_path),
        "planned_directories": sorted(set(planned_dirs)),
        "created_directories": sorted(set(created_dirs)),
        "next_commands": [
            f"source {env_path}",
            f"bash {repo_check_path}",
            "worldfoundry-eval zoo benchmarks --json",
            f"bash {validation_plan_path}",
        ],
    }
    report_path = output_root / "prepare_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local config scaffolding for embodied official-runtime benchmark assets.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark-id", action="append", choices=EMBODIED_BENCHMARK_IDS)
    parser.add_argument("--create-dirs", action="store_true", help="Create local dataset/checkpoint/result placeholder directories.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"manifest: {report['manifest_path']}")
        print(f"env: {report['env_path']}")
        print(f"repo_check: {report['repo_check_path']}")
        print(f"download_plan: {report['download_plan_path']}")
        print(f"validation_plan: {report['validation_plan_path']}")
        print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
