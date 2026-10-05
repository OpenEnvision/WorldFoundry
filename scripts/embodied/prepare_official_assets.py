#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
MODEL_CATALOG_ROOT = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog" / "vla_va_wam"
MODEL_PROFILE_ROOT = REPO_ROOT / "worldfoundry" / "data" / "models" / "runtime" / "profiles"
BENCHMARK_PROFILE_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official"
BENCHMARK_CATALOG_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog" / "embodied"
LOCAL_ASSETS_TEMPLATE = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "local_assets.example.yaml"

ACTIVE_BENCHMARK_IDS: tuple[str, ...] = (
    "behavior1k",
    "calvin",
    "kinetix",
    "libero",
    "libero-mem",
    "libero-plus",
    "libero-pro",
    "maniskill2",
    "mikasa",
    "molmospaces",
    "rlbench",
    "robocasa",
    "robocerebra",
    "robomme",
    "robotwin",
    "simpler-env",
    "vlabench",
)

BENCHMARK_REPO_URLS: dict[str, str] = {
    "behavior1k": "https://github.com/StanfordVL/OmniGibson",
    "calvin": "https://github.com/mees/calvin",
    "kinetix": "https://github.com/FLAIROx/Kinetix",
    "libero": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "libero-mem": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "libero-plus": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "libero-pro": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "maniskill2": "https://github.com/haosulab/ManiSkill",
    "mikasa": "https://github.com/CognitiveAISystems/MIKASA-Robo",
    "molmospaces": "https://github.com/allenai/MolmoBot",
    "rlbench": "https://github.com/stepjam/RLBench",
    "robocasa": "https://github.com/robocasa/robocasa",
    "robocerebra": "https://github.com/buaa-colalab/RoboCerebra",
    "robomme": "https://github.com/RoboMME/robomme_policy_learning",
    "robotwin": "https://github.com/RoboTwin-Platform/RoboTwin",
    "simpler-env": "https://github.com/simpler-env/SimplerEnv",
    "vlabench": "https://github.com/OpenMOSS/VLABench",
}

TOKEN_RE = re.compile(r"\$(?P<brace>\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\})|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)")
HF_REVISION_MARKER_SCHEMA = "worldfoundry-embodied-hf-revision-v1"


@dataclass(frozen=True)
class PrepareItem:
    category: str
    kind: str
    owner_id: str
    asset_id: str
    local_path: Path
    source: str | None = None
    revision: str | None = None
    role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.category, self.kind, self.owner_id, str(self.local_path))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def repo_slug(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")


def github_slug(repo_url: str) -> str:
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.replace("https://github.com/", "").replace("/", "--")


def normalized_repo_url(repo_url: str) -> str:
    url = repo_url.rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def target_env(data_root: Path, ckpt_root: Path, hfd_root: Path, hfd_dataset_root: Path) -> dict[str, str]:
    cache_dir = data_root
    return {
        **os.environ,
        "WORLDFOUNDRY_REPO_ROOT": str(REPO_ROOT),
        "WORLDFOUNDRY_BENCH_ROOT": str(REPO_ROOT),
        "WORLDFOUNDRY_CACHE_DIR": str(cache_dir),
        "WORLDFOUNDRY_DATA_DIR": str(data_root),
        "WORLDFOUNDRY_MODEL_DIR": str(ckpt_root),
        "WORLDFOUNDRY_CKPT_DIR": str(ckpt_root),
        "WORLDFOUNDRY_HFD_ROOT": str(hfd_root),
        "WORLDFOUNDRY_HFD_DATASET_ROOT": str(hfd_dataset_root),
        "WORLDFOUNDRY_ASSET_ROOT": str(data_root / "assets"),
        "WORLDFOUNDRY_OPENPI_ASSET_ROOT": str(ckpt_root / "openpi"),
        "WORLDFOUNDRY_MODEL_SOURCE_DIR": str(data_root / "repos"),
        "WORLDFOUNDRY_BENCHMARK_REPO_ROOT": str(data_root / "repos"),
        "WORLDFOUNDRY_ARTIFACT_DIR": str(data_root / "artifacts"),
        "HF_HOME": str(ckpt_root / ".cache" / "huggingface"),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
    }


def expand_path(value: str | Path, env: Mapping[str, str]) -> Path:
    raw = str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        return env.get(name, match.group(0))

    expanded = TOKEN_RE.sub(replace, raw)
    expanded = os.path.expanduser(expanded)
    path = Path(expanded)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def asset_ready(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0 and not path.name.endswith((".aria2", ".incomplete"))
    if not path.is_dir():
        return False
    hfd_manifest = path / ".hfd" / "manifest"
    if hfd_manifest.is_file():
        return hfd_manifest_ready(path, hfd_manifest)
    ignored_dirs = {".git", ".cache", ".hfd", "__pycache__"}
    ignored_files = {"README.md", "readme.md", ".gitattributes"}
    incomplete_suffixes = (".aria2", ".incomplete", ".part", ".tmp")
    saw_payload = False
    for child in path.rglob("*"):
        try:
            rel_parts = set(child.relative_to(path).parts)
        except ValueError:
            rel_parts = set()
        if ignored_dirs & rel_parts:
            continue
        if child.is_file() and child.name.endswith(incomplete_suffixes):
            return False
        if child.is_file() and child.name not in ignored_files and child.stat().st_size > 0:
            saw_payload = True
    return saw_payload


def hfd_manifest_ready(root: Path, manifest_path: Path) -> bool:
    incomplete_suffixes = (".aria2", ".incomplete", ".part", ".tmp")
    for child in root.rglob("*"):
        if child.is_file() and child.name.endswith(incomplete_suffixes):
            return False
    saw_manifest_entry = False
    try:
        lines = manifest_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            size_text, rel_path = line.split("\t", 1)
            expected_size = int(float(size_text))
        except ValueError:
            return False
        candidate = root / rel_path
        if not candidate.is_file():
            return False
        try:
            if candidate.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        saw_manifest_entry = True
    return saw_manifest_entry


def ensure_target_dir(path: Path) -> Path:
    if path.suffix and not path.exists():
        return path.parent
    return path


def discover_model_items(env: Mapping[str, str], selected_models: set[str] | None) -> list[PrepareItem]:
    items: list[PrepareItem] = []
    model_ids = sorted(path.stem for path in MODEL_CATALOG_ROOT.glob("*.yaml"))
    if selected_models:
        missing = sorted(selected_models - set(model_ids))
        if missing:
            raise ValueError(f"unknown embodied model ids: {', '.join(missing)}")
        model_ids = [model_id for model_id in model_ids if model_id in selected_models]

    for model_id in model_ids:
        profile_path = MODEL_PROFILE_ROOT / f"{model_id}.yaml"
        if not profile_path.is_file():
            items.append(
                PrepareItem(
                    "model",
                    "manual_checkpoint",
                    model_id,
                    "missing_runtime_profile",
                    expand_path("${WORLDFOUNDRY_CKPT_DIR}", env) / model_id,
                    metadata={"reason": f"missing runtime profile: {profile_path}"},
                )
            )
            continue
        profile = load_yaml(profile_path)
        checkpoints = profile.get("checkpoints") or []
        if not isinstance(checkpoints, list):
            continue
        for index, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, Mapping):
                continue
            role = str(checkpoint.get("role") or f"checkpoint_{index}")
            repo_id = checkpoint.get("repo_id")
            remote_path = checkpoint.get("path")
            local_raw = checkpoint.get("local_dir") or checkpoint.get("local_path")
            revision = checkpoint.get("revision")
            if isinstance(repo_id, str) and repo_id:
                local_path = expand_path(str(local_raw), env) if local_raw else expand_path("${WORLDFOUNDRY_HFD_ROOT}", env) / repo_slug(repo_id)
                items.append(
                    PrepareItem(
                        "model",
                        "hf_model",
                        model_id,
                        repo_id,
                        local_path,
                        source=repo_id,
                        revision=str(revision) if revision else None,
                        role=role,
                        metadata={"profile": str(profile_path.relative_to(REPO_ROOT)), **dict(checkpoint)},
                    )
                )
            elif isinstance(remote_path, str) and remote_path.startswith("gs://"):
                local_path = expand_path(str(local_raw), env) if local_raw else expand_path("${WORLDFOUNDRY_CKPT_DIR}", env) / model_id / role
                items.append(
                    PrepareItem(
                        "model",
                        "gcs_checkpoint",
                        model_id,
                        remote_path,
                        local_path,
                        source=remote_path,
                        role=role,
                        metadata={"profile": str(profile_path.relative_to(REPO_ROOT)), **dict(checkpoint)},
                    )
                )
            elif local_raw:
                items.append(
                    PrepareItem(
                        "model",
                        "manual_checkpoint",
                        model_id,
                        role,
                        expand_path(str(local_raw), env),
                        role=role,
                        metadata={"profile": str(profile_path.relative_to(REPO_ROOT)), **dict(checkpoint)},
                    )
                )
    return dedupe_items(items)


def load_template_benchmark_assets() -> dict[str, list[Mapping[str, Any]]]:
    if not LOCAL_ASSETS_TEMPLATE.is_file():
        return {}
    payload = load_yaml(LOCAL_ASSETS_TEMPLATE)
    result: dict[str, list[Mapping[str, Any]]] = {}
    for benchmark in payload.get("benchmarks") or []:
        if isinstance(benchmark, Mapping) and isinstance(benchmark.get("id"), str):
            assets = [asset for asset in benchmark.get("assets") or [] if isinstance(asset, Mapping)]
            result[str(benchmark["id"])] = assets
    return result


def template_repo_revisions(template_assets: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for assets in template_assets.values():
        for asset in assets:
            repo_url = asset.get("repo_url")
            revision = asset.get("revision")
            if isinstance(repo_url, str) and repo_url and isinstance(revision, str) and revision:
                revisions[normalized_repo_url(repo_url)] = revision
    return revisions


def load_benchmark_catalog_revisions() -> dict[str, dict[tuple[str, str], str]]:
    """Load canonical source and dataset pins from embodied catalog entries."""

    result: dict[str, dict[tuple[str, str], str]] = {}
    for benchmark_id in ACTIVE_BENCHMARK_IDS:
        path = BENCHMARK_CATALOG_ROOT / f"{benchmark_id}.yaml"
        if not path.is_file():
            continue
        payload = load_yaml(path)
        pins: dict[tuple[str, str], str] = {}

        def add_pin(kind: str, asset_id: str, revision: object) -> None:
            if not isinstance(revision, str) or not revision.strip():
                return
            key = (kind, asset_id)
            normalized_revision = revision.strip()
            previous = pins.get(key)
            if previous is not None and previous != normalized_revision:
                raise ValueError(
                    f"conflicting {benchmark_id} catalog revisions for {asset_id}: "
                    f"{previous} != {normalized_revision}"
                )
            pins[key] = normalized_revision

        official_sources = payload.get("official_sources") or {}
        if isinstance(official_sources, Mapping):
            github = official_sources.get("github") or {}
            if isinstance(github, Mapping) and isinstance(github.get("url"), str):
                add_pin("git", normalized_repo_url(str(github["url"])), github.get("head_sha"))
            for dataset in official_sources.get("huggingface_datasets") or []:
                if isinstance(dataset, Mapping) and isinstance(dataset.get("repo_id"), str):
                    add_pin("hf", str(dataset["repo_id"]), dataset.get("revision"))
        for dataset in payload.get("dataset_refs") or []:
            if isinstance(dataset, Mapping) and isinstance(dataset.get("repo_id"), str):
                add_pin("hf", str(dataset["repo_id"]), dataset.get("revision"))
        result[benchmark_id] = pins
    return result


def resolved_catalog_revision(
    *,
    benchmark_id: str,
    kind: str,
    asset_id: str,
    explicit_revision: object,
    catalog_revisions: Mapping[str, Mapping[tuple[str, str], str]],
) -> str | None:
    explicit = str(explicit_revision).strip() if explicit_revision else None
    catalog = catalog_revisions.get(benchmark_id, {}).get((kind, asset_id))
    if explicit is not None and catalog is not None and explicit != catalog:
        raise ValueError(
            f"{benchmark_id} revision for {asset_id} conflicts with catalog: {explicit} != {catalog}"
        )
    return explicit or catalog


def benchmark_environment_defaults(
    env: Mapping[str, str],
    selected_benchmarks: set[str] | None,
) -> dict[str, str]:
    """Resolve runtime environment variables declared by the asset template."""

    selected = set(ACTIVE_BENCHMARK_IDS) if selected_benchmarks is None else selected_benchmarks
    defaults: dict[str, str] = {}
    for benchmark_id, assets in load_template_benchmark_assets().items():
        if benchmark_id not in selected:
            continue
        for asset in assets:
            env_name = asset.get("env")
            path_raw = asset.get("path") or asset.get("local_path")
            if (
                isinstance(env_name, str)
                and re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_name)
                and isinstance(path_raw, str)
                and path_raw
            ):
                defaults.setdefault(env_name, str(env.get(env_name) or expand_path(path_raw, env)))
            split_env = asset.get("split_env")
            default_split = asset.get("default_split")
            if (
                isinstance(split_env, str)
                and re.fullmatch(r"[A-Z_][A-Z0-9_]*", split_env)
                and isinstance(default_split, (str, int, float))
            ):
                defaults.setdefault(split_env, str(env.get(split_env) or default_split))
    return defaults


def discover_benchmark_items(env: Mapping[str, str], selected_benchmarks: set[str] | None) -> list[PrepareItem]:
    benchmark_ids = list(ACTIVE_BENCHMARK_IDS)
    if selected_benchmarks:
        missing = sorted(selected_benchmarks - set(benchmark_ids))
        if missing:
            raise ValueError(f"unknown embodied benchmark ids: {', '.join(missing)}")
        benchmark_ids = [benchmark_id for benchmark_id in benchmark_ids if benchmark_id in selected_benchmarks]

    template_assets = load_template_benchmark_assets()
    repo_revisions = template_repo_revisions(template_assets)
    catalog_revisions = load_benchmark_catalog_revisions()
    items: list[PrepareItem] = []
    for benchmark_id in benchmark_ids:
        profile_path = BENCHMARK_PROFILE_ROOT / f"{benchmark_id}.yaml"
        seen_paths: set[Path] = set()
        for asset in template_assets.get(benchmark_id, []):
            path_raw = asset.get("path") or asset.get("local_path")
            if not isinstance(path_raw, str) or not path_raw:
                continue
            local_path = expand_path(path_raw, env)
            kind = str(asset.get("kind") or "asset")
            if kind == "repo" and isinstance(asset.get("repo_url"), str):
                repo_url = str(asset["repo_url"])
                items.append(
                    PrepareItem(
                        "benchmark",
                        "git_repo",
                        benchmark_id,
                        str(asset.get("id") or "official_repo"),
                        local_path,
                        source=repo_url,
                        revision=resolved_catalog_revision(
                            benchmark_id=benchmark_id,
                            kind="git",
                            asset_id=normalized_repo_url(repo_url),
                            explicit_revision=asset.get("revision"),
                            catalog_revisions=catalog_revisions,
                        ),
                        metadata=dict(asset),
                    )
                )
            elif kind in {"dataset", "simulator_asset"} and isinstance(asset.get("hf_dataset_id"), str):
                dataset_id = str(asset["hf_dataset_id"])
                items.append(
                    PrepareItem(
                        "benchmark",
                        "hf_dataset",
                        benchmark_id,
                        dataset_id,
                        local_path,
                        source=dataset_id,
                        revision=resolved_catalog_revision(
                            benchmark_id=benchmark_id,
                            kind="hf",
                            asset_id=dataset_id,
                            explicit_revision=asset.get("revision"),
                            catalog_revisions=catalog_revisions,
                        ),
                        metadata=dict(asset),
                    )
                )
            elif kind == "checkpoint" and isinstance(asset.get("hf_model_id"), str):
                items.append(
                    PrepareItem(
                        "benchmark",
                        "hf_model",
                        benchmark_id,
                        str(asset["hf_model_id"]),
                        local_path,
                        source=str(asset["hf_model_id"]),
                        revision=str(asset.get("revision")) if asset.get("revision") else None,
                        metadata=dict(asset),
                    )
                )
            elif kind not in {"dataset_split", "upstream_rollout_execution"}:
                items.append(
                    PrepareItem(
                        "benchmark",
                        "manual_asset",
                        benchmark_id,
                        str(asset.get("id") or kind),
                        local_path,
                        metadata=dict(asset),
                    )
                )
            seen_paths.add(local_path)

        if profile_path.is_file():
            profile = load_yaml(profile_path)
            required = profile.get("required_assets") or {}
            if isinstance(required, Mapping):
                for path_raw in required.get("local_dataset_paths") or []:
                    if not isinstance(path_raw, str):
                        continue
                    local_path = expand_path(path_raw, env)
                    if local_path not in seen_paths:
                        items.append(PrepareItem("benchmark", "manual_asset", benchmark_id, "required_dataset", local_path))
                        seen_paths.add(local_path)
                for path_raw in required.get("local_repo_paths") or []:
                    if not isinstance(path_raw, str):
                        continue
                    local_path = expand_path(path_raw, env)
                    if local_path in seen_paths:
                        continue
                    repo_url = BENCHMARK_REPO_URLS.get(benchmark_id)
                    if repo_url:
                        items.append(
                            PrepareItem(
                                "benchmark",
                                "git_repo",
                                benchmark_id,
                                "required_repo",
                                local_path,
                                source=repo_url,
                                revision=repo_revisions.get(normalized_repo_url(repo_url)),
                            )
                        )
                    else:
                        items.append(PrepareItem("benchmark", "manual_asset", benchmark_id, "required_repo", local_path))
                    seen_paths.add(local_path)

        repo_url = BENCHMARK_REPO_URLS.get(benchmark_id)
        repo_path = expand_path("${WORLDFOUNDRY_CACHE_DIR}", env) / "repos" / github_slug(repo_url) if repo_url else None
        if repo_url and repo_path and repo_path not in seen_paths:
            items.append(
                PrepareItem(
                    "benchmark",
                    "git_repo",
                    benchmark_id,
                    "official_repo",
                    repo_path,
                    source=repo_url,
                    revision=repo_revisions.get(normalized_repo_url(repo_url)),
                )
            )
            seen_paths.add(repo_path)
    return dedupe_items(items)


def dedupe_items(items: Iterable[PrepareItem]) -> list[PrepareItem]:
    by_key: dict[tuple[str, str, str, str], PrepareItem] = {}
    for item in items:
        by_key.setdefault(item.key, item)
    return sorted(by_key.values(), key=lambda item: (item.category, item.owner_id, item.kind, str(item.local_path)))


def tool_path(name: str) -> str | None:
    search_path = os.environ.get("PATH", "")
    local_bin = str(Path.home() / ".local" / "bin")
    paths = search_path.split(os.pathsep)
    if local_bin not in paths:
        paths.insert(0, local_bin)
    return shutil.which(name, path=os.pathsep.join(paths))


def run_command(command: list[str], *, env: Mapping[str, str], log_path: Path, timeout: int) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed = subprocess.run(
        command,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    duration = time.monotonic() - start
    log_path.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    return completed.returncode, duration


def hf_revision_marker_path(item: PrepareItem) -> Path:
    return item.local_path.parent / f".{item.local_path.name}.worldfoundry-hf-revision.json"


def hf_revision_marker_matches(item: PrepareItem, repo_type: str) -> bool:
    if not item.revision:
        return True
    try:
        payload = json.loads(hf_revision_marker_path(item).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return payload == {
        "schema": HF_REVISION_MARKER_SCHEMA,
        "repo_id": item.asset_id,
        "repo_type": repo_type,
        "revision": item.revision,
    }


def write_hf_revision_marker(item: PrepareItem, repo_type: str) -> Path:
    if not item.revision:
        raise ValueError("a pinned Hugging Face revision is required")
    marker = hf_revision_marker_path(item)
    write_json(
        marker,
        {
            "schema": HF_REVISION_MARKER_SCHEMA,
            "repo_id": item.asset_id,
            "repo_type": repo_type,
            "revision": item.revision,
        },
    )
    return marker


def prepare_hf_item(item: PrepareItem, args: argparse.Namespace, env: Mapping[str, str], log_dir: Path) -> dict[str, Any]:
    hf = tool_path("hfd") if args.hf_tool == "hfd" else (tool_path("hf") or tool_path("huggingface-cli"))
    repo_type = "dataset" if item.kind == "hf_dataset" else "model"
    log_path = log_dir / item.category / item.owner_id / f"{repo_slug(item.asset_id)}.log"
    row = base_row(item, log_path)
    row["repo_type"] = repo_type
    row["ready_before"] = asset_ready(item.local_path)
    row["revision_matches_before"] = hf_revision_marker_matches(item, repo_type)
    if args.skip_existing and row["ready_before"] and row["revision_matches_before"]:
        row["status"] = "ready"
        row["ready"] = True
        return row
    if args.plan_only:
        row["status"] = "planned"
        row["ready"] = row["ready_before"] and row["revision_matches_before"]
        row["command"] = hf_download_command(hf or args.hf_tool, item, repo_type, args)
        return row
    if not hf:
        row["status"] = "blocked_missing_tool"
        row["ready"] = False
        row["reason"] = f"{args.hf_tool} download tool is not installed"
        return row
    item.local_path.mkdir(parents=True, exist_ok=True)
    command = hf_download_command(hf, item, repo_type, args)
    row["command"] = command
    try:
        returncode, duration = run_command(command, env=env, log_path=log_path, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        row["status"] = "failed"
        row["ready"] = asset_ready(item.local_path)
        row["reason"] = f"timeout after {args.timeout_seconds}s"
        return row
    row["returncode"] = returncode
    row["duration_seconds"] = round(duration, 3)
    row["ready"] = asset_ready(item.local_path)
    if returncode == 0 and row["ready"] and item.revision:
        try:
            marker = write_hf_revision_marker(item, repo_type)
        except OSError as exc:
            row["ready"] = False
            row["reason"] = f"could not record pinned Hugging Face revision: {exc}"
        else:
            row["revision_marker"] = str(marker)
            row["revision_matches"] = True
    row["status"] = "ready" if returncode == 0 and row["ready"] else "failed"
    return row


def hf_download_command(hf: str, item: PrepareItem, repo_type: str, args: argparse.Namespace) -> list[str]:
    include_patterns = [str(pattern) for pattern in item.metadata.get("allow_patterns") or item.metadata.get("include_patterns") or []]
    exclude_patterns = [
        str(pattern) for pattern in item.metadata.get("exclude_patterns") or []
    ] + [".DS_Store", "*/.DS_Store"]
    if args.hf_tool == "hfd":
        command = [
            hf,
            item.asset_id,
        ]
        if include_patterns:
            command.extend(["--include", *include_patterns])
        command.extend(["--exclude", *exclude_patterns])
        command.extend(
            [
                "--local-dir",
                str(item.local_path),
                "--tool",
                args.hfd_backend,
            ]
        )
        if args.hfd_backend == "aria2c":
            command.extend(
                [
                    "-x",
                    str(args.hfd_threads),
                    "-j",
                    str(args.hfd_jobs),
                ]
            )
        if repo_type == "dataset":
            command.append("--dataset")
    else:
        command = [hf, "download", item.asset_id, "--local-dir", str(item.local_path), "--max-workers", str(args.hf_workers)]
        if repo_type != "model":
            command.extend(["--repo-type", repo_type])
        for pattern in include_patterns:
            command.extend(["--include", pattern])
        for pattern in exclude_patterns:
            command.extend(["--exclude", pattern])
    if item.revision:
        command.extend(["--revision", item.revision])
    return command


def git_head_revision(path: Path, git: str = "git") -> str | None:
    try:
        completed = subprocess.run(
            [git, "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = (completed.stdout or "").strip()
    return revision if completed.returncode == 0 and revision else None


def prepare_git_item(item: PrepareItem, args: argparse.Namespace, env: Mapping[str, str], log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / item.category / item.owner_id / f"{item.local_path.name}.git.log"
    row = base_row(item, log_path)
    repo_exists = (item.local_path / ".git").is_dir()
    head_before = git_head_revision(item.local_path) if repo_exists and item.revision else None
    row["head_before"] = head_before
    row["revision_matches_before"] = not item.revision or head_before == item.revision
    row["ready_before"] = repo_exists
    if args.plan_only:
        row["status"] = "planned"
        row["ready"] = row["ready_before"] and row["revision_matches_before"]
        if not repo_exists:
            row["command"] = git_command(item)
        if item.revision:
            row["fetch_command"] = git_fetch_command(item)
            row["checkout_command"] = git_checkout_command(item)
        return row
    git = tool_path("git")
    if not git:
        row["status"] = "blocked_missing_tool"
        row["ready"] = False
        row["reason"] = "git is not installed"
        return row
    if repo_exists:
        try:
            checkout_clean = git_checkout_is_clean(
                git,
                item.local_path,
                env=env,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            row["status"] = "failed"
            row["ready"] = False
            row["reason"] = f"timeout checking git checkout after {args.timeout_seconds}s"
            return row
        except RuntimeError as exc:
            row["status"] = "failed"
            row["ready"] = False
            row["reason"] = str(exc)
            return row
        row["checkout_clean_before"] = checkout_clean
        if not checkout_clean:
            row["status"] = "blocked_dirty_checkout"
            row["ready"] = False
            row["reason"] = (
                "git checkout contains tracked or untracked changes; "
                "clean or move them manually before preparing this pinned asset"
            )
            return row
    if args.skip_existing and row["ready_before"] and row["revision_matches_before"]:
        row["status"] = "ready"
        row["ready"] = True
        return row
    if repo_exists:
        returncode, duration = 0, 0.0
        row["reused_existing_repo"] = True
    else:
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        command = git_command(item, git=git)
        row["command"] = command
        try:
            returncode, duration = run_command(command, env=env, log_path=log_path, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            row["status"] = "failed"
            row["ready"] = (item.local_path / ".git").is_dir()
            row["reason"] = f"timeout after {args.timeout_seconds}s"
            return row
    row["returncode"] = returncode
    row["duration_seconds"] = round(duration, 3)
    if returncode == 0 and item.revision:
        fetch_log = log_path.with_suffix(".fetch.log")
        fetch_command = git_fetch_command(item, git=git)
        row["fetch_command"] = fetch_command
        try:
            fetch_returncode, fetch_duration = run_command(
                fetch_command, env=env, log_path=fetch_log, timeout=args.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            row["status"] = "failed"
            row["ready"] = (item.local_path / ".git").is_dir()
            row["reason"] = f"revision fetch timeout after {args.timeout_seconds}s"
            return row
        row["fetch_returncode"] = fetch_returncode
        row["fetch_duration_seconds"] = round(fetch_duration, 3)
        returncode = fetch_returncode

    if returncode == 0 and item.revision:
        checkout_log = log_path.with_suffix(".checkout.log")
        checkout_command = git_checkout_command(item, git=git)
        row["checkout_command"] = checkout_command
        try:
            checkout_returncode, checkout_duration = run_command(
                checkout_command, env=env, log_path=checkout_log, timeout=args.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            row["status"] = "failed"
            row["ready"] = (item.local_path / ".git").is_dir()
            row["reason"] = f"revision checkout timeout after {args.timeout_seconds}s"
            return row
        row["checkout_returncode"] = checkout_returncode
        row["checkout_duration_seconds"] = round(checkout_duration, 3)
        returncode = checkout_returncode
    row["head_after"] = git_head_revision(item.local_path, git=git) if item.revision else None
    row["revision_matches"] = not item.revision or row["head_after"] == item.revision
    row["ready"] = (item.local_path / ".git").is_dir() and row["revision_matches"]
    row["status"] = "ready" if returncode == 0 and row["ready"] else "failed"
    return row


def git_command(item: PrepareItem, git: str = "git") -> list[str]:
    if not item.source:
        return [git, "clone", str(item.local_path)]
    return [git, "clone", "--depth", "1", item.source, str(item.local_path)]


def git_fetch_command(item: PrepareItem, git: str = "git") -> list[str]:
    if not item.revision:
        raise ValueError("git fetch command requires a pinned revision")
    return [git, "-C", str(item.local_path), "fetch", "--depth", "1", "origin", item.revision]


def git_checkout_command(item: PrepareItem, git: str = "git") -> list[str]:
    if not item.revision:
        raise ValueError("git checkout command requires a pinned revision")
    return [git, "-C", str(item.local_path), "checkout", "--detach", "FETCH_HEAD"]


def git_checkout_is_clean(
    git: str,
    checkout: Path,
    *,
    env: Mapping[str, str],
    timeout: int,
) -> bool:
    completed = subprocess.run(
        [git, "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=normal"],
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown git status error"
        raise RuntimeError(f"could not inspect existing git checkout: {detail}")
    return not completed.stdout.strip()


def prepare_gcs_item(item: PrepareItem, args: argparse.Namespace, env: Mapping[str, str], log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / item.category / item.owner_id / f"{repo_slug(item.asset_id)}.gcs.log"
    row = base_row(item, log_path)
    row["ready_before"] = asset_ready(item.local_path)
    if args.skip_existing and row["ready_before"]:
        row["status"] = "ready"
        row["ready"] = True
        return row
    gsutil = tool_path("gsutil")
    gcloud = tool_path("gcloud")
    destination = f"{item.local_path}/"
    if args.plan_only:
        row["status"] = "planned"
        row["ready"] = row["ready_before"]
        row["command"] = ["gsutil", "-m", "rsync", "-r", item.asset_id, destination]
        return row
    if gsutil:
        command = [gsutil, "-m", "rsync", "-r", item.asset_id, destination]
    elif gcloud:
        command = [gcloud, "storage", "rsync", "--recursive", item.asset_id, destination]
    else:
        row["status"] = "blocked_missing_tool"
        row["ready"] = False
        row["reason"] = "gsutil/gcloud is not installed"
        return row
    item.local_path.mkdir(parents=True, exist_ok=True)
    row["command"] = command
    try:
        returncode, duration = run_command(command, env=env, log_path=log_path, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        row["status"] = "failed"
        row["ready"] = asset_ready(item.local_path)
        row["reason"] = f"timeout after {args.timeout_seconds}s"
        return row
    row["returncode"] = returncode
    row["duration_seconds"] = round(duration, 3)
    row["ready"] = asset_ready(item.local_path)
    row["status"] = "ready" if returncode == 0 and row["ready"] else "failed"
    return row


def prepare_manual_item(item: PrepareItem, args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / item.category / item.owner_id / f"{repo_slug(item.asset_id)}.manual.log"
    row = base_row(item, log_path)
    row["ready_before"] = asset_ready(item.local_path)
    if row["ready_before"]:
        row["status"] = "ready"
        row["ready"] = True
        return row
    if not args.plan_only and args.create_placeholders:
        ensure_target_dir(item.local_path).mkdir(parents=True, exist_ok=True)
    row["ready"] = asset_ready(item.local_path)
    row["status"] = "manual_required" if not row["ready"] else "ready"
    row["reason"] = "no public automated source is recorded in the runtime profile"
    return row


def base_row(item: PrepareItem, log_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "worldfoundry-embodied-official-asset-v1",
        "category": item.category,
        "kind": item.kind,
        "owner_id": item.owner_id,
        "asset_id": item.asset_id,
        "role": item.role,
        "source": item.source,
        "revision": item.revision,
        "local_path": str(item.local_path),
        "log_path": str(log_path),
        "metadata": dict(item.metadata),
    }


def prepare_one(item: PrepareItem, args: argparse.Namespace, env: Mapping[str, str], log_dir: Path) -> dict[str, Any]:
    if item.kind in {"hf_model", "hf_dataset"}:
        return prepare_hf_item(item, args, env, log_dir)
    if item.kind == "git_repo":
        return prepare_git_item(item, args, env, log_dir)
    if item.kind == "gcs_checkpoint":
        return prepare_gcs_item(item, args, env, log_dir)
    return prepare_manual_item(item, args, log_dir)


def write_env_file(
    path: Path,
    env: Mapping[str, str],
    *,
    benchmark_keys: Iterable[str] = (),
) -> None:
    generic_keys = (
        "WORLDFOUNDRY_REPO_ROOT",
        "WORLDFOUNDRY_DATA_DIR",
        "WORLDFOUNDRY_MODEL_DIR",
        "WORLDFOUNDRY_CKPT_DIR",
        "WORLDFOUNDRY_HFD_ROOT",
        "WORLDFOUNDRY_HFD_DATASET_ROOT",
        "WORLDFOUNDRY_ASSET_ROOT",
        "WORLDFOUNDRY_OPENPI_ASSET_ROOT",
        "WORLDFOUNDRY_MODEL_SOURCE_DIR",
        "WORLDFOUNDRY_BENCHMARK_REPO_ROOT",
        "WORLDFOUNDRY_CACHE_DIR",
        "WORLDFOUNDRY_ARTIFACT_DIR",
        "HF_HOME",
        "HF_HUB_ENABLE_HF_TRANSFER",
    )
    keys = (*generic_keys, *sorted(set(benchmark_keys) - set(generic_keys)))
    lines = [
        "# Source this before running WorldFoundry embodied evaluations.",
        f"# Generated at {utc_now_iso()}",
        "export PATH=\"$HOME/.local/bin:$PATH\"",
    ]
    for key in keys:
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[Mapping[str, Any]], *, item_count: int, mode: str, report_dir: Path) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status"))] = status_counts.get(str(row.get("status")), 0) + 1
        kind_counts[str(row.get("kind"))] = kind_counts.get(str(row.get("kind")), 0) + 1
    return {
        "schema_version": "worldfoundry-embodied-official-assets-summary-v1",
        "generated_at": utc_now_iso(),
        "mode": mode,
        "item_count": item_count,
        "row_count": len(rows),
        "status_counts": status_counts,
        "kind_counts": kind_counts,
        "ok": all(str(row.get("status")) == "ready" for row in rows),
        "report_dir": str(report_dir),
    }


def parse_csv(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.add(part)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare all WorldFoundry embodied benchmark assets and model checkpoints.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("WORLDFOUNDRY_DATA_DIR", WORKSPACE_ROOT / "data")))
    parser.add_argument("--ckpt-root", type=Path, default=Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", WORKSPACE_ROOT / "ckpt")))
    parser.add_argument("--hfd-root", type=Path)
    parser.add_argument("--hfd-dataset-root", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--model", action="append", help="Embodied model id to include. Defaults to all vla_va_wam models.")
    parser.add_argument("--benchmark", action="append", help="Embodied benchmark id to include. Defaults to all active simulator benchmarks.")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-placeholders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--hf-tool", choices=("hf", "hfd"), default="hf")
    parser.add_argument("--hf-workers", type=int, default=4)
    parser.add_argument("--hfd-backend", choices=("aria2c", "wget"), default="aria2c")
    parser.add_argument("--hfd-threads", type=int, default=4)
    parser.add_argument("--hfd-jobs", type=int, default=4)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--hf-username", default=os.environ.get("HF_USERNAME"))
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--no-models", action="store_true")
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    ckpt_root = args.ckpt_root.resolve()
    hfd_root = (args.hfd_root or ckpt_root / "hfd_models").resolve()
    hfd_dataset_root = (args.hfd_dataset_root or data_root / "datasets").resolve()
    report_dir = (args.report_dir or data_root / "embodied_prepare_reports").resolve()
    env = target_env(data_root, ckpt_root, hfd_root, hfd_dataset_root)
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = str(args.hf_endpoint)
    if args.hf_username:
        env["HF_USERNAME"] = str(args.hf_username)
    env["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + env.get("PATH", "")
    env["PATH"] = str(WORKSPACE_ROOT / ".tools" / "bin") + os.pathsep + env["PATH"]

    for path in (data_root, ckpt_root, hfd_root, hfd_dataset_root, data_root / "repos", data_root / "assets", report_dir):
        path.mkdir(parents=True, exist_ok=True)

    selected_models = parse_csv(args.model)
    selected_benchmarks = parse_csv(args.benchmark)
    benchmark_env = benchmark_environment_defaults(env, selected_benchmarks)
    for key, value in benchmark_env.items():
        env.setdefault(key, value)
    items: list[PrepareItem] = []
    if not args.no_models:
        items.extend(discover_model_items(env, selected_models))
    if not args.no_benchmarks:
        items.extend(discover_benchmark_items(env, selected_benchmarks))
    items = dedupe_items(items)

    env_path = report_dir / "embodied_env.sh"
    manifest_path = report_dir / "asset_manifest.json"
    report_jsonl_path = report_dir / "asset_prepare_report.jsonl"
    summary_path = report_dir / "asset_prepare_summary.json"
    write_env_file(env_path, env, benchmark_keys=benchmark_env)
    write_json(
        manifest_path,
        {
            "schema_version": "worldfoundry-embodied-official-asset-manifest-v1",
            "generated_at": utc_now_iso(),
            "data_root": str(data_root),
            "ckpt_root": str(ckpt_root),
            "hfd_root": str(hfd_root),
            "hfd_dataset_root": str(hfd_dataset_root),
            "items": [
                {
                    "category": item.category,
                    "kind": item.kind,
                    "owner_id": item.owner_id,
                    "asset_id": item.asset_id,
                    "role": item.role,
                    "source": item.source,
                    "revision": item.revision,
                    "local_path": str(item.local_path),
                    "metadata": dict(item.metadata),
                }
                for item in items
            ],
        },
    )

    log_dir = report_dir / "logs"
    rows: list[dict[str, Any]] = []
    if args.plan_only or args.max_workers <= 1:
        for item in items:
            rows.append(prepare_one(item, args, env, log_dir))
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = [pool.submit(prepare_one, item, args, env, log_dir) for item in items]
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row: (str(row.get("category")), str(row.get("owner_id")), str(row.get("kind")), str(row.get("local_path"))))

    write_jsonl(report_jsonl_path, rows)
    summary = summarize(rows, item_count=len(items), mode="plan" if args.plan_only else "execute", report_dir=report_dir)
    summary.update(
        {
            "env_path": str(env_path),
            "manifest_path": str(manifest_path),
            "report_jsonl_path": str(report_jsonl_path),
            "summary_path": str(summary_path),
        }
    )
    write_json(summary_path, summary)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(f"env: {env_path}")
        print(f"manifest: {manifest_path}")
        print(f"report: {report_jsonl_path}")
        print(f"summary: {summary_path}")
        print(json.dumps(summary["status_counts"], ensure_ascii=False, sort_keys=True))
    return 0 if summary["ok"] or args.plan_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
