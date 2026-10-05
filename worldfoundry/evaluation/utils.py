"""Shared evaluation utilities.

Sections: JSON/text IO re-exports, YAML manifest loading, repository/data
paths, and version/fingerprint capture.  Importing this module has no global
side effects (notably, it does not touch ``sys.path``; see
:func:`ensure_repo_root_on_sys_path` for the explicit opt-in).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.manifests import (
    MANIFEST_SUFFIXES,
    load_manifest,
    load_manifest_collection,
    manifest_paths,
)
from worldfoundry.core.io.paths import (
    BENCHMARKS_DATA_ROOT,
    DATA_ROOT,
    REPO_ROOT,
    hfd_dataset_root_path,
    package_data_path,
    package_data_root,
    package_root,
    project_root,
    resolve_worldfoundry_path,
)
from worldfoundry.core.io.formats.serialization import (
    append_jsonl,
    jsonable,
    read_json,
    read_json_object,
    read_json_or_jsonl,
    read_jsonl_objects,
    reset_jsonl,
    write_json,
    write_jsonl,
    write_text_file,
)
from worldfoundry.evaluation.api import (
    AGGREGATE_RESULT_SCHEMA_VERSION,
    ARTIFACT_REF_SCHEMA_VERSION,
    BENCHMARK_SPEC_SCHEMA_VERSION,
    GENERATION_REQUEST_SCHEMA_VERSION,
    GENERATION_RESULT_SCHEMA_VERSION,
    METRIC_RESULT_SCHEMA_VERSION,
    METRIC_SPEC_SCHEMA_VERSION,
    WORLD_MODEL_CONFIG_SCHEMA_VERSION,
    WORLD_MODEL_MANIFEST_SCHEMA_VERSION,
    WORLD_TASK_CONFIG_SCHEMA_VERSION,
)
from worldfoundry.evaluation.api.json_contract import sha256_file, to_plain

# ── JSON / text formatting helpers ─────────────────────────────────────


def mapping_or_empty(value: Any) -> dict[str, Any]:
    """Return a mutable mapping when the value is mapping-like."""
    return dict(value) if isinstance(value, Mapping) else {}


def format_value(value: Any) -> str:
    """Format a scalar or structured value for human-readable reports."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def escape_markdown_cell(value: Any) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return format_value(value).replace("|", "\\|").replace("\n", " ")


def write_text(path: str | Path, payload: str, *, atomic: bool = True) -> Path:
    """Write text to a destination path.

    Args:
        path: Destination path.
        payload: Text content to write.
        atomic: Whether to write through a temporary sibling before replacing.
    """
    return write_text_file(path, payload, atomic=atomic)


# ── YAML manifest loading ──────────────────────────────────────────────
#
# ``MANIFEST_SUFFIXES`` / ``load_manifest`` / ``manifest_paths`` /
# ``load_manifest_collection`` are re-exported above from
# ``worldfoundry.core.io.manifests`` (single canonical implementation; SA-10
# moved it below the evaluation layer so ``worldfoundry.runtime`` no longer
# imports this module).  Behavior and error messages are unchanged.


# ── Repository / data paths ────────────────────────────────────────────

WORLDFOUNDRY_PACKAGE_ROOT = package_root()
worldfoundry_repository_root = project_root
worldfoundry_data_root = package_data_root
worldfoundry_data_path = package_data_path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT
BENCHMARK_ZOO_DIR = BENCHMARKS_DATA_ROOT / "catalog"
BENCHMARK_TASK_ROOT = BENCHMARKS_DATA_ROOT / "tasks" / "external"
BENCHMARK_ASSETS_ROOT = BENCHMARKS_DATA_ROOT / "assets"
BENCHMARK_RUNTIME_PROFILE_DIR = BENCHMARKS_DATA_ROOT / "runtime_profiles"
MODEL_ZOO_DIR = DATA_ROOT / "models" / "catalog"
MODEL_RUNTIME_ROOT = DATA_ROOT / "models" / "runtime"
MODEL_RUNTIME_PROFILES_ROOT = MODEL_RUNTIME_ROOT / "profiles"
MODEL_RUNTIME_CONFIGS_ROOT = MODEL_RUNTIME_ROOT / "configs"
MODEL_RUNTIME_ENVIRONMENTS_ROOT = MODEL_RUNTIME_ROOT / "environments"
MODEL_RUNTIME_ASSETS_ROOT = MODEL_RUNTIME_ROOT / "assets"
TMP_ROOT = REPO_ROOT / "tmp"
CACHE_ROOT = REPO_ROOT / "cache"
HFD_DATASET_CACHE_ROOT = hfd_dataset_root_path()


def benchmark_task_sample_path(benchmark_id: str) -> Path | None:
    """Return a checked-in benchmark fixture result file."""
    for suffix in (".csv", ".jsonl", ".json", ".txt"):
        path = BENCHMARK_ASSETS_ROOT / benchmark_id / f"sample_results{suffix}"
        if path.is_file():
            return path
    for suffix in (".csv", ".jsonl", ".json", ".txt"):
        path = BENCHMARK_TASK_ROOT / f"{benchmark_id}.sample_results{suffix}"
        if path.is_file():
            return path
    return None


def worldfoundry_hfd_dataset_root() -> Path:
    """Resolve the benchmark Hugging Face dataset root.

    Explicit command-line arguments should still take precedence. This helper is
    only for defaults shared by benchmark download, data probes, and audits.
    """

    return hfd_dataset_root_path()


def ensure_repo_root_on_sys_path() -> Path:
    """Explicitly put the repository root on ``sys.path`` and return it.

    This used to happen implicitly whenever this module was imported, which
    polluted host processes embedding the evaluation framework (and, in
    installed deployments, could promote the site-packages parent directory to
    ``sys.path[0]``).  Callers that resolve repo-relative dynamic imports
    (benchmark/model discovery from a source checkout) must now opt in.
    """
    root_text = str(REPO_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return REPO_ROOT


# ── Version / fingerprint capture ──────────────────────────────────────

VERSION_CONTEXT_SCHEMA_VERSION = "worldfoundry-version-context-v2"
RUN_FINGERPRINT_SCHEMA_VERSION = "worldfoundry-run-fingerprint"
# Explicit engine revision: bump the numeric suffix when evaluation-engine
# behavior changes in a way that affects results, so run manifests written by
# different engine revisions are distinguishable (the package version alone is
# "unknown" in source checkouts).
EVALUATION_ENGINE_VERSION = "worldfoundry-eval-engine/1"


def _repo_root() -> Path:
    """Return the resolved repository root path."""
    return worldfoundry_repository_root()


@lru_cache(maxsize=32)
def package_version(distribution: str = "worldfoundry") -> str:
    """Retrieve the installed package version of the given distribution."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _run_git(root: Path, *args: str) -> str | None:
    """Run a git command in the specified directory, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *args),
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_diff_sha256(root: Path) -> str | None:
    """Hash the tracked working-tree diff without buffering it in memory.

    Untracked files are intentionally excluded.  A bounded probe keeps version
    capture from stalling runs on large or remote filesystems; callers retain
    ``dirty=True`` and record a null digest when the diff cannot be captured.
    """

    command = (
        "git",
        "-C",
        str(root),
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "HEAD",
        "--",
    )
    try:
        with tempfile.TemporaryFile() as diff_stream:
            result = subprocess.run(
                command,
                stdout=diff_stream,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            diff_stream.seek(0)
            digest = hashlib.sha256()
            while chunk := diff_stream.read(1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest()
    except (OSError, subprocess.SubprocessError):
        return None


def git_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Get status and commit metadata of the current git repository.

    Captured fresh on every call (no caching): version contexts and run
    manifests must record the git state at run time, and a long-lived process
    (service, notebook) may commit or dirty the tree between runs.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root()
    commit = _run_git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None, "dirty_diff_sha256": None}
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None else bool(status)
    return {
        "available": True,
        "commit": commit,
        "dirty": dirty,
        "dirty_diff_sha256": _git_diff_sha256(root) if dirty else None,
    }


def _callable_reference(value: Any) -> str:
    """Generate a qualified string identifier for a callable object."""
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(value)


def stable_json_dumps(value: Any) -> str:
    """Serialize a JSON-safe dictionary with stable key-sorting and no extra whitespace.

    Canonicalizes through :func:`~worldfoundry.evaluation.api.json_contract.to_plain`
    first so ``set``/``frozenset`` members are deterministically ordered (plain
    ``jsonable`` preserves set iteration order, which varies across processes).
    """
    return json.dumps(jsonable(to_plain(value)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Calculate and return the SHA-256 hash of the given string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    """Generate a stable, reproducible SHA-256 hash of any serializable object."""
    return sha256_text(stable_json_dumps(value))


def file_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 hash of a file on disk.

    Delegates to :func:`worldfoundry.evaluation.api.json_contract.sha256_file`
    (single canonical chunked implementation).
    """
    return sha256_file(path)


def contract_versions() -> dict[str, str]:
    """Retrieve current schema versions of all evaluation contracts."""
    return {
        "artifact_ref": ARTIFACT_REF_SCHEMA_VERSION,
        "generation_request": GENERATION_REQUEST_SCHEMA_VERSION,
        "generation_result": GENERATION_RESULT_SCHEMA_VERSION,
        "metric_spec": METRIC_SPEC_SCHEMA_VERSION,
        "metric_result": METRIC_RESULT_SCHEMA_VERSION,
        "aggregate_result": AGGREGATE_RESULT_SCHEMA_VERSION,
        "world_model_manifest": WORLD_MODEL_MANIFEST_SCHEMA_VERSION,
        "world_model_config": WORLD_MODEL_CONFIG_SCHEMA_VERSION,
        "world_task_config": WORLD_TASK_CONFIG_SCHEMA_VERSION,
        "benchmark_spec": BENCHMARK_SPEC_SCHEMA_VERSION,
    }


def _class_reference(value: Any) -> str:
    """Generate a qualified string identifier for the class of the given object."""
    cls = value if isinstance(value, type) else value.__class__
    return f"{cls.__module__}:{cls.__qualname__}"


def model_runner_fingerprint(model_runner: Any | None) -> dict[str, Any] | None:
    """Generate a serialized fingerprint metadata dictionary for a model runner."""
    if model_runner is None:
        return None
    payload: dict[str, Any] = {
        "class": _class_reference(model_runner),
        "model_id": str(getattr(model_runner, "model_id", "")),
        "runner_version": str(getattr(model_runner, "runner_version", getattr(model_runner, "version", ""))),
        "capabilities": sorted(str(item) for item in getattr(model_runner, "capabilities", ()) or ()),
        "parameters": jsonable(getattr(model_runner, "parameters", None)),
        "config": jsonable(getattr(model_runner, "config", None)),
    }
    describe = getattr(model_runner, "describe_capabilities", None)
    if callable(describe):
        try:
            described = describe()
        except Exception as exc:  # noqa: BLE001 - version capture must not fail a run.
            payload["describe_capabilities_error"] = f"{type(exc).__name__}: {exc}"
        else:
            payload["described_capabilities"] = jsonable(described)
    return payload


def metric_fingerprint(metric: Any) -> dict[str, Any]:
    """Generate a stable metadata fingerprint dictionary for a metric object."""
    return {
        "class": _class_reference(metric),
        "name": str(getattr(metric, "name", "") or getattr(metric, "__name__", "") or metric.__class__.__name__),
        "version": str(getattr(metric, "version", "")),
        "parameters": jsonable(getattr(metric, "parameters", None)),
        "config": jsonable(getattr(metric, "config", None)),
        "metric_ids": jsonable(getattr(metric, "metric_ids", None)),
        "required_artifacts": tuple(str(item) for item in getattr(metric, "required_artifacts", ()) or ()),
        "higher_is_better": getattr(metric, "higher_is_better", None),
    }


def metric_callable_fingerprint(metric: Any | None) -> dict[str, Any] | None:
    """Generate a fingerprint dictionary for a metric callable, returning None if metric is None."""
    if metric is None:
        return None
    return {
        "class": _class_reference(metric),
        "callable": _callable_reference(metric),
        "name": str(getattr(metric, "name", "") or getattr(metric, "__name__", "") or metric.__class__.__name__),
        "version": str(getattr(metric, "version", "")),
    }


def build_version_context(
    *,
    runner: str,
    benchmark: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    model_runner: Any | None = None,
    metrics: Sequence[Any] = (),
    metric: Any | None = None,
    engine_version: str = EVALUATION_ENGINE_VERSION,
    extra: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
    git_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a comprehensive version context dictionary capturing engine, runtime, and git state."""
    return {
        "schema_version": VERSION_CONTEXT_SCHEMA_VERSION,
        "engine_version": engine_version,
        "runner": runner,
        "worldfoundry_version": package_version(),
        "python": {
            "version": platform.python_version(),
        },
        "git": dict(git_context) if git_context is not None else git_metadata(repo_root),
        "contract_versions": contract_versions(),
        "benchmark": jsonable(benchmark or {}),
        "model": jsonable(model or {}),
        "dataset": jsonable(dataset or {}),
        "model_runner": model_runner_fingerprint(model_runner),
        "metrics": [metric_fingerprint(item) for item in metrics],
        "metric_callable": metric_callable_fingerprint(metric),
        "extra": jsonable(extra or {}),
    }


def build_run_fingerprint(
    *,
    version_context: Mapping[str, Any],
    requests: Sequence[Any] = (),
    results: Sequence[Any] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unique run fingerprint dictionary with stable hashes of context, requests, and results."""
    payload = {
        "version_context": version_context,
        "requests": [jsonable(item) for item in requests],
        "results": [jsonable(item) for item in results],
        "extra": jsonable(extra or {}),
    }
    return {
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "hash": stable_hash(payload),
        "version_context_hash": stable_hash(version_context),
        "request_count": len(requests),
        "result_count": len(results),
    }
