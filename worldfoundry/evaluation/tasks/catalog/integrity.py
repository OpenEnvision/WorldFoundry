"""Integrity checks for the public benchmark catalog inventory."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import (
    EXCLUDED_BENCHMARK_IDS,
    benchmark_runtime_profiles_by_id,
    formal_benchmark_ids,
)
from worldfoundry.evaluation.utils import BENCHMARK_RUNTIME_PROFILE_DIR, BENCHMARK_ZOO_DIR, REPO_ROOT, load_manifest

_RUNNER_SCRIPT_RE = re.compile(r"worldfoundry/evaluation/tasks/execution/[A-Za-z0-9_./-]+\.py")
_BENCHMARK_ID_ARG_RE = re.compile(r"--benchmark-id(?:=|\s+)([A-Za-z0-9_.-]+)")
_TABLE_BACKTICK_RE = re.compile(r"`([a-z0-9][a-z0-9_.-]*)`")
_DOC_EXTENSIONS = {".md", ".mdx", ".ts", ".tsx"}


@dataclass(frozen=True)
class BenchmarkInventoryIntegrityReport:
    benchmark_ids: tuple[str, ...]
    catalog_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    runtime_profile_ids: tuple[str, ...]
    missing_task_ids: tuple[str, ...]
    extra_task_ids: tuple[str, ...]
    missing_runtime_profile_ids: tuple[str, ...]
    extra_runtime_profile_ids: tuple[str, ...]
    runner_scripts: tuple[str, ...]
    missing_runner_scripts: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_task_ids
            or self.extra_task_ids
            or self.missing_runtime_profile_ids
            or self.extra_runtime_profile_ids
            or self.missing_runner_scripts
        )

    def failure_summary(self) -> str:
        parts = []
        for name in (
            "missing_task_ids",
            "extra_task_ids",
            "missing_runtime_profile_ids",
            "extra_runtime_profile_ids",
            "missing_runner_scripts",
        ):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={list(value)}")
        return "; ".join(parts) if parts else "ok"


@dataclass(frozen=True)
class DocsBenchmarkCoverageReport:
    benchmark_ids: tuple[str, ...]
    mentioned_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    unknown_refs: tuple[str, ...]
    excluded_refs: tuple[str, ...]
    docs_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing_ids or self.unknown_refs or self.excluded_refs)

    def failure_summary(self) -> str:
        parts = []
        for name in ("missing_ids", "unknown_refs", "excluded_refs"):
            value = getattr(self, name)
            if value:
                parts.append(f"{name}={list(value)}")
        return "; ".join(parts) if parts else "ok"


def _as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


_README_SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    ".pytest_cache",
    "__pycache__",
    ".next",
    "dist",
    ".turbo",
    ".venv",
    "venv",
}


def _tracked_repo_paths() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ()
    return tuple(part.decode() for part in completed.stdout.split(b"\0") if part)


def _walk_repo_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(REPO_ROOT).parts
        except ValueError:
            continue
        if any(part in _README_SKIP_DIR_NAMES for part in relative_parts):
            continue
        files.append(path)
    return tuple(files)


def _is_allowed_readme(relative: str) -> bool:
    name = Path(relative).name
    if relative == "README.md":
        return True
    # Eigen vendor license notes, not WorldFoundry documentation.
    return name == "README.txt" and "/eigen/" in relative.replace("\\", "/")


def extra_readme_issues() -> list[str]:
    """The only WorldFoundry README.md is the repository root file."""
    issues: list[str] = []
    tracked = _tracked_repo_paths()
    candidates = tracked or tuple(_as_repo_relative(path) for path in _walk_repo_files())
    for relative in candidates:
        name = Path(relative).name
        if not name.lower().startswith("readme"):
            continue
        if _is_allowed_readme(relative):
            continue
        issues.append(f"{relative}: the only allowed README.md is the repository root file")
    issues.extend(extra_evaluation_doc_issues(candidates))
    return issues


def extra_evaluation_doc_issues(candidates: Iterable[str] | None = None) -> list[str]:
    """Fumadocs owns documentation; evaluation trees must not grow markdown guides."""
    issues: list[str] = []
    paths = candidates if candidates is not None else (_tracked_repo_paths() or tuple(
        _as_repo_relative(path) for path in _walk_repo_files()
    ))
    for relative in paths:
        if not relative.startswith("worldfoundry/evaluation/"):
            continue
        # Catalog manifests link to this source-attribution record. It is
        # provenance metadata for vendored code, not a reader-facing guide.
        if Path(relative).name == "WORLDFOUNDRY_PROVENANCE.md":
            continue
        if relative.lower().endswith((".md", ".mdx")):
            issues.append(f"{relative}: evaluation docs belong in fumadocs, not evaluation/")
    return issues


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = load_manifest(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _iter_yaml_files(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root,)
    return tuple(sorted(path for path in root.glob("*.yaml") if path.is_file() and path.name != "_manifest.yaml"))


def _collect_task_ids(task_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in _iter_yaml_files(task_dir):
        benchmark_id = _load_yaml_mapping(path).get("benchmark")
        if benchmark_id:
            ids.add(str(benchmark_id))
    return ids


def _collect_runner_scripts(value: Any) -> set[str]:
    scripts: set[str] = set()
    if isinstance(value, str):
        scripts.update(_RUNNER_SCRIPT_RE.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            scripts.update(_collect_runner_scripts(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            scripts.update(_collect_runner_scripts(item))
    return scripts


def _collect_runner_scripts_from_paths(paths: Iterable[Path]) -> set[str]:
    scripts: set[str] = set()
    for path in paths:
        scripts.update(_collect_runner_scripts(_load_yaml_mapping(path)))
    return scripts


def build_benchmark_inventory_integrity(
    *,
    catalog_dir: str | Path | None = None,
    task_dir: str | Path,
    runtime_profile_path: str | Path | None = None,
) -> BenchmarkInventoryIntegrityReport:
    catalog_root = Path(catalog_dir or BENCHMARK_ZOO_DIR)
    task_root = Path(task_dir)
    runtime_root = Path(runtime_profile_path or (BENCHMARK_RUNTIME_PROFILE_DIR / "official"))

    benchmark_ids = set(formal_benchmark_ids(catalog_root))
    task_ids = _collect_task_ids(task_root)
    runtime_profile_ids = set(benchmark_runtime_profiles_by_id(runtime_root))

    catalog_paths = [path for shard in ("video", "embodied") for path in _iter_yaml_files(catalog_root / shard)]
    task_paths = list(_iter_yaml_files(task_root))
    runtime_paths = list(_iter_yaml_files(runtime_root if runtime_root.is_dir() else runtime_root.parent))
    runner_scripts = _collect_runner_scripts_from_paths((*catalog_paths, *task_paths, *runtime_paths))
    missing_runner_scripts = tuple(sorted(script for script in runner_scripts if not (REPO_ROOT / script).is_file()))

    return BenchmarkInventoryIntegrityReport(
        benchmark_ids=tuple(sorted(benchmark_ids)),
        catalog_ids=tuple(sorted(benchmark_ids)),
        task_ids=tuple(sorted(task_ids)),
        runtime_profile_ids=tuple(sorted(runtime_profile_ids)),
        missing_task_ids=tuple(sorted(benchmark_ids - task_ids)),
        extra_task_ids=tuple(sorted(task_ids - benchmark_ids)),
        missing_runtime_profile_ids=tuple(sorted(benchmark_ids - runtime_profile_ids)),
        extra_runtime_profile_ids=tuple(sorted(runtime_profile_ids - benchmark_ids)),
        runner_scripts=tuple(sorted(runner_scripts)),
        missing_runner_scripts=missing_runner_scripts,
    )


def _default_docs_paths() -> tuple[Path, ...]:
    benchmark_hub_dir = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmark-hub"
    candidates = (
        REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmarks.mdx",
        benchmark_hub_dir,
        REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-hub-data.ts",
        REPO_ROOT / "docs" / "fumadocs" / "components" / "benchmark-hub.tsx",
    )
    paths: list[Path] = []
    for path in candidates:
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix in _DOC_EXTENSIONS
                )
            )
    return tuple(dict.fromkeys(paths))


def _iter_docs_paths(roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
    if roots is None:
        return _default_docs_paths()
    paths: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file():
            paths.append(root)
            continue
        if root.is_dir():
            paths.extend(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in _DOC_EXTENSIONS))
    return tuple(dict.fromkeys(paths))


def _explicit_doc_refs(path: Path, text: str) -> tuple[str, ...]:
    refs: list[str] = []
    for match in _BENCHMARK_ID_ARG_RE.finditer(text):
        value = match.group(1).strip()
        if value and not value.startswith("<"):
            refs.append(value)
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        first_cell = line.split("|", 2)[1] if "|" in line[1:] else line
        if " / " not in first_cell:
            continue
        refs.extend(_TABLE_BACKTICK_RE.findall(first_cell))
    return tuple(dict.fromkeys(refs))


def build_docs_benchmark_coverage(
    *,
    docs_roots: Iterable[str | Path] | None = None,
    benchmark_ids: Iterable[str] | None = None,
) -> DocsBenchmarkCoverageReport:
    ids = tuple(sorted(str(item) for item in (benchmark_ids or formal_benchmark_ids())))
    id_set = set(ids)
    excluded_ids = set(EXCLUDED_BENCHMARK_IDS)
    docs_paths = _iter_docs_paths(docs_roots)

    mentioned: set[str] = set()
    unknown_refs: list[str] = []
    excluded_refs: list[str] = []
    for path in docs_paths:
        text = path.read_text(encoding="utf-8")
        for benchmark_id in ids:
            if benchmark_id in text:
                mentioned.add(benchmark_id)
        for ref in _explicit_doc_refs(path, text):
            rel = _as_repo_relative(path)
            if ref in id_set:
                mentioned.add(ref)
            elif ref in excluded_ids:
                excluded_refs.append(f"{rel}:{ref}")
            else:
                unknown_refs.append(f"{rel}:{ref}")

    return DocsBenchmarkCoverageReport(
        benchmark_ids=ids,
        mentioned_ids=tuple(sorted(mentioned)),
        missing_ids=tuple(sorted(id_set - mentioned)),
        unknown_refs=tuple(sorted(dict.fromkeys(unknown_refs))),
        excluded_refs=tuple(sorted(dict.fromkeys(excluded_refs))),
        docs_paths=tuple(_as_repo_relative(path) for path in docs_paths),
    )


# Video catalog ids that are documentation-only and must not claim a runnable
# dispatch path. Keep this explicit: catalog/registry drift is a ship bug.
CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS: frozenset[str] = frozenset()

# VIDEO_RUNNER_REGISTRY keys whose catalog YAML lives outside catalog/video/.
# larybench is an embodied catalog entry that still uses the video runner table.
VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG: frozenset[str] = frozenset({"larybench"})


def _declared_field(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_declared_field(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_declared_field(item) for item in value)
    return True


def _video_catalog_entry_is_runnable(entry: Mapping[str, Any]) -> bool:
    """Return True when a video catalog YAML claims a runnable dispatch path."""
    runner = entry.get("runner")
    runner = runner if isinstance(runner, Mapping) else {}
    integration = entry.get("integration")
    integration = integration if isinstance(integration, Mapping) else {}
    if _declared_field(runner.get("run_command")) or _declared_field(runner.get("official_script")):
        return True
    return str(integration.get("status") or "").strip().lower() == "integrated"


def _iter_video_catalog_yaml_entries() -> tuple[tuple[Path, str, Mapping[str, Any]], ...]:
    """Yield ``(path, benchmark_id, entry)`` for every video catalog YAML with a payload."""
    from .benchmark_catalog import DEFAULT_VIDEO_CATALOG_DIR, is_catalog_metadata_manifest
    from .schema import iter_benchmark_zoo_payloads

    rows: list[tuple[Path, str, Mapping[str, Any]]] = []
    if not DEFAULT_VIDEO_CATALOG_DIR.is_dir():
        return ()
    for path in sorted(DEFAULT_VIDEO_CATALOG_DIR.glob("*.yaml")):
        if is_catalog_metadata_manifest(path) or not path.is_file():
            continue
        payload = load_manifest(path)
        for entry in iter_benchmark_zoo_payloads(payload):
            benchmark_id = entry.get("id") or entry.get("benchmark_id")
            if isinstance(benchmark_id, str) and benchmark_id.strip():
                rows.append((path, benchmark_id.strip(), entry))
            else:
                rows.append((path, "", entry))
    return tuple(rows)


def _catalog_yaml_ids() -> dict[str, Path]:
    """Map every catalog ``id`` / ``benchmark_id`` to its YAML path."""
    from .benchmark_catalog import _catalog_benchmark_path_index, resolve_benchmark_catalog_root

    return dict(_catalog_benchmark_path_index(str(resolve_benchmark_catalog_root())))


def video_catalog_dispatch_issues() -> list[str]:
    """Explain video catalog ids against ``VIDEO_RUNNER_REGISTRY``.

    Runnable video catalog entries must be registered. Every registry key must
    have a catalog YAML. Catalog-only documentation benches and non-video-shard
    runner ids must be listed explicitly rather than skipped.
    """
    issues: list[str] = []
    try:
        from worldfoundry.evaluation.tasks.catalog.dispatch import VIDEO_RUNNER_REGISTRY
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        return [f"catalog runner registry unavailable: {type(exc).__name__}: {exc}"]

    registry_ids = set(VIDEO_RUNNER_REGISTRY)
    catalog_index = _catalog_yaml_ids()
    video_ids: set[str] = set()
    runnable_ids: set[str] = set()

    for path, benchmark_id, entry in _iter_video_catalog_yaml_entries():
        if not benchmark_id:
            issues.append(f"{_as_repo_relative(path)}: video catalog YAML is missing top-level id")
            continue
        video_ids.add(benchmark_id)
        if _video_catalog_entry_is_runnable(entry):
            runnable_ids.add(benchmark_id)

    for benchmark_id in sorted(CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS - video_ids):
        issues.append(f"{benchmark_id}: catalog-only docs allowlist has no video catalog YAML")
    for benchmark_id in sorted(CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS & runnable_ids):
        issues.append(
            f"{benchmark_id}: catalog-only docs allowlist claims runnable "
            "(runner.run_command, runner.official_script, or integration.status=integrated)"
        )
    for benchmark_id in sorted(CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS & registry_ids):
        issues.append(f"{benchmark_id}: catalog-only docs allowlist is also in VIDEO_RUNNER_REGISTRY")

    for benchmark_id in sorted((video_ids - registry_ids) - CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS):
        if benchmark_id in runnable_ids:
            issues.append(f"{benchmark_id}: runnable video catalog entry is missing from VIDEO_RUNNER_REGISTRY")
        else:
            issues.append(
                f"{benchmark_id}: video catalog id is missing from VIDEO_RUNNER_REGISTRY "
                "(add a runner or list it in CATALOG_ONLY_DOCS_VIDEO_BENCHMARKS)"
            )

    for benchmark_id in sorted(VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG & video_ids):
        issues.append(
            f"{benchmark_id}: VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG is stale; YAML is under catalog/video/"
        )
    for benchmark_id in sorted(VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG - registry_ids):
        issues.append(f"{benchmark_id}: VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG is not in VIDEO_RUNNER_REGISTRY")

    for benchmark_id in sorted((registry_ids - video_ids) - VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG):
        issues.append(
            f"{benchmark_id}: VIDEO_RUNNER_REGISTRY key has no video catalog YAML "
            f"(add catalog/video/{benchmark_id}.yaml or list it in VIDEO_RUNNER_IDS_OUTSIDE_VIDEO_CATALOG)"
        )
    for benchmark_id in sorted(registry_ids):
        if benchmark_id not in catalog_index:
            issues.append(f"{benchmark_id}: VIDEO_RUNNER_REGISTRY key has no catalog YAML")
    issues.extend(catalog_runner_table_issues())
    return issues


_FORBIDDEN_TABLE_NAMES = (
    "_LEGACY_VIDEO_RUNNER_REGISTRY",
    "_LEGACY_CLI_RUNNERS",
    "_LEGACY_BENCHMARK_INTEGRATION_REGISTRY",
    "GENERIC_RESULT_NORMALIZER_BENCHMARKS",
)

_TABLE_SOURCE_PATHS = (
    "worldfoundry/evaluation/tasks/catalog/dispatch.py",
    "worldfoundry/evaluation/tasks/catalog/workspace_registry/dispatch.py",
    "worldfoundry/evaluation/tasks/execution/orchestration/benchmark_runner.py",
)

_FORBIDDEN_SHIM_PATHS = (
    "worldfoundry/evaluation/tasks/catalog/workspace_registry/specs.py",
    "worldfoundry/evaluation/tasks/execution/framework/runner_registry.py",
    "worldfoundry/evaluation/tasks/execution/framework/integration.py",
    "worldfoundry/evaluation/tasks/execution/framework/in_tree_registry.py",
    "worldfoundry/evaluation/tasks/execution/framework/benchmark_contract_registry.py",
    "worldfoundry/evaluation/tasks/execution/orchestration/benchmark_routing.py",
    "worldfoundry/evaluation/tasks/execution/orchestration/run_mode.py",
)


def catalog_runner_table_issues() -> list[str]:
    """Catalog YAML must be the only runner/routing/scoring membership source."""
    issues: list[str] = []
    for rel in _TABLE_SOURCE_PATHS:
        path = REPO_ROOT / rel
        if not path.is_file():
            issues.append(f"{rel}: missing while checking catalog-only runner tables")
            continue
        text = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_TABLE_NAMES:
            if re.search(rf"^{name}\s*[:=]", text, re.M):
                issues.append(f"{rel}: handwritten {name} must not remain")

    for rel in _FORBIDDEN_SHIM_PATHS:
        if (REPO_ROOT / rel).is_file():
            issues.append(f"{rel}: leftover dispatch shim; import from catalog.dispatch or scoring_registry")

    try:
        from worldfoundry.evaluation.tasks.catalog.dispatch import (
            VIDEO_RUNNER_REGISTRY,
            official_runner_registry,
            official_runner_spec,
            scoring_benchmark_ids,
            script_to_module,
        )
        from worldfoundry.evaluation.tasks.execution.framework.scoring_registry import (
            has_benchmark_contract_evaluator,
            supported_in_tree_benchmark_ids,
        )
        from worldfoundry.evaluation.tasks.execution.framework.video_quality_specs import (
            _BLOCKED_REASON,
            supported_video_quality_benchmark_ids,
        )
    except Exception as exc:  # pragma: no cover - defensive catalog validation
        return [f"catalog runner table unavailable: {type(exc).__name__}: {exc}"]

    for benchmark_id, spec in official_runner_registry().items():
        if spec.script and not spec.workspace_dispatch and spec.module != script_to_module(spec.script):
            issues.append(
                f"{benchmark_id}: workspace module {spec.module!r} does not match derived "
                f"{script_to_module(spec.script)!r}"
            )
    if "pawbench" not in VIDEO_RUNNER_REGISTRY:
        issues.append("pawbench must stay a specialized video runner")
    if official_runner_spec("pawbench") is None or official_runner_spec("pawbench").result_dispatch == "embodied":
        issues.append("pawbench must not be routed as embodied/generic")
    if set(supported_in_tree_benchmark_ids()) != set(scoring_benchmark_ids("in_tree")):
        issues.append("in-tree registry membership is not sourced from runner.scoring.engines")
    if set(supported_video_quality_benchmark_ids()) != set(scoring_benchmark_ids("video_quality")):
        issues.append("video-quality membership is not sourced from runner.scoring.engines")
    if set(scoring_benchmark_ids("video_quality")) != set(_BLOCKED_REASON):
        issues.append("video-quality catalog engines do not match video_quality_specs engine config")
    physics_ids = set(scoring_benchmark_ids("physics_contract"))
    for benchmark_id in sorted(physics_ids):
        if not has_benchmark_contract_evaluator(benchmark_id):
            issues.append(f"{benchmark_id}: physics_contract engine has no catalog-backed writer")
    extra_contract = [
        benchmark_id
        for benchmark_id in VIDEO_RUNNER_REGISTRY
        if has_benchmark_contract_evaluator(benchmark_id) and benchmark_id not in physics_ids
    ]
    for benchmark_id in extra_contract:
        issues.append(f"{benchmark_id}: contract evaluator is not declared in runner.scoring.engines")

    runners_root = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners"
    for path in sorted(runners_root.glob("*/README.md")) + sorted(runners_root.glob("*/readme.md")):
        issues.append(f"{_as_repo_relative(path)}: runner-root README must live in fumadocs, not runners/")
    for path in sorted(runners_root.glob("*/runtime/**/README.md")) + sorted(runners_root.glob("*/runtime/**/readme.md")):
        issues.append(f"{_as_repo_relative(path)}: vendor README must not remain under runners/*/runtime/")
    for path in sorted(runners_root.glob("*/runtime/**/requirements.txt")):
        issues.append(
            f"{_as_repo_relative(path)}: pip requirements must live under "
            "worldfoundry/data/benchmarks/requirements/"
        )
    issues.extend(extra_readme_issues())
    return issues
