"""Shared helpers for per-benchmark official video runners.

Each benchmark keeps its own folder under ``runners/<bench>/`` with bench-specific
discovery, parsing, and CLI flags.  This module holds generic scorecard assembly
and in-tree official-runtime subprocess execution.

Sections:

* **Config dataclasses** — :class:`BenchRunnerConfig`, :class:`RunnerHooks`.
* **Path resolution** — runtime root, generated videos, official results.
* **Metric extraction** — tabular/generic parsers and catalog fallback.
* **Pipeline** — :func:`run_official_pipeline` and :func:`run_main` CLI entry.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    mean_numeric,
    normalize_unit_score,
    scalar_number,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from worldfoundry.evaluation.tasks.execution.framework.result_normalizer import OfficialResultsNormalizer
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, REPO_ROOT

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"

ExtractMetricsFn = Callable[[Any, Path], dict[str, dict[str, Any]]]
DiscoverResultsFn = Callable[[Path, Path | None], Path | None]
BuildCommandFn = Callable[..., list[str] | None]
ExtendParserFn = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class BenchRunnerConfig:
    """Static configuration shared by one official video runner."""

    benchmark_id: str
    display_name: str
    root_env: str
    results_path_env: str
    default_repo_subdir: str
    metric_order: tuple[str, ...]
    metric_specs: dict[str, dict[str, Any]]
    metric_aliases: dict[str, str]
    average_metric_id: str
    official_entry: str | None = None
    official_output_globs: tuple[str, ...] = ()
    generated_video_envs: tuple[str, ...] = ("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR",)
    requires_api_env: tuple[str, ...] = ()
    usage_epilog: str = ""


PrepareResultsFn = Callable[["BenchRunnerConfig", Path, Any, Path], Path]


@dataclass
class RunnerHooks:
    """Per-benchmark hooks plugged into :func:`run_official_pipeline`."""

    build_official_command: BuildCommandFn
    discover_official_results: DiscoverResultsFn | None = None
    extract_metrics: ExtractMetricsFn | None = None
    prepare_upstream_results: PrepareResultsFn | None = None
    extend_parser: ExtendParserFn | None = None


# ---------------------------------------------------------------------------
# Key normalization and path resolution
# ---------------------------------------------------------------------------


def canonical_key(value: Any) -> str:
    """Normalize metric or alias keys to lowercase underscore form."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())).strip("_")


def first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_repo_root(config: BenchRunnerConfig, explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_dir():
        return explicit.expanduser().resolve()
    env_root = env_path(config.root_env)
    if env_root is not None and env_root.is_dir():
        return env_root.expanduser().resolve()
    if config.default_repo_subdir:
        configured = Path(config.default_repo_subdir).expanduser()
        if configured.is_absolute() and configured.is_dir():
            return configured.resolve()
        in_tree = REPO_ROOT / configured
        if in_tree.is_dir():
            return in_tree.resolve()
    return None


def command_pythonpath(runtime_root: Path) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for path in (runtime_root, REPO_ROOT):
        resolved = str(path.resolve())
        if resolved not in seen:
            parts.append(resolved)
            seen.add(resolved)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def resolve_generated_video_dir(config: BenchRunnerConfig, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for name in config.generated_video_envs:
        candidate = env_path(name)
        if candidate is not None:
            return candidate.expanduser().resolve()
    return None


def resolve_results_path(config: BenchRunnerConfig, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return env_path(config.results_path_env)


def load_upstream_payload(path: Path) -> tuple[Any, str]:
    if path.is_dir():
        for suffix in (".json", ".jsonl", ".csv", ".tsv", ".xlsx", ".xls"):
            matches = sorted(path.rglob(f"*{suffix}"))
            if matches:
                return load_upstream_payload(matches[0])
        raise FileNotFoundError(f"no supported result files found under: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8")), "json"
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows, "jsonl"
    if suffix in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)], suffix.lstrip(".")
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("xlsx normalization requires pandas") from exc
        frame = pd.read_excel(path)
        return [dict(row) for row in frame.to_dict(orient="records")], "xlsx"
    raise ValueError(f"unsupported upstream result format: {path}")


def metric_id_from_key(key: Any, config: BenchRunnerConfig) -> str | None:
    normalized = canonical_key(key)
    if normalized in config.metric_aliases:
        return config.metric_aliases[normalized]
    if normalized in config.metric_specs:
        return normalized
    for alias, metric_id in config.metric_aliases.items():
        if alias in normalized or normalized in alias:
            return metric_id
    return None


def generic_extract_metrics(payload: Any, config: BenchRunnerConfig, source: str) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}

    def add(metric_id: str, raw_score: float | None, *, sample_count: int | None = None) -> None:
        if raw_score is None or metric_id in extracted:
            return
        normalized = normalize_unit_score(raw_score)
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_score,
            "normalized_score": normalized,
            "source": source,
            "sample_count": sample_count,
        }

    if isinstance(payload, list):
        buckets: dict[str, list[float]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                metric_id = metric_id_from_key(key, config)
                if metric_id is None:
                    continue
                score = scalar_number(value)
                if score is not None:
                    buckets.setdefault(metric_id, []).append(score)
        for metric_id, values in buckets.items():
            add(metric_id, mean_numeric(values), sample_count=len(values))
        return extracted

    if isinstance(payload, dict):
        for key, value in payload.items():
            metric_id = metric_id_from_key(key, config)
            if metric_id is not None:
                add(metric_id, scalar_number(value))
        for container_key in ("metrics", "leaderboard", "leaderboard_metrics", "scores", "results", "summary"):
            nested = payload.get(container_key)
            if isinstance(nested, dict):
                extracted.update(generic_extract_metrics(nested, config, source))
    return extracted


def catalog_fallback(
    config: BenchRunnerConfig,
    results_path: Path,
    extracted: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if extracted:
        return extracted
    try:
        entry = load_benchmark_zoo_registry(BENCHMARK_ZOO_DIR).get(config.benchmark_id)
    except Exception:
        return extracted
    normalization = OfficialResultsNormalizer.from_benchmark_entry(entry).normalize_file(str(results_path))
    for metric_id, row in normalization.scorecard_metrics().items():
        if not isinstance(row, dict):
            continue
        raw_score = scalar_number(row.get("raw_score") if row.get("raw_score") is not None else row.get("score"))
        normalized = row.get("normalized_score")
        if normalized is None:
            normalized = normalize_unit_score(raw_score)
        if raw_score is None and normalized is None:
            continue
        extracted[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_score if raw_score is not None else normalized,
            "normalized_score": normalized,
            "source": "official_results_normalizer",
            "sample_count": row.get("sample_count"),
        }
    return extracted


def compute_average(config: BenchRunnerConfig, extracted: dict[str, dict[str, Any]]) -> None:
    existing = extracted.get(config.average_metric_id)
    if existing is not None and existing.get("normalized_score") is not None:
        return
    component_ids = [metric_id for metric_id in config.metric_order if metric_id != config.average_metric_id]
    values = []
    for metric_id in component_ids:
        item = extracted.get(metric_id)
        if item is None or item.get("normalized_score") is None:
            return
        values.append(item["normalized_score"])
    average = mean_numeric(values)
    if average is not None:
        extracted[config.average_metric_id] = {
            "metric_id": config.average_metric_id,
            "raw_score": average,
            "normalized_score": average,
            "source": "mean_complete_component_metrics",
            "sample_count": len(values),
        }


def metric_row(
    metric_id: str,
    raw_score: float,
    *,
    source: str,
    sample_count: int | None = None,
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "raw_score": raw_score,
        "normalized_score": normalize_unit_score(raw_score),
        "source": source,
        "sample_count": sample_count,
    }


def merge_metric_id_score_rows(
    extracted: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    config: BenchRunnerConfig,
    results_path: Path,
) -> None:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        metric_id = metric_id_from_key(
            row.get("metric_id") or row.get("metric") or row.get("name"),
            config,
        )
        score = scalar_number(row.get("score") if row.get("score") is not None else row.get("value"))
        if metric_id is None or score is None:
            continue
        buckets.setdefault(metric_id, []).append(score)
    for metric_id, values in buckets.items():
        extracted[metric_id] = metric_row(
            metric_id,
            mean_numeric(values),
            source=str(results_path),
            sample_count=len(values),
        )


def apply_component_aggregates(benchmark_id: str, extracted: dict[str, dict[str, Any]]) -> None:
    from worldfoundry.evaluation.tasks.execution.framework.video_quality_registry import (
        get_video_quality_benchmark_config,
    )

    components = get_video_quality_benchmark_config(benchmark_id).get("aggregate_components", {})
    if not components:
        return
    available = {
        metric_id: float(item["normalized_score"])
        for metric_id, item in extracted.items()
        if item.get("normalized_score") is not None
    }
    changed = True
    while changed:
        changed = False
        for metric_id, component_ids in components.items():
            if metric_id in available:
                continue
            if not all(component_id in available for component_id in component_ids):
                continue
            score = mean_numeric([available[component_id] for component_id in component_ids])
            if score is None:
                continue
            extracted[metric_id] = metric_row(
                metric_id,
                score,
                source="component_aggregate",
                sample_count=len(component_ids),
            )
            available[metric_id] = score
            changed = True


def extract_tabular_official_metrics(
    payload: Any,
    results_path: Path,
    config: BenchRunnerConfig,
) -> dict[str, dict[str, Any]]:
    extracted = generic_extract_metrics(payload, config, str(results_path))
    if isinstance(payload, list):
        merge_metric_id_score_rows(extracted, [row for row in payload if isinstance(row, dict)], config, results_path)
    apply_component_aggregates(config.benchmark_id, extracted)
    return extracted


def normalizer_only_hooks(
    *,
    extract_metrics: ExtractMetricsFn,
    discover_official_results: DiscoverResultsFn | None = None,
    prepare_upstream_results: PrepareResultsFn | None = None,
    extend_parser: ExtendParserFn | None = None,
) -> RunnerHooks:
    return RunnerHooks(
        build_official_command=lambda **kwargs: None,
        discover_official_results=discover_official_results,
        extract_metrics=extract_metrics,
        prepare_upstream_results=prepare_upstream_results,
        extend_parser=extend_parser,
    )


def build_runner_config_from_contract(
    benchmark_id: str,
    *,
    root_env: str,
    results_path_env: str,
    default_repo_subdir: str,
    official_output_globs: tuple[str, ...] = (),
    official_entry: str | None = None,
    lower_is_better: frozenset[str] | None = None,
    metric_groups: dict[str, str] | None = None,
    sample_filename: str | None = None,
) -> BenchRunnerConfig:
    from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
    from worldfoundry.evaluation.tasks.execution.framework.video_quality_registry import (
        get_video_quality_benchmark_config,
    )

    contract = get_external_benchmark_contract(benchmark_id)
    config = get_video_quality_benchmark_config(benchmark_id)
    components = config.get("aggregate_components", {})
    metric_order: list[str] = []
    seen: set[str] = set()

    def add(metric_id: str) -> None:
        if metric_id not in seen:
            seen.add(metric_id)
            metric_order.append(metric_id)

    for aggregate_id, component_ids in components.items():
        for component_id in component_ids:
            add(component_id)
        add(aggregate_id)
    for metric_id in contract.metric_ids:
        add(metric_id)

    average_metric_id = next(
        (metric_id for metric_id in reversed(metric_order) if metric_id.endswith("_average")),
        f"{benchmark_id.replace('-', '_')}_average",
    )
    lower = lower_is_better or frozenset()
    groups = metric_groups or {}

    def metric_group(metric_id: str) -> str:
        if metric_id in groups:
            return groups[metric_id]
        if metric_id in components:
            return "aggregate" if metric_id.endswith("_average") else metric_id
        for aggregate_id, component_ids in components.items():
            if metric_id in component_ids:
                return aggregate_id
        return "official"

    metric_specs = {
        metric_id: {
            "name": metric_id.replace("_", " ").title(),
            "group": metric_group(metric_id),
            "higher_is_better": metric_id not in lower,
        }
        for metric_id in metric_order
    }
    sample_display_name = (
        f"sample_results{Path(sample_filename).suffix}"
        if sample_filename is not None
        else "sample_results.*"
    )
    return BenchRunnerConfig(
        benchmark_id=benchmark_id,
        display_name=contract.display_name,
        root_env=root_env,
        results_path_env=results_path_env,
        default_repo_subdir=default_repo_subdir,
        metric_order=tuple(metric_order),
        metric_specs=metric_specs,
        metric_aliases=dict(config.get("metric_aliases", {})),
        average_metric_id=average_metric_id,
        official_entry=official_entry,
        official_output_globs=official_output_globs,
        usage_epilog=f"Checked-in sample: worldfoundry/data/benchmarks/assets/{benchmark_id}/{sample_display_name}",
    )


def build_scorecard(
    *,
    config: BenchRunnerConfig,
    output_dir: Path,
    results_path: Path,
    extracted: dict[str, dict[str, Any]],
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int | None,
    blocked_reasons: list[str] | None = None,
    repo_root: Path | None = None,
    generated_video_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_scores_path = output_dir / "per_sample_scores.jsonl"

    compute_average(config, extracted)
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in config.metric_order:
        item = extracted.get(metric_id, {})
        meta = config.metric_specs.get(metric_id, {})
        normalized_score = item.get("normalized_score")
        row = {
            "metric_id": metric_id,
            "name": meta.get("name", metric_id),
            "available": normalized_score is not None,
            "raw_score": item.get("raw_score"),
            "normalized_score": normalized_score,
            "higher_is_better": meta.get("higher_is_better", True),
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
            "group": meta.get("group"),
        }
        if normalized_score is None:
            row["reason"] = "score_not_found_in_upstream_results"
        else:
            leaderboard[metric_id] = float(normalized_score)
        metric_rows.append(row)
        per_metric[metric_id] = row

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_scores_path, [])

    available_count = sum(1 for row in metric_rows if row["available"])
    normalizer_only = command is None and not blocked_reasons
    official_runtime_succeeded = command is not None and returncode == 0 and available_count > 0
    run_status = (
        "blocked"
        if blocked_reasons
        else "official_bounded"
        if official_runtime_succeeded
        else "normalized"
        if available_count
        else "failed"
    )
    runner_name = f"benchmark_zoo_{config.benchmark_id.replace('-', '_')}_official_runner"
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": runner_name,
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": config.benchmark_id,
            "name": config.display_name,
            "contract_only": False,
            "requires_upstream_runtime": True,
        },
        "dataset": {
            "upstream_results": str(results_path.resolve()),
            "repo_root": None if repo_root is None else str(repo_root),
            "generated_artifact_dir": None if generated_video_dir is None else str(generated_video_dir),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official benchmark validation; leaderboard parity requires full benchmark assets, generated videos, and judge/checkpoint configuration",
            ],
        },
        "metrics": {
            "leaderboard": leaderboard,
            "per_metric": per_metric,
            "summary": {
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": available_count > 0 and not blocked_reasons,
            "kind": f"official_{config.benchmark_id.replace('-', '_')}",
            "upstream_results": str(results_path.resolve()),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_runtime_succeeded": official_runtime_succeeded,
            "full_suite_complete": False,
            "scope": "bounded_or_unproven" if official_runtime_succeeded else "not_executed",
            "blocked_reasons": blocked_reasons or [],
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_scores_path.resolve()),
            "upstream_results": str(results_path.resolve()),
            "upstream_stdout": str((output_dir / "upstream_stdout.log").resolve()),
            "upstream_stderr": str((output_dir / "upstream_stderr.log").resolve()),
        },
        # The shared runner cannot infer canonical prompt/sample/metric coverage.
        # Benchmark-specific runners may make a full-suite claim only after an
        # explicit coverage check; a successful component run proves integration.
        "official_benchmark_verified": False,
        "integration_evidence": official_runtime_succeeded,
        "normalizer_only": normalizer_only,
        "normalization_ok": available_count > 0,
        "official_results_imported": normalizer_only and available_count > 0,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def discover_by_globs(search_roots: list[Path], patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            if "*" in pattern:
                matches = [Path(path) for path in glob.glob(str(root / pattern), recursive=True)]
            else:
                matches = sorted(root.glob(pattern))
            candidates.extend(path for path in matches if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def default_discover_official_results(
    config: BenchRunnerConfig,
    output_dir: Path,
    repo_root: Path | None,
) -> Path | None:
    if not config.official_output_globs:
        return None
    search_roots = [output_dir]
    if repo_root is not None:
        search_roots.append(repo_root)
    return discover_by_globs(search_roots, config.official_output_globs)


def missing_api_requirements(config: BenchRunnerConfig) -> list[str]:
    if not config.requires_api_env or first_env_value(*config.requires_api_env):
        return []
    return [f"one of {', '.join(config.requires_api_env)}"]


# ---------------------------------------------------------------------------
# Official pipeline and CLI
# ---------------------------------------------------------------------------


def run_official_pipeline(
    *,
    config: BenchRunnerConfig,
    hooks: RunnerHooks,
    args: Any,
) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    repo_root = resolve_repo_root(config, args.repo_root)
    command_root = repo_root or REPO_ROOT
    generated_video_dir = resolve_generated_video_dir(config, args.generated_video_dir)
    results_path = resolve_results_path(config, args.from_upstream_results)
    if results_path is not None and hooks.prepare_upstream_results is not None:
        results_path = hooks.prepare_upstream_results(config, results_path, args, output_dir)

    extract_fn = hooks.extract_metrics or (
        lambda payload, path: generic_extract_metrics(payload, config, str(path))
    )

    if results_path is not None:
        payload, _fmt = load_upstream_payload(results_path)
        extracted = extract_fn(payload, results_path)
        extracted = catalog_fallback(config, results_path, extracted)
        return build_scorecard(
            config=config,
            output_dir=output_dir,
            results_path=results_path,
            extracted=extracted,
            command=None,
            duration_seconds=None,
            returncode=0,
            repo_root=repo_root,
            generated_video_dir=generated_video_dir,
        )

    blocked = missing_api_requirements(config)
    if config.official_entry is None:
        blocked.append("no public official runtime repository confirmed for this benchmark")
    if args.run_official and generated_video_dir is None:
        blocked.append("generated video directory is required for official runtime execution")
    if args.run_official and repo_root is None and (config.root_env or config.default_repo_subdir):
        root_hint = f"set {config.root_env}" if config.root_env else "check the in-tree runtime path"
        blocked.append(f"official runtime root not found; {root_hint}")

    command: list[str] | None = None
    duration_seconds: float | None = None
    returncode: int | None = None

    if args.run_official and not blocked and generated_video_dir is not None:
        command = hooks.build_official_command(
            config=config,
            repo_root=command_root,
            generated_video_dir=generated_video_dir,
            output_dir=output_dir,
            args=args,
        )
        if command is None:
            blocked.append("official command could not be resolved from repo assets and environment")
        else:
            env = os.environ.copy()
            env["PYTHONPATH"] = command_pythonpath(command_root)
            if generated_video_dir is not None:
                for env_name in config.generated_video_envs:
                    env[env_name] = str(generated_video_dir)
            start = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=command_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            duration_seconds = time.monotonic() - start
            returncode = completed.returncode
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if hooks.discover_official_results is not None:
                discovered = hooks.discover_official_results(output_dir, repo_root)
            else:
                discovered = default_discover_official_results(config, output_dir, repo_root)
            if discovered is not None:
                results_path = discovered
            elif returncode != 0:
                blocked.append(f"official command failed with exit code {returncode}")

    if results_path is None:
        placeholder = output_dir / "upstream" / "blocked_results.json"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        write_json(placeholder, {"blocked": True, "blocked_reasons": blocked or ["missing upstream results path"]})
        return build_scorecard(
            config=config,
            output_dir=output_dir,
            results_path=placeholder,
            extracted={},
            command=command,
            duration_seconds=duration_seconds,
            returncode=returncode if returncode is not None else 1,
            blocked_reasons=blocked or ["missing upstream results"],
            repo_root=repo_root,
            generated_video_dir=generated_video_dir,
        )

    payload, _fmt = load_upstream_payload(results_path)
    extracted = extract_fn(payload, results_path)
    extracted = catalog_fallback(config, results_path, extracted)
    return build_scorecard(
        config=config,
        output_dir=output_dir,
        results_path=results_path,
        extracted=extracted,
        command=command,
        duration_seconds=duration_seconds,
        returncode=returncode if returncode is not None else 0,
        blocked_reasons=blocked if blocked and not extracted else None,
        repo_root=repo_root,
        generated_video_dir=generated_video_dir,
    )


def build_common_parser(config: BenchRunnerConfig, hooks: RunnerHooks) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run or normalize official {config.display_name} results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=config.usage_epilog or None,
    )
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", config.benchmark_id))
    parser.add_argument(
        "--official-results-path",
        "--results-path",
        dest="from_upstream_results",
        type=Path,
    )
    parser.add_argument("--generated-video-dir", type=Path, default=env_path(*config.generated_video_envs))
    repo_root_help = (
        f"defaults to ${config.root_env} or the in-tree runtime"
        if config.root_env
        else "optional in-tree runtime root for runners that use one"
    )
    parser.add_argument("--repo-root", type=Path, help=repo_root_help)
    run_official_help = (
        "run the configured in-tree official entrypoint"
        if config.official_entry
        else "build a scorecard from imported official results; no runner-local command is launched"
    )
    parser.add_argument("--run-official", action="store_true", help=run_official_help)
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="use checked-in sample results under worldfoundry/data/benchmarks/assets/<benchmark-id>/",
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_BENCHMARK_TIMEOUT", "7200")))
    parser.add_argument("--json", action="store_true")
    if hooks.extend_parser is not None:
        hooks.extend_parser(parser)
    return parser


def runner_result_payload(config: BenchRunnerConfig, scorecard: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    return {
        "ok": scorecard.get("official_benchmark_verified") and scorecard.get("integration_evidence"),
        "benchmark_id": config.benchmark_id,
        "output_dir": str(output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard.get("official_benchmark_verified"),
        "integration_evidence": scorecard.get("integration_evidence"),
        "normalization_ok": scorecard.get("normalization_ok"),
        "official_results_imported": scorecard.get("official_results_imported"),
    }


def run_main(config: BenchRunnerConfig, hooks: RunnerHooks, argv: list[str] | None = None) -> int:
    args = build_common_parser(config, hooks).parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    if getattr(args, "run_fixture", False):
        from worldfoundry.evaluation.utils import BENCHMARK_ASSETS_ROOT, benchmark_task_sample_path

        sample_path = benchmark_task_sample_path(config.benchmark_id)
        if sample_path is None:
            print(
                f"error: no checked-in sample for {config.benchmark_id} under {BENCHMARK_ASSETS_ROOT / config.benchmark_id}",
                file=sys.stderr,
            )
            return 2
        args.from_upstream_results = sample_path
    elif args.from_upstream_results is None and not args.run_official:
        env_results = resolve_results_path(config, None)
        if env_results is None:
            print(
                f"error: --official-results-path or {config.results_path_env} is required unless --run-official is set",
                file=sys.stderr,
            )
            return 2
        args.from_upstream_results = env_results
    try:
        scorecard = run_official_pipeline(config=config, hooks=hooks, args=args)
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = runner_result_payload(config, scorecard, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "ok" if result["ok"] or result["normalization_ok"] else "failed"
        print(f"{config.benchmark_id}: official validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] or result["normalization_ok"] else 1
