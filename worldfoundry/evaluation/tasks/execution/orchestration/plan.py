"""Specification and generation of evaluation RunPlans.

Provides standard definitions and parsers for declarative WorldFoundry execution run plans.
A RunPlan encapsulates the entire execution scope, defining tasks, models, datasets, and metrics.
Orchestrators parse RunPlans to materialize requests and execute model evaluations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.tasks.catalog.registry import TaskRegistryEntry, load_task_registry_from_paths
from worldfoundry.evaluation.tasks.datasets import (
    load_dataset_manifest,
    read_dataset_samples,
    resolve_dataset_samples_path,
    validate_dataset_manifest,
)
from worldfoundry.evaluation.tasks.execution.framework.in_tree_registry import target_benchmark_metrics
from worldfoundry.evaluation.tasks.metrics.registry import validate_metric_ids
from worldfoundry.evaluation.utils import jsonable, read_json_or_jsonl, write_json

from .cache import json_sha256
from .evaluate import EVALUATE_RUN_REQUEST_SCHEMA_VERSION, EvaluateRunRequest
from .materialize import materialize_generation_requests

# Canonical schema version for RunPlan structure tracking
RUN_PLAN_SCHEMA_VERSION = "worldfoundry-run-plan"


def _task_payload(entry: TaskRegistryEntry | None) -> dict[str, Any] | None:
    """Transforms a TaskRegistryEntry into a serializable dictionary format."""
    if entry is None:
        return None
    payload = entry.to_dict()
    payload["task_config"] = entry.task.to_dict()
    return payload


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    """Coerces various collection types or single strings into a tuple of clean strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _materialize_requests_from_task(plan: "RunPlan") -> tuple[GenerationRequest, ...] | None:
    """Processes task parameters and datasets from a RunPlan to materialize GenerationRequests.

    Locates the underlying dataset manifest or metadata file, reads the raw sample records,
    and formats them into strongly-typed GenerationRequests suitable for Model evaluation.
    """
    if not plan.task:
        return None
    task_config = plan.task.get("task_config")
    if not isinstance(task_config, Mapping):
        return None
    data = task_config.get("data")
    if not isinstance(data, Mapping):
        return None
    manifest_path = plan.dataset.get("manifest_path") or plan.materialization.get("dataset_manifest")
    if manifest_path:
        dataset_manifest = load_dataset_manifest(manifest_path)
        source_path = resolve_dataset_samples_path(dataset_manifest, manifest_path=manifest_path)
        samples = read_dataset_samples(source_path)
    else:
        if not plan.dataset.get("root"):
            return None
        metadata_path = data.get("metadata_path")
        if not metadata_path:
            return None

        dataset_root = Path(str(plan.dataset["root"]))
        source_path = Path(str(metadata_path))
        if not source_path.is_absolute():
            source_path = dataset_root / source_path
        samples = read_json_or_jsonl(source_path)
        if isinstance(samples, Mapping):
            samples = samples.get("samples", samples.get("items", samples.get("data", [])))
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError(f"task metadata source must contain a sequence of samples: {source_path}")
    limit = plan.materialization.get("limit")
    if limit is not None:
        samples = samples[: int(limit)]

    return materialize_generation_requests(
        samples,
        task_name=str(task_config.get("name") or plan.task.get("task_name") or "worldfoundry_task"),
        split=str(plan.dataset.get("split") or plan.materialization.get("split") or "default"),
        input_keys=_coerce_str_tuple(task_config.get("input_keys")),
        output_keys=_coerce_str_tuple(task_config.get("output_keys")) or ("generated_video",),
        generation_defaults=(
            task_config.get("generation_defaults")
            if isinstance(task_config.get("generation_defaults"), Mapping)
            else {}
        ),
        cache_policy=(
            plan.materialization.get("cache_policy")
            if isinstance(plan.materialization.get("cache_policy"), Mapping)
            else {}
        ),
    )


@dataclass(frozen=True)
class RunPlan:
    """A declarative blueprint defining the complete execution parameters for an evaluation run.

    Attributes:
        runner: Core orchestrating runner type (typically "evaluate").
        mode: Run mode (either "model" or "existing-results").
        output_dir: Target directory path where scores and scorecard are written.
        task: Structured parameters of the target task.
        model: Dictionary of model information and configurations.
        dataset: Dictionary of dataset information and paths.
        metrics: Tuple of metric IDs to evaluate.
        required_artifacts: Tuple of target file kinds expected to be produced.
        requests: Materialized list of generation requests.
        requests_path: Optional path to a pre-saved requests list.
        results_path: Optional path to pre-generated results under evaluation.
        materialization: Parameters governing task data resolution.
        run_id: Unique trace ID for this run.
        fail_on_sample_error: If True, halts execution immediately upon any evaluation error.
        write_artifacts_index: If True, outputs flat list indexing of all generated files.
        generation_cache_dir: Storage path for SQLite generation caching.
        generation_cache_mode: Cache mode ('off', 'read', 'write', 'read-write', 'refresh').
        generation_cache_namespace: Namespace partitioning keys.
        metadata: Arbitrary user-defined context dictionary.
        schema_version: Standard schema identification string.
    """
    runner: str
    mode: str
    output_dir: str
    task: Mapping[str, Any] | None = None
    model: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)
    metrics: tuple[str, ...] = ("artifact_count",)
    required_artifacts: tuple[str, ...] = ()
    requests: tuple[Any, ...] = ()
    requests_path: str | None = None
    results_path: str | None = None
    materialization: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    fail_on_sample_error: bool = False
    write_artifacts_index: bool = True
    generation_cache_dir: str | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "run_plan"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RUN_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validates schema versions and normalizes collections into standardized frozen tuples."""
        if self.schema_version != RUN_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported RunPlan schema_version: {self.schema_version}")
        object.__setattr__(self, "runner", str(self.runner))
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "output_dir", str(self.output_dir))
        object.__setattr__(self, "metrics", _coerce_str_tuple(self.metrics))
        object.__setattr__(self, "required_artifacts", _coerce_str_tuple(self.required_artifacts))
        object.__setattr__(self, "requests", tuple(jsonable(item) for item in (self.requests or ())))
        object.__setattr__(self, "model", dict(self.model or {}))
        object.__setattr__(self, "dataset", dict(self.dataset or {}))
        object.__setattr__(self, "materialization", dict(self.materialization or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def fingerprint(self) -> str:
        """Computes a unique SHA-256 fingerprint representing the exact configuration hash of this plan."""
        return str(self.to_dict()["fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        """Serializes the RunPlan into a standardized dictionary containing its configuration fingerprint."""
        payload = {
            "schema_version": self.schema_version,
            "runner": self.runner,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "task": jsonable(self.task),
            "model": jsonable(self.model),
            "dataset": jsonable(self.dataset),
            "metrics": list(self.metrics),
            "required_artifacts": list(self.required_artifacts),
            "requests": list(self.requests),
            "requests_path": self.requests_path,
            "results_path": self.results_path,
            "materialization": jsonable(self.materialization),
            "run_id": self.run_id,
            "fail_on_sample_error": self.fail_on_sample_error,
            "write_artifacts_index": self.write_artifacts_index,
            "generation_cache_dir": self.generation_cache_dir,
            "generation_cache_mode": self.generation_cache_mode,
            "generation_cache_namespace": self.generation_cache_namespace,
            "metadata": jsonable(self.metadata),
        }
        payload["fingerprint"] = json_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunPlan":
        """Reconstructs a RunPlan instance from a configuration dictionary, applying safe defaults."""
        return cls(
            runner=str(data.get("runner", "evaluate")),
            mode=str(data.get("mode", "existing-results")),
            output_dir=str(data["output_dir"]),
            task=data.get("task") if isinstance(data.get("task"), Mapping) else None,
            model=data.get("model") if isinstance(data.get("model"), Mapping) else {},
            dataset=data.get("dataset") if isinstance(data.get("dataset"), Mapping) else {},
            metrics=tuple(data.get("metrics") or ("artifact_count",)),
            required_artifacts=tuple(data.get("required_artifacts") or ()),
            requests=tuple(data.get("requests") or ()),
            requests_path=data.get("requests_path"),
            results_path=data.get("results_path"),
            materialization=data.get("materialization") if isinstance(data.get("materialization"), Mapping) else {},
            run_id=data.get("run_id"),
            fail_on_sample_error=bool(data.get("fail_on_sample_error", False)),
            write_artifacts_index=bool(data.get("write_artifacts_index", True)),
            generation_cache_dir=data.get("generation_cache_dir"),
            generation_cache_mode=str(data.get("generation_cache_mode", "off")),
            generation_cache_namespace=str(data.get("generation_cache_namespace", "run_plan")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {},
            schema_version=str(data.get("schema_version", RUN_PLAN_SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, payload: str) -> "RunPlan":
        """Loads and reconstructs a RunPlan from a raw JSON string."""
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("RunPlan JSON must be an object")
        return cls.from_dict(data)


def load_run_plan(path: str | Path) -> RunPlan:
    """Reads and parses a RunPlan directly from a JSON file on disk."""
    return RunPlan.from_json(Path(path).read_text(encoding="utf-8"))


def write_run_plan(plan: RunPlan, path: str | Path) -> Path:
    """Writes a RunPlan instance to a specified JSON file on disk."""
    destination = Path(path)
    write_json(destination, plan.to_dict(), atomic=False)
    return destination


def build_run_plan(
    *,
    output_dir: str | Path,
    mode: str = "existing-results",
    task_entry: TaskRegistryEntry | None = None,
    dataset_root: str | Path | None = None,
    dataset_manifest: str | Path | None = None,
    dataset_id: str | None = None,
    split: str = "default",
    requests_path: str | Path | None = None,
    results_path: str | Path | None = None,
    model_id: str | None = None,
    model_runner: str | None = None,
    model_manifest_dir: str | Path | None = None,
    model_variant_id: str | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    model_runtime: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    metrics: Sequence[str] = ("artifact_count",),
    required_artifacts: Sequence[str] = (),
    limit: int | None = None,
    materialize_requests: bool = False,
    run_id: str | None = None,
    fail_on_sample_error: bool = False,
    write_artifacts_index: bool = True,
    generation_cache_dir: str | Path | None = None,
    generation_cache_mode: str = "off",
    generation_cache_namespace: str = "run_plan",
    metadata: Mapping[str, Any] | None = None,
) -> RunPlan:
    """Programmatically constructs a complete, cohesive RunPlan instance with all parameters populated.

    Validates, merges, and formats model configurations, dataset structures, and evaluation boundaries
    into a structured RunPlan. If `materialize_requests` is True, reads samples and populates
    requests inline.
    """
    model = {
        "model_id": model_id,
        "model_runner": model_runner,
        "model_manifest_dir": None if model_manifest_dir is None else str(model_manifest_dir),
        "model_variant_id": model_variant_id,
        "model_parameters": dict(model_parameters or {}),
        "model_runtime": dict(model_runtime or {}),
        "model_config": dict(model_config or {}) if model_config is not None else None,
    }
    dataset_manifest_payload: dict[str, Any] = {}
    if dataset_manifest is not None:
        manifest = load_dataset_manifest(dataset_manifest)
        dataset_manifest_payload = {
            "manifest_path": str(dataset_manifest),
            "dataset_id": manifest.dataset_id,
            "split": manifest.split,
            "root": manifest.root,
            "samples_path": manifest.samples_path,
            "sample_count": manifest.sample_count,
            "sha256": manifest.sha256,
            "sample_ids_sha256": manifest.sample_ids_sha256,
        }

    dataset = {
        "root": None if dataset_root is None else str(dataset_root),
        "dataset_id": dataset_id,
        "split": split,
        **dataset_manifest_payload,
    }
    if dataset_id is not None:
        dataset["dataset_id"] = dataset_id
    if split != "default" or "split" not in dataset:
        dataset["split"] = split
    materialization = {
        "source": (
            "dataset_manifest"
            if task_entry is not None and dataset_manifest is not None
            else "task_yaml"
            if task_entry is not None and dataset_root is not None
            else None
        ),
        "limit": limit,
        "split": split,
        "dataset_manifest": None if dataset_manifest is None else str(dataset_manifest),
    }
    plan = RunPlan(
        runner="evaluate",
        mode=mode,
        output_dir=str(output_dir),
        task=_task_payload(task_entry),
        model={key: value for key, value in model.items() if value is not None},
        dataset={key: value for key, value in dataset.items() if value is not None},
        metrics=tuple(metrics),
        required_artifacts=tuple(required_artifacts),
        requests_path=None if requests_path is None else str(requests_path),
        results_path=None if results_path is None else str(results_path),
        materialization={key: value for key, value in materialization.items() if value is not None},
        run_id=run_id,
        fail_on_sample_error=fail_on_sample_error,
        write_artifacts_index=write_artifacts_index,
        generation_cache_dir=None if generation_cache_dir is None else str(generation_cache_dir),
        generation_cache_mode=generation_cache_mode,
        generation_cache_namespace=generation_cache_namespace,
        metadata=dict(metadata or {}),
    )
    if materialize_requests and task_entry is not None and (
        dataset_root is not None or dataset_manifest is not None
    ):
        requests = tuple(request.to_dict() for request in (_materialize_requests_from_task(plan) or ()))
        plan = replace(plan, requests=requests)
    return plan


def build_run_plan_from_task_registry(
    *,
    task_name: str,
    task_roots: Sequence[str | Path],
    output_dir: str | Path,
    benchmark: str | None = None,
    recursive: bool = False,
    root_dir: str | Path | None = None,
    **kwargs: Any,
) -> RunPlan:
    """Resolves a task definition from local filesystem catalogs and programmatically builds its RunPlan."""
    registry = load_task_registry_from_paths(task_roots, recursive=recursive, root_dir=root_dir)
    entry = registry.get(task_name, benchmark=benchmark)
    return build_run_plan(output_dir=output_dir, task_entry=entry, **kwargs)


def evaluate_request_from_run_plan(plan: RunPlan) -> EvaluateRunRequest:
    """Translates a standardized RunPlan configuration into a concrete EvaluateRunRequest payload.

    Provides the primary bridging mechanism for executing evaluations planned inside RunPlan files.
    """
    if plan.runner != "evaluate":
        raise ValueError(f"unsupported run plan runner: {plan.runner}")
    model = dict(plan.model)
    dataset = dict(plan.dataset)
    requests = tuple(plan.requests) if plan.requests else _materialize_requests_from_task(plan)
    benchmark = None
    if plan.task:
        benchmark = {
            "benchmark_name": plan.task.get("benchmark_name"),
            "task_name": plan.task.get("task_name"),
            "protocol": plan.task.get("protocol"),
            "evaluation_protocol": plan.task.get("evaluation_protocol"),
            "source_path": plan.task.get("source_path"),
        }
    return EvaluateRunRequest(
        output_dir=plan.output_dir,
        mode=plan.mode,
        requests=requests,
        requests_path=plan.requests_path,
        results_path=plan.results_path,
        metrics=plan.metrics,
        required_artifacts=plan.required_artifacts,
        benchmark=benchmark,
        model={"model_id": model.get("model_id"), "model_runner": model.get("model_runner")},
        dataset=dataset,
        benchmark_id=(plan.task or {}).get("benchmark_name"),
        model_id=model.get("model_id"),
        model_runner=model.get("model_runner"),
        model_zoo_manifest_dir=model.get("model_manifest_dir"),
        model_variant_id=model.get("model_variant_id"),
        model_parameters=model.get("model_parameters"),
        model_runtime=model.get("model_runtime"),
        model_config=model.get("model_config"),
        dataset_id=dataset.get("dataset_id"),
        run_id=plan.run_id,
        fail_on_sample_error=plan.fail_on_sample_error,
        write_artifacts_index=plan.write_artifacts_index,
        generation_cache_dir=plan.generation_cache_dir,
        generation_cache_mode=plan.generation_cache_mode,
        generation_cache_namespace=plan.generation_cache_namespace,
        run_metadata=plan.metadata,
        schema_version=EVALUATE_RUN_REQUEST_SCHEMA_VERSION,
    )


def validate_run_plan(plan: RunPlan | Mapping[str, Any]) -> dict[str, Any]:
    """Validates structural integrity, metrics bindings, and dataset paths defined in a RunPlan.

    Performs comprehensive diagnostic checks:
    - Verifies runner type boundaries.
    - Validates presence of dataset files or requests.
    - Resolves and verifies correctness of all registered metrics.
    - Returns a list of structural warnings or issues found.
    """
    try:
        run_plan = plan if isinstance(plan, RunPlan) else RunPlan.from_dict(plan)
        issues: list[str] = []
        if run_plan.runner != "evaluate":
            issues.append(f"unsupported runner: {run_plan.runner}")
        if run_plan.mode in {"existing-results", "existing", "results"} and not run_plan.results_path:
            issues.append("existing-results mode requires results_path")
        if run_plan.mode in {"model", "generate"} and not (run_plan.requests or run_plan.requests_path or run_plan.task):
            issues.append("model mode requires requests, requests_path, or task materialization")
        if run_plan.task and run_plan.materialization.get("source") == "task_yaml" and not run_plan.dataset.get("root"):
            issues.append("task materialization from YAML requires dataset.root")
        dataset_validation = None
        manifest_path = run_plan.dataset.get("manifest_path") or run_plan.materialization.get("dataset_manifest")
        if manifest_path:
            dataset_validation = validate_dataset_manifest(manifest_path)
            if not dataset_validation["ok"]:
                for issue in dataset_validation["issues"]:
                    issues.append(f"dataset manifest: {issue}")
        task_config = run_plan.task.get("task_config") if isinstance(run_plan.task, Mapping) else {}
        benchmark_id = None
        if isinstance(task_config, Mapping):
            item = task_config.get("metadata", {}).get("benchmark_id") if isinstance(task_config.get("metadata"), Mapping) else None
            benchmark_id = item or task_config.get("benchmark_name")
        if benchmark_id is None and isinstance(run_plan.task, Mapping):
            benchmark_id = run_plan.task.get("benchmark_name")
        benchmark_key = None if benchmark_id is None else str(benchmark_id).strip().lower()
        in_tree_metrics = target_benchmark_metrics().get(benchmark_key or "", ())
        regular_metrics = tuple(metric for metric in run_plan.metrics if metric not in in_tree_metrics)
        in_tree_resolved = [
            {
                "metric_id": metric,
                "canonical_metric_id": metric,
                "registry_id": "benchmark_zoo_in_tree_evaluator",
                "parameterized": False,
            }
            for metric in run_plan.metrics
            if metric in in_tree_metrics
        ]
        metric_validation = validate_metric_ids(regular_metrics)
        metric_validation["metrics"] = [*in_tree_resolved, *metric_validation["metrics"]]
        metric_validation["ok"] = not metric_validation["unknown_metrics"]
        for metric_id in metric_validation["unknown_metrics"]:
            issues.append(f"unsupported metric: {metric_id!r}")
        return {
            "ok": not issues,
            "schema_version": run_plan.schema_version,
            "fingerprint": run_plan.fingerprint,
            "issues": issues,
            "metrics": metric_validation,
            "dataset": dataset_validation,
        }
    except Exception as exc:  # noqa: BLE001 - validation should return structured errors.
        return {
            "ok": False,
            "schema_version": None,
            "fingerprint": None,
            "issues": [f"{type(exc).__name__}: {exc}"],
        }
