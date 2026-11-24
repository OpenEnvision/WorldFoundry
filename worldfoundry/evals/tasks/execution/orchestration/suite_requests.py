"""Request, result, and plan records for the model × benchmark suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.catalog.schema import BenchmarkZooEntry
from worldfoundry.evals.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_ZOO_DIR,
)

MODEL_BENCHMARK_SUITE_SCHEMA_VERSION = "worldfoundry-model-benchmark-suite"


MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION = "worldfoundry-model-benchmark-suite-result"


MODEL_BENCHMARK_SUITE_SCORECARDS_SCHEMA_VERSION = "worldfoundry-model-benchmark-suite-scorecards"


DEFAULT_BENCHMARK_ZOO_DIR = BENCHMARK_ZOO_DIR


DEFAULT_MODEL_ZOO_DIR = MODEL_ZOO_DIR


DEFAULT_SUITE_PRESET_PATH: Path | None = None


# Artifact kinds used when matching model outputs to benchmark inputs.
_GENERIC_OUTPUT_ARTIFACTS = (
    "generated_video",
    "predicted_video",
    "generated_world",
    "generated_3d_asset",
    "generated_4d_scene",
    "action_trace",
    "actions",
    "rollout_video",
    "action_tokens",
)


@dataclass(frozen=True)
class ModelBenchmarkSuiteRequest:
    """Configuration for a model × benchmark matrix sweep."""

    output_dir: str | Path
    benchmark_manifest_dir: str | Path = DEFAULT_BENCHMARK_ZOO_DIR
    model_manifest_dir: str | Path | None = DEFAULT_MODEL_ZOO_DIR
    suite_ids: Sequence[str] = ()
    suite_preset_path: str | Path | None = None
    model_ids: Sequence[str] = ()
    benchmark_ids: Sequence[str] = ()
    benchmark_integration_status: str | None = None
    model_integration_status: str | None = None
    mode: str = "official-run"
    execute: bool = True
    model_workers: int = 1
    worker_cuda_devices: Sequence[str] = ()
    skip_incompatible: bool = True
    fail_on_skipped: bool = False
    model_runner: str | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    requests_path: str | Path | None = None
    task_name: str | None = None
    task_roots: Sequence[str | Path] | None = None
    task_benchmark: str | None = None
    task_recursive: bool = False
    task_root_dir: str | Path | None = None
    dataset_root: str | Path | None = None
    dataset_id: str | None = None
    split: str = "default"
    num_samples: int | None = None
    generated_artifact_dir: str | Path | None = None
    output_artifact: str | None = None
    required_artifacts: Sequence[str] | None = None
    metrics: Sequence[str] = ("artifact_count", "required_artifacts_present")
    metric_batch_size: int = 32
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "model_benchmark_suite"
    benchmark_timeout_seconds: float | None = None
    benchmark_workdir: str | Path | None = None
    benchmark_env: Mapping[str, Any] | None = None
    benchmark_parameters: Mapping[str, Any] | None = None
    materialize_placeholders: bool | None = None
    contract_fixture: bool = False
    fail_on_generation_error: bool = False
    run_id: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class ModelBenchmarkSuiteResult:
    """Aggregated suite status, cell records, and artifact paths."""

    schema_version: str
    status: str
    exit_code: int
    run_fingerprint: str
    output_dir: Path
    suite_manifest_path: Path
    suite_report_path: Path
    summary: Mapping[str, Any]
    cells: Sequence[Mapping[str, Any]]
    artifacts: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        """Return True when ``exit_code == 0``."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize suite result to a plain dict."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "run_fingerprint": self.run_fingerprint,
            "output_dir": str(self.output_dir),
            "suite_manifest_path": str(self.suite_manifest_path),
            "suite_report_path": str(self.suite_report_path),
            "summary": dict(self.summary),
            "cells": [dict(cell) for cell in self.cells],
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class _SuiteCellPlan:
    """Internal plan for one model × benchmark matrix cell."""

    model_id: str
    requested_model_id: str
    known_model: bool
    benchmark: BenchmarkZooEntry
    benchmark_manifest_path: Path
    model_output_artifacts: tuple[str, ...]
    benchmark_acceptable_artifacts: tuple[str, ...]
    output_artifact: str | None
    required_artifacts: tuple[str, ...]
    compatibility: str
    reason: str | None
    evaluation_provenance: Mapping[str, Any]
    cell_fingerprint: str

    def to_base_cell(self) -> dict[str, Any]:
        """Export stable cell metadata for suite manifests."""
        return {
            "model_id": self.model_id,
            "requested_model_id": self.requested_model_id,
            "benchmark_id": self.benchmark.benchmark_id,
            "benchmark_manifest_path": str(self.benchmark_manifest_path),
            "model_output_artifacts": list(self.model_output_artifacts),
            "benchmark_acceptable_artifacts": list(self.benchmark_acceptable_artifacts),
            "output_artifact": self.output_artifact,
            "required_artifacts": list(self.required_artifacts),
            "compatibility": self.compatibility,
            "provenance": dict(self.evaluation_provenance),
            "cell_fingerprint": self.cell_fingerprint,
        }


@dataclass(frozen=True)
class _ModelWorkerPlan:
    """Resolved process count and immutable CUDA affinity for suite workers."""

    requested_workers: int
    worker_count: int
    cuda_device_groups: tuple[str | None, ...]
    device_source: str
    use_spawn_workers: bool
    cpu_threads_per_worker: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "spawn_per_model" if self.use_spawn_workers else "in_process",
            "start_method": "spawn" if self.use_spawn_workers else None,
            "requested_workers": self.requested_workers,
            "worker_count": self.worker_count,
            "cuda_device_groups": [group for group in self.cuda_device_groups if group is not None],
            "device_source": self.device_source,
            "cpu_threads_per_worker": self.cpu_threads_per_worker,
        }
