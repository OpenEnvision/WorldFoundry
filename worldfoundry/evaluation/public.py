"""Stable open-source evaluation API for WorldFoundry.

Benchmark integration is pragmatic: scorer/normalizer logic lives under
``worldfoundry/evaluation/tasks/execution/runners/``.  Straightforward metrics run
in-tree; model-backed judges and reusable foundation models live under
``worldfoundry/base_models``.  Generate videos with any integrated WorldFoundry
model, then score through the same package.

Typical flow::

    from worldfoundry.evaluation.public import run_model_benchmark

    # 1) generate with an integrated model id from the model catalog
    # 2) score in-tree via the benchmark runner
    run_model_benchmark(..., benchmark_mode="official-run")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.evaluation.framework import WorldFoundryRunRequest, WorldFoundryRunResult, run_worldfoundry
from worldfoundry.evaluation.runner import (
    EvaluateRunRequest,
    EvaluateRunResult,
    ModelBenchmarkRunRequest,
    ModelBenchmarkRunResult,
    ModelBenchmarkSuiteRequest,
    ModelBenchmarkSuiteResult,
    execute_evaluate_run,
    run_model_benchmark,
    run_model_benchmark_suite,
)
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution
from worldfoundry.evaluation.tasks.catalog.dispatch import (
    BENCHMARK_INTEGRATION_REGISTRY,
    BenchmarkIntegrationSpec,
    IntegrationTier,
    VIDEO_RUNNER_REGISTRY,
    VideoRunnerSpec,
    integration_spec,
    video_runner_spec as _video_runner_spec,
)
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, REPO_ROOT


def list_video_benchmarks(*, catalog_dir: str | Path | None = None) -> list[str]:
    """Return sorted video benchmark ids from the checked-in catalog."""
    root = Path(catalog_dir or REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog" / "video")
    return sorted(path.stem for path in root.glob("*.yaml") if path.stem != "_manifest")


def video_runner_spec(benchmark_id: str) -> VideoRunnerSpec | None:
    """Return the registered official runner spec for a video benchmark, if any."""
    return _video_runner_spec(benchmark_id)


def run_benchmark(
    benchmark_id: str,
    *,
    output_dir: str | Path,
    mode: str = "official-run",
    generated_artifact_dir: str | Path | None = None,
    manifest_path: str | Path = BENCHMARK_ZOO_DIR,
    **kwargs: Any,
) -> Any:
    """Run a benchmark through the unified official runner stack.

    Modes mirror ``worldfoundry-eval zoo benchmark-run``:

    - ``normalizer``: normalize caller-provided official results
    - ``official-validation``: run the benchmark's bounded validation command
    - ``official-run``: invoke upstream official runtime when assets are available
    """
    return run_benchmark_execution(
        benchmark_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        mode=mode,
        generated_artifact_dir=generated_artifact_dir,
        **kwargs,
    )


def normalize_upstream_results(
    benchmark_id: str,
    results_path: str | Path,
    *,
    output_dir: str | Path,
    generated_artifact_dir: str | Path | None = None,
    manifest_path: str | Path = BENCHMARK_ZOO_DIR,
    **kwargs: Any,
) -> Any:
    """Normalize official upstream result files into a WorldFoundry scorecard."""
    return run_benchmark(
        benchmark_id,
        output_dir=output_dir,
        mode="normalizer",
        generated_artifact_dir=generated_artifact_dir,
        manifest_path=manifest_path,
        official_results_path=str(results_path),
        **kwargs,
    )


def benchmark_integration_spec(benchmark_id: str) -> BenchmarkIntegrationSpec | None:
    """Return the in-tree integration spec for a video benchmark."""
    return integration_spec(benchmark_id)


__all__ = [
    "BENCHMARK_INTEGRATION_REGISTRY",
    "BenchmarkIntegrationSpec",
    "IntegrationTier",
    "EvaluateRunRequest",
    "EvaluateRunResult",
    "ModelBenchmarkRunRequest",
    "ModelBenchmarkRunResult",
    "ModelBenchmarkSuiteRequest",
    "ModelBenchmarkSuiteResult",
    "VIDEO_RUNNER_REGISTRY",
    "VideoRunnerSpec",
    "WorldFoundryRunRequest",
    "WorldFoundryRunResult",
    "execute_evaluate_run",
    "list_video_benchmarks",
    "normalize_upstream_results",
    "benchmark_integration_spec",
    "run_benchmark",
    "run_benchmark_execution",
    "run_model_benchmark",
    "run_model_benchmark_suite",
    "run_worldfoundry",
    "video_runner_spec",
]
