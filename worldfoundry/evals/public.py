"""Stable open-source evaluation API for WorldFoundry.

Benchmark integration is pragmatic: scorer/normalizer logic lives under
``worldfoundry/evals/tasks/execution/runners/``.  Straightforward metrics run
in-tree; model-backed judges and reusable foundation models live under
``worldfoundry/base_models``.  Generate videos with any integrated WorldFoundry
model, then score through the same package.

Typical flow::

    from worldfoundry.evals.public import run_model_benchmark

    # 1) generate with an integrated model id from the model catalog
    # 2) score in-tree via the benchmark runner
    run_model_benchmark(..., benchmark_mode="official-run")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worldfoundry.evals.tasks.catalog.dispatch import (
    BENCHMARK_INTEGRATION_REGISTRY,
    VIDEO_RUNNER_REGISTRY,
    BenchmarkIntegrationSpec,
    IntegrationTier,
    VideoRunnerSpec,
    integration_spec,
)
from worldfoundry.evals.tasks.catalog.dispatch import (
    video_runner_spec as _video_runner_spec,
)
from worldfoundry.evals.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution
from worldfoundry.evals.tasks.execution.orchestration.evaluate import (
    EvaluateRunRequest,
    EvaluateRunResult,
    execute_evaluate_run,
)
from worldfoundry.evals.tasks.execution.orchestration.model_benchmark import (
    ModelBenchmarkRunRequest,
    ModelBenchmarkRunResult,
    run_model_benchmark,
)
from worldfoundry.evals.tasks.execution.orchestration.model_benchmark_suite import run_model_benchmark_suite
from worldfoundry.evals.tasks.execution.orchestration.run import (
    WorldFoundryRunRequest,
    WorldFoundryRunResult,
    run_worldfoundry,
)
from worldfoundry.evals.tasks.execution.orchestration.suite_requests import (
    ModelBenchmarkSuiteRequest,
    ModelBenchmarkSuiteResult,
)
from worldfoundry.evals.utils import BENCHMARK_ZOO_DIR, REPO_ROOT


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

    Modes mirror ``wf zoo benchmark-run``:

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
    "VIDEO_RUNNER_REGISTRY",
    "BenchmarkIntegrationSpec",
    "EvaluateRunRequest",
    "EvaluateRunResult",
    "IntegrationTier",
    "ModelBenchmarkRunRequest",
    "ModelBenchmarkRunResult",
    "ModelBenchmarkSuiteRequest",
    "ModelBenchmarkSuiteResult",
    "VideoRunnerSpec",
    "WorldFoundryRunRequest",
    "WorldFoundryRunResult",
    "benchmark_integration_spec",
    "execute_evaluate_run",
    "list_video_benchmarks",
    "normalize_upstream_results",
    "run_benchmark",
    "run_benchmark_execution",
    "run_model_benchmark",
    "run_model_benchmark_suite",
    "run_worldfoundry",
    "video_runner_spec",
]
