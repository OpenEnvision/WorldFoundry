"""Runs single model inference generations combined with subsequent benchmark score evaluations.

This module provides the orchestrator that takes a model (HuggingFace, custom, etc.), resolves
its configuration, generates outputs (e.g. videos), materializes output files, runs those files
through official benchmark validation tools on the host system, and writes an integrated scorecard.
"""

from __future__ import annotations

import base64
import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.api.artifacts import local_path_for_uri
from worldfoundry.evaluation.reporting import (
    RUN_SUMMARY_SCHEMA_VERSION,
    build_comparison_identity,
    write_run_manifest_artifacts,
)
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import resolve_benchmark_manifest_path
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution
from worldfoundry.evaluation.tasks.execution.orchestration.run_mode import normalize_benchmark_run_mode
from worldfoundry.evaluation.utils import BENCHMARK_TASK_ROOT, stable_hash, write_json, write_jsonl

from .benchmark_generation import get_benchmark_generation_adapter
from .evaluate import EvaluateRunRequest, EvaluateRunResult, execute_evaluate_run
from .fidelity import model_benchmark_fidelity
from .plan import build_run_plan_from_task_registry, evaluate_request_from_run_plan, write_run_plan

MODEL_BENCHMARK_RUN_SCHEMA_VERSION = "worldfoundry-model-benchmark-run"
MODEL_BENCHMARK_RESULT_SCHEMA_VERSION = "worldfoundry-model-benchmark-result"
DEFAULT_BENCHMARK_TASK_ROOT = BENCHMARK_TASK_ROOT
CONTRACT_VALIDATION_ID = "contract-validation"
RESERVED_BENCHMARK_PARAMETERS = frozenset({"output_dir", "manifest_path", "mode", "generated_artifact_dir"})

# Static MP4 video binary encoded in base64 utilized to fill placeholders during contract runs
_CONTRACT_FIXTURE_MP4_B64 = "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"
_CONTRACT_FIXTURE_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov"}


@dataclass(frozen=True)
class ModelBenchmarkRunRequest:
    """Configures execution parameters to run models and official benchmarks together.

    Attributes:
        output_dir: Main output workspace path.
        benchmark_id: Canonical target benchmark identifier (from the zoo).
        benchmark_manifest_path: Path on disk pointing to the benchmark zoo catalog shard.
        benchmark_mode: Target execution mode ('official-validation', 'official-run').
        model_id: Target model identifier under evaluation.
        model_runner: Overriding model runner implementation.
        model_zoo_manifest_dir: Path to Model Zoo manifest directory.
        model_variant_id: selected weights variant ID.
        model_parameters: Hyperparameters passed to the model runner.
        model_runtime: System configurations passed to model runner environment.
        model_config: Raw model parameter mappings.
        requests: Custom pre-materialized request inputs.
        requests_path: Preflight requests path on disk.
        task_name: Task key evaluated within the registry.
        task_roots: Local paths scanned to discover task yaml catalogs.
        task_benchmark: Scope constraint for registry lookups.
        task_recursive: If True, recursively scans task_roots.
        task_root_dir: Anchor directory for relative catalog paths.
        dataset_root: Physical path to dataset files on disk.
        dataset_id: Selected dataset identifier.
        split: Selected dataset split (e.g. "default", "validation").
        num_samples: Maximum count of samples evaluated.
        generated_artifact_dir: Custom output directory for generated artifacts.
        output_artifact: Override representing the expected generated asset type.
        required_artifacts: List of artifact kinds expected to be verified.
        metrics: Sequence of scorecard metrics computed.
        generation_cache_dir: Custom directory path for SQLite generation caching.
        generation_cache_mode: Caching mode.
        generation_cache_namespace: Caching namespace partition.
        run_id: Unique trace ID.
        benchmark_timeout_seconds: Bounded timeout for running official benchmarks.
        benchmark_workdir: Target working directory for benchmark runners.
        benchmark_env: Custom environment overrides for benchmark runners.
        materialize_placeholders: If True, copies placeholder media during contract runs.
        contract_fixture: If True, allows running mock contract validators without actual models.
        fail_on_generation_error: If True, fails immediately if any generation fails.
    """

    output_dir: str | Path
    benchmark_id: str
    benchmark_manifest_path: str | Path
    benchmark_mode: str = "official-run"
    model_id: str = ""
    model_runner: str | None = None
    model_zoo_manifest_dir: str | Path | None = None
    model_variant_id: str | None = None
    model_parameters: Mapping[str, Any] | None = None
    model_runtime: Mapping[str, Any] | None = None
    model_config: Mapping[str, Any] | Any | None = None
    requests: Sequence[Any] | None = None
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
    output_artifact: str = "generated_video"
    required_artifacts: Sequence[str] = ("generated_video",)
    metrics: Sequence[str] = ("artifact_count", "required_artifacts_present")
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "off"
    generation_cache_namespace: str = "model_benchmark"
    run_id: str | None = None
    benchmark_timeout_seconds: float | None = None
    benchmark_workdir: str | Path | None = None
    benchmark_env: Mapping[str, Any] | None = None
    benchmark_parameters: Mapping[str, Any] | None = None
    materialize_placeholders: bool | None = None
    contract_fixture: bool = False
    fail_on_generation_error: bool = False
    evaluation_kind: str = "benchmark_model"
    leaderboard_candidate: bool = True
    evaluation_provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ModelBenchmarkRunResult:
    """Encapsulates the aggregated summary, report paths, and exit codes of single run.

    Attributes:
        schema_version: Standard schema identification string.
        status: Run status ("succeeded" or "failed").
        exit_code: Process exit code.
        output_dir: Output workspace directory.
        run_manifest_path: Path to serialised run manifest JSON.
        generation_result: Detailed result metrics returned from generation.
        benchmark_result: Scorecard details returned from official evaluation.
        generated_artifact_dir: Directory containing copied generated artifact files.
        artifact_manifest_path: Path to serialized manifest indexing generated artifacts.
        artifacts: Map of produced summary and report paths.
    """

    schema_version: str
    status: str
    exit_code: int
    output_dir: Path
    run_manifest_path: Path
    generation_result: EvaluateRunResult | None
    benchmark_result: Mapping[str, Any]
    generated_artifact_dir: Path
    artifact_manifest_path: Path
    artifacts: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        """Determines if both generation and evaluation completed with success."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Converts the run outcome into a plain, serializable dictionary."""
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["run_manifest_path"] = str(self.run_manifest_path)
        payload["generated_artifact_dir"] = str(self.generated_artifact_dir)
        payload["artifact_manifest_path"] = str(self.artifact_manifest_path)
        payload["generation_result"] = None if self.generation_result is None else self.generation_result.to_dict()
        payload["benchmark_result"] = dict(self.benchmark_result)
        payload["artifacts"] = dict(self.artifacts)
        payload["ok"] = self.ok
        return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Safely decodes and parses list records from a JSONL file."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _default_task_roots(benchmark_manifest_path: str | Path) -> tuple[Path, ...]:
    """Resolves relative fallback directories scanned to search for task YAML definitions."""
    manifest_path = resolve_benchmark_manifest_path(benchmark_manifest_path)
    sibling_tasks = manifest_path.parent / "tasks"
    if sibling_tasks.exists():
        return (sibling_tasks,)
    return (DEFAULT_BENCHMARK_TASK_ROOT,)


def _default_requests(benchmark_id: str, output_artifact: str) -> tuple[GenerationRequest, ...]:
    """Generates a list containing one generic fallback GenerationRequest during contract runs."""
    return (
        GenerationRequest(
            sample_id="sample-0000",
            task_name=benchmark_id,
            inputs={"prompt": "WorldFoundry contract fixture"},
            output_schema={output_artifact: {"kind": output_artifact}},
        ),
    )


def _safe_name(value: str) -> str:
    """Sanitizes strings for safe usage in file paths and manifest IDs."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._") or "artifact"


def _artifact_suffix(name: str, uri: str) -> str:
    """Determines the standard file extension to apply when copying intermediate generated files."""
    suffix = Path(uri).suffix
    if suffix:
        return suffix
    lowered = name.lower()
    if "video" in lowered:
        return ".mp4"
    if "image" in lowered:
        return ".png"
    return ".bin"


def _write_placeholder_artifact(destination: Path, metadata: Mapping[str, Any]) -> None:
    """Writes a mock placeholder file to act as model generation output during contract runs."""
    if destination.suffix.lower() in _CONTRACT_FIXTURE_VIDEO_SUFFIXES:
        destination.write_bytes(base64.b64decode(_CONTRACT_FIXTURE_MP4_B64))
        return
    destination.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _placeholder_allowed(mode: str, explicit: bool | None, *, contract_fixture: bool = False) -> bool:
    """Dictates whether writing fake media placeholders is permitted under the active configuration."""
    if explicit is not None:
        return bool(explicit)
    return mode == "contract" and contract_fixture


def _task_plan_path(root: Path) -> Path:
    """Returns the filepath representing the intermediate task execution plan."""
    return root / "task_run_plan.json"


def _run_generation_from_task_registry(request: ModelBenchmarkRunRequest, root: Path) -> EvaluateRunResult:
    """Materializes requests and runs model generations utilising the local task catalog registry."""
    if not request.task_name:
        raise ValueError("task_name is required for task-registry materialization")
    if request.dataset_root is None:
        raise ValueError("task-registry model-benchmark runs require dataset_root/data_path to materialize requests")

    task_roots = (
        tuple(Path(path) for path in request.task_roots)
        if request.task_roots
        else _default_task_roots(request.benchmark_manifest_path)
    )
    plan = build_run_plan_from_task_registry(
        task_name=request.task_name,
        task_roots=task_roots,
        output_dir=root / "generation",
        benchmark=request.task_benchmark or request.benchmark_id,
        recursive=request.task_recursive,
        root_dir=request.task_root_dir,
        mode="model",
        dataset_root=request.dataset_root,
        dataset_id=request.dataset_id or f"{request.benchmark_id}:generated",
        split=request.split,
        model_id=request.model_id,
        model_runner=request.model_runner,
        model_manifest_dir=request.model_zoo_manifest_dir,
        model_variant_id=request.model_variant_id,
        model_parameters=request.model_parameters,
        model_runtime=request.model_runtime,
        model_config=request.model_config,
        metrics=tuple(request.metrics),
        required_artifacts=tuple(request.required_artifacts),
        generation_cache_dir=request.generation_cache_dir,
        generation_cache_mode=request.generation_cache_mode,
        generation_cache_namespace=request.generation_cache_namespace,
        limit=request.num_samples,
        materialize_requests=True,
        run_id=None if request.run_id is None else f"{request.run_id}:generation",
        fail_on_sample_error=request.fail_on_generation_error,
    )
    if not plan.requests:
        raise ValueError(
            "task-registry materialization produced zero requests; check task metadata_path, dataset_root/data_path, "
            "split, and num_samples"
        )
    write_run_plan(plan, _task_plan_path(root))
    return execute_evaluate_run(evaluate_request_from_run_plan(plan))


def _materialize_generated_artifacts(
    *,
    generation_output_dir: Path,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str,
    allow_placeholders: bool,
) -> tuple[int, int]:
    """Copies physical output files from generation outputs into a unified benchmark input folder."""
    rows: list[dict[str, Any]] = []
    official_names: dict[str, str] = {}
    requests_path = generation_output_dir / "requests.jsonl"
    if requests_path.is_file():
        for request_row in _read_jsonl(requests_path):
            inputs = request_row.get("inputs")
            official_name = inputs.get("official_video_name") if isinstance(inputs, Mapping) else None
            sample_id = request_row.get("sample_id")
            if sample_id is not None and official_name:
                # Official layouts may choose the basename but never escape the run directory.
                official_names[str(sample_id)] = Path(str(official_name)).name
    results = [GenerationResult.from_dict(row) for row in _read_jsonl(generation_output_dir / "results.jsonl")]
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        for name, artifact in result.artifacts.items():
            if output_artifact and name != output_artifact:
                continue
            suffix = _artifact_suffix(name, artifact.uri)
            official_name = official_names.get(result.sample_id)
            destination_name = official_name or f"{_safe_name(result.sample_id)}__{_safe_name(name)}{suffix}"
            destination = generated_artifact_dir / destination_name
            source_path = local_path_for_uri(artifact.uri)
            row = {
                "sample_id": result.sample_id,
                "artifact_name": name,
                "source_uri": artifact.uri,
                "destination": str(destination),
                "status": "missing",
                "placeholder": False,
            }
            if source_path is not None and source_path.is_file():
                if source_path.resolve() != destination.resolve():
                    shutil.copy2(source_path, destination)
                row["status"] = "copied"
            elif allow_placeholders:
                _write_placeholder_artifact(
                    destination,
                    {
                        "placeholder": True,
                        "sample_id": result.sample_id,
                        "artifact_name": name,
                        "source_uri": artifact.uri,
                    },
                )
                row["status"] = "placeholder"
                row["placeholder"] = True
            rows.append(row)

    write_jsonl(artifact_manifest_path, rows, atomic=False)
    materialized_count = sum(1 for row in rows if row["status"] in {"copied", "placeholder"})
    placeholder_count = sum(1 for row in rows if row["status"] == "placeholder")
    return materialized_count, placeholder_count


def _materialize_contract_validation_artifacts(
    *,
    generated_artifact_dir: Path,
    artifact_manifest_path: Path,
    output_artifact: str,
) -> tuple[int, int]:
    """Generates dummy placeholder artifacts directly when contract_fixture is executed in mock mode."""
    artifact_name = output_artifact or "generated_artifact"
    destination = (
        generated_artifact_dir / f"sample-0000__{_safe_name(artifact_name)}{_artifact_suffix(artifact_name, '')}"
    )
    generated_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_placeholder_artifact(
        destination,
        {
            "placeholder": True,
            "sample_id": "sample-0000",
            "artifact_name": artifact_name,
            "source": "benchmark_contract_validation",
        },
    )
    write_jsonl(
        artifact_manifest_path,
        [
            {
                "sample_id": "sample-0000",
                "artifact_name": artifact_name,
                "source_uri": "",
                "destination": str(destination),
                "status": "placeholder",
                "placeholder": True,
            }
        ],
        atomic=False,
    )
    return 1, 1


def _run_generation(request: ModelBenchmarkRunRequest, root: Path) -> EvaluateRunResult | None:
    """Materialize benchmark requests and run the selected in-tree model."""
    if request.generated_artifact_dir is not None:
        return None

    requests = request.requests
    if requests is None and request.requests_path is None:
        if request.task_name:
            return _run_generation_from_task_registry(request, root)
        if request.contract_fixture:
            return None
        if request.benchmark_mode == "contract":
            raise ValueError(
                "model-benchmark contract runs require generated inputs or contract_fixture=True"
            )
        adapter = get_benchmark_generation_adapter(request.benchmark_id)
        if adapter is not None:
            requests = adapter.materialize_requests(
                limit=request.num_samples,
                dataset_root=request.dataset_root,
                split=request.split,
            )
            if not requests:
                raise ValueError(
                    f"{request.benchmark_id} prompt materialization produced zero requests; "
                    f"{adapter.missing_requests_hint}."
                )
        else:
            raise ValueError(
                "model-benchmark runs require generated inputs. Provide task_name+dataset_root, "
                "requests_path, generated_artifact_dir, or choose a benchmark with a registered "
                "generation adapter."
            )

    return execute_evaluate_run(
        EvaluateRunRequest(
            output_dir=root / "generation",
            mode="model",
            requests=requests,
            requests_path=request.requests_path,
            metrics=tuple(request.metrics),
            required_artifacts=tuple(request.required_artifacts),
            benchmark={
                "suite": "benchmark_zoo",
                "benchmark_name": request.benchmark_id,
                "task_type": request.task_name or request.benchmark_id,
                "evaluation_protocol": "model_generation",
            },
            model_id=request.model_id,
            model_runner=request.model_runner,
            model_zoo_manifest_dir=request.model_zoo_manifest_dir,
            model_variant_id=request.model_variant_id,
            model_parameters=request.model_parameters,
            model_runtime=request.model_runtime,
            model_config=request.model_config,
            generation_cache_dir=request.generation_cache_dir,
            generation_cache_mode=request.generation_cache_mode,
            generation_cache_namespace=request.generation_cache_namespace,
            dataset_id=f"{request.benchmark_id}:generated",
            run_id=None if request.run_id is None else f"{request.run_id}:generation",
            fail_on_sample_error=request.fail_on_generation_error,
        )
    )


def _model_benchmark_run_summary(
    *,
    request: ModelBenchmarkRunRequest,
    status: str,
    mode: str,
    root: Path,
    materialized_count: int,
    placeholder_count: int,
    generation_result: EvaluateRunResult | None,
    benchmark_payload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Compiles the primary run summary record, detailing generation and benchmark execution metrics."""
    sample_count = generation_result.sample_count if generation_result is not None else materialized_count
    successful_samples = (
        generation_result.successful_sample_count if generation_result is not None else materialized_count
    )
    failed_samples = generation_result.failed_sample_count if generation_result is not None else 0
    generation_success_rate = successful_samples / sample_count if sample_count else 0.0
    benchmark_ok = 1.0 if benchmark_payload.get("ok") is True else 0.0
    provenance = dict(request.evaluation_provenance or {})
    provenance_claim = provenance.get("claim")
    provenance_claim = provenance_claim if isinstance(provenance_claim, Mapping) else {}
    provenance_candidate = (
        provenance_claim.get("leaderboard_candidate") is True
        if provenance
        else request.leaderboard_candidate
    )
    effective_candidate = request.leaderboard_candidate and provenance_candidate
    leaderboard_valid = (
        status == "succeeded"
        and benchmark_payload.get("ok") is True
        and not request.contract_fixture
        and placeholder_count == 0
        and effective_candidate
    )
    score_valid = (
        status == "succeeded"
        and benchmark_payload.get("ok") is True
        and not request.contract_fixture
        and placeholder_count == 0
    )
    leaderboard_blockers = _model_benchmark_leaderboard_blockers(status, benchmark_payload)
    if request.contract_fixture and "model-benchmark run used contract fixture" not in leaderboard_blockers:
        leaderboard_blockers.append("model-benchmark run used contract fixture")
    if placeholder_count and "generated artifacts include placeholders" not in leaderboard_blockers:
        leaderboard_blockers.append("generated artifacts include placeholders")
    if not effective_candidate:
        fidelity = provenance.get("fidelity")
        fidelity = fidelity if isinstance(fidelity, Mapping) else {}
        if fidelity.get("data") not in (None, "official"):
            blocker = "custom data is not leaderboard-comparable"
        elif fidelity.get("evaluation") not in (None, "official"):
            blocker = "modified evaluation protocol is not leaderboard-comparable"
        else:
            blocker = "evaluation provenance is not leaderboard-comparable"
        if blocker not in leaderboard_blockers:
            leaderboard_blockers.append(blocker)
    leaderboard = {
        "materialized_artifact_count": float(materialized_count),
        "placeholder_artifact_count": float(placeholder_count),
        "real_artifact_count": float(max(materialized_count - placeholder_count, 0)),
        "generation_success_rate": float(generation_success_rate),
        "benchmark_ok": benchmark_ok,
    }
    benchmark_parameters = dict(request.benchmark_parameters or {})
    benchmark_revision = benchmark_parameters.get("revision")
    protocol_parameters = {
        key: value for key, value in benchmark_parameters.items() if key != "revision"
    }
    benchmark_summary = {
        "benchmark_id": request.benchmark_id,
        "benchmark_name": request.benchmark_id,
        "benchmark_revision": benchmark_revision,
        "task_type": request.task_name or f"{request.benchmark_id}:model_benchmark",
        "suite": "benchmark_zoo",
        "evaluation_protocol": f"model_benchmark:{mode}",
        "protocol_revision": benchmark_revision,
        "protocol_config_hash": stable_hash(protocol_parameters),
        "evaluation_kind": request.evaluation_kind,
    }
    dataset_summary = {
        "dataset_id": request.dataset_id or f"{request.benchmark_id}:generated",
        "name": request.dataset_id or f"{request.benchmark_id}:generated",
        "split": request.split,
        "sample_count": sample_count,
    }
    metrics_summary = {
        "leaderboard": leaderboard,
        "per_metric": {
            metric_id: {"mean": value, "higher_is_better": True} for metric_id, value in leaderboard.items()
        },
        "summary": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": [],
        },
    }
    comparison_identity = build_comparison_identity(
        benchmark=benchmark_summary,
        dataset=dataset_summary,
        metrics=metrics_summary,
        provenance=provenance,
        evaluation_kind=request.evaluation_kind,
    )
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "source_schema_version": MODEL_BENCHMARK_RUN_SCHEMA_VERSION,
        "run": {
            "run_id": request.run_id,
            "status": status,
            "started_at": None,
            "finished_at": None,
            "worldfoundry_version": None,
            "run_fingerprint": None,
        },
        "benchmark": benchmark_summary,
        "model": {
            "model_id": request.model_id,
            "model_name": request.model_id,
            "model_type": "model_benchmark",
        },
        "dataset": dataset_summary,
        "counts": {
            "sample_count": sample_count,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "failed_sample_ids": [],
        },
        "generation": {
            "successful": successful_samples,
            "failed": failed_samples,
            "materialized_artifact_count": materialized_count,
            "placeholder_artifact_count": placeholder_count,
            "real_artifact_count": max(materialized_count - placeholder_count, 0),
        },
        "metrics": metrics_summary,
        "leaderboard": leaderboard,
        "evaluation": {
            "kind": request.evaluation_kind,
            "mode": comparison_identity["evaluation_mode"],
        },
        "provenance": provenance,
        "comparison_identity": comparison_identity,
        "eligibility": {
            "score_valid": score_valid,
            "leaderboard_valid": leaderboard_valid,
            "leaderboard_eligible": leaderboard_valid,
            "reasons": leaderboard_blockers,
            "blocking_reasons": leaderboard_blockers,
        },
        "artifacts": {str(key): value for key, value in artifacts.items() if value not in (None, "")},
        "wrapper": {
            "output_dir": str(root),
            "benchmark_mode": mode,
            "evaluation_kind": request.evaluation_kind,
            "leaderboard_candidate": effective_candidate,
            "materialized_artifact_count": materialized_count,
            "placeholder_artifact_count": placeholder_count,
            "contract_fixture": request.contract_fixture,
        },
    }


def _model_benchmark_leaderboard_blockers(status: str, benchmark_payload: Mapping[str, Any]) -> list[str]:
    """Compiles blocking eligibility reasons if a run cannot be published to official leaderboards."""
    if status != "succeeded":
        return [status]
    if benchmark_payload.get("ok") is True:
        return []

    metadata = benchmark_payload.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if metadata.get("contract_only") is True:
        return ["benchmark runner ran in contract-only mode"]
    if metadata.get("normalizer_only") is True:
        return ["benchmark runner produced normalizer-only evidence"]
    if benchmark_payload.get("official_benchmark_verified") is not True:
        return ["official benchmark verification missing"]
    if benchmark_payload.get("integration_evidence") is not True:
        return ["benchmark integration evidence missing"]
    return ["benchmark runner did not produce leaderboard-valid evidence"]


def run_model_benchmark(
    request: ModelBenchmarkRunRequest | Mapping[str, Any] | None = None, **kwargs: Any
) -> ModelBenchmarkRunResult:
    """Executes a complete 1:1 model generation and benchmark scoring sequence.

    First triggers target model generation (unless generated_artifact_dir is supplied directly),
    materializes the files, dispatches host execution commands to score outputs, and compiles scorecard summaries.

    Args:
        request: Configured ModelBenchmarkRunRequest payload.
        **kwargs: Inline overrides merged directly into request properties.

    Returns:
        The generated ModelBenchmarkRunResult summary.
    """
    if isinstance(request, ModelBenchmarkRunRequest):
        if kwargs:
            payload = asdict(request)
            payload.update(kwargs)
            request = ModelBenchmarkRunRequest(**payload)
    else:
        payload = dict(kwargs)
        if isinstance(request, Mapping):
            payload = {**dict(request), **payload}
        request = ModelBenchmarkRunRequest(**payload)

    mode = normalize_benchmark_run_mode(request.benchmark_mode)
    if mode != "contract" and request.evaluation_provenance is None:
        fidelity = model_benchmark_fidelity(
            benchmark_mode=mode,
            custom_data=any(
                value not in (None, "")
                for value in (
                    request.requests_path,
                    request.task_name,
                    request.dataset_root,
                    request.dataset_id,
                    request.generated_artifact_dir,
                )
            ),
            sample_limited=request.num_samples is not None,
            benchmark_parameters=request.benchmark_parameters,
            producer=(
                "catalog_model"
                if request.model_zoo_manifest_dir is not None
                else "custom_model"
            ),
        )
        request = replace(
            request,
            leaderboard_candidate=request.leaderboard_candidate and fidelity.leaderboard_candidate,
            evaluation_provenance=fidelity.to_dict(),
        )
    if not request.model_id and not request.contract_fixture:
        raise ValueError("model-benchmark runs require model_id unless contract_fixture=True is set.")
    root = Path(request.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "model_benchmark_run.json"
    run_manifest_path = root / "run_manifest.json"
    environment_path = root / "environment.json"
    env_requirements_path = root / "env_requirements.json"
    summary_path = root / "summary.json"
    artifact_manifest_path = root / "generated_artifacts.jsonl"

    generation_result = _run_generation(request, root)
    if generation_result is not None and generation_result.successful_sample_count == 0:
        raise RuntimeError(
            "model generation produced no successful samples; benchmark evaluation was not started. "
            f"Inspect {generation_result.output_dir / 'results.jsonl'} for per-sample errors."
        )
    if generation_result is not None and request.fail_on_generation_error and generation_result.exit_code:
        raise RuntimeError(
            "model generation failed and fail_on_generation_error=True; benchmark evaluation was not started. "
            f"Inspect {generation_result.output_dir / 'results.jsonl'} for per-sample errors."
        )
    placeholder_count = 0
    if request.generated_artifact_dir is not None:
        generated_artifact_dir = Path(request.generated_artifact_dir).expanduser().resolve()
        materialized_count = (
            len([path for path in generated_artifact_dir.rglob("*") if path.is_file()])
            if generated_artifact_dir.exists()
            else 0
        )
        write_jsonl(artifact_manifest_path, [], atomic=False)
    else:
        generated_artifact_dir = root / "generated_artifacts"
        if request.contract_fixture and generation_result is None:
            materialized_count, placeholder_count = _materialize_contract_validation_artifacts(
                generated_artifact_dir=generated_artifact_dir,
                artifact_manifest_path=artifact_manifest_path,
                output_artifact=request.output_artifact,
            )
        else:
            adapter = get_benchmark_generation_adapter(request.benchmark_id)
            materialized = (
                None
                if adapter is None
                else adapter.materialize_artifacts(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                )
            )
            if materialized is None:
                materialized = _materialize_generated_artifacts(
                    generation_output_dir=root / "generation",
                    generated_artifact_dir=generated_artifact_dir,
                    artifact_manifest_path=artifact_manifest_path,
                    output_artifact=request.output_artifact,
                    allow_placeholders=_placeholder_allowed(
                        mode,
                        request.materialize_placeholders,
                        contract_fixture=request.contract_fixture,
                    ),
                )
            materialized_count, placeholder_count = materialized

    if materialized_count == 0:
        raise RuntimeError(
            "no generated artifacts were available; benchmark evaluation was not started. "
            f"Expected media under {generated_artifact_dir}."
        )

    benchmark_parameters = dict(request.benchmark_parameters or {})
    if request.dataset_root is not None:
        benchmark_parameters.setdefault("dataset_root", str(request.dataset_root))
    if request.num_samples is not None and not {"limit", "num_samples"}.intersection(benchmark_parameters):
        benchmark_parameters["limit"] = request.num_samples
    if request.model_id is not None:
        benchmark_parameters.setdefault("model_id", request.model_id)
    if request.split is not None:
        benchmark_parameters.setdefault("split", request.split)
    reserved_benchmark_parameters = RESERVED_BENCHMARK_PARAMETERS.intersection(benchmark_parameters)
    if reserved_benchmark_parameters:
        names = ", ".join(sorted(reserved_benchmark_parameters))
        raise ValueError(f"benchmark_parameters cannot override reserved fields: {names}")
    benchmark_result = run_benchmark_execution(
        request.benchmark_id,
        output_dir=root / "benchmark",
        manifest_path=resolve_benchmark_manifest_path(request.benchmark_manifest_path, request.benchmark_id),
        mode=mode,
        generated_artifact_dir=generated_artifact_dir,
        timeout_seconds=request.benchmark_timeout_seconds,
        workdir=request.benchmark_workdir,
        env_overrides=dict(request.benchmark_env or {}),
        **benchmark_parameters,
    )
    benchmark_payload = benchmark_result.to_dict()
    generation_exit_code = 0 if generation_result is None else generation_result.exit_code
    artifact_exit_code = (
        1
        if generation_result is not None and generation_result.successful_sample_count > 0 and materialized_count == 0
        else 0
    )
    official_mode_requires_evidence = mode in {"official-validation", "official-run"}
    benchmark_exit_code = 0 if not official_mode_requires_evidence or benchmark_result.ok else 1
    exit_code = generation_exit_code or artifact_exit_code or benchmark_exit_code
    status = "succeeded" if exit_code == 0 else "failed"
    sample_count = generation_result.sample_count if generation_result is not None else materialized_count
    successful_samples = (
        generation_result.successful_sample_count if generation_result is not None else materialized_count
    )
    failed_samples = generation_result.failed_sample_count if generation_result is not None else 0
    artifacts = {
        "run_manifest": str(manifest_path),
        "standard_run_manifest": str(run_manifest_path),
        "environment": str(environment_path),
        "env_requirements": str(env_requirements_path),
        "run_summary": str(summary_path),
        "generated_artifact_dir": str(generated_artifact_dir),
        "generated_artifact_manifest": str(artifact_manifest_path),
        "benchmark_scorecard": benchmark_payload.get("scorecard_path"),
    }
    if generation_result is not None:
        artifacts["generation_scorecard"] = str(generation_result.scorecard_path)
    task_run_plan_path = _task_plan_path(root)
    if task_run_plan_path.is_file():
        artifacts["task_run_plan"] = str(task_run_plan_path)

    manifest = {
        "schema_version": MODEL_BENCHMARK_RUN_SCHEMA_VERSION,
        "run_id": request.run_id,
        "status": status,
        "benchmark_id": request.benchmark_id,
        "benchmark_mode": mode,
        "evaluation_kind": request.evaluation_kind,
        "leaderboard_candidate": request.leaderboard_candidate,
        "provenance": dict(request.evaluation_provenance or {}),
        "benchmark_parameters": benchmark_parameters,
        "model_id": request.model_id or CONTRACT_VALIDATION_ID,
        "task": {
            "task_name": request.task_name,
            "task_benchmark": request.task_benchmark or (request.benchmark_id if request.task_name else None),
            "task_roots": [str(path) for path in request.task_roots] if request.task_roots else [],
            "dataset_root": None if request.dataset_root is None else str(request.dataset_root),
            "dataset_id": request.dataset_id,
            "split": request.split,
            "num_samples": request.num_samples,
        },
        "output_dir": str(root),
        "generated_artifact_dir": str(generated_artifact_dir),
        "materialized_artifact_count": materialized_count,
        "placeholder_artifact_count": placeholder_count,
        "generation": None if generation_result is None else generation_result.to_dict(),
        "benchmark": benchmark_payload,
        "artifacts": dict(artifacts),
    }
    write_json(manifest_path, manifest, atomic=False)
    write_run_manifest_artifacts(
        output_dir=root,
        base_manifest={
            "schema_version": "worldfoundry-run-manifest",
            "run_id": request.run_id,
            "runner": "model_benchmark_runner",
            "status": status,
            "exit_code": exit_code,
            "output_dir": str(root),
            "benchmark": {
                "benchmark_id": request.benchmark_id,
                "benchmark_mode": mode,
                "manifest_path": str(
                    resolve_benchmark_manifest_path(request.benchmark_manifest_path, request.benchmark_id)
                ),
            },
            "model": {
                "model_id": request.model_id or CONTRACT_VALIDATION_ID,
                "model_runner": request.model_runner,
                "variant_id": request.model_variant_id,
            },
            "dataset": {
                "dataset_id": request.dataset_id or f"{request.benchmark_id}:generated",
                "split": request.split,
                "sample_count": sample_count,
            },
            "sample_count": sample_count,
            "successful_sample_count": successful_samples,
            "failed_sample_count": failed_samples,
            "artifacts": dict(artifacts),
            "provenance": dict(request.evaluation_provenance or {}),
        },
        config={
            "benchmark_mode": mode,
            "output_artifact": request.output_artifact,
            "required_artifacts": tuple(request.required_artifacts),
            "metrics": tuple(request.metrics),
            "materialize_placeholders": request.materialize_placeholders,
            "contract_fixture": request.contract_fixture,
            "placeholder_artifact_count": placeholder_count,
            "benchmark_timeout_seconds": request.benchmark_timeout_seconds,
            "model_parameters": dict(request.model_parameters or {}),
            "model_runtime": dict(request.model_runtime or {}),
            "benchmark_env": dict(request.benchmark_env or {}),
        },
        required_paths=(
            request.benchmark_manifest_path,
            *((request.generated_artifact_dir,) if request.generated_artifact_dir is not None else ()),
        ),
        cache_paths={
            "generated_artifact_dir": generated_artifact_dir,
            "benchmark_output_dir": root / "benchmark",
            "generation_output_dir": root / "generation",
        },
        package_names=("worldfoundry", "numpy", "pandas"),
        manifest_path=run_manifest_path,
        environment_path=environment_path,
        env_requirements_path=env_requirements_path,
    )
    write_json(
        summary_path,
        _model_benchmark_run_summary(
            request=request,
            status=status,
            mode=mode,
            root=root,
            materialized_count=materialized_count,
            placeholder_count=placeholder_count,
            generation_result=generation_result,
            benchmark_payload=benchmark_payload,
            artifacts=artifacts,
        ),
        atomic=False,
    )

    return ModelBenchmarkRunResult(
        schema_version=MODEL_BENCHMARK_RESULT_SCHEMA_VERSION,
        status=status,
        exit_code=exit_code,
        output_dir=root,
        run_manifest_path=manifest_path,
        generation_result=generation_result,
        benchmark_result=benchmark_payload,
        generated_artifact_dir=generated_artifact_dir,
        artifact_manifest_path=artifact_manifest_path,
        artifacts=artifacts,
    )


__all__ = [
    "CONTRACT_VALIDATION_ID",
    "DEFAULT_BENCHMARK_TASK_ROOT",
    "MODEL_BENCHMARK_RESULT_SCHEMA_VERSION",
    "MODEL_BENCHMARK_RUN_SCHEMA_VERSION",
    "ModelBenchmarkRunRequest",
    "ModelBenchmarkRunResult",
    "run_model_benchmark",
]
