"""Planning, resuming, and running one model × benchmark suite cell."""

from __future__ import annotations

import gc
import inspect
import json
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from worldfoundry.core.io.formats.serialization import read_jsonl_objects
from worldfoundry.core.observability.logging_setup import get_logger
from worldfoundry.evals.api import is_generation_result_successful
from worldfoundry.evals.tasks.catalog.benchmark_catalog import resolve_benchmark_manifest_path
from worldfoundry.evals.tasks.catalog.schema import BenchmarkZooEntry
from worldfoundry.evals.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evals.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evals.utils import (
    jsonable,
)
from worldfoundry.zoo.catalog import load_model_zoo_registry
from worldfoundry.zoo.catalog.manifest import model_zoo_entry_to_world_model_manifest

from .cache import generation_cache_payload, json_sha256
from .evaluate import _coerce_generation_requests, _load_optional_source, _model_execution_identity
from .fidelity import model_benchmark_fidelity
from .model_benchmark import (
    CONTRACT_VALIDATION_ID,
    ModelBenchmarkRunRequest,
    prepare_benchmark_generation,
    run_model_benchmark,
)
from .run_session import resume_generation, session_paths
from .suite_presets import _preset_values, _selected_suite_presets
from .suite_requests import _GENERIC_OUTPUT_ARTIFACTS, ModelBenchmarkSuiteRequest, _SuiteCellPlan


class _SuiteGenerationMemoRunner:
    """Reuse an identical successful request batch within one model lease.

    A suite commonly scores the same generated samples with many independent
    benchmarks. Persistent generation caching is intentionally optional, but
    rerunning the model for every scorer is both wasteful and unfair for
    stochastic generators. Memoization is batch-level rather than sample-level
    so stateful world models remain correct when output depends on request order
    or prior requests in the same rollout.
    """

    def __init__(self, runner: Any) -> None:
        self._runner = runner
        self._batches: dict[str, tuple[Any, ...]] = {}
        self.batch_hits = 0
        self.batch_misses = 0
        self.requests_reused = 0
        self.requests_executed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    @staticmethod
    def _batch_key(requests: Sequence[Any]) -> str | None:
        payloads: list[Any] = []
        for request in requests:
            policy = dict(getattr(request, "cache_policy", None) or {})
            policy_mode = policy.get("mode", policy.get("cache"))
            if policy_mode is False or (
                policy_mode is not None and str(policy_mode).strip().lower() in {"off", "false", "disabled", "none"}
            ):
                return None
            payloads.append(generation_cache_payload(request))
        try:
            return json_sha256(payloads)
        except (TypeError, ValueError):
            # A custom request extension may contain a process-local object.
            # It remains runnable but is not safe to identify for reuse.
            return None

    def generate(self, requests: Sequence[Any]) -> list[Any]:
        return list(self.generate_iter(requests))

    def generate_iter(self, requests: Sequence[Any]) -> Iterator[Any]:
        """Forward complete results, memoizing only a successful whole batch."""
        rows = tuple(requests)
        if not rows:
            return
        batch_key = self._batch_key(rows)
        if batch_key is not None:
            cached = self._batches.get(batch_key)
            if cached is not None:
                self.batch_hits += 1
                self.requests_reused += len(rows)
                yield from cached
                return

        self.batch_misses += 1
        self.requests_executed += len(rows)
        stream = getattr(self._runner, "generate_iter", None)
        outputs = stream(rows) if callable(stream) else self._runner.generate(rows)
        generated = []
        for result in outputs:
            generated.append(result)
            yield result
        if (
            batch_key is not None
            and len(generated) == len(rows)
            and all(is_generation_result_successful(result) for result in generated)
        ):
            self._batches[batch_key] = tuple(generated)

    def stats(self) -> dict[str, int]:
        return {
            "batch_hits": self.batch_hits,
            "batch_misses": self.batch_misses,
            "requests_reused": self.requests_reused,
            "requests_executed": self.requests_executed,
        }


def _safe_name(value: str) -> str:
    """Sanitize a string for filesystem-safe cell directory names."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return cleaned.strip("._") or "item"


def _fingerprint_request(request: ModelBenchmarkSuiteRequest) -> str:
    """Hash declarative suite fields (excluding output dir, run id, cache paths)."""
    payload = dict(jsonable(asdict(request)))
    payload.pop("output_dir", None)
    payload.pop("run_id", None)
    payload.pop("resume", None)
    payload.pop("metric_batch_size", None)
    payload.pop("generation_cache_dir", None)
    payload.pop("generation_cache_mode", None)
    payload.pop("generation_cache_namespace", None)
    payload.pop("model_workers", None)
    payload.pop("worker_cuda_devices", None)
    if request.requests_path is not None:
        payload["materialized_requests"] = _load_optional_source(None, request.requests_path)
    return json_sha256(payload)


def _fingerprint_cell(
    *,
    run_fingerprint: str,
    model_id: str,
    benchmark_id: str,
    output_artifact: str | None,
    required_artifacts: Sequence[str],
    mode: str,
    benchmark_manifest: Mapping[str, Any],
) -> str:
    """Builds a deterministic caching/run fingerprint for a specific 1:1 Model-to-Benchmark test cell."""
    return json_sha256(
        {
            "run_fingerprint": run_fingerprint,
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "output_artifact": output_artifact,
            "required_artifacts": list(required_artifacts),
            "mode": mode,
            "benchmark_manifest": benchmark_manifest,
        }
    )


def _coerce_request(
    request: ModelBenchmarkSuiteRequest | Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> ModelBenchmarkSuiteRequest:
    """Safely merges mapping parameters into a strict `ModelBenchmarkSuiteRequest` format."""
    if isinstance(request, ModelBenchmarkSuiteRequest):
        if not kwargs:
            return request
        return replace(request, **kwargs)
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    return ModelBenchmarkSuiteRequest(**payload)


def _selected_benchmarks(request: ModelBenchmarkSuiteRequest) -> tuple[BenchmarkZooEntry, ...]:
    """Maps selected benchmark IDs or status requirements onto concrete BenchmarkZooEntries."""
    registry = load_benchmark_zoo_registry(request.benchmark_manifest_dir)
    presets = _selected_suite_presets(request)
    benchmark_ids = tuple(dict.fromkeys((*_preset_values(presets, "benchmark_ids"), *request.benchmark_ids)))
    if benchmark_ids:
        return tuple(registry.get(item) for item in benchmark_ids)
    entries = registry.list()
    if request.benchmark_integration_status is not None:
        entries = [item for item in entries if item.integration_status == request.benchmark_integration_status]
    return tuple(entries)


def _selected_model_ids(request: ModelBenchmarkSuiteRequest) -> tuple[str, ...]:
    """Maps selected model IDs onto list of strings, falling back to validation fixtures if necessary."""
    presets = _selected_suite_presets(request)
    model_ids = tuple(
        dict.fromkeys((*_preset_values(presets, "model_ids"), *[str(item) for item in request.model_ids]))
    )
    if model_ids:
        return model_ids
    manifest_dir = Path(request.model_manifest_dir) if request.model_manifest_dir is not None else None
    if manifest_dir is None or not manifest_dir.exists():
        return (CONTRACT_VALIDATION_ID,) if request.contract_fixture else ()
    registry = load_model_zoo_registry(manifest_dir)
    entries = registry.list()
    if request.model_integration_status is not None:
        entries = [item for item in entries if item.integration_status == request.model_integration_status]
    runnable = [item.model_id for item in entries if item.runner_target or any(v.runner_target for v in item.variants)]
    if runnable:
        return tuple(runnable)
    return (CONTRACT_VALIDATION_ID,) if request.contract_fixture else ()


def _benchmark_input_keys(entry: BenchmarkZooEntry) -> tuple[str, ...]:
    """Retrieves expected task input keys declared by the benchmark's API contract."""
    try:
        return get_external_benchmark_contract(entry.benchmark_id).input_keys
    except KeyError:
        if entry.runner_target:
            try:
                from worldfoundry.evals.tasks.catalog.specs import benchmark_zoo_entry_to_benchmark_spec

                spec = benchmark_zoo_entry_to_benchmark_spec(entry)
                if spec.tasks:
                    return tuple(spec.tasks[0].input_keys)
            except Exception:  # fall back to generated videos.
                pass
    return ("generated_video_dir",)


def _acceptable_artifacts_for_benchmark(entry: BenchmarkZooEntry) -> tuple[str, ...]:
    """Derives expected intermediate file artifact types acceptable to evaluate this benchmark."""
    keys = {item.lower() for item in _benchmark_input_keys(entry)}
    artifacts: list[str] = []
    if any("policy_results" in key or "rollout" in key for key in keys):
        artifacts.extend(["action_trace", "actions", "rollout_video"])
    if any("world_outputs" in key for key in keys):
        artifacts.extend(["generated_world", "generated_video", "generated_3d_asset", "generated_4d_scene"])
    if any("generated_views" in key or "camera_metadata" in key for key in keys):
        artifacts.extend(["generated_3d_asset", "generated_4d_scene", "generated_video"])
    if any("video" in key or "generated_video" in key for key in keys):
        artifacts.extend(["generated_video", "predicted_video", "rollout_video"])
    if any("action_tokens" in key or "latent_action" in key for key in keys):
        artifacts.extend(["action_tokens", "plan_trace"])
    return tuple(dict.fromkeys(artifacts)) or ("generated_video",)


def _model_outputs(model_id: str, model_manifest_dir: str | Path | None) -> tuple[tuple[str, ...], bool, str]:
    """Retrieves standard outputs and metadata declared by a model zoo manifest entry."""
    if model_id == CONTRACT_VALIDATION_ID:
        return _GENERIC_OUTPUT_ARTIFACTS, False, model_id
    if model_manifest_dir is None:
        return (), False, model_id
    manifest_dir = Path(model_manifest_dir)
    if not manifest_dir.exists():
        return (), False, model_id
    try:
        entry = load_model_zoo_registry(manifest_dir).get(model_id)
    except Exception:  # custom runner/model id not present in model-zoo.
        return (), False, model_id
    manifest = model_zoo_entry_to_world_model_manifest(entry)
    return tuple(manifest.output_artifacts), True, entry.model_id


def _cell_artifact_selection(
    *,
    benchmark: BenchmarkZooEntry,
    model_outputs: Sequence[str],
    known_model: bool,
    output_override: str | None,
    required_override: Sequence[str] | None,
) -> tuple[str | None, tuple[str, ...], str, str | None]:
    """Checks model-to-benchmark schema compatibility and selects appropriate transfer artifacts."""
    acceptable = _acceptable_artifacts_for_benchmark(benchmark)
    outputs = tuple(model_outputs)
    if output_override:
        output_artifact = output_override
    elif known_model:
        output_artifact = next((item for item in acceptable if item in outputs), None)
    else:
        output_artifact = acceptable[0]

    required = tuple(str(item) for item in required_override) if required_override is not None else ()
    if output_artifact and not required:
        required = (output_artifact,)

    if output_artifact is None:
        return None, required, "incompatible", f"model outputs {list(outputs)} do not satisfy {list(acceptable)}"
    if known_model and output_artifact not in outputs and "generated_artifact" not in outputs:
        return (
            output_artifact,
            required,
            "incompatible",
            (f"model outputs {list(outputs)} do not include required artifact {output_artifact!r}"),
        )
    missing_required = [
        item for item in required if known_model and item not in outputs and "generated_artifact" not in outputs
    ]
    if missing_required:
        return (
            output_artifact,
            required,
            "incompatible",
            (f"model outputs {list(outputs)} do not include required artifacts {missing_required}"),
        )
    return output_artifact, required, "compatible" if known_model else "unknown", None


def _benchmark_unavailable_reason(entry: BenchmarkZooEntry, *, mode: str) -> str | None:
    """Return why a benchmark cannot be attempted through the suite runner.

    Runtime availability and evidence strength are deliberately separate.  An
    integrated runner may be useful for a bounded run (and fail closed when an
    asset is missing) before the full official benchmark has been verified.
    The resulting scorecard carries that evidence boundary; planning must not
    turn a conservative verification claim into an execution ban.
    """
    if mode == "contract" and entry.runner_target:
        return None
    if entry.integration_status == "integrated" and entry.runner_target:
        if mode == "official-run":
            # The workspace registry is the executable argument contract for
            # built-in model -> generated-artifact -> benchmark runs.  Keep
            # result importers usable in normalizer mode without advertising
            # them as raw-artifact evaluators in a matrix plan.
            from worldfoundry.evals.tasks.catalog.workspace_registry import workspace_benchmark_runtime_hint

            runtime_hint = workspace_benchmark_runtime_hint(entry.benchmark_id)
            if runtime_hint:
                if (
                    runtime_hint.get("supports_official_runtime") is True
                    and runtime_hint.get("accepts_generated_artifacts") is True
                ):
                    return None
                return (
                    "benchmark has an existing-result normalizer/importer but no official runtime "
                    "that accepts generic generated artifacts; use normalizer mode or the "
                    "benchmark-specific prepared layout"
                )
            # A new manifest need not edit the central registry when it ships
            # its own explicit official-run command.
            if entry.run_command:
                return None
            return "benchmark has no executable official-run route for generated artifacts"
        if mode in {"official-validation", "normalizer"} and (entry.validation_command or entry.run_command):
            return None
    return (
        f"benchmark is {entry.integration_status}/{entry.verification_status}; "
        "an integrated benchmark with a mode-compatible executable route is required"
    )


def _benchmark_aware_provenance(
    provenance: Mapping[str, Any],
    benchmark: BenchmarkZooEntry,
) -> dict[str, Any]:
    """Bound a planned claim by the benchmark's checked-in evidence.

    Fidelity describes the requested protocol, while catalog evidence records
    what the in-tree benchmark implementation has actually established.  A
    bounded official component can therefore remain executable without being
    advertised as benchmark-comparable or leaderboard-eligible.
    """

    payload = dict(provenance)
    claim = dict(payload.get("claim") or {})
    reasons: list[str] = list(payload.get("reasons") or ())
    full_official_evidence = bool(
        benchmark.official_benchmark_verified
        and benchmark.integration_evidence
        and benchmark.verification_status == "verified"
    )
    if not full_official_evidence:
        claim["level"] = "diagnostic"
        claim["leaderboard_candidate"] = False
        reasons.append("benchmark catalog does not establish full official-suite evidence")
    elif not benchmark.leaderboard_valid:
        claim["leaderboard_candidate"] = False
        reasons.append("benchmark catalog is not leaderboard-valid")
    payload["claim"] = claim
    payload["reasons"] = list(dict.fromkeys(str(reason) for reason in reasons if str(reason).strip()))
    return payload


def _model_manifest_dir_for_cell(
    model_id: str, model_manifest_dir: str | Path | None, known_model: bool
) -> str | Path | None:
    """Filters model manifest dir search paths depending on model catalog recognition."""
    if known_model:
        return model_manifest_dir
    return None


def _resolve_suite_model_runner(
    request: ModelBenchmarkSuiteRequest,
    plan: _SuiteCellPlan,
) -> Any:
    """Resolve one model lease shared by all executable cells for that model."""
    from worldfoundry.evals.models.runners.resolver import resolve_model_zoo_runner, resolve_world_model_runner

    manifest_dir = _model_manifest_dir_for_cell(
        plan.requested_model_id,
        request.model_manifest_dir,
        plan.known_model,
    )
    if manifest_dir is not None:
        resolved = resolve_model_zoo_runner(
            plan.requested_model_id,
            manifest_dir=manifest_dir,
            variant_id=request.model_variant_id,
            parameters=request.model_parameters,
            runtime=request.model_runtime,
        )
    else:
        resolved = resolve_world_model_runner(
            plan.requested_model_id,
            runner=request.model_runner,
            parameters=request.model_parameters,
            runtime=request.model_runtime,
            config=request.model_config,
        )
    return replace(resolved, runner=_SuiteGenerationMemoRunner(resolved.runner))


def _suite_generation_memo_stats(resolved: Any | None) -> dict[str, int] | None:
    runner = None if resolved is None else getattr(resolved, "runner", None)
    stats = getattr(runner, "stats", None)
    if not isinstance(runner, _SuiteGenerationMemoRunner) or not callable(stats):
        return None
    return stats()


def _cleanup_suite_model_runner(resolved: Any | None) -> None:
    """Release a shared runner exactly once after its model's suite cells."""
    if resolved is None:
        return
    cleanup = getattr(resolved.runner, "cleanup", None)
    try:
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:  # process exit remains the hard release boundary.
                warnings.warn(f"model runner cleanup failed: {exc}", RuntimeWarning, stacklevel=2)
    finally:
        # Do not import Torch just to clean up a CPU-only worker.  When a model
        # already loaded it, return unused allocator and IPC blocks before this
        # worker leases its GPU group to the next model.
        gc.collect()
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        is_initialized = getattr(cuda, "is_initialized", None)
        if callable(is_initialized) and is_initialized():
            try:
                empty_cache = getattr(cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
                ipc_collect = getattr(cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
            except RuntimeError:
                # A runner may have torn down its CUDA context during cleanup;
                # process exit remains the final release boundary.
                pass


def _reset_suite_model_runner(resolved: Any) -> bool:
    """Reset per-run state without unloading a resident suite model."""
    runner = resolved.runner
    reset = getattr(runner, "reset_for_evaluation", None)
    if callable(reset):
        reset()
        return True

    # A few policy runners expose a conventional zero-argument reset instead
    # of the evaluation-specific hook. Never guess arguments for other reset
    # contracts (for example simulator resets that require episode metadata).
    reset = getattr(runner, "reset", None)
    if not callable(reset):
        return False
    try:
        inspect.signature(reset).bind()
    except (TypeError, ValueError):
        return False
    reset()
    return True


def _cell_run_id(request: ModelBenchmarkSuiteRequest, model_id: str, benchmark_id: str) -> str | None:
    """Builds a formatted unique run trace ID bound to a specific model x benchmark cell."""
    if not request.run_id:
        return None
    return f"{request.run_id}:{model_id}:{benchmark_id}"


def _cell_dir(root: Path, model_id: str, benchmark_id: str) -> Path:
    """Return stable output directory for one matrix cell."""
    return root / "runs" / f"{_safe_name(model_id)}__{_safe_name(benchmark_id)}"


def _load_previous_suite_cells(root: Path) -> dict[str, Mapping[str, Any]]:
    """Load prior ``suite_manifest.json`` cells keyed by ``cell_fingerprint``."""
    manifest_path = root / "suite_manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = payload.get("cells") or []
    return {
        str(cell["cell_fingerprint"]): cell
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("cell_fingerprint")
    }


def _cell_artifact_state(artifacts: Mapping[str, Any]) -> dict[str, list[int]] | None:
    """Record required reports and staged media so deleted or replaced outputs cannot resume."""
    files: set[Path] = set()
    for value in artifacts.values():
        path = Path(str(value))
        if not path.exists():
            return None
        if path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
        else:
            files.add(path)
    return {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(files)}


def _resume_cell(
    cell_dir: Path,
    expected_cell: Mapping[str, Any] | None,
    *,
    request: ModelBenchmarkRunRequest,
) -> Mapping[str, Any] | None:
    """Return cached cell payload when resume can skip re-execution."""
    if expected_cell is None or expected_cell.get("status") != "succeeded":
        return None
    manifest_path = cell_dir / "model_benchmark_run.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "succeeded":
        return None
    artifacts = dict(payload.get("artifacts") or {})
    benchmark_payload = dict(payload.get("benchmark") or {})
    generation_payload = payload.get("generation")
    state = _cell_artifact_state(artifacts)
    if not state or state != expected_cell.get("artifact_state"):
        return None
    if generation_payload is not None:
        prepared = prepare_benchmark_generation(request, cell_dir)
        if prepared is None:
            return None
        requests = _coerce_generation_requests(_load_optional_source(prepared.requests, prepared.requests_path))
        paths = session_paths(cell_dir / "generation")
        if not paths["requests"].is_file() or list(read_jsonl_objects(paths["requests"])) != [
            row.to_dict() for row in requests
        ]:
            return None
        resolved_config = None
        if prepared.model_zoo_manifest_dir is not None and prepared.model_id is not None:
            from worldfoundry.evals.models.runners.resolver import resolve_model_zoo_config

            resolved_config = resolve_model_zoo_config(
                prepared.model_id,
                manifest_dir=prepared.model_zoo_manifest_dir,
                variant_id=prepared.model_variant_id,
                parameters=prepared.model_parameters,
                runtime=prepared.model_runtime,
            )
        identity = _model_execution_identity(prepared, resolved_zoo_config=resolved_config)
        if len(resume_generation(paths, requests, identity)) != len(requests):
            return None
    return {
        "status": "succeeded",
        "artifact_state": state,
        "exit_code": 0,
        "resumed": True,
        "run_dir": str(cell_dir),
        "run_manifest_path": str(manifest_path),
        "run_summary_path": artifacts.get("run_summary"),
        "generated_artifact_dir": payload.get("generated_artifact_dir"),
        "artifact_manifest_path": artifacts.get("generated_artifact_manifest"),
        "benchmark_scorecard_path": benchmark_payload.get("scorecard_path"),
        "generation_scorecard_path": None if generation_payload is None else generation_payload.get("scorecard_path"),
        "artifacts": artifacts,
    }


def _plan_cell(
    request: ModelBenchmarkSuiteRequest,
    *,
    run_fingerprint: str,
    model_id: str,
    canonical_model_id: str,
    known_model: bool,
    model_outputs: Sequence[str],
    benchmark: BenchmarkZooEntry,
) -> _SuiteCellPlan:
    """Resolve artifact compatibility and fingerprint for one matrix cell."""
    benchmark_unavailable_reason = _benchmark_unavailable_reason(benchmark, mode=request.mode)
    output_artifact, required_artifacts, compatibility, reason = _cell_artifact_selection(
        benchmark=benchmark,
        model_outputs=model_outputs,
        known_model=known_model,
        output_override=request.output_artifact,
        required_override=request.required_artifacts,
    )
    fidelity = model_benchmark_fidelity(
        benchmark_mode=request.mode,
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
        producer="catalog_model" if known_model else "custom_model",
    )
    evaluation_provenance = _benchmark_aware_provenance(fidelity.to_dict(), benchmark)
    return _SuiteCellPlan(
        model_id=canonical_model_id,
        requested_model_id=model_id,
        known_model=known_model,
        benchmark=benchmark,
        benchmark_manifest_path=resolve_benchmark_manifest_path(request.benchmark_manifest_dir, benchmark.benchmark_id),
        model_output_artifacts=tuple(model_outputs),
        benchmark_acceptable_artifacts=_acceptable_artifacts_for_benchmark(benchmark),
        output_artifact=output_artifact,
        required_artifacts=tuple(required_artifacts),
        compatibility="benchmark_unavailable" if benchmark_unavailable_reason else compatibility,
        reason=reason or benchmark_unavailable_reason,
        evaluation_provenance=evaluation_provenance,
        cell_fingerprint=_fingerprint_cell(
            run_fingerprint=run_fingerprint,
            model_id=canonical_model_id,
            benchmark_id=benchmark.benchmark_id,
            output_artifact=output_artifact,
            required_artifacts=required_artifacts,
            mode=request.mode,
            benchmark_manifest=benchmark.to_dict(),
        ),
    )


def _cell_request(
    request: ModelBenchmarkSuiteRequest,
    *,
    root: Path,
    plan: _SuiteCellPlan,
) -> ModelBenchmarkRunRequest:
    """Apply cell bindings to the shared single-run request fields."""
    values = {
        field.name: getattr(request, field.name)
        for field in fields(ModelBenchmarkRunRequest)
        if hasattr(request, field.name)
    }
    values.update(
        output_dir=_cell_dir(root, plan.model_id, plan.benchmark.benchmark_id),
        benchmark_id=plan.benchmark.benchmark_id,
        benchmark_manifest_path=plan.benchmark_manifest_path,
        benchmark_mode=request.mode,
        model_id=plan.requested_model_id,
        model_zoo_manifest_dir=_model_manifest_dir_for_cell(
            plan.requested_model_id,
            request.model_manifest_dir,
            plan.known_model,
        ),
        output_artifact=plan.output_artifact,
        required_artifacts=plan.required_artifacts,
        run_id=_cell_run_id(request, plan.model_id, plan.benchmark.benchmark_id),
        evaluation_provenance=plan.evaluation_provenance,
        leaderboard_candidate=bool(dict(plan.evaluation_provenance.get("claim") or {}).get("leaderboard_candidate")),
    )
    return ModelBenchmarkRunRequest(**values)


def _run_cell(
    request: ModelBenchmarkSuiteRequest,
    *,
    root: Path,
    plan: _SuiteCellPlan,
    resolved_runner: Any | None = None,
    runner_factory: Callable[[], Any] | None = None,
    runner_reused: bool | None = None,
    runner_state_reset: bool | None = None,
) -> Mapping[str, Any]:
    """Run :func:`run_model_benchmark` for one planned cell."""
    if plan.output_artifact is None:
        raise ValueError("cannot run a matrix cell without a selected output artifact")
    cell_dir = _cell_dir(root, plan.model_id, plan.benchmark.benchmark_id)
    cell_run_id = _cell_run_id(request, plan.model_id, plan.benchmark.benchmark_id)
    logger = get_logger(__name__).bind(
        run_id=cell_run_id,
        model_id=plan.model_id,
        benchmark_id=plan.benchmark.benchmark_id,
        phase="suite.cell",
    )
    logger.event(
        "INFO",
        "suite.cell.started",
        "Model-benchmark suite cell started",
        output_dir=str(cell_dir),
        runner_reused=runner_reused,
    )
    result = run_model_benchmark(
        _cell_request(request, root=root, plan=plan),
        resolved_runner=resolved_runner,
        runner_factory=runner_factory,
        cleanup_runner=resolved_runner is None and runner_factory is None,
    )
    logger.event(
        "INFO" if result.exit_code == 0 else "ERROR",
        "suite.cell.finished",
        "Model-benchmark suite cell finished",
        status=result.status,
        exit_code=result.exit_code,
    )
    payload = result.to_dict()
    artifacts = dict(payload.get("artifacts") or {})
    return {
        "artifact_state": _cell_artifact_state(artifacts),
        "model_id": plan.model_id,
        "requested_model_id": plan.requested_model_id,
        "benchmark_id": plan.benchmark.benchmark_id,
        "status": result.status,
        "exit_code": result.exit_code,
        "output_artifact": plan.output_artifact,
        "required_artifacts": list(plan.required_artifacts),
        "run_dir": str(cell_dir),
        "run_manifest_path": str(result.run_manifest_path),
        "run_summary_path": artifacts.get("run_summary"),
        "generated_artifact_dir": payload.get("generated_artifact_dir"),
        "artifact_manifest_path": payload.get("artifact_manifest_path"),
        "benchmark_scorecard_path": payload["benchmark_result"].get("scorecard_path"),
        "generation_scorecard_path": (
            None if payload.get("generation_result") is None else payload["generation_result"].get("scorecard_path")
        ),
        "resumed": False,
        "runner_reused": runner_reused,
        "runner_state_reset": runner_state_reset,
        "artifacts": artifacts,
    }
