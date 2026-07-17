"""Model × benchmark matrix suite orchestrator.

Expands model-zoo and benchmark-zoo selections into a cartesian grid of cells,
checks artifact compatibility, runs :func:`run_model_benchmark` per cell, and
aggregates index/comparison dashboards.

Sections:

* **DTOs** — suite request/result dataclasses and internal cell plans.
* **Planning** — preset loading, compatibility checks, fingerprints.
* **Execution** — cell run/resume and suite artifact writers.
* **Public API** — :func:`run_model_benchmark_suite` entry point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.catalog.manifest import model_zoo_entry_to_world_model_manifest
from worldfoundry.evaluation.tasks.catalog.benchmark_catalog import resolve_benchmark_manifest_path
from worldfoundry.evaluation.tasks.catalog.schema import BenchmarkZooEntry
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.utils import (
    BENCHMARK_ZOO_DIR,
    MODEL_ZOO_DIR,
    jsonable,
    load_manifest,
    write_json,
    write_jsonl,
    write_text,
)

from .cache import json_sha256
from .fidelity import model_benchmark_fidelity
from .model_benchmark import CONTRACT_VALIDATION_ID, ModelBenchmarkRunRequest, run_model_benchmark

# ---------------------------------------------------------------------------
# Schema constants and artifact compatibility
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Suite request / result DTOs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


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
    payload.pop("generation_cache_dir", None)
    payload.pop("generation_cache_mode", None)
    payload.pop("generation_cache_namespace", None)
    return json_sha256(payload)


def _fingerprint_cell(
    *,
    run_fingerprint: str,
    model_id: str,
    benchmark_id: str,
    output_artifact: str | None,
    required_artifacts: Sequence[str],
    mode: str,
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
        payload = asdict(request)
        payload.update(kwargs)
        return ModelBenchmarkSuiteRequest(**payload)
    payload = dict(kwargs)
    if isinstance(request, Mapping):
        payload = {**dict(request), **payload}
    return ModelBenchmarkSuiteRequest(**payload)


def _lookup_key(value: str) -> str:
    """Normalizes string aliases (e.g. replacing underscores with hyphens)."""
    return str(value).strip().lower().replace("_", "-")


def _load_suite_presets(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Loads optional declarative model-benchmark suite combinations from an external YAML manifest."""
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    payload = load_manifest(source)
    if not isinstance(payload, Mapping):
        raise TypeError(f"suite preset file must be a mapping: {source}")
    raw_suites = payload.get("suites", [])
    if not isinstance(raw_suites, list):
        raise TypeError(f"suite preset file suites must be a list: {source}")
    suites: dict[str, dict[str, Any]] = {}
    for item in raw_suites:
        if not isinstance(item, Mapping) or not item.get("id"):
            continue
        suite = dict(item)
        suite_id = str(suite["id"])
        keys = [suite_id, *[str(alias) for alias in suite.get("aliases") or ()]]
        for key in keys:
            suites[_lookup_key(key)] = suite
    return suites


def list_model_benchmark_suite_presets(path: str | Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """List unique suite presets from the suites YAML file."""
    suites = _load_suite_presets(path)
    unique: dict[str, Mapping[str, Any]] = {}
    for suite in suites.values():
        suite_id = str(suite.get("id"))
        unique.setdefault(suite_id, suite)
    return tuple(unique[key] for key in sorted(unique))


def get_model_benchmark_suite_preset(suite_id: str, path: str | Path | None = None) -> Mapping[str, Any]:
    """Look up one suite preset by id or alias."""
    suites = _load_suite_presets(path)
    try:
        return suites[_lookup_key(suite_id)]
    except KeyError as exc:
        known = ", ".join(str(item.get("id")) for item in list_model_benchmark_suite_presets(path))
        raise KeyError(f"unknown model-benchmark suite preset {suite_id!r}; known: {known}") from exc


def _selected_suite_presets(request: ModelBenchmarkSuiteRequest) -> tuple[Mapping[str, Any], ...]:
    """Retrieves a compiled list of all presets selected inside the suite request."""
    if not request.suite_ids:
        return ()
    selected = []
    for suite_id in request.suite_ids:
        selected.append(get_model_benchmark_suite_preset(str(suite_id), request.suite_preset_path))
    return tuple(selected)


def _preset_values(presets: Sequence[Mapping[str, Any]], key: str) -> tuple[str, ...]:
    """Extracts unique string values associated with a specific preset key (e.g. 'model_ids')."""
    values: list[str] = []
    for preset in presets:
        for item in preset.get(key) or ():
            text = str(item)
            if text not in values:
                values.append(text)
    return tuple(values)


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
                from worldfoundry.evaluation.tasks.catalog.specs import benchmark_zoo_entry_to_benchmark_spec

                spec = benchmark_zoo_entry_to_benchmark_spec(entry)
                if spec.tasks:
                    return tuple(spec.tasks[0].input_keys)
            except Exception:  # noqa: BLE001 - fall back to generated videos.
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
    except Exception:  # noqa: BLE001 - custom runner/model id not present in model-zoo.
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
            from ..runners.workspace_registry import workspace_benchmark_runtime_hint

            runtime_hint = workspace_benchmark_runtime_hint(entry.benchmark_id)
            if runtime_hint:
                if runtime_hint.get("supports_official_runtime") is True and runtime_hint.get(
                    "accepts_generated_artifacts"
                ) is True:
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
    reasons = list(payload.get("reasons") or ())
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


def _resume_cell(cell_dir: Path, expected_cell: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
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
    return {
        "status": "succeeded",
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
        ),
    )


def _run_cell(
    request: ModelBenchmarkSuiteRequest,
    *,
    root: Path,
    plan: _SuiteCellPlan,
) -> Mapping[str, Any]:
    """Run :func:`run_model_benchmark` for one planned cell."""
    if plan.output_artifact is None:
        raise ValueError("cannot run a matrix cell without a selected output artifact")
    cell_dir = _cell_dir(root, plan.model_id, plan.benchmark.benchmark_id)
    model_parameters = dict(request.model_parameters or {})
    model_runtime = dict(request.model_runtime or {})
    result = run_model_benchmark(
        ModelBenchmarkRunRequest(
            output_dir=cell_dir,
            benchmark_id=plan.benchmark.benchmark_id,
            benchmark_manifest_path=plan.benchmark_manifest_path,
            benchmark_mode=request.mode,
            model_id=plan.requested_model_id,
            model_runner=request.model_runner,
            model_zoo_manifest_dir=_model_manifest_dir_for_cell(
                plan.requested_model_id,
                request.model_manifest_dir,
                plan.known_model,
            ),
            model_variant_id=request.model_variant_id,
            model_parameters=model_parameters,
            model_runtime=model_runtime,
            model_config=request.model_config,
            requests_path=request.requests_path,
            task_name=request.task_name,
            task_roots=request.task_roots,
            task_benchmark=request.task_benchmark,
            task_recursive=request.task_recursive,
            task_root_dir=request.task_root_dir,
            dataset_root=request.dataset_root,
            dataset_id=request.dataset_id,
            split=request.split,
            num_samples=request.num_samples,
            generated_artifact_dir=request.generated_artifact_dir,
            output_artifact=plan.output_artifact,
            required_artifacts=plan.required_artifacts,
            metrics=tuple(request.metrics),
            generation_cache_dir=request.generation_cache_dir,
            generation_cache_mode=request.generation_cache_mode,
            generation_cache_namespace=request.generation_cache_namespace,
            run_id=_cell_run_id(request, plan.model_id, plan.benchmark.benchmark_id),
            benchmark_timeout_seconds=request.benchmark_timeout_seconds,
            benchmark_workdir=request.benchmark_workdir,
            benchmark_env=request.benchmark_env,
            benchmark_parameters=request.benchmark_parameters,
            materialize_placeholders=request.materialize_placeholders,
            contract_fixture=request.contract_fixture,
            fail_on_generation_error=request.fail_on_generation_error,
            evaluation_provenance=plan.evaluation_provenance,
            leaderboard_candidate=bool(
                dict(plan.evaluation_provenance.get("claim") or {}).get("leaderboard_candidate")
            ),
        )
    )
    payload = result.to_dict()
    artifacts = dict(payload.get("artifacts") or {})
    return {
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
        "artifacts": artifacts,
    }


def _suite_summary(cells: Sequence[Mapping[str, Any]], *, execute: bool) -> dict[str, Any]:
    """Aggregates execution states (succeeded/failed/skipped) across all matrix cells."""
    counts: dict[str, int] = {}
    for cell in cells:
        status = str(cell.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "total": len(cells),
        "execute": execute,
        "planned": counts.get("planned", 0),
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "status_counts": counts,
        "models": sorted({str(cell.get("model_id")) for cell in cells if cell.get("model_id")}),
        "benchmarks": sorted({str(cell.get("benchmark_id")) for cell in cells if cell.get("benchmark_id")}),
    }


def build_markdown_suite_report(payload: Mapping[str, Any]) -> str:
    """Generates a human-readable Markdown summary representing the entire matrix sweep."""
    summary = dict(payload.get("summary") or {})
    lines = [
        "# WorldFoundry Model x Benchmark Suite",
        "",
        f"- Status: {payload.get('status')}",
        f"- Total cells: {summary.get('total', 0)}",
        f"- Succeeded: {summary.get('succeeded', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Skipped: {summary.get('skipped', 0)}",
        "",
        "| Model | Benchmark | Artifact | Compatibility | Status | Run |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in payload.get("cells") or ():
        if not isinstance(cell, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in (
                    cell.get("model_id", ""),
                    cell.get("benchmark_id", ""),
                    cell.get("output_artifact", ""),
                    cell.get("compatibility", ""),
                    cell.get("status", ""),
                    cell.get("run_dir", ""),
                )
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _suite_scorecard_rows(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collect per-cell scorecard paths for the suite index."""
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        artifacts = dict(cell.get("artifacts") or {})
        benchmark_scorecard = cell.get("benchmark_scorecard_path") or artifacts.get("benchmark_scorecard")
        generation_scorecard = cell.get("generation_scorecard_path") or artifacts.get("generation_scorecard")
        if benchmark_scorecard in (None, "") and generation_scorecard in (None, ""):
            continue
        rows.append(
            {
                "index": index,
                "model_id": cell.get("model_id"),
                "requested_model_id": cell.get("requested_model_id"),
                "benchmark_id": cell.get("benchmark_id"),
                "status": cell.get("status"),
                "run_dir": cell.get("run_dir"),
                "run_summary_path": cell.get("run_summary_path"),
                "benchmark_scorecard_path": benchmark_scorecard,
                "generation_scorecard_path": generation_scorecard,
            }
        )
    return rows


def _run_summary_paths_and_labels(cells: Sequence[Mapping[str, Any]]) -> tuple[list[Path], list[str]]:
    """Extract ``run_summary`` paths and ``model:benchmark`` labels."""
    paths: list[Path] = []
    labels: list[str] = []
    for cell in cells:
        summary_path = cell.get("run_summary_path")
        if summary_path in (None, ""):
            continue
        path = Path(str(summary_path))
        if not path.is_file():
            continue
        paths.append(path)
        labels.append(f"{cell.get('model_id')}:{cell.get('benchmark_id')}")
    return paths, labels


def _write_empty_comparison(
    output_json: Path,
    output_md: Path,
    *,
    issue: str = "no completed run summaries found",
) -> dict[str, Any]:
    """Write a valid empty comparison artifact with an actionable reason."""

    from worldfoundry.evaluation.reporting import RUN_COMPARISON_SCHEMA_VERSION, build_markdown_comparison

    payload = {
        "schema_version": RUN_COMPARISON_SCHEMA_VERSION,
        "run_count": 0,
        "baseline": None,
        "benchmarks": [],
        "datasets": [],
        "metric_ids": [],
        "available_metric_ids": [],
        "common_metric_ids": [],
        "runs": [],
        "rows": [],
        "metrics": {},
        "best_by_metric": {},
        "issues": [issue],
        "artifacts": {
            "comparison_json": str(output_json.resolve()),
            "comparison_markdown": str(output_md.resolve()),
        },
    }
    write_json(output_json, payload, atomic=False)
    write_text(output_md, build_markdown_comparison(payload), atomic=False)
    return payload


def _write_suite_artifacts(root: Path, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write suite scorecard index, run browser, and comparison artifacts."""
    from worldfoundry.evaluation.reporting import (
        build_markdown_run_index,
        write_run_browser,
        write_run_comparison,
        write_run_index,
    )

    scorecard_rows = _suite_scorecard_rows(cells)
    scorecards_json = root / "scorecards" / "scorecards.json"
    scorecards_jsonl = root / "scorecards" / "scorecards.jsonl"
    scorecards_payload = {
        "schema_version": MODEL_BENCHMARK_SUITE_SCORECARDS_SCHEMA_VERSION,
        "scorecard_count": len(scorecard_rows),
        "rows": scorecard_rows,
    }
    write_json(scorecards_json, scorecards_payload, atomic=False)
    write_jsonl(scorecards_jsonl, scorecard_rows, atomic=False)

    summary_paths, labels = _run_summary_paths_and_labels(cells)
    index_json = root / "index" / "index.json"
    index_jsonl = root / "index" / "index.jsonl"
    index_md = root / "index" / "index.md"
    index_html = root / "index" / "index.html"
    index_roots: Sequence[str | Path] = summary_paths if summary_paths else (root,)
    index = write_run_index(index_roots, output_json=index_json, output_jsonl=index_jsonl)
    write_text(index_md, build_markdown_run_index(index), atomic=False)
    write_run_browser(index, index_html)

    comparison_json = root / "comparison" / "comparison.json"
    comparison_md = root / "comparison" / "comparison.md"
    completed_benchmark_ids = {
        str(cell.get("benchmark_id"))
        for cell in cells
        if cell.get("run_summary_path") and Path(str(cell["run_summary_path"])).is_file()
    }
    if len(completed_benchmark_ids) > 1:
        comparison = _write_empty_comparison(
            comparison_json,
            comparison_md,
            issue="suite spans multiple benchmarks; select one benchmark from index.json before comparing runs",
        )
    elif summary_paths:
        try:
            comparison = write_run_comparison(
                summary_paths,
                labels=labels,
                output_json=comparison_json,
                output_md=comparison_md,
            )
        except ValueError as exc:
            comparison = _write_empty_comparison(comparison_json, comparison_md, issue=str(exc))
    else:
        comparison = _write_empty_comparison(comparison_json, comparison_md)

    return {
        "scorecards_json": str(scorecards_json),
        "scorecards_jsonl": str(scorecards_jsonl),
        "index_json": str(index_json),
        "index_jsonl": str(index_jsonl),
        "index_markdown": str(index_md),
        "index_html": str(index_html),
        "comparison_json": str(comparison_json),
        "comparison_markdown": str(comparison_md),
        "scorecard_count": len(scorecard_rows),
        "indexed_run_count": index.get("run_count", 0),
        "comparison_run_count": comparison.get("run_count", 0),
    }


def run_model_benchmark_suite(
    request: ModelBenchmarkSuiteRequest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ModelBenchmarkSuiteResult:
    """Execute or plan a model × benchmark matrix sweep.

    Execution flow:

    * Expand selected models × benchmarks (or suite presets).
    * Skip or block incompatible artifact pairings.
    * Run or resume each cell via :func:`run_model_benchmark`.
    * Aggregate index, comparison, and ``suite_manifest.json``.
    """
    suite_request = _coerce_request(request, kwargs)
    if suite_request.contract_fixture and suite_request.mode != "contract":
        suite_request = replace(suite_request, mode="contract")
    root = Path(suite_request.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_fingerprint = _fingerprint_request(suite_request)
    previous_cells = _load_previous_suite_cells(root) if suite_request.resume else {}
    selected_model_ids = _selected_model_ids(suite_request)
    if not selected_model_ids:
        raise ValueError(
            "model-benchmark suites require at least one model id. Pass --model, choose a suite preset "
            "that declares model_ids, or set contract_fixture=True to run benchmark contract validation cells."
        )

    cells: list[dict[str, Any]] = []
    for model_id in selected_model_ids:
        model_outputs, known_model, canonical_model_id = _model_outputs(model_id, suite_request.model_manifest_dir)
        for benchmark in _selected_benchmarks(suite_request):
            plan = _plan_cell(
                suite_request,
                run_fingerprint=run_fingerprint,
                model_id=model_id,
                canonical_model_id=canonical_model_id,
                known_model=known_model,
                model_outputs=model_outputs,
                benchmark=benchmark,
            )
            base_cell = plan.to_base_cell()
            if plan.compatibility == "benchmark_unavailable" and suite_request.skip_incompatible:
                cells.append({**base_cell, "status": "skipped", "exit_code": 0, "reason": plan.reason})
                continue
            if plan.reason and suite_request.skip_incompatible:
                cells.append({**base_cell, "status": "skipped", "exit_code": 0, "reason": plan.reason})
                continue
            if not suite_request.execute:
                status = "planned" if not plan.reason else "blocked"
                cells.append(
                    {**base_cell, "status": status, "exit_code": 0 if not plan.reason else 1, "reason": plan.reason}
                )
                continue
            if plan.output_artifact is None:
                cells.append({**base_cell, "status": "failed", "exit_code": 1, "reason": plan.reason})
                continue
            try:
                if suite_request.resume:
                    resumed = _resume_cell(
                        _cell_dir(root, plan.model_id, plan.benchmark.benchmark_id),
                        previous_cells.get(plan.cell_fingerprint),
                    )
                    if resumed is not None:
                        cells.append({**base_cell, **dict(resumed)})
                        continue
                run_cell = _run_cell(
                    suite_request,
                    root=root,
                    plan=plan,
                )
                cells.append({**base_cell, **dict(run_cell)})
            except Exception as exc:  # noqa: BLE001 - keep suite execution moving across cells.
                cells.append({**base_cell, "status": "failed", "exit_code": 1, "reason": str(exc)})

    summary = _suite_summary(cells, execute=suite_request.execute)
    artifacts = _write_suite_artifacts(root, cells)
    failed = int(summary["failed"])
    skipped = int(summary["skipped"])
    exit_code = 1 if failed or (suite_request.fail_on_skipped and skipped) else 0
    if not suite_request.execute and not failed:
        status = "planned"
    elif failed:
        status = "failed"
    elif skipped and not any(cell["status"] == "succeeded" for cell in cells):
        status = "skipped"
    else:
        status = "succeeded"
    payload = {
        "schema_version": MODEL_BENCHMARK_SUITE_SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "run_fingerprint": run_fingerprint,
        "request": jsonable(asdict(suite_request)),
        "summary": summary,
        "cells": cells,
        "artifacts": artifacts,
    }
    manifest_path = root / "suite_manifest.json"
    report_path = root / "suite_report.md"
    write_json(manifest_path, payload, atomic=False)
    write_text(report_path, build_markdown_suite_report(payload), atomic=False)
    return ModelBenchmarkSuiteResult(
        schema_version=MODEL_BENCHMARK_SUITE_RESULT_SCHEMA_VERSION,
        status=status,
        exit_code=exit_code,
        run_fingerprint=run_fingerprint,
        output_dir=root,
        suite_manifest_path=manifest_path,
        suite_report_path=report_path,
        summary=summary,
        cells=tuple(cells),
        artifacts=artifacts,
    )
