"""User-facing evaluation intents compiled onto the existing execution runners.

This module is the single boundary shared by CLI, TUI, and Workspace.  It owns
intent validation and preflight only; model inference, benchmark execution, and
metric calculation remain in their established runners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import yaml

from worldfoundry.evaluation.models.catalog.registry import load_model_zoo_registry
from worldfoundry.evaluation.tasks.catalog.zoo_registry import load_benchmark_zoo_registry
from worldfoundry.evaluation.tasks.datasets import validate_dataset_manifest
from worldfoundry.evaluation.tasks.execution.framework.in_tree_registry import target_benchmark_metrics
from worldfoundry.evaluation.tasks.metrics.registry import default_metric_registry
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR, write_json

from .benchmark_generation import get_benchmark_generation_adapter
from .evaluate import EvaluateRunRequest, execute_evaluate_run
from .fidelity import (
    EvaluationFidelity,
    artifact_scoring_fidelity,
    metric_only_fidelity,
    model_benchmark_fidelity,
    reproduction_fidelity,
)
from .materialize import materialize_requests_from_dataset_manifest
from .model_benchmark import (
    RESERVED_BENCHMARK_PARAMETERS,
    ModelBenchmarkRunRequest,
    run_model_benchmark,
)
from .reproduction_profiles import (
    DEFAULT_REPRODUCTION_PROFILE_ROOT,
    resolve_reproduction_profile,
)

PREPARED_EVALUATION_SCHEMA_VERSION = "worldfoundry-prepared-evaluation-v2"
REPRODUCTION_RECIPE_SCHEMA_VERSION = "worldfoundry-reproduction-recipe-v1"


@dataclass(frozen=True)
class PreparationIssue:
    """One actionable preflight diagnostic."""

    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ModelBenchmarkIntent:
    """Run an in-tree model on an official benchmark protocol."""

    output_dir: str | Path
    model_id: str
    benchmark_id: str
    model_variant_id: str | None = None
    requests_path: str | Path | None = None
    task_name: str | None = None
    dataset_root: str | Path | None = None
    dataset_id: str | None = None
    num_samples: int | None = None
    model_parameters: Mapping[str, Any] = field(default_factory=dict)
    model_runtime: Mapping[str, Any] = field(default_factory=dict)
    benchmark_env: Mapping[str, Any] = field(default_factory=dict)
    benchmark_parameters: Mapping[str, Any] = field(default_factory=dict)
    benchmark_mode: str = "official-run"
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "read-write"
    benchmark_manifest_dir: str | Path = BENCHMARK_ZOO_DIR
    model_manifest_dir: str | Path = MODEL_ZOO_DIR
    run_id: str | None = None


@dataclass(frozen=True)
class ScoreArtifactsIntent:
    """Apply a complete benchmark metric protocol to user-provided artifacts."""

    output_dir: str | Path
    benchmark_id: str
    artifact_dir: str | Path
    dataset_id: str | None = None
    benchmark_env: Mapping[str, Any] = field(default_factory=dict)
    benchmark_parameters: Mapping[str, Any] = field(default_factory=dict)
    benchmark_mode: str = "official-run"
    benchmark_manifest_dir: str | Path = BENCHMARK_ZOO_DIR
    leaderboard_candidate: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class GenerateAndScoreIntent:
    """Generate from a user dataset and apply executable metric ids."""

    output_dir: str | Path
    model_id: str
    dataset_manifest: str | Path
    metrics: Sequence[str]
    model_variant_id: str | None = None
    benchmark_id: str | None = None
    task_name: str = "custom-dataset"
    input_keys: Sequence[str] = ()
    output_keys: Sequence[str] = ("generated_video",)
    required_artifacts: Sequence[str] = ()
    generation_defaults: Mapping[str, Any] = field(default_factory=dict)
    model_parameters: Mapping[str, Any] = field(default_factory=dict)
    model_runtime: Mapping[str, Any] = field(default_factory=dict)
    model_manifest_dir: str | Path = MODEL_ZOO_DIR
    num_samples: int | None = None
    generation_cache_dir: str | Path | None = None
    generation_cache_mode: str = "read-write"
    run_id: str | None = None


@dataclass(frozen=True)
class ScoreResultsIntent:
    """Apply executable metric ids to WorldFoundry JSON/JSONL results."""

    output_dir: str | Path
    results_path: str | Path
    metrics: Sequence[str]
    requests_path: str | Path | None = None
    benchmark_id: str | None = None
    dataset_id: str | None = None
    required_artifacts: Sequence[str] = ()
    run_id: str | None = None


@dataclass(frozen=True)
class ReproduceIntent:
    """Execute an immutable model/benchmark recipe."""

    output_dir: str | Path
    recipe_path: str | Path | None = None
    profile_id: str | None = None
    benchmark_id: str | None = None
    profile_root: str | Path = DEFAULT_REPRODUCTION_PROFILE_ROOT


EvaluationIntent: TypeAlias = (
    ModelBenchmarkIntent | ScoreArtifactsIntent | GenerateAndScoreIntent | ScoreResultsIntent | ReproduceIntent
)
ExecutionRequest: TypeAlias = ModelBenchmarkRunRequest | EvaluateRunRequest


@dataclass(frozen=True)
class ReproductionRecipe:
    """Sparse model x benchmark configuration captured from an official source."""

    recipe_id: str
    benchmark: Mapping[str, Any]
    model: Mapping[str, Any]
    generation: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    reference: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    schema_version: str = REPRODUCTION_RECIPE_SCHEMA_VERSION

    @classmethod
    def from_path(cls, path: str | Path) -> "ReproductionRecipe":
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"reproduction recipe must be a mapping: {source}")
        schema_version = str(payload.get("schema_version") or REPRODUCTION_RECIPE_SCHEMA_VERSION)
        if schema_version != REPRODUCTION_RECIPE_SCHEMA_VERSION:
            raise ValueError(f"unsupported reproduction recipe schema: {schema_version}")
        benchmark = payload.get("benchmark")
        model = payload.get("model")
        if not isinstance(benchmark, Mapping) or not benchmark.get("id"):
            raise ValueError("reproduction recipe requires benchmark.id")
        if not isinstance(model, Mapping) or not model.get("id"):
            raise ValueError("reproduction recipe requires model.id")
        return cls(
            recipe_id=str(payload.get("id") or source.stem),
            benchmark=dict(benchmark),
            model=dict(model),
            generation=_mapping(payload.get("generation")),
            evaluation=_mapping(payload.get("evaluation")),
            data=_mapping(payload.get("data")),
            reference=_mapping(payload.get("reference")),
            source_path=source,
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class PreparedEvaluation:
    """Resolved request plus diagnostics shared by every frontend."""

    intent_kind: str
    classification: str
    request: ExecutionRequest | None
    fidelity: EvaluationFidelity
    issues: tuple[PreparationIssue, ...] = ()
    config_sources: Mapping[str, Any] = field(default_factory=dict)
    leaderboard_candidate: bool = False
    schema_version: str = PREPARED_EVALUATION_SCHEMA_VERSION

    @property
    def ready(self) -> bool:
        return self.request is not None and not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_kind": self.intent_kind,
            "classification": self.classification,
            "ready": self.ready,
            "leaderboard_candidate": self.leaderboard_candidate,
            "provenance": self.fidelity.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "config_sources": _jsonable(self.config_sources),
            "execution": None
            if self.request is None
            else {
                "kind": "model_benchmark" if isinstance(self.request, ModelBenchmarkRunRequest) else "evaluate",
                "request": _jsonable(asdict(self.request)),
            },
        }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prepared(
    *,
    intent_kind: str,
    classification: str,
    request: ExecutionRequest | None,
    fidelity: EvaluationFidelity,
    issues: Sequence[PreparationIssue] = (),
    config_sources: Mapping[str, Any] | None = None,
    leaderboard_candidate: bool = False,
) -> PreparedEvaluation:
    """Attach one policy decision to an existing executable request.

    The caller may request leaderboard candidacy, but provenance policy can only
    narrow that request.  Runner-specific evidence makes the final decision.
    """

    effective_candidate = bool(leaderboard_candidate and fidelity.leaderboard_candidate)
    provenance = fidelity.to_dict()
    resolved_request = request
    if isinstance(request, ModelBenchmarkRunRequest):
        resolved_request = replace(
            request,
            leaderboard_candidate=effective_candidate,
            evaluation_provenance=provenance,
        )
    elif isinstance(request, EvaluateRunRequest):
        resolved_request = replace(
            request,
            run_metadata={
                **dict(request.run_metadata or {}),
                "evaluation_provenance": provenance,
            },
        )
    return PreparedEvaluation(
        intent_kind=intent_kind,
        classification=classification,
        request=resolved_request,
        fidelity=fidelity,
        issues=tuple(issues),
        config_sources=dict(config_sources or {}),
        leaderboard_candidate=effective_candidate,
    )


def _issue(issues: list[PreparationIssue], severity: str, code: str, message: str) -> None:
    issues.append(PreparationIssue(severity=severity, code=code, message=message))


def _resolve_relative(value: Any, source: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return str((source.parent / path).resolve() if not path.is_absolute() else path.resolve())


def _resolve_benchmark(benchmark_id: str, manifest_dir: str | Path, issues: list[PreparationIssue]):
    try:
        entry = load_benchmark_zoo_registry(manifest_dir).get(benchmark_id)
    except Exception as exc:  # noqa: BLE001 - returned as an actionable preflight issue.
        _issue(issues, "error", "benchmark_not_found", str(exc))
        return None
    if entry.integration_status != "integrated":
        _issue(
            issues,
            "error",
            "benchmark_not_integrated",
            f"{entry.benchmark_id} is {entry.integration_status}",
        )
    if entry.verification_status != "verified":
        _issue(
            issues,
            "warning",
            "benchmark_not_verified",
            f"{entry.benchmark_id} is {entry.verification_status}; scores may not reproduce official evidence",
        )
    return entry


def _resolve_model(
    model_id: str,
    manifest_dir: str | Path,
    variant_id: str | None,
    issues: list[PreparationIssue],
):
    try:
        entry = load_model_zoo_registry(manifest_dir).get(model_id)
    except Exception as exc:  # noqa: BLE001 - returned as an actionable preflight issue.
        _issue(issues, "error", "model_not_found", str(exc))
        return None
    variants = {variant.variant_id: variant for variant in entry.variants}
    if variant_id and variant_id not in variants:
        _issue(
            issues,
            "error",
            "model_variant_not_found",
            f"model {entry.model_id} has no variant {variant_id!r}",
        )
    selected = variants.get(variant_id) if variant_id else None
    if not entry.runner_target and (selected is None or not selected.runner_target):
        _issue(issues, "error", "model_runner_missing", f"model {entry.model_id} has no in-tree runner")
    return entry


def _validate_path(
    value: str | Path | None,
    *,
    issues: list[PreparationIssue],
    code: str,
    directory: bool = False,
) -> Path | None:
    if value is None:
        _issue(issues, "error", code, "required path was not provided")
        return None
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        _issue(issues, "error", code, f"{kind} does not exist: {path}")
        return None
    return path


def _validate_metrics(
    metrics: Sequence[str],
    benchmark_id: str | None,
    issues: list[PreparationIssue],
) -> tuple[str, ...]:
    requested = tuple(str(metric) for metric in metrics)
    if not requested:
        _issue(issues, "error", "metrics_missing", "select at least one metric")
        return ()
    benchmark_metrics = target_benchmark_metrics().get((benchmark_id or "").strip().casefold())
    if benchmark_metrics is not None:
        unknown = [metric for metric in requested if metric not in benchmark_metrics]
        if unknown:
            _issue(
                issues,
                "error",
                "metric_not_in_benchmark",
                f"unsupported metrics for {benchmark_id}: {', '.join(unknown)}",
            )
        return requested

    registry = default_metric_registry()
    validation = registry.validate_ids(requested)
    for metric in validation["unknown_metrics"]:
        _issue(issues, "error", "metric_not_found", f"unknown metric id: {metric}")
    for item in validation["metrics"]:
        entry = registry.resolve_key(item["metric_id"])
        if entry.family != "existing_results":
            _issue(
                issues,
                "error",
                "metric_requires_runner",
                f"metric {item['metric_id']} is cataloged but has no generic existing-results executor; "
                "select its benchmark metric suite instead",
            )
    return tuple(item["canonical_metric_id"] for item in validation["metrics"])


def _validate_benchmark_parameters(
    parameters: Mapping[str, Any],
    issues: list[PreparationIssue],
) -> None:
    reserved = RESERVED_BENCHMARK_PARAMETERS.intersection(parameters)
    if reserved:
        _issue(
            issues,
            "error",
            "reserved_benchmark_parameter",
            f"benchmark parameters cannot override: {', '.join(sorted(reserved))}",
        )


def _prepare_model_benchmark(
    intent: ModelBenchmarkIntent,
    *,
    classification: str = "protocol_compliant",
    config_sources: Mapping[str, Any] | None = None,
    leaderboard_candidate: bool = True,
    fidelity: EvaluationFidelity | None = None,
) -> PreparedEvaluation:
    issues: list[PreparationIssue] = []
    benchmark = _resolve_benchmark(intent.benchmark_id, intent.benchmark_manifest_dir, issues)
    model = _resolve_model(intent.model_id, intent.model_manifest_dir, intent.model_variant_id, issues)
    _validate_benchmark_parameters(intent.benchmark_parameters, issues)

    requests_path = None
    if intent.requests_path is not None:
        requests_path = _validate_path(
            intent.requests_path,
            issues=issues,
            code="requests_path_missing",
        )
    elif intent.task_name:
        _validate_path(intent.dataset_root, issues=issues, code="dataset_root_missing", directory=True)
    else:
        adapter = get_benchmark_generation_adapter(intent.benchmark_id)
        if adapter is None:
            _issue(
                issues,
                "error",
                "generation_adapter_missing",
                f"{intent.benchmark_id} has no registered prompt provider; provide requests_path or task_name+dataset_root",
            )
        else:
            try:
                if not adapter.materialize_requests(limit=1, dataset_root=intent.dataset_root):
                    _issue(issues, "error", "benchmark_prompts_empty", adapter.missing_requests_hint)
            except Exception as exc:  # noqa: BLE001 - asset readiness belongs in preflight.
                _issue(issues, "error", "benchmark_prompts_unavailable", f"{type(exc).__name__}: {exc}")

    resolved_fidelity = fidelity or model_benchmark_fidelity(
        benchmark_mode=intent.benchmark_mode,
        custom_data=any(
            value not in (None, "")
            for value in (intent.requests_path, intent.task_name, intent.dataset_root, intent.dataset_id)
        ),
        sample_limited=intent.num_samples is not None,
        benchmark_parameters=intent.benchmark_parameters,
    )
    request = None
    if benchmark is not None and model is not None:
        request = ModelBenchmarkRunRequest(
            output_dir=intent.output_dir,
            benchmark_id=benchmark.benchmark_id,
            benchmark_manifest_path=intent.benchmark_manifest_dir,
            benchmark_mode=intent.benchmark_mode,
            model_id=model.model_id,
            model_zoo_manifest_dir=intent.model_manifest_dir,
            model_variant_id=intent.model_variant_id,
            model_parameters=dict(intent.model_parameters),
            model_runtime=dict(intent.model_runtime),
            requests_path=requests_path,
            task_name=intent.task_name,
            dataset_root=intent.dataset_root,
            dataset_id=intent.dataset_id,
            num_samples=intent.num_samples,
            generation_cache_dir=intent.generation_cache_dir,
            generation_cache_mode=intent.generation_cache_mode,
            benchmark_env=dict(intent.benchmark_env),
            benchmark_parameters=dict(intent.benchmark_parameters),
            run_id=intent.run_id,
            evaluation_kind=classification,
            leaderboard_candidate=leaderboard_candidate,
        )
    return _prepared(
        intent_kind="model_benchmark",
        classification=classification,
        request=request,
        fidelity=resolved_fidelity,
        issues=issues,
        config_sources=config_sources,
        leaderboard_candidate=leaderboard_candidate,
    )


def _prepare_score_artifacts(intent: ScoreArtifactsIntent) -> PreparedEvaluation:
    issues: list[PreparationIssue] = []
    fidelity = artifact_scoring_fidelity(benchmark_mode=intent.benchmark_mode)
    benchmark = _resolve_benchmark(intent.benchmark_id, intent.benchmark_manifest_dir, issues)
    _validate_benchmark_parameters(intent.benchmark_parameters, issues)
    artifact_dir = _validate_path(
        intent.artifact_dir,
        issues=issues,
        code="artifact_dir_missing",
        directory=True,
    )
    if artifact_dir is not None and not any(path.is_file() for path in artifact_dir.rglob("*")):
        _issue(issues, "error", "artifact_dir_empty", f"artifact directory is empty: {artifact_dir}")
    if intent.leaderboard_candidate:
        _issue(
            issues,
            "warning",
            "leaderboard_candidate_downgraded",
            "artifact coverage is not verified as official benchmark data; this run is diagnostic",
        )
    request = None
    if benchmark is not None and artifact_dir is not None:
        request = ModelBenchmarkRunRequest(
            output_dir=intent.output_dir,
            benchmark_id=benchmark.benchmark_id,
            benchmark_manifest_path=intent.benchmark_manifest_dir,
            benchmark_mode=intent.benchmark_mode,
            model_id="user-provided-artifacts",
            generated_artifact_dir=artifact_dir,
            dataset_id=intent.dataset_id or "user-provided-artifacts",
            benchmark_env=dict(intent.benchmark_env),
            benchmark_parameters=dict(intent.benchmark_parameters),
            run_id=intent.run_id,
            evaluation_kind="custom_dataset_metric_evaluation",
            leaderboard_candidate=intent.leaderboard_candidate,
        )
    return _prepared(
        intent_kind="score_artifacts",
        classification="custom_dataset_metric_evaluation",
        request=request,
        fidelity=fidelity,
        issues=issues,
        config_sources={"artifacts": None if artifact_dir is None else str(artifact_dir)},
        leaderboard_candidate=intent.leaderboard_candidate,
    )


def _prepare_generate_and_score(intent: GenerateAndScoreIntent) -> PreparedEvaluation:
    issues: list[PreparationIssue] = []
    fidelity = metric_only_fidelity(producer="catalog_model")
    model = _resolve_model(intent.model_id, intent.model_manifest_dir, intent.model_variant_id, issues)
    manifest_path = _validate_path(
        intent.dataset_manifest,
        issues=issues,
        code="dataset_manifest_missing",
    )
    metrics = _validate_metrics(intent.metrics, intent.benchmark_id, issues)
    requests = ()
    dataset_id = None
    dataset_validation: Mapping[str, Any] | None = None
    if manifest_path is not None:
        dataset_validation = validate_dataset_manifest(manifest_path)
        for message in dataset_validation.get("issues", ()):
            _issue(issues, "error", "dataset_manifest_invalid", str(message))
        for message in dataset_validation.get("warnings", ()):
            _issue(issues, "warning", "dataset_manifest_warning", str(message))
        try:
            if dataset_validation.get("ok"):
                materialized = materialize_requests_from_dataset_manifest(
                    manifest_path,
                    task_name=intent.task_name,
                    input_keys=intent.input_keys,
                    output_keys=intent.output_keys,
                    generation_defaults=intent.generation_defaults,
                    limit=intent.num_samples,
                )
                requests = materialized.requests
                dataset_id = materialized.benchmark_name
                if not requests:
                    _issue(issues, "error", "dataset_empty", "dataset manifest produced zero requests")
        except Exception as exc:  # noqa: BLE001 - dataset errors are preflight diagnostics.
            _issue(issues, "error", "dataset_materialization_failed", f"{type(exc).__name__}: {exc}")
    request = None
    if model is not None and requests and metrics:
        request = EvaluateRunRequest(
            output_dir=intent.output_dir,
            mode="model",
            requests=requests,
            metrics=metrics,
            required_artifacts=tuple(intent.required_artifacts),
            benchmark_id=intent.benchmark_id,
            model_id=model.model_id,
            model_zoo_manifest_dir=intent.model_manifest_dir,
            model_variant_id=intent.model_variant_id,
            model_parameters=dict(intent.model_parameters),
            model_runtime=dict(intent.model_runtime),
            dataset_id=dataset_id,
            generation_cache_dir=intent.generation_cache_dir,
            generation_cache_mode=intent.generation_cache_mode,
            run_id=intent.run_id,
        )
    return _prepared(
        intent_kind="generate_and_score",
        classification="custom_dataset_metric_evaluation",
        request=request,
        fidelity=fidelity,
        issues=issues,
        config_sources={
            "dataset_manifest": None if manifest_path is None else str(manifest_path),
            "dataset_validation": _jsonable(dataset_validation),
        },
    )


def _prepare_score_results(intent: ScoreResultsIntent) -> PreparedEvaluation:
    issues: list[PreparationIssue] = []
    fidelity = metric_only_fidelity(producer="imported_results")
    results_path = _validate_path(intent.results_path, issues=issues, code="results_path_missing")
    requests_path = None
    if intent.requests_path is not None:
        requests_path = _validate_path(intent.requests_path, issues=issues, code="requests_path_missing")
    metrics = _validate_metrics(intent.metrics, intent.benchmark_id, issues)
    request = None
    if results_path is not None and metrics:
        request = EvaluateRunRequest(
            output_dir=intent.output_dir,
            mode="existing-results",
            results_path=results_path,
            requests_path=requests_path,
            metrics=metrics,
            required_artifacts=tuple(intent.required_artifacts),
            benchmark_id=intent.benchmark_id,
            dataset_id=intent.dataset_id,
            run_id=intent.run_id,
        )
    return _prepared(
        intent_kind="score_results",
        classification="custom_results_metric_evaluation",
        request=request,
        fidelity=fidelity,
        issues=issues,
        config_sources={"results": None if results_path is None else str(results_path)},
    )


def _prepare_reproduction(intent: ReproduceIntent) -> PreparedEvaluation:
    try:
        selectors = sum(
            value not in (None, "")
            for value in (intent.recipe_path, intent.profile_id, intent.benchmark_id)
        )
        if selectors != 1:
            raise ValueError("select exactly one of recipe_path, profile_id, or benchmark_id")
        recipe_path = (
            Path(intent.recipe_path)
            if intent.recipe_path is not None
            else resolve_reproduction_profile(
                profile_id=intent.profile_id,
                benchmark_id=intent.benchmark_id,
                root=intent.profile_root,
            )
        )
        recipe = ReproductionRecipe.from_path(recipe_path)
    except Exception as exc:  # noqa: BLE001 - malformed recipes are returned as preflight issues.
        issue = PreparationIssue("error", "recipe_invalid", f"{type(exc).__name__}: {exc}")
        fidelity = EvaluationFidelity(
            producer="catalog_model",
            generation="unknown",
            data="unknown",
            evaluation="unknown",
            runtime="unknown",
            reference="unknown",
            reasons=("reproduction recipe could not be resolved",),
        )
        return _prepared(
            intent_kind="reproduce",
            classification="reproduction",
            request=None,
            fidelity=fidelity,
            issues=(issue,),
            leaderboard_candidate=False,
        )
    assert recipe.source_path is not None
    source = recipe.source_path
    model_parameters = {**_mapping(recipe.model.get("parameters")), **dict(recipe.generation)}
    if recipe.model.get("revision"):
        model_parameters["revision"] = str(recipe.model["revision"])
    model_runtime = _mapping(recipe.model.get("runtime"))
    benchmark_env = _mapping(recipe.evaluation.get("env"))
    benchmark_parameters = _mapping(recipe.evaluation.get("parameters"))
    if recipe.benchmark.get("revision"):
        benchmark_parameters["revision"] = str(recipe.benchmark["revision"])
    recipe_fidelity = reproduction_fidelity(
        benchmark_mode=str(recipe.evaluation.get("mode") or "official-run"),
        benchmark_revision=bool(recipe.benchmark.get("revision")),
        model_revision=bool(recipe.model.get("revision")),
        data=recipe.data,
        evaluation_parameters=_mapping(recipe.evaluation.get("parameters")),
        reference=recipe.reference,
    )
    requests_path = _resolve_relative(recipe.data.get("requests_path"), source)
    data_source = _jsonable(recipe.data)
    if requests_path is not None:
        requests_source = Path(requests_path)
        data_source = {
            **dict(data_source),
            "requests_path": requests_path,
            "requests_sha256": _sha256(requests_source) if requests_source.is_file() else None,
        }
    model_intent = ModelBenchmarkIntent(
        output_dir=intent.output_dir,
        model_id=str(recipe.model["id"]),
        model_variant_id=recipe.model.get("variant"),
        benchmark_id=str(recipe.benchmark["id"]),
        requests_path=requests_path,
        task_name=recipe.data.get("task_name"),
        dataset_root=_resolve_relative(recipe.data.get("root"), source),
        dataset_id=recipe.data.get("id"),
        num_samples=recipe.data.get("num_samples"),
        model_parameters=model_parameters,
        model_runtime=model_runtime,
        benchmark_env=benchmark_env,
        benchmark_parameters=benchmark_parameters,
        benchmark_mode=str(recipe.evaluation.get("mode") or "official-run"),
        generation_cache_mode="read-write",
        run_id=recipe.recipe_id,
    )
    prepared = _prepare_model_benchmark(
        model_intent,
        classification="reproduction",
        config_sources={
            "recipe_id": recipe.recipe_id,
            "recipe_path": str(source),
            "recipe_sha256": _sha256(source),
            "benchmark": _jsonable(recipe.benchmark),
            "model": _jsonable(recipe.model),
            "data": data_source,
            "reference": _jsonable(recipe.reference),
        },
        leaderboard_candidate=bool(recipe.reference.get("leaderboard_valid", False)),
        fidelity=recipe_fidelity,
    )
    return replace(prepared, intent_kind="reproduce")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_evaluation(intent: EvaluationIntent) -> PreparedEvaluation:
    """Compile one user intent into an executable request without loading a model."""

    if isinstance(intent, ReproduceIntent):
        return _prepare_reproduction(intent)
    if isinstance(intent, ModelBenchmarkIntent):
        return _prepare_model_benchmark(intent)
    if isinstance(intent, ScoreArtifactsIntent):
        return _prepare_score_artifacts(intent)
    if isinstance(intent, GenerateAndScoreIntent):
        return _prepare_generate_and_score(intent)
    if isinstance(intent, ScoreResultsIntent):
        return _prepare_score_results(intent)
    raise TypeError(f"unsupported evaluation intent: {type(intent).__name__}")


def execute_prepared_evaluation(prepared: PreparedEvaluation) -> Any:
    """Execute a ready prepared evaluation through the established runner."""

    if not prepared.ready or prepared.request is None:
        messages = "; ".join(issue.message for issue in prepared.issues if issue.severity == "error")
        raise ValueError(f"evaluation preflight failed: {messages or 'request is not executable'}")
    output_dir = Path(prepared.request.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "prepared_evaluation.json", prepared.to_dict(), atomic=False)
    if isinstance(prepared.request, ModelBenchmarkRunRequest):
        return run_model_benchmark(prepared.request)
    return execute_evaluate_run(prepared.request)


def execute_evaluation(intent: EvaluationIntent) -> Any:
    """Prepare and execute one evaluation intent."""

    return execute_prepared_evaluation(prepare_evaluation(intent))


__all__ = [
    "EvaluationIntent",
    "EvaluationFidelity",
    "GenerateAndScoreIntent",
    "ModelBenchmarkIntent",
    "PREPARED_EVALUATION_SCHEMA_VERSION",
    "PreparedEvaluation",
    "PreparationIssue",
    "REPRODUCTION_RECIPE_SCHEMA_VERSION",
    "ReproduceIntent",
    "ReproductionRecipe",
    "ScoreArtifactsIntent",
    "ScoreResultsIntent",
    "execute_evaluation",
    "execute_prepared_evaluation",
    "prepare_evaluation",
]
