"""Evaluator classes and registry mapping external benchmark metrics to execution backends.

This module resolves how third-party benchmarks are checked or computed (e.g. locally or blocked).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.api import MetricResult
from worldfoundry.evaluation.api.json_contract import to_plain

from ....contracts import ExternalBenchmarkContract, list_external_benchmark_contracts
from .bindings import FORMULA_EVALUATOR_BINDINGS, success_metric_bindings
from .local_evaluators import LOCAL_EVALUATORS, _failure_result, _metric_key, _tuple_of_str

JsonValue = Any
MetricEvaluationCallable = Callable[["ExternalMetricEvaluationRequest", "ExternalMetricEvaluatorEntry"], MetricResult]
BLOCKED_EVALUATION_KINDS = frozenset({"blocked", "judge_required", "api_required", "external_runtime_required"})


def _benchmark_key(value: str) -> str:
    """Normalize a benchmark ID key.

    Args:
        value: Raw benchmark ID.

    Returns:
        Normalized lowercase key.
    """
    return value.strip().casefold()


def _blocked_result(request: "ExternalMetricEvaluationRequest", entry: "ExternalMetricEvaluatorEntry") -> MetricResult:
    """
    Build a structured blocked metric result for judge/API/upstream metrics.

    Args:
        request: Current metric evaluation request.
        entry: Evaluator registry entry describing the blocker.

    Returns:
        A MetricResult indicating the metric is blocked.
    """
    skip_reason = entry.blocked_reason or entry.evaluation_kind
    return MetricResult(
        sample_id=request.sample_id,
        metric_id=request.metric_id,
        valid=False,
        coverage=0.0,
        skip_reason=skip_reason,
        diagnostics={
            "status": "blocked",
            "benchmark_id": request.benchmark_id,
            "metric_id": request.metric_id,
            "reason": skip_reason,
            "requires_judge": entry.requires_judge,
            "requires_api": entry.requires_api,
            "requires_upstream_runtime": entry.requires_upstream_runtime,
            "description": entry.description,
        },
    )


@dataclass(frozen=True)
class ExternalMetricEvaluationRequest:
    """
    Represents a request to evaluate an external benchmark metric.

    This dataclass encapsulates all necessary information for an external
    metric evaluation, including benchmark and metric IDs, generated artifacts,
    task metadata, reference data, and sample identification.
    """

    benchmark_id: str
    metric_id: str
    generated_artifact_manifest: JsonValue = None
    task_metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    reference: Mapping[str, JsonValue] = field(default_factory=dict)
    sample_id: str = "external-benchmark:sample"
    artifact_base_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """
        Ensures that certain fields are properly cast to their expected types
        after initialization, making the dataclass truly immutable with correct types.
        """
        # Using object.__setattr__ to bypass dataclass's frozen property for type coercion
        object.__setattr__(self, "benchmark_id", str(self.benchmark_id))
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "task_metadata", dict(self.task_metadata or {}))
        object.__setattr__(self, "reference", dict(self.reference or {}))
        object.__setattr__(self, "sample_id", str(self.sample_id))


@dataclass(frozen=True)
class ExternalMetricEvaluatorEntry:
    """
    Defines an entry in the external metric evaluator registry.

    This dataclass specifies how a particular external benchmark metric should
    be evaluated, including its kind (e.g., local, blocked), required artifacts,
    and any dependencies like judges or external APIs.
    """

    benchmark_id: str
    metric_id: str
    evaluation_kind: str = "blocked"
    local_evaluator: str | None = None
    required_artifacts: tuple[str, ...] = ()
    requires_judge: bool = False
    requires_api: bool = False
    requires_upstream_runtime: bool = True
    blocked_reason: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """
        Ensures that certain fields are properly cast to their expected types
        after initialization, making the dataclass truly immutable with correct types.
        """
        # Using object.__setattr__ to bypass dataclass's frozen property for type coercion
        object.__setattr__(self, "benchmark_id", str(self.benchmark_id))
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "evaluation_kind", str(self.evaluation_kind))
        object.__setattr__(self, "required_artifacts", _tuple_of_str(self.required_artifacts))
        if self.local_evaluator is not None:
            object.__setattr__(self, "local_evaluator", str(self.local_evaluator))
        if self.blocked_reason is not None:
            object.__setattr__(self, "blocked_reason", str(self.blocked_reason))

    @property
    def is_local(self) -> bool:
        """
        Indicates if the evaluation kind is 'local_deterministic'.

        Returns:
            True if the evaluation kind is 'local_deterministic', False otherwise.
        """
        return self.evaluation_kind == "local_deterministic"

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the dataclass instance to a plain dictionary.

        Returns:
            A dictionary representation of the evaluator entry.
        """
        return to_plain(self)

    def evaluate(self, request: ExternalMetricEvaluationRequest) -> MetricResult:
        """
        Evaluate one metric request into a MetricResult.

        Args:
            request: Benchmark, metric, artifact manifest, metadata, and reference payload.

        Returns:
            A MetricResult object containing the evaluation outcome.
        """
        # Handle cases where evaluation is explicitly blocked
        if self.evaluation_kind in BLOCKED_EVALUATION_KINDS:
            return _blocked_result(request, self)
        # Handle unsupported evaluation kinds
        if self.evaluation_kind != "local_deterministic":
            return _failure_result(
                request,
                skip_reason="unsupported_metric_evaluator",
                message=f"unsupported external metric evaluator kind: {self.evaluation_kind}",
                diagnostics={"evaluation_kind": self.evaluation_kind},
            )
        # Handle cases where a local evaluator is specified but not registered
        if self.local_evaluator not in LOCAL_EVALUATORS:
            return _failure_result(
                request,
                skip_reason="invalid_metric_evaluator",
                message=f"unknown local evaluator: {self.local_evaluator}",
                diagnostics={"local_evaluator": self.local_evaluator},
            )
        # If all checks pass, execute the local evaluator
        return LOCAL_EVALUATORS[self.local_evaluator](request, self)


class ExternalMetricEvaluatorRegistry:
    """
    Registry for external benchmark metric evaluators.

    This class manages a collection of `ExternalMetricEvaluatorEntry` instances,
    allowing for registration, lookup, and evaluation of metrics based on their
    benchmark and metric IDs. It can be initialized with built-in evaluators
    and those derived from external benchmark contracts.
    """

    def __init__(
        self,
        entries: Sequence[ExternalMetricEvaluatorEntry | Mapping[str, Any]] = (),
        *,
        include_builtins: bool = True,
        include_external_contracts: bool = True,
    ) -> None:
        """
        Initializes the ExternalMetricEvaluatorRegistry.

        Args:
            entries: A sequence of initial evaluator entries or mappings to register.
            include_builtins: If True, include default, generic evaluators like
                              'artifact_count' and 'required_artifacts_present'.
            include_external_contracts: If True, load evaluators based on discovered
                                        external benchmark contracts.
        """
        self._entries: dict[tuple[str, str], ExternalMetricEvaluatorEntry] = {}
        if include_builtins:
            # Register common, generic local evaluators
            self.register(
                ExternalMetricEvaluatorEntry(
                    benchmark_id="*",
                    metric_id="artifact_count",
                    evaluation_kind="local_deterministic",
                    local_evaluator="artifact_count",
                    requires_upstream_runtime=False,
                    description="Counts generated artifacts in a manifest.",
                )
            )
            self.register(
                ExternalMetricEvaluatorEntry(
                    benchmark_id="*",
                    metric_id="required_artifacts_present",
                    evaluation_kind="local_deterministic",
                    local_evaluator="required_artifacts_present",
                    requires_upstream_runtime=False,
                    description="Checks that required generated artifacts are present.",
                )
            )
        if include_external_contracts:
            # Register evaluators derived from external benchmark contracts
            for contract in list_external_benchmark_contracts():
                for entry in _entries_from_contract(contract):
                    self.register(entry)
            # Register evaluators for formula-based metrics
            _register_formula_evaluators(self)
        # Register any user-provided entries last, allowing them to override
        for entry in entries:
            self.register(entry)

    def register(
        self,
        entry: ExternalMetricEvaluatorEntry | Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> ExternalMetricEvaluatorEntry:
        """
        Register one external metric evaluator.

        Args:
            entry: Evaluator entry or mapping.
            replace: Whether to replace an existing entry if a key conflict occurs.

        Returns:
            The registered `ExternalMetricEvaluatorEntry` instance.

        Raises:
            ValueError: If an entry with the same benchmark_id and metric_id already
                        exists and `replace` is False.
        """
        evaluator = entry if isinstance(entry, ExternalMetricEvaluatorEntry) else ExternalMetricEvaluatorEntry(**entry)
        key = (_benchmark_key(evaluator.benchmark_id), _metric_key(evaluator.metric_id))
        # Prevent overwriting existing entries unless 'replace' is explicitly True
        if key in self._entries and not replace:
            raise ValueError(
                f"external metric evaluator already exists: {evaluator.benchmark_id}:{evaluator.metric_id}"
            )
        self._entries[key] = evaluator
        return evaluator

    def list(self, benchmark_id: str | None = None) -> tuple[ExternalMetricEvaluatorEntry, ...]:
        """
        List registered evaluator entries.

        Args:
            benchmark_id: Optional benchmark ID filter. If provided, only entries
                          matching this benchmark ID or wildcard ("*") are returned.

        Returns:
            A tuple of matching `ExternalMetricEvaluatorEntry` instances.
        """
        if benchmark_id is None:
            return tuple(self._entries.values())
        key = _benchmark_key(benchmark_id)
        # Filter entries based on exact benchmark ID or wildcard match
        return tuple(entry for entry in self._entries.values() if _benchmark_key(entry.benchmark_id) in {key, "*"})

    def get(self, benchmark_id: str, metric_id: str) -> ExternalMetricEvaluatorEntry:
        """
        Resolve a benchmark metric evaluator.

        Attempts to find an exact match first, then falls back to a wildcard benchmark_id match.

        Args:
            benchmark_id: External benchmark id.
            metric_id: Metric id from the benchmark contract or generic local metrics.

        Returns:
            The matched `ExternalMetricEvaluatorEntry`.

        Raises:
            KeyError: If no evaluator is found for the given benchmark and metric IDs.
        """
        exact_key = (_benchmark_key(benchmark_id), _metric_key(metric_id))
        wildcard_key = ("*", _metric_key(metric_id))
        # Prioritize exact benchmark_id match
        if exact_key in self._entries:
            return self._entries[exact_key]
        # Fallback to wildcard benchmark_id match if no exact match is found
        if wildcard_key in self._entries:
            return self._entries[wildcard_key]
        raise KeyError(f"unknown external metric evaluator: {benchmark_id}:{metric_id}")

    def evaluate(self, request: ExternalMetricEvaluationRequest) -> MetricResult:
        """
        Evaluate one request through the resolved evaluator.

        Args:
            request: Benchmark, metric, artifacts, task metadata, and reference payload.

        Returns:
            A `MetricResult` object from the evaluation.
        """
        entry = self.get(request.benchmark_id, request.metric_id)
        return entry.evaluate(request)


def _contract_blocked_reason(contract: ExternalBenchmarkContract) -> tuple[str, bool, bool]:
    """
    Infer the explicit blocked category for a contract-only metric.

    Analyzes contract text to determine if a judge or API is required for evaluation.

    Args:
        contract: External benchmark contract metadata.

    Returns:
        A tuple containing:
        - The inferred blocked reason string.
        - A boolean indicating if a judge is required.
        - A boolean indicating if an API is required.
    """
    # Combine relevant contract fields into a single lowercase string for keyword search
    text = " ".join(
        (
            contract.display_name,
            *contract.input_keys,
            *contract.output_keys,
            *contract.notes,
        )
    ).casefold()
    requires_api = any(token in text for token in ("api", "gpt", "gemini", "openai"))
    requires_judge = any(token in text for token in ("judge", "vlm", "mllm", "autoeval", "checkpoint"))
    if requires_api:
        return "judge_api_required", requires_judge, True
    if requires_judge:
        return "judge_required", True, False
    return "external_runtime_required", False, False


def _entries_from_contract(contract: ExternalBenchmarkContract) -> tuple[ExternalMetricEvaluatorEntry, ...]:
    """
    Build blocked evaluator entries for one external benchmark contract.

    These entries signify that the metrics defined in the contract require
    external processing (e.g., judge, API, or external runtime) and cannot
    be evaluated locally by default.

    Args:
        contract: External benchmark contract metadata.

    Returns:
        A tuple of `ExternalMetricEvaluatorEntry` instances, one for each metric
        defined in the contract, marked as blocked.
    """
    blocked_reason, requires_judge, requires_api = _contract_blocked_reason(contract)
    return tuple(
        ExternalMetricEvaluatorEntry(
            benchmark_id=contract.benchmark_id,
            metric_id=metric_id,
            evaluation_kind="blocked",
            requires_judge=requires_judge,
            requires_api=requires_api,
            requires_upstream_runtime=contract.requires_upstream_runtime,
            blocked_reason=blocked_reason,
            description=f"{contract.display_name} metric requires external benchmark evaluation.",
        )
        for metric_id in contract.metric_ids
    )


def _register_formula_evaluators(registry: ExternalMetricEvaluatorRegistry) -> None:
    """Register custom formula based local metric evaluators to the provided registry.

    These evaluators typically implement metric calculations that can be performed
    locally based on existing data, rather than requiring external systems.

    Args:
        registry: Target ExternalMetricEvaluatorRegistry to register within.
    """
    # Register formula evaluators from FORMULA_EVALUATOR_BINDINGS
    for benchmark_id, metric_id, evaluator, description in FORMULA_EVALUATOR_BINDINGS:
        registry.register(
            ExternalMetricEvaluatorEntry(
                benchmark_id=benchmark_id,
                metric_id=metric_id,
                evaluation_kind="local_deterministic",
                local_evaluator=evaluator,
                requires_upstream_runtime=False,
                blocked_reason=None,
                description=description,
            ),
            replace=True,  # Allow replacing existing entries if a formula provides a local override
        )
    # Register success rate metric evaluators from success_metric_bindings
    for benchmark_id, metric_id in success_metric_bindings():
        registry.register(
            ExternalMetricEvaluatorEntry(
                benchmark_id=benchmark_id,
                metric_id=metric_id,
                evaluation_kind="local_deterministic",
                local_evaluator="success_rate",
                requires_upstream_runtime=False,
                description="Embodied benchmark success-rate aggregation from existing episode/result rows.",
            ),
            replace=True,  # Allow replacing existing entries
        )


_DEFAULT_EXTERNAL_METRIC_EVALUATOR_REGISTRY = ExternalMetricEvaluatorRegistry()


def default_external_metric_evaluator_registry() -> ExternalMetricEvaluatorRegistry:
    """Access the global default ExternalMetricEvaluatorRegistry instance.

    This function provides a singleton instance of the registry, pre-populated
    with built-in and contract-derived evaluators.

    Returns:
        The default registry.
    """
    return _DEFAULT_EXTERNAL_METRIC_EVALUATOR_REGISTRY


def list_external_metric_evaluators(benchmark_id: str | None = None) -> tuple[ExternalMetricEvaluatorEntry, ...]:
    """Retrieve list of registered ExternalMetricEvaluatorEntries.

    Args:
        benchmark_id: Optional filter for a specific benchmark. If provided,
                      only evaluators for that benchmark or generic ("*") will be listed.

    Returns:
        Tuple of registered evaluator entries.
    """
    return default_external_metric_evaluator_registry().list(benchmark_id=benchmark_id)


def get_external_metric_evaluator(benchmark_id: str, metric_id: str) -> ExternalMetricEvaluatorEntry:
    """Lookup and retrieve a specific ExternalMetricEvaluatorEntry.

    Args:
        benchmark_id: Target benchmark ID.
        metric_id: Target metric ID.

    Returns:
        The matched ExternalMetricEvaluatorEntry.

    Raises:
        KeyError: If no evaluator is found for the given benchmark and metric IDs.
    """
    return default_external_metric_evaluator_registry().get(benchmark_id, metric_id)


def evaluate_external_metric(
    benchmark_id: str,
    metric_id: str,
    *,
    generated_artifact_manifest: JsonValue = None,
    task_metadata: Mapping[str, JsonValue] | None = None,
    reference: Mapping[str, JsonValue] | None = None,
    sample_id: str = "external-benchmark:sample",
    artifact_base_dir: str | Path | None = None,
) -> MetricResult:
    """
    Evaluate one external benchmark metric request.

    This is a convenience function that uses the default global registry
    to find and execute the appropriate evaluator for a given metric.

    Args:
        benchmark_id: External benchmark id.
        metric_id: Metric id to evaluate.
        generated_artifact_manifest: Generated artifacts from a model run.
        task_metadata: Benchmark/task metadata, including optional required_artifacts.
        reference: Reference answer or metadata payload.
        sample_id: Stable sample id for the structured metric result.
        artifact_base_dir: Optional base directory for relative artifact URIs.

    Returns:
        A `MetricResult` object from the evaluation.
    """
    request = ExternalMetricEvaluationRequest(
        benchmark_id=benchmark_id,
        metric_id=metric_id,
        generated_artifact_manifest=generated_artifact_manifest,
        task_metadata=dict(task_metadata or {}),
        reference=dict(reference or {}),
        sample_id=sample_id,
        artifact_base_dir=artifact_base_dir,
    )
    return default_external_metric_evaluator_registry().evaluate(request)
