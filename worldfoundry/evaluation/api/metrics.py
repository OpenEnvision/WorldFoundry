"""Contracts and specifications for sample metrics and aggregate benchmark results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from worldfoundry.evaluation.api.json_contract import JsonContract, copy_mapping, tuple_of_str

from .artifacts import ArtifactRef, coerce_artifact_refs
from .generation import GenerationRequest, GenerationResult

METRIC_SPEC_SCHEMA_VERSION = "worldfoundry-metric-spec"
METRIC_RESULT_SCHEMA_VERSION = "worldfoundry-metric-result"
AGGREGATE_RESULT_SCHEMA_VERSION = "worldfoundry-aggregate-result"


@dataclass(frozen=True, init=False)
class MetricSpec(JsonContract):
    """Declarative metadata for one metric implementation."""

    id: str
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    description: str = ""
    version: str = "1.0"
    family: str = ""
    capability: str = ""
    requires_reference: bool = False
    required_artifacts: tuple[str, ...] = ()
    output_unit: str = ""
    higher_is_better: bool | None = None
    normalizer: str | None = None
    aggregator: str = "mean"
    statistics: tuple[str, ...] = ("mean",)
    primary: bool = False
    weight: float = 1.0
    implementation: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = METRIC_SPEC_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        id: str | None = None,
        *,
        metric_id: str | None = None,
        aliases: Sequence[str] = (),
        display_name: str = "",
        description: str = "",
        version: str = "1.0",
        family: str = "",
        capability: str = "",
        requires_reference: bool = False,
        required_artifacts: Sequence[str] = (),
        output_unit: str = "",
        higher_is_better: bool | None = None,
        normalizer: str | None = None,
        aggregator: str = "mean",
        statistics: Sequence[str] = ("mean",),
        primary: bool = False,
        weight: float = 1.0,
        implementation: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = METRIC_SPEC_SCHEMA_VERSION,
    ) -> None:
        resolved_id = id if id is not None else metric_id
        if not resolved_id:
            raise ValueError("MetricSpec requires id or metric_id.")
        if schema_version != METRIC_SPEC_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MetricSpec schema_version: {schema_version}")
        object.__setattr__(self, "id", str(resolved_id))
        object.__setattr__(self, "aliases", tuple_of_str(aliases))
        object.__setattr__(self, "display_name", str(display_name))
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "version", str(version))
        object.__setattr__(self, "family", str(family))
        object.__setattr__(self, "capability", str(capability))
        object.__setattr__(self, "requires_reference", bool(requires_reference))
        object.__setattr__(self, "required_artifacts", tuple_of_str(required_artifacts))
        object.__setattr__(self, "output_unit", str(output_unit))
        object.__setattr__(self, "higher_is_better", higher_is_better)
        object.__setattr__(self, "normalizer", normalizer)
        object.__setattr__(self, "aggregator", str(aggregator))
        object.__setattr__(self, "statistics", tuple_of_str(statistics))
        object.__setattr__(self, "primary", bool(primary))
        object.__setattr__(self, "weight", float(weight))
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "tags", tuple_of_str(tags))
        object.__setattr__(self, "metadata", copy_mapping(metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def metric_id(self) -> str:
        return self.id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSpec":
        return cls(
            id=data.get("id"),
            metric_id=data.get("metric_id"),
            aliases=data.get("aliases", ()),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            family=data.get("family", ""),
            capability=data.get("capability", ""),
            requires_reference=data.get("requires_reference", False),
            required_artifacts=data.get("required_artifacts", ()),
            output_unit=data.get("output_unit", ""),
            higher_is_better=data.get("higher_is_better"),
            normalizer=data.get("normalizer"),
            aggregator=data.get("aggregator", "mean"),
            statistics=data.get("statistics", ("mean",)),
            primary=data.get("primary", False),
            weight=data.get("weight", 1.0),
            implementation=data.get("implementation"),
            tags=data.get("tags", ()),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", METRIC_SPEC_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class MetricResult(JsonContract):
    """Per-sample metric output."""

    sample_id: str
    metric_id: str
    raw_value: Any = None
    normalized_value: float | None = None
    components: Mapping[str, Any] = field(default_factory=dict)
    valid: bool = True
    coverage: float = 1.0
    skip_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    judge_trace: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = METRIC_RESULT_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MetricResult schema_version: {self.schema_version}")
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "components", copy_mapping(self.components))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "diagnostics", copy_mapping(self.diagnostics))
        object.__setattr__(self, "artifact_refs", coerce_artifact_refs(self.artifact_refs))
        object.__setattr__(self, "judge_trace", copy_mapping(self.judge_trace))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricResult":
        return cls(
            sample_id=str(data["sample_id"]),
            metric_id=str(data["metric_id"]),
            raw_value=data.get("raw_value"),
            normalized_value=data.get("normalized_value"),
            components=data.get("components"),
            valid=data.get("valid", True),
            coverage=data.get("coverage", 1.0),
            skip_reason=data.get("skip_reason"),
            diagnostics=data.get("diagnostics"),
            artifact_refs=data.get("artifact_refs"),
            judge_trace=data.get("judge_trace"),
            schema_version=data.get("schema_version", METRIC_RESULT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class AggregateResult(JsonContract):
    """Aggregate metric output over a sample set."""

    metric_id: str
    n_total: int = 0
    n_valid: int = 0
    n_skipped: int = 0
    raw_stats: Mapping[str, Any] = field(default_factory=dict)
    normalized_stats: Mapping[str, Any] = field(default_factory=dict)
    confidence_interval: Mapping[str, Any] = field(default_factory=dict)
    stderr: float | None = None
    skip_breakdown: Mapping[str, int] = field(default_factory=dict)
    valid: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = AGGREGATE_RESULT_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __post_init__(self) -> None:
        if self.schema_version != AGGREGATE_RESULT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported AggregateResult schema_version: {self.schema_version}")
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "n_total", int(self.n_total))
        object.__setattr__(self, "n_valid", int(self.n_valid))
        object.__setattr__(self, "n_skipped", int(self.n_skipped))
        object.__setattr__(self, "raw_stats", copy_mapping(self.raw_stats))
        object.__setattr__(self, "normalized_stats", copy_mapping(self.normalized_stats))
        object.__setattr__(self, "confidence_interval", copy_mapping(self.confidence_interval))
        object.__setattr__(
            self,
            "skip_breakdown",
            {str(key): int(value) for key, value in (self.skip_breakdown or {}).items()},
        )
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "diagnostics", copy_mapping(self.diagnostics))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AggregateResult":
        return cls(
            metric_id=str(data["metric_id"]),
            n_total=data.get("n_total", 0),
            n_valid=data.get("n_valid", 0),
            n_skipped=data.get("n_skipped", 0),
            raw_stats=data.get("raw_stats"),
            normalized_stats=data.get("normalized_stats"),
            confidence_interval=data.get("confidence_interval"),
            stderr=data.get("stderr"),
            skip_breakdown=data.get("skip_breakdown"),
            valid=data.get("valid", True),
            diagnostics=data.get("diagnostics"),
            schema_version=data.get("schema_version", AGGREGATE_RESULT_SCHEMA_VERSION),
        )


def aggregate_mean(metric_id: str, rows: Sequence[MetricResult]) -> AggregateResult:
    """Mean numeric metric values while retaining validity and skip counts."""
    raw = [
        float(row.raw_value)
        for row in rows
        if row.valid and isinstance(row.raw_value, Real) and not isinstance(row.raw_value, bool)
    ]
    normalized = [float(row.normalized_value) for row in rows if row.valid and row.normalized_value is not None]
    values = (row.normalized_value if row.normalized_value is not None else row.raw_value for row in rows if row.valid)
    n_valid = sum(isinstance(value, Real) and not isinstance(value, bool) for value in values)
    return AggregateResult(
        metric_id=metric_id,
        n_total=len(rows),
        n_valid=n_valid,
        n_skipped=len(rows) - n_valid,
        valid=bool(raw or normalized),
        raw_stats={"mean": sum(raw) / len(raw)} if raw else {},
        normalized_stats={"mean": sum(normalized) / len(normalized)} if normalized else {},
        skip_breakdown=dict(Counter(row.skip_reason or "invalid" for row in rows if not row.valid)),
    )


@runtime_checkable
class Metric(Protocol):
    """Minimum metric implementation surface."""

    name: str
    version: str
    required_artifacts: tuple[str, ...]
    higher_is_better: bool | None

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> MetricResult:
        ...

    def aggregate(self, results: Sequence[MetricResult]) -> AggregateResult:
        ...


@runtime_checkable
class BatchMetric(Metric, Protocol):
    """Optional metric extension for accelerator-efficient sample batches."""

    def compute_batch(
        self,
        requests: Sequence[GenerationRequest],
        results: Sequence[GenerationResult],
    ) -> Sequence[Any] | Mapping[str, Any]:
        """Return one metric output per request, in order or keyed by sample id."""
        ...


def align_batch_metric_outputs(outputs: Any, sample_ids: Sequence[str]) -> tuple[Any, ...]:
    """Validate and align ordered or sample-id-keyed batch metric outputs."""

    expected_ids = tuple(str(sample_id) for sample_id in sample_ids)
    if isinstance(outputs, Mapping):
        keyed = {str(sample_id): value for sample_id, value in outputs.items()}
        missing = [sample_id for sample_id in expected_ids if sample_id not in keyed]
        unexpected = sorted(set(keyed).difference(expected_ids))
        if missing or unexpected:
            raise ValueError(
                "batch metric output sample ids do not match requests: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return tuple(keyed[sample_id] for sample_id in expected_ids)
    if not isinstance(outputs, Iterable) or isinstance(outputs, (str, bytes, bytearray)):
        raise TypeError("batch metric output must be an iterable or a sample-id mapping")
    aligned = tuple(outputs)
    if len(aligned) != len(expected_ids):
        raise ValueError(
            f"batch metric returned {len(aligned)} outputs for {len(expected_ids)} requests"
        )
    return aligned
