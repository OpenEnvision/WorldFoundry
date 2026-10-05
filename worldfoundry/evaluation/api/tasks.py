from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api.json_contract import JsonContract, copy_mapping, tuple_of_str

from .metrics import MetricSpec

EVALUATION_PROTOCOL_SCHEMA_VERSION = "worldfoundry-evaluation-protocol"
WORLD_TASK_CONFIG_SCHEMA_VERSION = "worldfoundry-task"
BENCHMARK_SPEC_SCHEMA_VERSION = "worldfoundry-benchmark"


@dataclass(frozen=True)
class EvaluationProtocolSpec(JsonContract):
    """Evaluation protocol entry for a task or benchmark catalog task.

    The public API keeps simple string protocols supported, while catalog
    surfaces can use this structured form to attach metric ids, groups, and
    protocol-specific metadata.
    """

    name: str
    metric_ids: tuple[str, ...] = ()
    metric_groups: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVALUATION_PROTOCOL_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvaluationProtocolSpec":
        name = data.get("name", data.get("id", data.get("protocol", data.get("type"))))
        if not name:
            raise ValueError("EvaluationProtocolSpec requires name, id, protocol, or type.")
        known_keys = {
            "name",
            "id",
            "protocol",
            "type",
            "metric_ids",
            "metrics",
            "metric_groups",
            "metadata",
            "schema_version",
        }
        metadata = dict(data.get("metadata", {}))
        metadata.update({str(key): value for key, value in data.items() if key not in known_keys})
        return cls(
            name=str(name),
            metric_ids=tuple_of_str(data.get("metric_ids", data.get("metrics", ()))),
            metric_groups=tuple_of_str(data.get("metric_groups", ())),
            metadata=metadata,
            schema_version=str(data.get("schema_version", EVALUATION_PROTOCOL_SCHEMA_VERSION)),
        )

    from_dict = from_mapping

    @classmethod
    def coerce_many(cls, value: Any) -> tuple["EvaluationProtocolSpec", ...]:
        if value is None:
            return ()
        if isinstance(value, EvaluationProtocolSpec):
            return (value,)
        if isinstance(value, str):
            return (cls(name=value),)
        if isinstance(value, Mapping):
            if "items" in value and isinstance(value.get("items"), Sequence):
                return tuple(cls.coerce_many(item)[0] for item in value["items"])
            return (cls.from_mapping(value),)

        result: list[EvaluationProtocolSpec] = []
        for item in value:
            result.extend(cls.coerce_many(item))
        return tuple(result)


@dataclass(frozen=True, init=False)
class WorldTaskConfig(JsonContract):
    """Task-level data and generation contract."""

    name: str
    protocol: str = "open_loop"
    evaluation_protocol: str = "reference_metrics"
    capability_track: str = "core_video"
    schema_type: str = "sample"
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ("generated_video",)
    metric_ids: tuple[str, ...] = ()
    metric_groups: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)
    generation_defaults: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = WORLD_TASK_CONFIG_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        name: str | None = None,
        *,
        task_id: str | None = None,
        protocol: str = "open_loop",
        evaluation_protocol: str = "reference_metrics",
        capability_track: str = "core_video",
        schema_type: str = "sample",
        input_keys: Sequence[str] = (),
        output_keys: Sequence[str] = ("generated_video",),
        metric_ids: Sequence[str] = (),
        metric_groups: Sequence[str] = (),
        tags: Sequence[str] = (),
        description: str = "",
        data: Mapping[str, Any] | None = None,
        generation_defaults: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = WORLD_TASK_CONFIG_SCHEMA_VERSION,
    ) -> None:
        resolved_name = name if name is not None else task_id
        if not resolved_name:
            raise ValueError("WorldTaskConfig requires name or task_id.")
        if schema_version != WORLD_TASK_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported WorldTaskConfig schema_version: {schema_version}")
        object.__setattr__(self, "name", str(resolved_name))
        object.__setattr__(self, "protocol", str(protocol))
        object.__setattr__(self, "evaluation_protocol", str(evaluation_protocol))
        object.__setattr__(self, "capability_track", str(capability_track))
        object.__setattr__(self, "schema_type", str(schema_type))
        object.__setattr__(self, "input_keys", tuple_of_str(input_keys))
        object.__setattr__(self, "output_keys", tuple_of_str(output_keys))
        object.__setattr__(self, "metric_ids", tuple_of_str(metric_ids))
        object.__setattr__(self, "metric_groups", tuple_of_str(metric_groups))
        object.__setattr__(self, "tags", tuple_of_str(tags))
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "data", copy_mapping(data))
        object.__setattr__(self, "generation_defaults", copy_mapping(generation_defaults))
        object.__setattr__(self, "metadata", copy_mapping(metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def task_id(self) -> str:
        return self.name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldTaskConfig":
        return cls(
            name=data.get("name"),
            task_id=data.get("task_id"),
            protocol=data.get("protocol", "open_loop"),
            evaluation_protocol=data.get("evaluation_protocol", "reference_metrics"),
            capability_track=data.get("capability_track", "core_video"),
            schema_type=data.get("schema_type", "sample"),
            input_keys=data.get("input_keys", ()),
            output_keys=data.get("output_keys", ("generated_video",)),
            metric_ids=data.get("metric_ids", data.get("metrics", ())),
            metric_groups=data.get("metric_groups", ()),
            tags=data.get("tags", ()),
            description=data.get("description", ""),
            data=data.get("data"),
            generation_defaults=data.get("generation_defaults"),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", WORLD_TASK_CONFIG_SCHEMA_VERSION),
        )


@dataclass(frozen=True, init=False)
class BenchmarkSpec(JsonContract):
    """Benchmark-level collection of tasks, metrics, and dataset metadata."""

    name: str
    version: str = "1.0"
    tasks: tuple[WorldTaskConfig, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    splits: tuple[str, ...] = ("default",)
    tags: tuple[str, ...] = ()
    description: str = ""
    dataset_root: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = BENCHMARK_SPEC_SCHEMA_VERSION
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        name: str | None = None,
        *,
        benchmark_id: str | None = None,
        version: str = "1.0",
        tasks: Sequence[WorldTaskConfig | Mapping[str, Any]] = (),
        metrics: Sequence[MetricSpec | Mapping[str, Any]] = (),
        splits: Sequence[str] = ("default",),
        tags: Sequence[str] = (),
        description: str = "",
        dataset_root: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = BENCHMARK_SPEC_SCHEMA_VERSION,
    ) -> None:
        resolved_name = name if name is not None else benchmark_id
        if not resolved_name:
            raise ValueError("BenchmarkSpec requires name or benchmark_id.")
        if schema_version != BENCHMARK_SPEC_SCHEMA_VERSION:
            raise ValueError(f"Unsupported BenchmarkSpec schema_version: {schema_version}")
        object.__setattr__(self, "name", str(resolved_name))
        object.__setattr__(self, "version", str(version))
        object.__setattr__(
            self,
            "tasks",
            tuple(
                task if isinstance(task, WorldTaskConfig) else WorldTaskConfig.from_dict(task)
                for task in (tasks or ())
            ),
        )
        object.__setattr__(
            self,
            "metrics",
            tuple(
                metric if isinstance(metric, MetricSpec) else MetricSpec.from_dict(metric)
                for metric in (metrics or ())
            ),
        )
        object.__setattr__(self, "splits", tuple_of_str(splits))
        object.__setattr__(self, "tags", tuple_of_str(tags))
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "metadata", copy_mapping(metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def benchmark_id(self) -> str:
        return self.name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkSpec":
        return cls(
            name=data.get("name"),
            benchmark_id=data.get("benchmark_id"),
            version=data.get("version", "1.0"),
            tasks=data.get("tasks", ()),
            metrics=data.get("metrics", ()),
            splits=data.get("splits", ("default",)),
            tags=data.get("tags", ()),
            description=data.get("description", ""),
            dataset_root=data.get("dataset_root"),
            metadata=data.get("metadata"),
            schema_version=data.get("schema_version", BENCHMARK_SPEC_SCHEMA_VERSION),
        )
