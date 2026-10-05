"""Evaluation Task and Benchmark Catalog System.

This module provides the data models, parsing routines, and registry logic
that map external YAML configurations to strongly-typed WorldFoundry evaluation contracts.

Key abstractions:
- `WorldTaskConfig`: A strongly-typed contract representing a single evaluation task.
- `BenchmarkSpec`: A container holding multiple related `WorldTaskConfig`s along with top-level
  benchmark metadata (versions, splits, metrics).
- `CatalogRegistry`: An in-memory, indexed registry allowing querying benchmarks and tasks
  by tag, protocol, or identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from worldfoundry.evaluation.api.json_contract import (
    JsonContract,
    copy_mapping,
    require_mapping,
    tuple_of_str,
)
from worldfoundry.evaluation.api.tasks import EvaluationProtocolSpec

CATALOG_TASK_SCHEMA_VERSION = "worldfoundry-catalog-task"
CATALOG_BENCHMARK_SCHEMA_VERSION = "worldfoundry-catalog-benchmark"

_copy_mapping = copy_mapping
_require_mapping = require_mapping
_tuple_of_str = tuple_of_str


@dataclass(frozen=True, init=False)
class WorldTaskConfig(JsonContract):
    """Immutable, strongly-typed contract defining a specific evaluation task.

    A `WorldTaskConfig` fully describes the expected input/output signatures, the
    protocol used for evaluation (e.g., open-loop video generation or closed-loop control),
    the associated metric groups, and metadata needed to execute the task safely.
    """
    name: str
    protocol: str
    evaluation_protocol: tuple[EvaluationProtocolSpec, ...]
    capability_track: str
    schema_type: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    metric_ids: tuple[str, ...]
    metric_groups: tuple[str, ...]
    tags: tuple[str, ...]
    description: str
    data: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    metadata: Mapping[str, Any]
    schema_version: str
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        name: str | None = None,
        *,
        task_id: str | None = None,
        protocol: str = "open_loop",
        evaluation_protocol: Any = (),
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
        schema_version: str = CATALOG_TASK_SCHEMA_VERSION,
    ) -> None:
        """Initializes and validates a task contract.

        Args:
            name: The canonical name of the task.
            task_id: An alias for `name` for backwards compatibility. If `name` is None, `task_id` is used.
            protocol: The execution protocol (e.g., 'open_loop', 'closed_loop').
            evaluation_protocol: Specifications for how evaluation should be conducted.
            capability_track: The capability track this task belongs to (e.g., 'core_video').
            schema_type: The type of schema this task uses (e.g., 'sample').
            input_keys: Keys expected in the input data.
            output_keys: Keys expected in the output data.
            metric_ids: List of specific metric identifiers associated with this task.
            metric_groups: Groups of metrics relevant to this task.
            tags: Arbitrary tags for filtering and categorization.
            description: A human-readable description of the task.
            data: Arbitrary task-specific configuration data.
            generation_defaults: Default parameters for generation for this task.
            metadata: Arbitrary additional metadata.
            schema_version: The schema version string for this task configuration.

        Raises:
            ValueError: If neither `name` nor `task_id` is provided.
        """
        # Resolve the task name, prioritizing 'name' then 'task_id'.
        resolved_name = name if name is not None else task_id
        if not resolved_name:
            raise ValueError("WorldTaskConfig requires name or task_id.")
        # Use object.__setattr__ to bypass dataclass's frozen property during initialization.
        object.__setattr__(self, "name", str(resolved_name))
        object.__setattr__(self, "protocol", str(protocol))
        object.__setattr__(self, "evaluation_protocol", EvaluationProtocolSpec.coerce_many(evaluation_protocol))
        object.__setattr__(self, "capability_track", str(capability_track))
        object.__setattr__(self, "schema_type", str(schema_type))
        object.__setattr__(self, "input_keys", _tuple_of_str(input_keys))
        object.__setattr__(self, "output_keys", _tuple_of_str(output_keys))
        object.__setattr__(self, "metric_ids", _tuple_of_str(metric_ids))
        object.__setattr__(self, "metric_groups", _tuple_of_str(metric_groups))
        object.__setattr__(self, "tags", _tuple_of_str(tags))
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "data", _copy_mapping(data))
        object.__setattr__(self, "generation_defaults", _copy_mapping(generation_defaults))
        object.__setattr__(self, "metadata", _copy_mapping(metadata))
        object.__setattr__(self, "schema_version", str(schema_version))

    @property
    def task_id(self) -> str:
        """Alias for `name`, kept for backwards-compatibility with legacy task APIs."""
        return self.name

    @property
    def evaluation_protocol_names(self) -> tuple[str, ...]:
        """Returns all configured evaluation protocol names."""
        return tuple(protocol.name for protocol in self.evaluation_protocol)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorldTaskConfig":
        """Constructs a `WorldTaskConfig` safely from a raw dictionary/JSON mapping.

        Args:
            data: A mapping (e.g., dict) containing the task configuration.

        Returns:
            A `WorldTaskConfig` instance.
        """
        return cls(
            name=data.get("name"),
            task_id=data.get("task_id"),
            protocol=data.get("protocol", "open_loop"),
            evaluation_protocol=data.get("evaluation_protocol", data.get("evaluation_protocols", ())),
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
            schema_version=data.get("schema_version", CATALOG_TASK_SCHEMA_VERSION),
        )

    from_dict = from_mapping


def _coerce_tasks(value: Any) -> tuple[WorldTaskConfig, ...]:
    """Helper to coerce arbitrary inputs (mappings or sequences) into a tuple of `WorldTaskConfig`.

    Args:
        value: The raw task config(s) value, can be None, Mapping, or Sequence.

    Returns:
        A tuple of validated WorldTaskConfig instances.
    """
    if value is None:
        return ()
    # If the value is a mapping, it means tasks are defined as a dictionary
    # where keys are task names and values are task configurations.
    if isinstance(value, Mapping):
        tasks = []
        for name, task_data in value.items():
            task_mapping = dict(_require_mapping(task_data, f"tasks[{name!r}]"))
            task_mapping.setdefault("name", name)  # Ensure task has a name, defaulting to key if not present.
            tasks.append(WorldTaskConfig.from_mapping(task_mapping))
        return tuple(tasks)
    # If the value is a sequence, it means tasks are defined as a list of configurations.
    return tuple(
        item if isinstance(item, WorldTaskConfig) else WorldTaskConfig.from_mapping(_require_mapping(item, "task"))
        for item in value
    )


@dataclass(frozen=True, init=False)
class BenchmarkSpec(JsonContract):
    """Top-level specification for an evaluation benchmark.

    Wraps multiple `WorldTaskConfig` instances together along with shared evaluation metadata
    like splits, tags, dataset root locations, and supported metrics.
    """
    name: str
    version: str
    tasks: tuple[WorldTaskConfig, ...]
    metrics: tuple[Mapping[str, Any], ...]
    splits: tuple[str, ...]
    tags: tuple[str, ...]
    description: str
    dataset_root: str | None
    metadata: Mapping[str, Any]
    schema_version: str
    __hash__ = JsonContract.__hash__

    def __init__(
        self,
        name: str | None = None,
        *,
        benchmark_id: str | None = None,
        version: str = "1.0",
        tasks: Any = (),
        metrics: Sequence[Mapping[str, Any]] = (),
        splits: Sequence[str] = ("default",),
        tags: Sequence[str] = (),
        description: str = "",
        dataset_root: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = CATALOG_BENCHMARK_SCHEMA_VERSION,
    ) -> None:
        """Initializes and validates a benchmark specification.

        Args:
            name: The canonical name of the benchmark.
            benchmark_id: An alias for `name` for backwards compatibility. If `name` is None, `benchmark_id` is used.
            version: The version string of the benchmark.
            tasks: A sequence or mapping of `WorldTaskConfig` instances or their raw data.
            metrics: A sequence of metric configuration mappings relevant to this benchmark.
            splits: A sequence of supported data splits (e.g., 'train', 'validation', 'test').
            tags: Arbitrary tags for filtering and categorization.
            description: A human-readable description of the benchmark.
            dataset_root: The root path for benchmark datasets.
            metadata: Arbitrary additional metadata.
            schema_version: The schema version string for this benchmark configuration.

        Raises:
            ValueError: If neither `name` nor `benchmark_id` is provided.
        """
        # Resolve the benchmark name, prioritizing 'name' then 'benchmark_id'.
        resolved_name = name if name is not None else benchmark_id
        if not resolved_name:
            raise ValueError("BenchmarkSpec requires name or benchmark_id.")
        # Use object.__setattr__ to bypass dataclass's frozen property during initialization.
        object.__setattr__(self, "name", str(resolved_name))
        object.__setattr__(self, "version", str(version))
        object.__setattr__(self, "tasks", _coerce_tasks(tasks))
        object.__setattr__(self, "metrics", tuple(_copy_mapping(metric) for metric in (metrics or ())))
        object.__setattr__(self, "splits", _tuple_of_str(splits))
        object.__setattr__(self, "tags", _tuple_of_str(tags))
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "metadata", _copy_mapping(metadata))
        object.__setattr__(self, "schema_version", str(schema_version))

    @property
    def benchmark_id(self) -> str:
        """Alias for `name`, kept for backwards-compatibility with legacy benchmark APIs."""
        return self.name

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BenchmarkSpec":
        """Constructs a `BenchmarkSpec` safely from a raw dictionary/JSON mapping.

        Args:
            data: A mapping (e.g., dict) containing the benchmark configuration.

        Returns:
            A `BenchmarkSpec` instance.
        """
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
            schema_version=data.get("schema_version", CATALOG_BENCHMARK_SCHEMA_VERSION),
        )

    from_dict = from_mapping


def coerce_task_config(data: Mapping[str, Any]) -> WorldTaskConfig:
    """Coerces a task-config-style mapping into the catalog format.

    Normalizes legacy top-level keys like `dataset_root` or `output_dir` into the `metadata` block
    to maintain strict schema separation while maintaining backwards-compatibility.

    Args:
        data: A raw mapping representing a task configuration.

    Returns:
        A `WorldTaskConfig` instance.
    """
    metadata = dict(data.get("metadata", {}))
    # Migrate legacy top-level keys to the metadata block.
    for key in ("dataset_root", "output_dir"):
        if key in data:
            metadata[key] = data[key]

    return WorldTaskConfig(
        name=data.get("name"),
        task_id=data.get("task_id"),
        protocol=data.get("protocol", "open_loop"),
        evaluation_protocol=data.get("evaluation_protocol", ()),
        capability_track=data.get("capability_track", "core_video"),
        schema_type=data.get("schema_type", "sample"),
        input_keys=data.get("input_keys", ()),
        output_keys=data.get("output_keys", ("generated_video",)),
        metric_ids=data.get("metric_ids", data.get("metrics", ())),
        metric_groups=data.get("metric_groups", ()),
        tags=data.get("tags", ()),
        description=data.get("description", ""),
        data=data.get("data", {}),
        generation_defaults=data.get("generation_defaults", {}),
        metadata=metadata,
        schema_version=data.get("schema_version", CATALOG_TASK_SCHEMA_VERSION),
    )


def _iter_tasks(tasks: Any) -> list[Mapping[str, Any]]:
    """Helper yielding task mappings from heterogeneous inputs (dicts or lists).

    Args:
        tasks: Raw dictionary mapping task IDs to task definitions, or list of tasks.

    Returns:
        A list of task dictionary mappings.

    Raises:
        TypeError: If a task in a tasks dictionary is not a mapping.
    """
    if tasks is None:
        return []
    # If tasks are provided as a dictionary, iterate through items
    # and ensure each task config has a 'name' field.
    if isinstance(tasks, Mapping):
        result = []
        for name, task_data in tasks.items():
            if not isinstance(task_data, Mapping):
                raise TypeError(f"tasks[{name!r}] must be a mapping")
            task_mapping = dict(task_data)
            task_mapping.setdefault("name", name)
            result.append(task_mapping)
        return result
    return list(tasks)


def coerce_benchmark_config(data: Mapping[str, Any]) -> BenchmarkSpec:
    """Coerces a catalog benchmark mapping with embedded task dictionaries into a `BenchmarkSpec`.

    This function processes a raw mapping, ensuring that nested task configurations
    are also coerced into `WorldTaskConfig` instances.

    Args:
        data: A raw mapping representing a benchmark configuration.

    Returns:
        A `BenchmarkSpec` instance.
    """
    return BenchmarkSpec(
        name=data.get("name"),
        benchmark_id=data.get("benchmark_id"),
        version=data.get("version", "1.0"),
        tasks=[coerce_task_config(task) for task in _iter_tasks(data.get("tasks", ()))],
        metrics=data.get("metrics", ()),
        splits=data.get("splits", ("default",)),
        tags=data.get("tags", ()),
        description=data.get("description", ""),
        dataset_root=data.get("dataset_root"),
        metadata=data.get("metadata", {}),
        schema_version=data.get("schema_version", "worldfoundry-catalog-benchmark"),
    )


class CatalogRegistryError(ValueError):
    """Base error for catalog registry failures."""


class DuplicateCatalogKeyError(CatalogRegistryError):
    """Raised when a benchmark key is registered more than once."""


class UnknownCatalogKeyError(KeyError):
    """Raised when a benchmark or task lookup cannot be resolved."""


def _normalise_key(value: str, field_name: str = "catalog key") -> str:
    """Enforces and standardizes registry keys via lowercase stripping.

    Args:
        value: The string key to normalize.
        field_name: Contextual field name used in formatting error messages.

    Returns:
        The normalized key in lowercase.

    Raises:
        TypeError: If value is not a string.
        ValueError: If value is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    # Strip whitespace and convert to lowercase for case-insensitive matching.
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value.casefold()


def _coerce_benchmark(value: BenchmarkSpec | Mapping[str, Any]) -> BenchmarkSpec:
    """Safely coerces mappings into `BenchmarkSpec` instances.

    Args:
        value: Either an existing BenchmarkSpec or a Mapping.

    Returns:
        A BenchmarkSpec instance.

    Raises:
        TypeError: If value is neither a BenchmarkSpec nor a Mapping.
    """
    if isinstance(value, BenchmarkSpec):
        return value
    if isinstance(value, Mapping):
        return BenchmarkSpec.from_mapping(value)
    raise TypeError(f"expected BenchmarkSpec or mapping, got {type(value).__name__}")


def _protocol_matches(task: WorldTaskConfig, protocol: str, protocol_kind: str) -> bool:
    """Checks if a task matches a target execution or evaluation protocol.

    Args:
        task: The task config object.
        protocol: Protocol string to match.
        protocol_kind: Type of protocol ('any', 'execution', or 'evaluation').

    Returns:
        True if the protocol matches the kind and task, False otherwise.

    Raises:
        ValueError: If protocol_kind is not one of 'any', 'execution', or 'evaluation'.
    """
    protocol_key = _normalise_key(protocol, "protocol")
    if protocol_kind not in {"any", "execution", "evaluation"}:
        raise ValueError("protocol_kind must be 'any', 'execution', or 'evaluation'")

    # Check for match against the task's primary execution protocol.
    if protocol_kind in {"any", "execution"} and task.protocol.casefold() == protocol_key:
        return True
    # Check for match against any of the task's evaluation protocols.
    if protocol_kind in {"any", "evaluation"}:
        return any(item.name.casefold() == protocol_key for item in task.evaluation_protocol)
    return False


class CatalogRegistry:
    """In-memory benchmark catalog with task, tag, and protocol indexes.
    
    Provides a rich set of filtering methods to quickly resolve `BenchmarkSpec` and `WorldTaskConfig` 
    instances via tags, execution protocols, or benchmark/task identifiers.
    """

    def __init__(self, benchmarks: Iterable[BenchmarkSpec | Mapping[str, Any]] = ()) -> None:
        """Initializes the catalog registry with an optional set of benchmarks.

        Args:
            benchmarks: An iterable of `BenchmarkSpec` instances or raw mappings to register.
        """
        self._benchmarks: dict[str, BenchmarkSpec] = {}
        self._order: list[str] = []  # Maintain insertion order for consistent listing.
        for benchmark in benchmarks:
            self.register(benchmark)

    def __contains__(self, benchmark: object) -> bool:
        """Checks if a benchmark name exists in the registry.

        Args:
            benchmark: The benchmark name key to look up.

        Returns:
            True if registered, False otherwise.
        """
        if not isinstance(benchmark, str):
            return False
        return benchmark.strip().casefold() in self._benchmarks

    def __iter__(self) -> Iterator[BenchmarkSpec]:
        """Iterates over all registered benchmarks in insertion order."""
        return iter(self.list_benchmarks())

    def __len__(self) -> int:
        """Returns the total number of registered benchmarks."""
        return len(self._order)

    def register(self, benchmark: BenchmarkSpec | Mapping[str, Any]) -> BenchmarkSpec:
        """Registers a new benchmark into the registry, preventing name collisions.

        Args:
            benchmark: The `BenchmarkSpec` instance or a mapping to be registered.

        Returns:
            The registered `BenchmarkSpec` instance.

        Raises:
            DuplicateCatalogKeyError: If a benchmark with the same name (case-insensitive) is already registered.
        """
        spec = _coerce_benchmark(benchmark)
        key = _normalise_key(spec.name, "benchmark")
        if key in self._benchmarks:
            raise DuplicateCatalogKeyError(f"duplicate benchmark: {spec.name!r}")
        self._benchmarks[key] = spec
        self._order.append(key)
        return spec

    def get_benchmark(self, benchmark: str) -> BenchmarkSpec:
        """Retrieves a BenchmarkSpec by name.

        Args:
            benchmark: Name of the benchmark spec.

        Returns:
            The registered BenchmarkSpec.

        Raises:
            UnknownCatalogKeyError: If benchmark name is not found in the registry.
        """
        key = _normalise_key(benchmark, "benchmark")
        try:
            return self._benchmarks[key]
        except KeyError as exc:
            raise UnknownCatalogKeyError(f"unknown benchmark: {benchmark!r}") from exc

    def get_task(self, task: str, *, benchmark: str | None = None) -> WorldTaskConfig:
        """Retrieves a single task configuration, raising errors if ambiguous or missing.

        Args:
            task: The name of the task to retrieve.
            benchmark: If provided, restricts the search to a specific benchmark.

        Returns:
            The `WorldTaskConfig` instance.

        Raises:
            UnknownCatalogKeyError: If the task cannot be found.
            CatalogRegistryError: If the task name is ambiguous (found in multiple benchmarks
                                  when no benchmark filter is provided).
        """
        matches = self.find_tasks(task=task, benchmark=benchmark)
        if not matches:
            raise UnknownCatalogKeyError(f"unknown task: {task!r}")
        if len(matches) > 1:
            names = ", ".join(benchmark.name for benchmark, _ in matches)
            raise CatalogRegistryError(f"task {task!r} exists in multiple benchmarks: {names}")
        return matches[0][1]

    def list_benchmarks(self) -> list[BenchmarkSpec]:
        """Lists all registered benchmarks.

        Returns:
            A list of BenchmarkSpec objects in their registration order.
        """
        return [self._benchmarks[key] for key in self._order]

    def list_tasks(self, *, benchmark: str | None = None) -> list[WorldTaskConfig]:
        """Lists all task configurations.

        Args:
            benchmark: Optional benchmark filter name. If provided, only lists tasks from that benchmark.

        Returns:
            A list of WorldTaskConfig objects.
        """
        return [task for _, task in self.find_tasks(benchmark=benchmark)]

    def find_benchmarks(
        self,
        *,
        task: str | None = None,
        tag: str | None = None,
        protocol: str | None = None,
        protocol_kind: str = "any",
    ) -> list[BenchmarkSpec]:
        """Filters benchmarks by specific tasks, tags, or protocol bindings.

        Args:
            task: Name of a task that must be present in the benchmark.
            tag: A tag that must be associated with the benchmark or any of its tasks.
            protocol: A protocol (execution or evaluation) that must be supported by any task in the benchmark.
            protocol_kind: Specifies which kind of protocol to match ('any', 'execution', or 'evaluation').

        Returns:
            A list of matching `BenchmarkSpec` objects.
        """
        matches: list[BenchmarkSpec] = []
        for benchmark in self.list_benchmarks():
            if tag is not None and not self._benchmark_has_tag(benchmark, tag):
                continue
            if task is not None and not self._benchmark_has_task(benchmark, task):
                continue
            if protocol is not None and not any(
                _protocol_matches(item, protocol, protocol_kind) for item in benchmark.tasks
            ):
                continue
            matches.append(benchmark)
        return matches

    def find_tasks(
        self,
        *,
        benchmark: str | None = None,
        task: str | None = None,
        tag: str | None = None,
        protocol: str | None = None,
        protocol_kind: str = "any",
    ) -> list[tuple[BenchmarkSpec, WorldTaskConfig]]:
        """Returns a list of (BenchmarkSpec, WorldTaskConfig) tuples matching all provided filters.

        Args:
            benchmark: Optional name of a specific benchmark to search within.
            task: Optional name of a specific task to match.
            tag: Optional tag to filter tasks by (either on the benchmark or task level).
            protocol: Optional protocol name to filter tasks by.
            protocol_kind: Type of protocol ('any', 'execution', 'evaluation').

        Returns:
            A list of tuples, each containing a matching `BenchmarkSpec` and `WorldTaskConfig`.
        """
        # Determine the set of benchmarks to search.
        benchmark_specs = [self.get_benchmark(benchmark)] if benchmark is not None else self.list_benchmarks()
        # Normalize task and tag keys for case-insensitive comparison.
        task_key = _normalise_key(task, "task") if task is not None else None
        tag_key = _normalise_key(tag, "tag") if tag is not None else None

        matches: list[tuple[BenchmarkSpec, WorldTaskConfig]] = []
        for benchmark_spec in benchmark_specs:
            # Prepare benchmark-level tags for efficient lookup.
            benchmark_tags = {item.casefold() for item in benchmark_spec.tags}
            for task_spec in benchmark_spec.tasks:
                if task_key is not None and task_spec.name.casefold() != task_key:
                    continue
                # Check if the tag matches either a benchmark-level tag or a task-level tag.
                if tag_key is not None and tag_key not in benchmark_tags and tag_key not in {
                    item.casefold() for item in task_spec.tags
                }:
                    continue
                if protocol is not None and not _protocol_matches(task_spec, protocol, protocol_kind):
                    continue
                matches.append((benchmark_spec, task_spec))
        return matches

    def benchmarks_by_tag(self, tag: str) -> list[BenchmarkSpec]:
        """Finds all benchmarks containing a specific tag.

        Args:
            tag: Tag name to match.

        Returns:
            A list of matching BenchmarkSpec objects.
        """
        return self.find_benchmarks(tag=tag)

    def benchmarks_by_protocol(self, protocol: str, *, protocol_kind: str = "any") -> list[BenchmarkSpec]:
        """Finds benchmarks matched to a specific execution or evaluation protocol.

        Args:
            protocol: Protocol string name.
            protocol_kind: Type of protocol ('any', 'execution', 'evaluation').

        Returns:
            A list of matching BenchmarkSpec objects.
        """
        return self.find_benchmarks(protocol=protocol, protocol_kind=protocol_kind)

    def tasks_by_tag(self, tag: str, *, benchmark: str | None = None) -> list[WorldTaskConfig]:
        """Finds tasks matching a given tag.

        Args:
            tag: Tag name to search.
            benchmark: Optional benchmark filter name.

        Returns:
            A list of matching WorldTaskConfig objects.
        """
        return [task for _, task in self.find_tasks(benchmark=benchmark, tag=tag)]

    def tasks_by_protocol(
        self,
        protocol: str,
        *,
        benchmark: str | None = None,
        protocol_kind: str = "any",
    ) -> list[WorldTaskConfig]:
        """Finds tasks matching a specific protocol.

        Args:
            protocol: Protocol string name.
            benchmark: Optional benchmark filter name.
            protocol_kind: Type of protocol ('any', 'execution', 'evaluation').

        Returns:
            A list of matching WorldTaskConfig objects.
        """
        return [
            task
            for _, task in self.find_tasks(
                benchmark=benchmark,
                protocol=protocol,
                protocol_kind=protocol_kind,
            )
        ]

    @staticmethod
    def _benchmark_has_tag(benchmark: BenchmarkSpec, tag: str) -> bool:
        """Determines if the benchmark or any of its tasks is annotated with the given tag.

        Args:
            benchmark: The `BenchmarkSpec` to check.
            tag: The tag string to search for.

        Returns:
            True if the tag is found, False otherwise.
        """
        key = _normalise_key(tag, "tag")
        # Check benchmark-level tags first.
        if key in {item.casefold() for item in benchmark.tags}:
            return True
        # If not found at benchmark level, check all task-level tags.
        return any(key in {item.casefold() for item in task.tags} for task in benchmark.tasks)

    @staticmethod
    def _benchmark_has_task(benchmark: BenchmarkSpec, task: str) -> bool:
        """Determines if the benchmark contains a task with the given name.

        Args:
            benchmark: The `BenchmarkSpec` to check.
            task: The name of the task to search for.

        Returns:
            True if a task with the given name is found within the benchmark, False otherwise.
        """
        key = _normalise_key(task, "task")
        return any(item.name.casefold() == key for item in benchmark.tasks)


__all__ = [
    "CATALOG_BENCHMARK_SCHEMA_VERSION",
    "CATALOG_TASK_SCHEMA_VERSION",
    "BenchmarkSpec",
    "CatalogRegistry",
    "CatalogRegistryError",
    "DuplicateCatalogKeyError",
    "EvaluationProtocolSpec",
    "UnknownCatalogKeyError",
    "WorldTaskConfig",
    "coerce_benchmark_config",
    "coerce_task_config",
]