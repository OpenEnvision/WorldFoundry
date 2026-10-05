"""Utilities for loading and parsing WorldFoundry task and benchmark YAML configuration files.

This module provides functions to load YAML files that define tasks and benchmarks for the
WorldFoundry framework. It includes support for schema version validation,
inheritance (using an 'extends' mechanism), and normalization of raw YAML
data into `WorldTaskConfig` and `BenchmarkSpec` objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.tasks.catalog import CATALOG_TASK_SCHEMA_VERSION, BenchmarkSpec, WorldTaskConfig

# Define schema version identifiers for task and benchmark YAML files.
TASK_YAML_SCHEMA_VERSION = "worldfoundry-task"
BENCHMARK_YAML_SCHEMA_VERSION = "worldfoundry-benchmark"

# Supported schema versions for task YAML files, including legacy catalog versions.
SUPPORTED_TASK_YAML_SCHEMA_VERSIONS = frozenset(
    {
        TASK_YAML_SCHEMA_VERSION,
        CATALOG_TASK_SCHEMA_VERSION,
    }
)
# Supported schema versions for benchmark YAML files, including legacy catalog versions.
SUPPORTED_BENCHMARK_YAML_SCHEMA_VERSIONS = frozenset(
    {
        BENCHMARK_YAML_SCHEMA_VERSION,
        "worldfoundry-catalog-benchmark",
    }
)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Validates and returns the input value as a Mapping, raising a TypeError if it is not.

    Args:
        value: The value to validate.
        context: Context string used in compiling the error message.

    Returns:
        The validated Mapping.

    Raises:
        TypeError: If value is not a Mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Reads and parses a YAML file from disk, ensuring it represents a mapping.

    Args:
        path: Path to the target YAML file.

    Returns:
        A dictionary parsed from the YAML content.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is in pyproject.
        raise RuntimeError("Loading worldfoundry-task YAML requires PyYAML.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    # If the YAML file is empty or contains only comments, safe_load returns None.
    # Treat this as an empty dictionary.
    if payload is None:
        payload = {}
    return dict(_require_mapping(payload, str(path)))


def _safe_relative_path(value: str, *, source_path: Path, root_dir: Path) -> Path:
    """Safely resolves a relative extends path relative to a source file, ensuring it does not escape root_dir.

    Args:
        value: Relative path string.
        source_path: Path of the source YAML file.
        root_dir: Allowed root directory boundary.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If value is absolute or escapes the root directory.
    """
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError(f"extends path must be relative, got: {value}")
    
    # Resolve the candidate path relative to the source file's parent directory.
    candidate = (source_path.parent / raw_path).resolve()
    root = root_dir.resolve()
    try:
        # Check if the candidate path is a sub-path of the root directory.
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"extends path escapes task root: {value}") from exc
    return candidate


def _extends_list(value: Any) -> list[str]:
    """Coerces a raw extends value (None, single string, or sequence of strings) into a list of strings.

    Args:
        value: Raw input extends value.

    Returns:
        A list of extends path strings.

    Raises:
        TypeError: If value is of unsupported type.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    # Check if value is a sequence, but not a string or bytes-like object.
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    raise TypeError("extends must be a string or sequence of strings")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merges two dictionary mappings.

    Args:
        base: The base dictionary mapping.
        override: The dictionary whose keys override the base values.

    Returns:
        A new merged dictionary.
    """
    # Start with a shallow copy of the base dictionary.
    merged = {str(key): item for key, item in base.items()}
    for key, value in override.items():
        key = str(key)
        # If both base and override values for the same key are mappings,
        # recursively merge them. Otherwise, override the base value.
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_mapping_with_extends(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Loads a YAML file and resolves its dynamic parent inheritance hierarchy.

    The `extends` property in a YAML file allows it to inherit properties
    from other YAML files. This function resolves that chain of inheritance
    by recursively merging parent payloads.

    Args:
        path: Path of the source YAML file.
        root_dir: Base boundary directory for relative path security. All
            extended paths must resolve within this directory. If None,
            the parent directory of `path` is used as the root.

    Returns:
        The fully resolved and merged dictionary payload.
    """
    source_path = Path(path).resolve()
    # Determine the effective root directory for path security.
    root = Path(root_dir).resolve() if root_dir is not None else source_path.parent.resolve()
    return _load_yaml_mapping_with_extends(source_path, root_dir=root, seen=())


def _load_yaml_mapping_with_extends(
    source_path: Path,
    *,
    root_dir: Path,
    seen: tuple[Path, ...],
) -> dict[str, Any]:
    """Helper implementing cyclic dependency check and file parsing for extends inheritance.

    This function recursively loads YAML files, handling the `extends` property
    to merge configurations from parent files. It also detects and prevents
    cyclic dependencies in the inheritance chain.

    Args:
        source_path: Path of the target file.
        root_dir: Base boundary directory for relatives.
        seen: Visited files in the current resolution chain to detect cycles.

    Returns:
        The resolved and merged dictionary.

    Raises:
        ValueError: If a cyclic path reference is detected.
        FileNotFoundError: If the source path is missing.
    """
    # Check for cyclic dependencies in the extends chain.
    if source_path in seen:
        cycle = " -> ".join(str(path) for path in (*seen, source_path))
        raise ValueError(f"cyclic task YAML extends: {cycle}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    payload = _read_yaml_mapping(source_path)
    merged: dict[str, Any] = {}
    
    # Process 'extends' property: recursively load and merge parent configurations.
    for extends_path in _extends_list(payload.get("extends")):
        parent_path = _safe_relative_path(extends_path, source_path=source_path, root_dir=root_dir)
        parent_payload = _load_yaml_mapping_with_extends(
            parent_path,
            root_dir=root_dir,
            seen=(*seen, source_path),  # Add current path to seen for cycle detection.
        )
        merged = _deep_merge(merged, parent_payload)

    # Merge the current file's payload (excluding 'extends') over any inherited values.
    own_payload = {str(key): value for key, value in payload.items() if key != "extends"}
    return _deep_merge(merged, own_payload)


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Coerces a raw sequence/string value into a tuple of string items.

    Args:
        value: Input sequence, string, or None.

    Returns:
        A tuple of string items.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _name_from(data: Mapping[str, Any], *keys: str) -> str | None:
    """Safely extracts the first non-empty string value found under a list of alternate key names.

    Args:
        data: The dictionary mapping.
        *keys: Key names to check in order of preference.

    Returns:
        The first found string name, or None if no non-empty value is found.
    """
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None


def _protocol_name(protocol: Any) -> str:
    """Parses and returns a normalized protocol name from mapping/string values.

    Handles various ways a protocol might be specified in the YAML.

    Args:
        protocol: The raw protocol object or string.

    Returns:
        The parsed protocol name string. Defaults to "open_loop" if not found.
    """
    if isinstance(protocol, Mapping):
        # Prioritize 'type', then 'name', 'id', 'protocol', finally default.
        return str(
            protocol.get("type")
            or protocol.get("name")
            or protocol.get("id")
            or protocol.get("protocol")
            or "open_loop"
        )
    if protocol:
        return str(protocol)
    return "open_loop"


def _output_keys(data: Mapping[str, Any], protocol: Any) -> tuple[str, ...]:
    """Resolves output keys from task schema and protocol properties.

    Looks for `output_keys` directly in the task data, then tries to infer
    from the protocol's output artifacts.

    Args:
        data: The task data mapping.
        protocol: The protocol configuration, which can be a string or mapping.

    Returns:
        A tuple of expected output artifact keys. Defaults to `("generated_video",)`
        if no specific output keys are found.
    """
    explicit = data.get("output_keys")
    if explicit is not None:
        return _string_tuple(explicit)
    
    if isinstance(protocol, Mapping):
        # Look for output artifacts under 'output_artifacts' or 'outputs' in the protocol.
        artifacts = protocol.get("output_artifacts") or protocol.get("outputs")
        if isinstance(artifacts, Mapping):
            return tuple(str(key) for key in artifacts)
        # If artifacts are a sequence, extract names from each item.
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray)):
            names = []
            for item in artifacts:
                if isinstance(item, Mapping):
                    # Extract name from mapping item, prioritizing 'name', 'id', 'kind'.
                    names.append(str(item.get("name") or item.get("id") or item.get("kind") or "artifact"))
                else:
                    names.append(str(item))
            return tuple(names)
    return ("generated_video",)


def _metric_id_from(metric: Any) -> str | None:
    """Extracts a metric identifier string from raw string or dictionary metric definitions.

    Args:
        metric: The raw metric object, which can be a string or a mapping.

    Returns:
        The extracted metric identifier string, or None if not found.
    """
    if isinstance(metric, str):
        return metric
    if isinstance(metric, Mapping):
        return _name_from(metric, "id", "metric_id", "name", "metric")
    return None


def _metric_ids(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Aggregates all metric identifiers declared for a task.

    Looks for `metric_ids` directly, or extracts them from a list of `metrics` definitions.

    Args:
        data: The raw task data mapping.

    Returns:
        A tuple of metric ID strings.
    """
    explicit = data.get("metric_ids")
    if explicit is not None:
        return _string_tuple(explicit)
    
    metric_ids = []
    # Iterate through individual metric definitions to extract their IDs.
    for metric in data.get("metrics") or ():
        metric_id = _metric_id_from(metric)
        if metric_id:
            metric_ids.append(metric_id)
    return tuple(metric_ids)


def _metric_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Coerces metric configurations into a tuple of mappings.

    This function handles metrics defined as a dictionary where keys are metric IDs
    and values are their configurations, or as a list of metric configuration mappings.

    Args:
        value: Input metrics config (None, a mapping, or a sequence of mappings/strings).

    Returns:
        A tuple of metric dictionary mappings, each with an "id" field.

    Raises:
        TypeError: If value has an unsupported type.
    """
    if value is None:
        return ()
    if isinstance(value, Mapping):
        metrics = []
        # Convert a dictionary of metrics to a list of mappings, ensuring each has an 'id'.
        for name, metric in value.items():
            if isinstance(metric, Mapping):
                metric_mapping = dict(metric)
            else:
                metric_mapping = {"value": metric} # If metric is scalar, wrap it.
            metric_mapping.setdefault("id", str(name))
            metrics.append(metric_mapping)
        return tuple(metrics)
    # If value is a sequence, ensure each item is a mapping.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_require_mapping(metric, "metric") for metric in value)
    raise TypeError("metrics must be a mapping or sequence of mappings")


def _task_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    """Coerces task definitions into a tuple of task mappings.

    This function handles tasks defined as a dictionary where keys are task names
    and values are their configurations, or as a list of task configuration mappings.

    Args:
        value: Input tasks config (None, a mapping, or a sequence of mappings).

    Returns:
        A tuple of task dictionary mappings, each with a "name" field.

    Raises:
        TypeError: If value has an unsupported type.
    """
    if value is None:
        return ()
    if isinstance(value, Mapping):
        tasks = []
        # Convert a dictionary of tasks to a list of mappings, ensuring each has a 'name'.
        for name, task in value.items():
            task_mapping = dict(_require_mapping(task, f"task {name}"))
            task_mapping.setdefault("name", str(name))
            tasks.append(task_mapping)
        return tuple(tasks)
    # If value is a sequence, ensure each item is a mapping.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_require_mapping(task, "task") for task in value)
    raise TypeError("tasks must be a mapping or sequence of mappings")


def _dataset_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Resolves and extracts dataset details from modern dataset specs or legacy data properties.

    Prioritizes a top-level `dataset` mapping for manifest, media_root, and split,
    falling back to or merging with a legacy `data` mapping.

    Args:
        data: The task configuration mapping.

    Returns:
        A dictionary containing dataset parameters.
    """
    dataset = data.get("dataset")
    # If 'dataset' is not a mapping, fall back to the legacy 'data' field.
    if not isinstance(dataset, Mapping):
        return dict(data.get("data") or {})
    
    task_data = dict(data.get("data") or {}) # Start with legacy 'data' values.
    # Map specific keys from the 'dataset' mapping to their equivalents in 'task_data'.
    if "manifest" in dataset:
        task_data.setdefault("metadata_path", dataset["manifest"])
    if "metadata_path" in dataset:
        task_data.setdefault("metadata_path", dataset["metadata_path"])
    if "media_root" in dataset:
        task_data.setdefault("media_root", dataset["media_root"])
    if "split" in dataset:
        task_data.setdefault("split", dataset["split"])
    return task_data


def _generation_defaults(data: Mapping[str, Any], protocol: Any) -> dict[str, Any]:
    """Combines protocol and task level generation parameters into a defaults dictionary.

    Merges generation defaults from the protocol, then `generation` field, then `generation_defaults`
    field from the task data, with later sources overriding earlier ones.

    Args:
        data: The task configuration mapping.
        protocol: The protocol configuration (can be a string or mapping).

    Returns:
        A dictionary containing merged generation parameters.
    """
    defaults = {}
    # Apply protocol-level generation defaults first.
    if isinstance(protocol, Mapping) and isinstance(protocol.get("generation_defaults"), Mapping):
        defaults.update(protocol["generation_defaults"])
    # Apply task-level 'generation' field.
    if isinstance(data.get("generation"), Mapping):
        defaults.update(data["generation"])
    # Apply task-level 'generation_defaults' field (overrides 'generation').
    if isinstance(data.get("generation_defaults"), Mapping):
        defaults.update(data["generation_defaults"])
    return defaults


def _evaluation_protocol(data: Mapping[str, Any]) -> Any:
    """Extracts the evaluation protocol structure from task configuration properties.

    Prioritizes `evaluation_protocol` then searches within an `evaluation` mapping.

    Args:
        data: The task configuration mapping.

    Returns:
        The extracted evaluation protocol structure. Can be a string, mapping, or sequence.
    """
    if "evaluation_protocol" in data:
        return data["evaluation_protocol"]
    evaluation = data.get("evaluation")
    if isinstance(evaluation, Mapping):
        # Look for 'protocols', then 'protocol', then 'type'.
        return evaluation.get("protocols") or evaluation.get("protocol") or evaluation.get("type") or ()
    return ()


def world_task_config_from_yaml_mapping(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> WorldTaskConfig:
    """Constructs and normalizes a WorldTaskConfig from a raw dictionary mapping.

    This function takes a dictionary (typically parsed from a YAML file) and
    transforms it into a structured `WorldTaskConfig` object, applying defaults
    and performing necessary type coercions.

    Args:
        data: The raw dictionary mapping parsed from YAML or JSON.
        source_path: Optional path to the source catalog file, used for metadata.

    Returns:
        The normalized and validated WorldTaskConfig object.

    Raises:
        ValueError: If the schema_version is unsupported.
    """
    schema_version = str(data.get("schema_version", TASK_YAML_SCHEMA_VERSION))
    # Validate the schema version against supported versions.
    if schema_version not in SUPPORTED_TASK_YAML_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported task YAML schema_version: {schema_version}")

    protocol = data.get("protocol", "open_loop")
    metadata = dict(data.get("metadata") or {})
    # Populate metadata from top-level fields for backwards compatibility or common use.
    for key in (
        "suite",
        "benchmark_name",
        "groups",
        "runtime_requirements",
        "artifact_contract",
        "dataset",
    ):
        if key in data:
            metadata[key] = data[key]
    metadata["source_schema_version"] = schema_version
    if source_path is not None:
        metadata["source_path"] = str(Path(source_path))

    return WorldTaskConfig(
        name=_name_from(data, "task", "name", "task_id"),
        protocol=_protocol_name(protocol),
        evaluation_protocol=_evaluation_protocol(data),
        capability_track=str(data.get("capability_track", data.get("track", "core_video"))),
        schema_type=str(data.get("schema_type", "sample")),
        input_keys=data.get("input_keys", ()),
        output_keys=_output_keys(data, protocol),
        metric_ids=_metric_ids(data),
        metric_groups=data.get("metric_groups", ()),
        tags=data.get("tags", ()),
        description=str(data.get("description", "")),
        data=_dataset_data(data),
        generation_defaults=_generation_defaults(data, protocol),
        metadata=metadata,
        # Force current catalog schema version for consistency in internal representation.
        schema_version=CATALOG_TASK_SCHEMA_VERSION, 
    )


def benchmark_spec_from_yaml_mapping(
    data: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> BenchmarkSpec:
    """Constructs and normalizes a BenchmarkSpec from a raw dictionary mapping containing tasks.

    This function takes a dictionary (typically parsed from a YAML file) and
    transforms it into a structured `BenchmarkSpec` object, recursively creating
    `WorldTaskConfig` objects for any nested task definitions.

    Args:
        data: The raw dictionary mapping containing benchmark and nested task fields.
        source_path: Optional path to the source catalog file, used for metadata.

    Returns:
        The normalized and validated BenchmarkSpec object.

    Raises:
        ValueError: If the schema_version is unsupported.
    """
    schema_version = str(data.get("schema_version", BENCHMARK_YAML_SCHEMA_VERSION))
    # Validate the schema version against supported versions.
    if schema_version not in SUPPORTED_BENCHMARK_YAML_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported benchmark YAML schema_version: {schema_version}")

    # Convert raw task mappings into WorldTaskConfig objects.
    tasks = [
        world_task_config_from_yaml_mapping(task, source_path=source_path)
        for task in _task_mappings(data.get("tasks"))
    ]
    metadata = dict(data.get("metadata") or {})
    metadata["source_schema_version"] = schema_version
    if source_path is not None:
        metadata["source_path"] = str(Path(source_path))
    return BenchmarkSpec(
        name=_name_from(data, "benchmark", "name", "benchmark_id"),
        version=str(data.get("version", "1.0")),
        tasks=tasks,
        # Convert raw metric configurations into dictionary mappings.
        metrics=tuple(dict(metric) for metric in _metric_mappings(data.get("metrics"))),
        splits=data.get("splits", ("default",)),
        tags=data.get("tags", ()),
        description=str(data.get("description", "")),
        dataset_root=data.get("dataset_root"),
        metadata=metadata,
    )


def load_world_task_yaml(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> WorldTaskConfig:
    """Loads and parses a single WorldTaskConfig from a YAML file, resolving inheritance (extends).

    Args:
        path: Path to the task YAML file.
        root_dir: Base directory for relative file references. All 'extends'
            paths must resolve within this directory. If None, the parent
            directory of `path` is used as the root.

    Returns:
        The loaded and validated WorldTaskConfig object.
    """
    payload = load_yaml_mapping_with_extends(path, root_dir=root_dir)
    return world_task_config_from_yaml_mapping(payload, source_path=path)


def load_benchmark_yaml(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> BenchmarkSpec:
    """Loads and parses a BenchmarkSpec from a YAML file, resolving inheritance (extends).

    Args:
        path: Path to the benchmark YAML file.
        root_dir: Base directory for relative file references. All 'extends'
            paths must resolve within this directory. If None, the parent
            directory of `path` is used as the root.

    Returns:
        The loaded and validated BenchmarkSpec object.
    """
    payload = load_yaml_mapping_with_extends(path, root_dir=root_dir)
    return benchmark_spec_from_yaml_mapping(payload, source_path=path)


def load_catalog_yaml(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> WorldTaskConfig | BenchmarkSpec:
    """Polymorphically loads a catalog YAML file, deciding whether it is a single task or benchmark.

    This function attempts to infer the type of the YAML file (task or benchmark)
    based on the presence of a 'tasks' field or the `schema_version`.

    Args:
        path: Path to the catalog YAML file.
        root_dir: Base directory for relative file references. All 'extends'
            paths must resolve within this directory. If None, the parent
            directory of `path` is used as the root.

    Returns:
        A WorldTaskConfig or BenchmarkSpec object depending on file schema contents.
    """
    payload = load_yaml_mapping_with_extends(path, root_dir=root_dir)
    schema_version = str(payload.get("schema_version", TASK_YAML_SCHEMA_VERSION))
    # Determine if the payload represents a benchmark based on 'tasks' key or schema version.
    if "tasks" in payload or schema_version in SUPPORTED_BENCHMARK_YAML_SCHEMA_VERSIONS:
        return benchmark_spec_from_yaml_mapping(payload, source_path=path)
    return world_task_config_from_yaml_mapping(payload, source_path=path)


# Alias for load_world_task_yaml for convenience.
load_task_yaml = load_world_task_yaml


__all__ = [
    "BENCHMARK_YAML_SCHEMA_VERSION",
    "SUPPORTED_BENCHMARK_YAML_SCHEMA_VERSIONS",
    "SUPPORTED_TASK_YAML_SCHEMA_VERSIONS",
    "TASK_YAML_SCHEMA_VERSION",
    "benchmark_spec_from_yaml_mapping",
    "load_benchmark_yaml",
    "load_catalog_yaml",
    "load_task_yaml",
    "load_world_task_yaml",
    "load_yaml_mapping_with_extends",
    "world_task_config_from_yaml_mapping",
]