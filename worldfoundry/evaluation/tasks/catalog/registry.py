"""Provides a registry for WorldEval tasks, allowing them to be discovered, loaded, and managed.

This module defines the TaskRegistry, which is responsible for loading task and benchmark
definitions from YAML files, normalizing keys, and providing lookup and filtering capabilities.
It handles conversion between raw YAML file paths, structured task configurations,
and an internal registry format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from worldfoundry.evaluation.tasks.catalog import BenchmarkSpec, CatalogRegistry, WorldTaskConfig

from .yaml import load_catalog_yaml


class TaskRegistryError(ValueError):
    """Base error for task-registry failures."""


class DuplicateTaskRegistryKeyError(TaskRegistryError):
    """Raised when a benchmark/task pair is registered more than once."""


class UnknownTaskRegistryKeyError(KeyError):
    """Raised when a task or benchmark lookup cannot be resolved."""


def _normalise_key(value: str, field_name: str = "task registry key") -> str:
    """Standardizes registry keys via lowercase stripping.

    Args:
        value: The string registry key.
        field_name: Contextual field name used in formatting error messages.

    Returns:
        The normalized key in lowercase.

    Raises:
        TypeError: If value is not a string.
        ValueError: If value is empty or only whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value.casefold()


def _as_path_tuple(paths: Iterable[str | Path] | str | Path | None) -> tuple[Path, ...]:
    """Coerces various path inputs (None, single path, or collection of paths) into a tuple of Path objects.

    Args:
        paths: The input paths.

    Returns:
        A tuple of Path objects.
    """
    if paths is None:
        return ()
    if isinstance(paths, (str, Path)):
        return (Path(paths),)
    return tuple(Path(path) for path in paths)


def iter_task_yaml_paths(
    roots: Iterable[str | Path] | str | Path,
    *,
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Recursively or shallowly iterates and resolves all task YAML files under given root paths.

    Args:
        roots: A single path or iterable of paths (files or directories) to scan.
        recursive: If True, recursively search directories for YAML/YML files.

    Returns:
        A sorted tuple of resolved, unique absolute Path objects pointing to YAML files.
    """
    paths: list[Path] = []
    for root in _as_path_tuple(roots):
        if root.is_file():
            # If the root itself is a file, add it directly.
            paths.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise NotADirectoryError(root)
        # Choose the appropriate glob method based on the 'recursive' flag.
        iterator = root.rglob("*") if recursive else root.glob("*")
        # Extend the list with paths that are files and have a YAML/YML suffix.
        paths.extend(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
    # Convert paths to absolute, unique, and sorted Path objects.
    return tuple(sorted(dict.fromkeys(path.resolve() for path in paths)))


def _benchmark_name_for_task(task: WorldTaskConfig, source_path: Path | None = None) -> str:
    """Infers the canonical benchmark name for a task using task metadata or source path stem.

    Args:
        task: The task config object.
        source_path: The optional path to the task's YAML manifest source file.

    Returns:
        The benchmark name string.
    """
    metadata = dict(task.metadata or {})
    # Check for common keys in task metadata to determine benchmark name.
    for key in ("benchmark_name", "benchmark", "suite"):
        value = metadata.get(key)
        if value:
            return str(value)
    # If no metadata key found, use the source file's stem (filename without extension).
    if source_path is not None:
        return source_path.stem
    # As a fallback, use the task's own name.
    return task.name


@dataclass(frozen=True)
class TaskRegistryEntry:
    """An entry in the task registry representing a cataloged task mapped to a benchmark.

    Attributes:
        benchmark_name: Canonical name of the benchmark this task belongs to.
        task: The strongly-typed WorldTaskConfig config object.
        source_path: Path to the YAML manifest file containing this task.
    """
    benchmark_name: str
    task: WorldTaskConfig
    source_path: Path | None = None

    @property
    def task_name(self) -> str:
        """Helper alias returning the name of the wrapped task."""
        return self.task.name

    @property
    def key(self) -> tuple[str, str]:
        """A compound resolution key: tuple of (normalized_benchmark_name, normalized_task_name)."""
        return (
            _normalise_key(self.benchmark_name, "benchmark"),
            _normalise_key(self.task_name, "task"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the task registry entry into a plain Python dictionary.

        Returns:
            A dictionary containing the serialized entry fields.
        """
        return {
            "benchmark_name": self.benchmark_name,
            "task_name": self.task.name,
            "name": self.task.name,  # Redundant but kept for backward compatibility if needed
            "source_path": None if self.source_path is None else str(self.source_path),
            "schema_version": self.task.schema_version,
            "protocol": self.task.protocol,
            "evaluation_protocol": [item.name for item in self.task.evaluation_protocol],
            "capability_track": self.task.capability_track,
            "schema_type": self.task.schema_type,
            "input_keys": list(self.task.input_keys),
            "output_keys": list(self.task.output_keys),
            "metric_ids": list(self.task.metric_ids),
            "metric_groups": list(self.task.metric_groups),
            "tags": list(self.task.tags),
            "description": self.task.description,
            "data": dict(self.task.data),
            "generation_defaults": dict(self.task.generation_defaults),
            "metadata": dict(self.task.metadata),
        }


class TaskRegistry:
    """Filesystem-backed registry over catalog task YAML definitions.

    This class manages a collection of TaskRegistryEntry objects, providing mechanisms
    to register, list, and retrieve tasks based on various criteria.
    It ensures uniqueness of benchmark/task pairs and allows conversion to a
    `CatalogRegistry` for broader system integration.
    """

    def __init__(self, entries: Iterable[TaskRegistryEntry] = ()) -> None:
        """Initializes the task registry, optionally pre-populating with registry entries."""
        self._entries: dict[tuple[str, str], TaskRegistryEntry] = {}
        self._order: list[tuple[str, str]] = []
        for entry in entries:
            self.register(entry)

    def __len__(self) -> int:
        """Returns the total number of registered tasks."""
        return len(self._order)

    def register(
        self,
        entry: TaskRegistryEntry,
        *,
        replace: bool = False,
    ) -> TaskRegistryEntry:
        """Registers a new TaskRegistryEntry.

        Args:
            entry: The entry to register.
            replace: If True, allows overwriting existing entries with matching keys.

        Returns:
            The registered entry.

        Raises:
            DuplicateTaskRegistryKeyError: If entry already exists and replace is False.
        """
        key = entry.key
        if not replace and key in self._entries:
            benchmark, task = key
            raise DuplicateTaskRegistryKeyError(
                f"duplicate task registry entry: benchmark={benchmark!r} task={task!r}"
            )
        # Add key to maintain registration order if it's a new entry.
        if key not in self._entries:
            self._order.append(key)
        self._entries[key] = entry
        return entry

    def list(
        self,
        *,
        benchmark: str | None = None,
        tag: str | None = None,
        protocol: str | None = None,
    ) -> list[TaskRegistryEntry]:
        """Lists registered TaskRegistryEntry objects matching optional filters.

        Args:
            benchmark: Optional benchmark name filter. Case-insensitive.
            tag: Optional tag filter. Case-insensitive.
            protocol: Optional protocol filter (main protocol or evaluation protocols). Case-insensitive.

        Returns:
            A list of matching TaskRegistryEntry objects in registration order.
        """
        # Normalize filter keys for case-insensitive comparison.
        benchmark_key = _normalise_key(benchmark, "benchmark") if benchmark else None
        tag_key = _normalise_key(tag, "tag") if tag else None
        protocol_key = _normalise_key(protocol, "protocol") if protocol else None

        entries: list[TaskRegistryEntry] = []
        for key in self._order:
            entry = self._entries[key]
            # Apply benchmark filter.
            if benchmark_key is not None and _normalise_key(entry.benchmark_name, "benchmark") != benchmark_key:
                continue
            # Apply tag filter.
            if tag_key is not None and tag_key not in {item.casefold() for item in entry.task.tags}:
                continue
            # Apply protocol filter (main protocol or any evaluation protocol).
            if protocol_key is not None:
                protocol_names = {
                    entry.task.protocol.casefold(),
                    *(item.name.casefold() for item in entry.task.evaluation_protocol),
                }
                if protocol_key not in protocol_names:
                    continue
            entries.append(entry)
        return entries

    def get(self        , task: str, *, benchmark: str | None = None) -> TaskRegistryEntry:
        """Retrieves a single task entry by name.

        If `benchmark` is provided, it attempts to find a task with that name within the specified benchmark.
        If `benchmark` is not provided, it searches for a uniquely named task across all benchmarks.

        Args:
            task: Name of the task to retrieve. Case-insensitive.
            benchmark: Optional benchmark name to disambiguate tasks with the same name. Case-insensitive.

        Returns:
            The unique matching TaskRegistryEntry.

        Raises:
            UnknownTaskRegistryKeyError: If no tasks match the criteria.
            TaskRegistryError: If multiple tasks match the criteria (ambiguous lookup).
        """
        # Filter tasks by the given benchmark (if any) and then by task name.
        matches = [
            entry
            for entry in self.list(benchmark=benchmark)
            if entry.task.name.casefold() == _normalise_key(task, "task")
        ]
        if not matches:
            # No task found matching the criteria.
            raise UnknownTaskRegistryKeyError(f"unknown task: {task!r}")
        if len(matches) > 1:
            # Multiple tasks found, indicating an ambiguous lookup.
            benchmarks = ", ".join(entry.benchmark_name for entry in matches)
            raise TaskRegistryError(f"task {task!r} exists in multiple benchmarks: {benchmarks}")
        return matches[0]

    def to_catalog_registry(self) -> CatalogRegistry:
        """Converts the task registry into an in-memory CatalogRegistry with grouped BenchmarkSpecs.

        Groups tasks by their `benchmark_name` and collects source paths into benchmark metadata.

        Returns:
            A CatalogRegistry containing the grouped benchmark specs.
        """
        benchmarks: dict[str, list[WorldTaskConfig]] = {}
        metadata_by_benchmark: dict[str, dict[str, Any]] = {}
        for entry in self.list():
            # Group tasks by their benchmark name.
            benchmarks.setdefault(entry.benchmark_name, []).append(entry.task)
            # Collect source paths for each benchmark's metadata.
            metadata = metadata_by_benchmark.setdefault(entry.benchmark_name, {})
            if entry.source_path is not None:
                metadata.setdefault("source_paths", []).append(str(entry.source_path))
        # Create BenchmarkSpec objects from the grouped tasks and collected metadata.
        specs = [
            BenchmarkSpec(
                name=benchmark_name,
                tasks=tasks,
                metadata=metadata_by_benchmark.get(benchmark_name),
            )
            for benchmark_name, tasks in benchmarks.items()
        ]
        return CatalogRegistry(specs)


def _entries_from_loaded_catalog(
    loaded: WorldTaskConfig | BenchmarkSpec,
    *,
    source_path: Path,
) -> tuple[TaskRegistryEntry, ...]:
    """Converts a loaded catalog object (task config or benchmark spec) to registry entries.

    Args:
        loaded: The parsed WorldTaskConfig or BenchmarkSpec.
        source_path: Path of the loaded YAML manifest.

    Returns:
        A tuple of parsed TaskRegistryEntry objects.

    Raises:
        TypeError: If the loaded object is neither a WorldTaskConfig nor a BenchmarkSpec.
    """
    if isinstance(loaded, WorldTaskConfig):
        # If the loaded object is a single task configuration, create one entry.
        return (
            TaskRegistryEntry(
                benchmark_name=_benchmark_name_for_task(loaded, source_path),
                task=loaded,
                source_path=source_path,
            ),
        )
    if isinstance(loaded, BenchmarkSpec):
        # If the loaded object is a benchmark specification, create an entry for each of its tasks.
        return tuple(
            TaskRegistryEntry(
                benchmark_name=loaded.name,
                task=task,
                source_path=source_path,
            )
            for task in loaded.tasks
        )
    raise TypeError(f"unsupported catalog object: {type(loaded).__name__}")


def load_task_registry_from_paths(
    roots: Iterable[str | Path] | str | Path,
    *,
    recursive: bool = False,
    root_dir: str | Path | None = None,
) -> TaskRegistry:
    """Discovers and compiles task registry entries from a set of catalog path roots.

    This function scans specified directories and/or files for YAML manifests
    defining tasks or benchmarks, parses them, and populates a `TaskRegistry`.

    Args:
        roots: Files or directories containing YAML manifests. Can be a single path or an iterable.
        recursive: If True, recursively scans under the roots for YAML files.
        root_dir: Base directory for resolving relative file references within YAML manifests.
                  Defaults to the manifest's parent directory if not provided.

    Returns:
        An initialized TaskRegistry containing all discovered entries.
    """
    entries: list[TaskRegistryEntry] = []
    # Iterate over all identified task YAML files.
    for path in iter_task_yaml_paths(roots, recursive=recursive):
        # Load and parse the YAML file into a WorldTaskConfig or BenchmarkSpec.
        # Use the path's parent as root_dir if not explicitly provided.
        loaded = load_catalog_yaml(path, root_dir=root_dir or path.parent)
        # Convert the loaded object(s) into TaskRegistryEntry instances.
        entries.extend(_entries_from_loaded_catalog(loaded, source_path=path))
    # Initialize and return a TaskRegistry with all the gathered entries.
    return TaskRegistry(entries)


def validate_task_yaml_file(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validates the structure and schema of a given task YAML catalog file.

    Attempts to load and parse the YAML file, returning a report indicating success
    or failure along with relevant details.

    Args:
        path: Path to the task YAML catalog file to validate.
        root_dir: Optional root directory for extends resolution within the YAML file.

    Returns:
        A dictionary report summarizing validation outcome, error type/text if any,
        and catalog details like kind, benchmark names, task names, count, and schema version.
    """
    source_path = Path(path)
    try:
        # Attempt to load and parse the YAML file.
        loaded = load_catalog_yaml(source_path, root_dir=root_dir)
        # Convert the loaded object(s) into TaskRegistryEntry objects for detail extraction.
        entries = _entries_from_loaded_catalog(loaded, source_path=source_path.resolve())
    except Exception as exc:  # noqa: BLE001 - validation reports the precise loader error.
        # If any exception occurs during loading or parsing, report it as a validation failure.
        return {
            "ok": False,
            "path": str(source_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    # If loading and parsing are successful, return a success report with extracted details.
    return {
        "ok": True,
        "path": str(source_path),
        "kind": "benchmark" if isinstance(loaded, BenchmarkSpec) else "task",
        "benchmark_names": sorted({entry.benchmark_name for entry in entries}),
        "task_names": [entry.task.name for entry in entries],
        "task_count": len(entries),
        "schema_version": loaded.schema_version,
    }