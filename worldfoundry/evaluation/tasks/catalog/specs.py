"""Utilities for converting Benchmark Zoo entries to evaluation API BenchmarkSpec and WorldTaskConfig objects.

This module provides functions to transform internal benchmark-zoo schema representations into the public
`BenchmarkSpec` and `WorldTaskConfig` contracts used by the evaluation API, facilitating
the dynamic loading and registration of benchmarks. It also includes utilities for
loading and writing these specifications from/to manifest files.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.evaluation.api import BenchmarkSpec, MetricSpec, WorldTaskConfig

from .schema import BenchmarkZooEntry, load_entries


def _coerce_entry(value: BenchmarkZooEntry | Mapping[str, Any]) -> BenchmarkZooEntry:
    """Coerces a raw benchmark-zoo entry representation into a BenchmarkZooEntry instance.

    Args:
        value: Either an existing BenchmarkZooEntry or a Mapping representing it.

    Returns:
        A BenchmarkZooEntry instance.

    Raises:
        TypeError: If value is of unsupported type.
    """
    if isinstance(value, BenchmarkZooEntry):
        return value
    if isinstance(value, Mapping):
        return BenchmarkZooEntry.from_dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return BenchmarkZooEntry.from_dict(to_dict())
    raise TypeError(f"expected BenchmarkZooEntry or mapping, got {type(value).__name__}")


def _load_benchmark_contract(target: str | None) -> Any | None:
    """Dynamically imports and loads an external benchmark contract from its module target string.

    Args:
        target: A target string of the form "module.path:contract_name".

    Returns:
        The loaded contract attribute, or None if importing fails.
    """
    if not target or ":" not in target:
        return None
    module_name, _, attr_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        # Return None if the module or attribute cannot be imported/found.
        return None


def _contract_tuple(contract: Any | None, attr: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Retrieves and coerces an attribute value from a contract object into a string tuple.

    Args:
        contract: The contract object to query.
        attr: Name of the attribute.
        fallback: Fallback value to return if attribute is missing.

    Returns:
        A tuple of string values.
    """
    value = getattr(contract, attr, None)
    if value is None:
        return fallback
    if isinstance(value, str):
        return (value,)
    try:
        # Attempt to convert an iterable value to a tuple of strings.
        return tuple(str(item) for item in value)
    except TypeError:
        # Fallback if the value is not iterable.
        return fallback


def benchmark_zoo_entry_to_benchmark_spec(value: BenchmarkZooEntry | Mapping[str, Any]) -> BenchmarkSpec:
    """Converts a single benchmark-zoo entry into the public evaluation API BenchmarkSpec contract.

    This function processes an internal `BenchmarkZooEntry` (or its raw dictionary representation)
    and transforms it into the structured `BenchmarkSpec` and `WorldTaskConfig` objects
    required by the `worldfoundry` evaluation API. It handles dynamic loading of benchmark
    contracts and mapping various entry attributes to the public spec.

    Args:
        value: A `BenchmarkZooEntry` instance or a dictionary mapping representing one.

    Returns:
        The converted `BenchmarkSpec` object.
    """

    entry = _coerce_entry(value)
    # Dynamically load the benchmark contract specified by `runner_target`.
    benchmark_contract = _load_benchmark_contract(entry.runner_target)

    # Create a tuple of MetricSpec objects from the entry's metrics list.
    metric_specs = tuple(
        MetricSpec(
            metric_id=metric.metric_id,
            display_name=metric.name or metric.metric_id,
            description=metric.description or "",
            family=entry.benchmark_id,
            higher_is_better=metric.higher_is_better,
            normalizer=metric.normalizer,
            aggregator=metric.aggregator,
            output_unit=metric.output_unit or "",
            primary=metric.primary,
            weight=metric.weight,
            tags=entry.tags,
            metadata=metric.to_dict(),
        )
        for metric in entry.metrics
    )
    metric_ids = tuple(metric.metric_id for metric in entry.metrics)

    # Populate a data dictionary with dataset and source references if available.
    data: dict[str, Any] = {}
    if entry.hf_dataset_id:
        data["hf_dataset_id"] = entry.hf_dataset_id
    if entry.hf_dataset_ids:
        data["hf_dataset_ids"] = list(entry.hf_dataset_ids)
    if entry.source.official_repo_url:
        data["official_repo_url"] = entry.source.official_repo_url
    if entry.data_refs.get("task_yaml"):
        data["task_yaml"] = entry.data_refs["task_yaml"]

    # Determine if the benchmark requires an upstream runtime, defaulting to True.
    requires_upstream_runtime = bool(getattr(benchmark_contract, "requires_upstream_runtime", True))

    # Check multiple conditions to determine if the official runtime for the benchmark is validated.
    official_runtime_validated = (
        entry.integration_status == "integrated"
        and entry.verification_status == "verified"
        and entry.official_benchmark_verified
        and entry.integration_evidence
    )

    # Extract metadata from the dynamically loaded benchmark contract if it exists and has a to_dict method.
    contract_metadata: dict[str, Any] = {}
    if benchmark_contract is not None and hasattr(benchmark_contract, "to_dict"):
        contract_metadata = benchmark_contract.to_dict()

    # Construct the WorldTaskConfig object for the benchmark.
    task = WorldTaskConfig(
        name=entry.benchmark_id,
        protocol="external_benchmark",
        evaluation_protocol="external_benchmark_contract",
        # Set capability track from domains, defaulting to "world_model".
        capability_track=",".join(entry.domains) if entry.domains else "world_model",
        schema_type="generated_artifact_set",
        # Retrieve input and output keys from the contract or use fallbacks.
        input_keys=_contract_tuple(benchmark_contract, "input_keys", ("generated_video_dir",)),
        output_keys=_contract_tuple(benchmark_contract, "output_keys", ("scorecard", "raw_results")),
        metric_ids=metric_ids,
        metric_groups=entry.tags,
        tags=entry.tags,
        description=f"Benchmark runner surface for {entry.name or entry.benchmark_id}.",
        data=data,
        # Populate comprehensive metadata for the task, including various statuses and references.
        metadata={
            "benchmark_id": entry.benchmark_id,
            "source_kind": "benchmark_zoo",
            "benchmark_zoo_id": entry.benchmark_id,
            "ready_now_command": entry.ready_now_command,
            "one_click_command": entry.one_click_command,
            "source_status": entry.source_status,
            "open_source_status": entry.open_source_status,
            "release_status": entry.release_status,
            "maturity": entry.maturity,
            "integration_status": entry.integration_status,
            "official_benchmark_verified": entry.official_benchmark_verified,
            "integration_evidence": entry.integration_evidence,
            "leaderboard_valid": entry.leaderboard_valid,
            "base_model_dependencies": entry.base_model_dependencies,
            "optional_base_model_dependencies": entry.optional_base_model_dependencies,
            "requires_auth": entry.requires_auth,
            "requires": entry.requires,
            "blockers": entry.blockers,
            "contract_only_surface": not official_runtime_validated,
            "requires_upstream_runtime": requires_upstream_runtime,
            "official_runtime_validated": official_runtime_validated,
            "benchmark_contract": contract_metadata,
            "runner": entry.runner.to_dict(),
            "dataset": entry.dataset.to_dict(),
            "dataset_refs": [item.to_dict() for item in entry.dataset_refs],
            "data_refs": dict(entry.data_refs),
            "runner_availability": dict(entry.runner_availability),
        },
    )

    # Construct the final BenchmarkSpec object.
    return BenchmarkSpec(
        benchmark_id=entry.benchmark_id,
        version="1.0",
        tasks=(task,),  # A BenchmarkSpec can contain multiple tasks, here it's a single task.
        metrics=metric_specs,
        splits=("default",),
        tags=entry.tags,
        description=entry.notes[0] if entry.notes else "",
        dataset_root=entry.dataset.path,
        # Populate comprehensive metadata for the benchmark, reflecting its status and dependencies.
        metadata={
            "name": entry.name,
            "ready_now_command": entry.ready_now_command,
            "one_click_command": entry.one_click_command,
            "source": entry.source.to_dict(),
            "dataset": entry.dataset.to_dict(),
            "dataset_refs": [item.to_dict() for item in entry.dataset_refs],
            "open_source_status": entry.open_source_status,
            "release_status": entry.release_status,
            "maturity": entry.maturity,
            "integration_status": entry.integration_status,
            "official_benchmark_verified": entry.official_benchmark_verified,
            "integration_evidence": entry.integration_evidence,
            "leaderboard_valid": entry.leaderboard_valid,
            "base_model_dependencies": entry.base_model_dependencies,
            "optional_base_model_dependencies": entry.optional_base_model_dependencies,
            "requires": entry.requires,
            "blockers": entry.blockers,
            "contract_only_surface": not official_runtime_validated,
            "requires_upstream_runtime": requires_upstream_runtime,
            "official_runtime_validated": official_runtime_validated,
            "runner": entry.runner.to_dict(),
            "data_refs": dict(entry.data_refs),
            "runner_availability": dict(entry.runner_availability),
            "notes": entry.notes,
        },
    )


def benchmark_zoo_entries_to_benchmark_specs(
    entries: Iterable[BenchmarkZooEntry | Mapping[str, Any]],
) -> tuple[BenchmarkSpec, ...]:
    """Converts multiple benchmark-zoo entries or raw mappings into public BenchmarkSpec configurations.

    Args:
        entries: An iterable of `BenchmarkZooEntry` instances or dictionary mappings representing them.

    Returns:
        A tuple of converted `BenchmarkSpec` objects.
    """
    return tuple(benchmark_zoo_entry_to_benchmark_spec(entry) for entry in entries)


def external_benchmark_contract_for_zoo_entry(
    value: BenchmarkZooEntry | Mapping[str, Any],
) -> Any | None:
    """Resolve the static contract object referenced by a catalog zoo entry."""
    from worldfoundry.evaluation.tasks.contracts.registry import ExternalBenchmarkContract

    entry = _coerce_entry(value)
    contract = _load_benchmark_contract(entry.runner_target)
    if isinstance(contract, ExternalBenchmarkContract):
        return contract
    return None


def external_benchmark_contract_for_id(benchmark_id: str) -> Any | None:
    """Resolve a benchmark contract via catalog ``runner_target`` when present."""
    from .zoo_registry import load_benchmark_zoo_registry

    try:
        entry = load_benchmark_zoo_registry().get(benchmark_id)
    except KeyError:
        return None
    return external_benchmark_contract_for_zoo_entry(entry)


def load_benchmark_specs(path: str | Path) -> tuple[BenchmarkSpec, ...]:
    """Loads and compiles public BenchmarkSpec objects from a benchmark-zoo manifest path.

    Args:
        path: Path to the benchmark-zoo manifest file (e.g., a YAML or JSON file
              containing multiple `BenchmarkZooEntry` definitions).

    Returns:
        A tuple of loaded `BenchmarkSpec` objects.
    """
    # Load raw entries from the specified path and then convert them to BenchmarkSpecs.
    return benchmark_zoo_entries_to_benchmark_specs(load_entries(path))


def write_benchmark_specs_json(entries_path: str | Path, output_path: str | Path) -> Path:
    """Loads benchmark-zoo entries from a manifest and exports their BenchmarkSpec representations to a JSON file.

    Args:
        entries_path: Path to the input benchmark-zoo manifest file.
        output_path: Path to the output JSON file where the `BenchmarkSpec` objects will be written.

    Returns:
        The `Path` to the exported JSON file.
    """
    specs = load_benchmark_specs(entries_path)
    destination = Path(output_path)
    # Ensure the parent directory for the output file exists.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Convert BenchmarkSpec objects to dictionaries for JSON serialization.
    payload = [spec.to_dict() for spec in specs]
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


BENCHMARK_ZOO_SUITE = "benchmark_zoo"
BENCHMARK_ZOO_BACKEND = "external_benchmark_contract"
_BENCHMARK_ZOO_SUITE_ALIASES = frozenset({"benchmark_zoo", "zoo"})


def benchmark_zoo_entry_to_cli_task_dict(
    value: BenchmarkZooEntry | Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Convert a benchmark-zoo entry into the CLI task-list payload shape."""
    entry = _coerce_entry(value)
    spec = benchmark_zoo_entry_to_benchmark_spec(entry)
    task = spec.tasks[0]
    metadata = dict(task.metadata)
    manifest_reference = (
        f"{manifest_path.resolve()}#{entry.benchmark_id}"
        if manifest_path is not None
        else entry.benchmark_id
    )
    evaluation_protocol_value = task.evaluation_protocol
    if isinstance(evaluation_protocol_value, str):
        evaluation_protocol = evaluation_protocol_value
    else:
        protocol_names = getattr(task, "evaluation_protocol_names", ())
        evaluation_protocol = protocol_names[0] if protocol_names else BENCHMARK_ZOO_BACKEND
    return {
        "task_type": entry.benchmark_id,
        "benchmark_name": entry.benchmark_id,
        "suite": BENCHMARK_ZOO_SUITE,
        "backend": BENCHMARK_ZOO_BACKEND,
        "task_yaml_path": manifest_reference,
        "manifest_reference": manifest_reference,
        "name": task.name,
        "protocol": task.protocol,
        "capability_track": task.capability_track,
        "schema_type": task.schema_type,
        "evaluation_protocol": evaluation_protocol,
        "input_keys": list(task.input_keys),
        "output_keys": list(task.output_keys),
        "metric_groups": list(task.metric_groups),
        "description": task.description or spec.description,
        "has_eval_prompt": False,
        "source_kind": "benchmark_zoo",
        "benchmark_zoo_id": entry.benchmark_id,
        "contract_only_surface": bool(metadata.get("contract_only_surface", True)),
        "requires_upstream_runtime": bool(metadata.get("requires_upstream_runtime", True)),
        "official_runtime_validated": bool(metadata.get("official_runtime_validated", False)),
    }


def validate_benchmark_zoo_cli_task(
    value: BenchmarkZooEntry | Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate benchmark-zoo metadata for the legacy ``validate`` CLI command."""
    _ = (dataset_root, limit)
    if isinstance(value, Mapping) and value.get("source_kind") == "benchmark_zoo" and "task_type" in value:
        return {**dict(value), "ok": True}
    payload = benchmark_zoo_entry_to_cli_task_dict(value)
    payload["ok"] = True
    return payload


def _filter_benchmark_zoo_entries(
    entries: Iterable[BenchmarkZooEntry],
    *,
    task_type: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    source_kind: str | None = None,
    registry=None,
) -> list[BenchmarkZooEntry]:
    if source_kind == "task_yaml":
        return []
    items = list(entries)
    if task_type is not None:
        resolved = registry.resolve_key(task_type) if registry is not None else task_type
        items = [item for item in items if item.benchmark_id == resolved]
    if suite is not None and suite.lower() not in _BENCHMARK_ZOO_SUITE_ALIASES:
        return []
    if backend is not None and backend != BENCHMARK_ZOO_BACKEND:
        return []
    if source_kind is not None and source_kind != "benchmark_zoo":
        return []
    return items


def list_benchmark_zoo_cli_tasks(
    *,
    task_type: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    source_kind: str | None = None,
    manifest_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List benchmark-zoo entries using the CLI task payload shape."""
    from .zoo_registry import load_benchmark_zoo_registry

    registry = load_benchmark_zoo_registry(manifest_dir)
    entries = _filter_benchmark_zoo_entries(
        registry.list(),
        task_type=task_type,
        suite=suite,
        backend=backend,
        source_kind=source_kind,
        registry=registry,
    )
    return [benchmark_zoo_entry_to_cli_task_dict(entry) for entry in entries]


def get_benchmark_zoo_cli_task(
    task_type: str,
    benchmark_name: str | None = None,
    *,
    manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one benchmark-zoo entry into the CLI task payload shape."""
    from .zoo_registry import UnknownBenchmarkZooKeyError, load_benchmark_zoo_registry

    registry = load_benchmark_zoo_registry(manifest_dir)
    lookup_keys = [key for key in (benchmark_name, task_type) if key]
    matches: list[BenchmarkZooEntry] = []
    seen: set[str] = set()
    for key in lookup_keys:
        try:
            entry = registry.get(key)
        except UnknownBenchmarkZooKeyError:
            continue
        if entry.benchmark_id not in seen:
            matches.append(entry)
            seen.add(entry.benchmark_id)
    if not matches:
        available = registry.keys()
        raise KeyError(
            f"Unknown benchmark '{task_type}/{benchmark_name or task_type}'. "
            f"Available entries: {available}"
        )
    resolved_task_type = registry.resolve_key(task_type)
    for entry in matches:
        aliases = {entry.benchmark_id, *(registry.aliases_for(entry.benchmark_id))}
        if benchmark_name is None:
            if entry.benchmark_id == resolved_task_type:
                return benchmark_zoo_entry_to_cli_task_dict(entry)
            continue
        if benchmark_name in aliases or benchmark_name == entry.benchmark_id:
            return benchmark_zoo_entry_to_cli_task_dict(entry)
    available = [f"{item}/{item}" for item in registry.keys()]
    raise KeyError(
        f"Unknown benchmark '{task_type}/{benchmark_name}'. "
        f"Available entries: {available}"
    )


def build_benchmark_zoo_catalog_registry(
    *,
    task_type: str | None = None,
    suite: str | None = None,
    backend: str | None = None,
    source_kind: str | None = None,
    manifest_dir: str | Path | None = None,
):
    """Build a CatalogRegistry view over benchmark-zoo BenchmarkSpec objects."""
    from . import BenchmarkSpec as CatalogBenchmarkSpec
    from . import CatalogRegistry
    from .zoo_registry import load_benchmark_zoo_registry

    registry = load_benchmark_zoo_registry(manifest_dir)
    entries = _filter_benchmark_zoo_entries(
        registry.list(),
        task_type=task_type,
        suite=suite,
        backend=backend,
        source_kind=source_kind,
        registry=registry,
    )
    catalog_specs = tuple(
        CatalogBenchmarkSpec.from_mapping(spec.to_dict())
        for spec in benchmark_zoo_entries_to_benchmark_specs(entries)
    )
    return CatalogRegistry(catalog_specs)
