"""Standardized protocols and input structures for external benchmark evaluation.

This module defines common types and classes used to map and transfer evaluation results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from worldfoundry.evals.api import MetricResult

JsonValue = Any


@dataclass(frozen=True)
class BenchmarkMetricInput:
    """Normalized input envelope for benchmark metric implementations.

    Provides a standard, immutable interface to pass extracted dataset results, evaluation
    manifests, and external references to downstream metric evaluator routines.
    """

    benchmark_id: str
    metric_id: str
    sample_id: str = "external-benchmark:sample"
    payload: JsonValue = None
    records: tuple[Mapping[str, JsonValue], ...] = ()
    reference: Mapping[str, JsonValue] = field(default_factory=dict)
    task_metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalizes and validates input attributes after object initialization.

        Ensures `benchmark_id`, `metric_id`, and `sample_id` are strings,
        `reference` and `task_metadata` are dictionaries, and `records`
        is a tuple of mappings derived from the `records` or `payload` fields.
        """
        # Ensure string types for identifiers and convert to dict for mappings
        object.__setattr__(self, "benchmark_id", str(self.benchmark_id))
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "reference", dict(self.reference or {}))
        object.__setattr__(self, "task_metadata", dict(self.task_metadata or {}))
        # Process records from either 'records' or 'payload' to a standardized tuple format
        object.__setattr__(self, "records", _records(self.records or self.payload))


@runtime_checkable
class BenchmarkMetricProtocol(Protocol):
    """Callable protocol for benchmark-specific metric adapters.

    Any custom metric plug-in written for WorldFoundry must implement this protocol,
    explicitly declaring its supported `metric_ids` and providing a standardized `compute` method.
    """

    metric_ids: tuple[str, ...]

    def compute(self, inputs: BenchmarkMetricInput) -> MetricResult:
        """Computes a formal MetricResult from the standardized inputs payload."""


def records_from_payload(value: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    """Extracts record rows from common official-result payload shapes.

    Args:
        value: Any JSON-decoded payload.

    Returns:
        Tuple of row dictionaries.
    """
    return _records(value)


def payload_from_request_parts(
    *,
    generated_artifact_manifest: JsonValue = None,
    task_metadata: Mapping[str, JsonValue] | None = None,
    reference: Mapping[str, JsonValue] | None = None,
) -> JsonValue:
    """Resolves the primary metric payload from WorldFoundry request fragments.

    Heuristically explores provided references and task metadata to find the richest
    dict-of-records or array-of-results to act as the primary metric calculation source.

    Args:
        generated_artifact_manifest: The primary generated artifact manifest,
            often containing the raw evaluation results.
        task_metadata: Additional metadata related to the evaluation task.
        reference: Reference data, often containing ground truth or expected outcomes.

    Returns:
        The extracted primary payload for metric calculation, or `generated_artifact_manifest`
        if no suitable payload is found in other containers.
    """
    # Iterate through potential containers (reference, task_metadata) to find structured results.
    # This prioritizes richer, explicitly named result fields over raw manifests.
    for container in (reference or {}, task_metadata or {}):
        # Search for common keys indicating a collection of results or records.
        for key in ("official_results", "scores", "records", "rows", "results", "data"):
            if key in container:
                return container[key]
    # If no structured results are found in reference or task metadata, fall back to the raw manifest.
    return generated_artifact_manifest


def _records(value: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    """Helper to convert nested payload shapes into a standardized tuple of row mappings.

    Args:
        value: Any JSON-decoded payload.

    Returns:
        Tuple of row dictionaries.
    """
    # If the value is a dictionary (mapping), try to extract records from common keys.
    if isinstance(value, Mapping):
        # Look for lists or tuples of records under common result keys.
        for key in ("scores", "records", "rows", "results", "official_results"):
            records = value.get(key)
            # If a sequence (and not a string/bytes) is found, filter for mappings and return.
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
                return tuple(item for item in records if isinstance(item, Mapping))
        # If no specific key yields a sequence, check if the dictionary values themselves are records.
        if all(isinstance(item, Mapping) for item in value.values()):
            return tuple(item for item in value.values() if isinstance(item, Mapping))
        return ()
    # If the value is a sequence (list/tuple), assume it directly contains records.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, Mapping))
    # If none of the above, return an empty tuple.
    return ()


__all__ = [
    "BenchmarkMetricInput",
    "BenchmarkMetricProtocol",
    "payload_from_request_parts",
    "records_from_payload",
]
