"""Universal framework to normalize and aggregate external official evaluation results.

This module parses standard JSON, JSONL, and CSV export formats, maps arbitrary upstream fields
to canonical metric schemas, and handles aggregations to build standardized scorecards.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.evals.api import MetricResult
from worldfoundry.evals.tasks.catalog.schema import BenchmarkMetricSpec, BenchmarkZooEntry
from worldfoundry.evals.tasks.execution.framework.normalizers import apply_normalizer

JsonValue = Any
OFFICIAL_RESULTS_NORMALIZER_SCHEMA_VERSION = "worldfoundry-official-results-normalizer"
DEFAULT_SAMPLE_ID_FIELDS = ("sample_id", "id", "uid", "question_id", "prompt_id", "video_id", "task", "task_id")
METRIC_ID_FIELDS = ("metric_id", "metric", "metric_name", "name", "key", "leaderboard_key", "dimension", "category")
VALUE_FIELDS = ("score", "value", "normalized_value", "mean", "accuracy", "metric_value", "raw_score")
RECORD_CONTAINER_KEYS = (
    "samples",
    "records",
    "items",
    "results",
    "data",
    "per_sample_scores",
    "per_sample_metrics",
    "metrics",
)


@dataclass(frozen=True)
class OfficialMetricMapping:
    """
    Defines how an official result field maps to a canonical metric.

    This dataclass specifies the metric ID, source fields, required fields,
    normalizer, aggregation method, sample ID fields, and optional category
    filtering criteria for transforming raw official evaluation results
    into standardized MetricResult objects.

    Attributes:
        metric_id: The canonical ID for the metric.
        source_fields: A tuple of potential field names in the source data
                       that contain the metric value.
        required_fields: A tuple of field names that must be present in a record
                         for it to be considered valid for this metric.
        normalizer: An optional normalizer function name to apply to the raw value.
        aggregation: The aggregation method (e.g., "mean", "sum") to use when
                     combining per-sample results into an aggregate score.
        sample_id_fields: A tuple of potential field names to use for identifying
                          individual samples.
        category_field: An optional field name that holds category information
                        for filtering records.
        category_values: A tuple of accepted values for `category_field` to
                         filter records specific to this metric.
        value_field: An optional explicit field name for the metric's value,
                     overriding `source_fields` if specified.
        metadata: Additional metadata associated with this mapping.
    """

    metric_id: str
    source_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    normalizer: str | None = None
    aggregation: str = "mean"
    sample_id_fields: tuple[str, ...] = DEFAULT_SAMPLE_ID_FIELDS
    category_field: str | None = None
    category_values: tuple[str, ...] = ()
    value_field: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization to ensure attribute types and default values are set correctly.

        This method converts various attributes to their canonical forms (e.g., tuples of strings)
        and handles default assignments.
        """
        object.__setattr__(self, "metric_id", str(self.metric_id))
        object.__setattr__(self, "source_fields", _tuple_of_str(self.source_fields))
        object.__setattr__(self, "required_fields", _tuple_of_str(self.required_fields))
        object.__setattr__(self, "aggregation", str(self.aggregation or "mean"))
        object.__setattr__(self, "sample_id_fields", _tuple_of_str(self.sample_id_fields) or DEFAULT_SAMPLE_ID_FIELDS)
        if self.category_field is not None:
            object.__setattr__(self, "category_field", str(self.category_field))
        object.__setattr__(self, "category_values", _tuple_of_str(self.category_values))
        if self.value_field is not None:
            object.__setattr__(self, "value_field", str(self.value_field))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_metric_spec(cls, metric: BenchmarkMetricSpec) -> OfficialMetricMapping:
        """
        Build an official-result mapping from benchmark catalog metric schema.

        Args:
            metric: Benchmark metric schema carrying leaderboard field aliases.
        """
        hook = dict(metric.official_results or {})
        # Determine source fields, prioritizing `source_fields` or `fields` from the hook,
        # then falling back to metric names from the BenchmarkMetricSpec.
        source_fields = _dedupe_text(
            _tuple_of_str(hook.get("source_fields"))
            or _tuple_of_str(hook.get("fields"))
            or (
                metric.raw_metric_name,
                metric.leaderboard_key,
                metric.metric_id,
            )
        )
        value_field = hook.get("value_field") or hook.get("score_field")
        # If a value_field is specified but no source_fields, use the value_field as the primary source field.
        if value_field is not None and not source_fields:
            source_fields = (str(value_field),)
        required_fields = _tuple_of_str(hook.get("required_fields") or hook.get("required_columns"))
        return cls(
            metric_id=metric.metric_id,
            source_fields=source_fields,
            required_fields=required_fields,
            normalizer=hook.get("normalizer", metric.normalizer),
            aggregation=str(hook.get("aggregation", hook.get("aggregator", metric.aggregator))),
            sample_id_fields=_tuple_of_str(hook.get("sample_id_fields") or hook.get("sample_id_field"))
            or DEFAULT_SAMPLE_ID_FIELDS,
            category_field=_optional_str(hook.get("category_field")),
            category_values=_tuple_of_str(hook.get("category_values") or hook.get("category_value")),
            value_field=_optional_str(value_field),
            metadata={"metric_schema": metric.to_dict(), "official_results": hook},
        )


@dataclass(frozen=True)
class OfficialResultsNormalization:
    """
    Represents the complete normalized output of an official evaluation.

    This dataclass encapsulates the original benchmark ID, source path,
    raw records, and the generated per-sample and aggregate MetricResult
    objects after normalization.

    Attributes:
        benchmark_id: The ID of the benchmark associated with these results.
        source_path: The path to the original official results file, if applicable.
        records: The raw, canonicalized records parsed from the source.
        per_sample_results: A tuple of MetricResult objects, one for each
                            relevant sample in the source data.
        aggregate_results: A tuple of MetricResult objects, representing
                           the aggregated scores for each metric.
        schema_version: The version of the normalization schema used.
    """

    benchmark_id: str
    source_path: str | None
    records: tuple[Mapping[str, JsonValue], ...]
    per_sample_results: tuple[MetricResult, ...]
    aggregate_results: tuple[MetricResult, ...]
    schema_version: str = OFFICIAL_RESULTS_NORMALIZER_SCHEMA_VERSION

    @property
    def metric_results(self) -> tuple[MetricResult, ...]:
        """
        Combines and returns all per-sample and aggregate metric results.
        """
        return (*self.aggregate_results, *self.per_sample_results)

    def raw_metric_rows(self) -> list[dict[str, JsonValue]]:
        """
        Return MetricResult-compatible rows for raw_metric_table.jsonl.

        Args:
            None.
        """
        return [_metric_raw_row(result.to_dict()) for result in self.metric_results]

    def scorecard_metrics(self) -> dict[str, dict[str, JsonValue]]:
        """
        Return scorecard per_metric entries from aggregate results.

        Args:
            None.
        """
        return {result.metric_id: _metric_scorecard_entry(result.to_dict()) for result in self.aggregate_results}


class OfficialResultsNormalizer:
    """
    Manages the normalization of official evaluation results.

    This class takes a set of metric mappings and can process raw data
    (from files or in-memory records) to produce standardized `MetricResult`
    objects, handling field mapping, normalization, and aggregation.
    """

    def __init__(
        self,
        benchmark_id: str,
        mappings: Sequence[OfficialMetricMapping | Mapping[str, JsonValue]],
        *,
        requested_metric_ids: Sequence[str] | None = None,
    ) -> None:
        """
        Initialize the OfficialResultsNormalizer.

        Args:
            benchmark_id: Benchmark id recorded in diagnostics and aggregate sample ids.
            mappings: Metric field mappings declared by benchmark schema or evaluator code.
            requested_metric_ids: Optional subset/order of metrics to normalize.
        """
        self.benchmark_id = str(benchmark_id)
        # Convert raw mappings (dict or OfficialMetricMapping) into a dictionary
        # of OfficialMetricMapping objects, keyed by metric_id.
        self.mappings = {
            str(mapping.metric_id if isinstance(mapping, OfficialMetricMapping) else mapping["metric_id"]): (
                mapping if isinstance(mapping, OfficialMetricMapping) else OfficialMetricMapping(**mapping)
            )
            for mapping in mappings
        }
        # Determine the order of metrics to process, either explicitly requested or all available.
        self.requested_metric_ids = tuple(str(item) for item in (requested_metric_ids or self.mappings))

    @classmethod
    def from_benchmark_entry(
        cls,
        entry: BenchmarkZooEntry,
        *,
        requested_metric_ids: Sequence[str] | None = None,
    ) -> OfficialResultsNormalizer:
        """
        Build a normalizer from benchmark catalog metric declarations.

        Args:
            entry: Benchmark-zoo entry with metric schemas.
            requested_metric_ids: Optional subset/order of metric ids.
        """
        return cls(
            entry.benchmark_id,
            tuple(OfficialMetricMapping.from_metric_spec(metric) for metric in entry.metrics),
            requested_metric_ids=requested_metric_ids,
        )

    def normalize_file(self, path: str | Path) -> OfficialResultsNormalization:
        """
        Load and normalize one JSON, JSONL, or CSV official results file.

        Args:
            path: Official result export path.
        """
        source_path = Path(path)
        # Load records from the specified file path, attempting to parse CSV, JSONL, or JSON.
        records, blocked_reason, diagnostics = load_official_result_records(source_path)

        # If loading was blocked for any reason (e.g., file not found, empty, invalid format),
        # create blocked MetricResult entries for all requested metrics.
        if blocked_reason is not None:
            aggregate_results = tuple(
                self._blocked_metric(metric_id, blocked_reason, diagnostics) for metric_id in self.requested_metric_ids
            )
            return OfficialResultsNormalization(
                benchmark_id=self.benchmark_id,
                source_path=str(source_path),
                records=(),
                per_sample_results=(),
                aggregate_results=aggregate_results,
            )
        # If records were loaded successfully, proceed to normalize them.
        return self.normalize_records(records, source_path=str(source_path))

    def normalize_records(
        self,
        records: Sequence[Mapping[str, JsonValue]],
        *,
        source_path: str | None = None,
    ) -> OfficialResultsNormalization:
        """
        Normalize already loaded official result records.

        Args:
            records: Mapping records from JSON, JSONL, CSV, or upstream code.
            source_path: Optional source path for diagnostics.
        """
        # Ensure records are consistent dictionaries for processing.
        canonical_records = tuple(dict(record) for record in records if isinstance(record, Mapping))
        per_sample_results: list[MetricResult] = []
        aggregate_results: list[MetricResult] = []

        # Process each requested metric.
        for metric_id in self.requested_metric_ids:
            mapping = self.mappings.get(metric_id)
            # If a mapping is not found for the requested metric, create a blocked result.
            if mapping is None:
                aggregate_results.append(
                    self._blocked_metric(
                        metric_id,
                        "unknown_metric",
                        {"known_metric_ids": sorted(self.mappings)},
                    )
                )
                continue

            # Check if any required fields are missing across the records for this metric.
            missing_fields = _missing_required_fields(canonical_records, mapping)
            if missing_fields:
                aggregate_results.append(
                    self._blocked_metric(
                        metric_id,
                        "missing_required_official_result_field",
                        {
                            "missing_fields": list(missing_fields),
                            "required_fields": list(mapping.required_fields),
                            "source_fields": list(mapping.source_fields),
                            "source_path": source_path,
                        },
                    )
                )
                continue

            # Extract per-sample results for the current metric.
            metric_sample_results = self._sample_results(canonical_records, mapping, source_path=source_path)
            per_sample_results.extend(metric_sample_results)

            # Aggregate the per-sample results into a single metric result.
            aggregate_results.append(self._aggregate_metric(mapping, metric_sample_results, source_path=source_path))

        return OfficialResultsNormalization(
            benchmark_id=self.benchmark_id,
            source_path=source_path,
            records=canonical_records,
            per_sample_results=tuple(per_sample_results),
            aggregate_results=tuple(aggregate_results),
        )

    def _sample_results(
        self,
        records: Sequence[Mapping[str, JsonValue]],
        mapping: OfficialMetricMapping,
        *,
        source_path: str | None,
    ) -> tuple[MetricResult, ...]:
        """
        Convert matching official result rows into per-sample MetricResult rows.

        Args:
            records: Canonical official result records.
            mapping: Metric field mapping to resolve.
            source_path: Optional source path for diagnostics.
        """
        sample_results: list[MetricResult] = []
        for index, record in enumerate(records):
            # Determine if the record matches a "long-format" metric row (metric ID in a specific field)
            # or a "short-format" where the record itself is the metric.
            if _long_metric_row_matches(record, mapping):
                field_name = _first_present_value_field(record, mapping)
            else:
                # If short-format, check if the record's category matches the metric's mapping.
                if not _record_matches_metric(record, mapping):
                    continue
                field_name = _first_present_field(record, mapping)

            # If no suitable field containing a value is found, skip this record.
            if field_name is None:
                continue

            value = record.get(field_name)
            # Attempt to convert the value to a numeric type; skip if not convertible.
            numeric = _numeric_value(value)
            if numeric is None:
                continue

            # Determine sample ID, using default if not found.
            sample_id = _sample_id(record, mapping.sample_id_fields) or f"{self.benchmark_id}:official:{index}"
            sample_results.append(
                MetricResult(
                    sample_id=sample_id,
                    metric_id=mapping.metric_id,
                    raw_value=value,
                    normalized_value=apply_normalizer(mapping.normalizer, numeric),
                    components={
                        "source_field": field_name,
                        "category_field": mapping.category_field,
                        "category_values": list(mapping.category_values),
                    },
                    diagnostics={
                        "benchmark_id": self.benchmark_id,
                        "source_path": source_path,
                        "status": "official_result_imported",
                        "official_results_normalizer": OFFICIAL_RESULTS_NORMALIZER_SCHEMA_VERSION,
                    },
                )
            )
        return tuple(sample_results)

    def _aggregate_metric(
        self,
        mapping: OfficialMetricMapping,
        sample_results: Sequence[MetricResult],
        *,
        source_path: str | None,
    ) -> MetricResult:
        """
        Aggregate per-sample official rows for one metric.

        Args:
            mapping: Metric field mapping to aggregate.
            sample_results: Per-sample MetricResult rows for the metric.
            source_path: Optional source path for diagnostics.
        """
        # If no sample results were found for this metric, return a blocked result.
        if not sample_results:
            return self._blocked_metric(
                mapping.metric_id,
                "missing_required_official_result_field",
                {
                    "required_fields": list(mapping.required_fields or mapping.source_fields),
                    "source_fields": list(mapping.source_fields),
                    "category_field": mapping.category_field,
                    "category_values": list(mapping.category_values),
                    "source_path": source_path,
                },
            )

        # Extract normalized numeric values from valid sample results for aggregation.
        values = [
            float(result.normalized_value)
            for result in sample_results
            if result.valid
            and isinstance(result.normalized_value, (int, float))
            and not isinstance(result.normalized_value, bool)
        ]

        # If no numeric values could be extracted, return a blocked result.
        if not values:
            return self._blocked_metric(
                mapping.metric_id,
                "non_numeric_official_result_field",
                {"source_fields": list(mapping.source_fields), "source_path": source_path},
            )

        # Perform the specified aggregation on the collected numeric values.
        value = _aggregate_values(values, mapping.aggregation)
        return MetricResult(
            sample_id=f"{self.benchmark_id}:official_results",
            metric_id=mapping.metric_id,
            raw_value=value,
            normalized_value=value,
            components={
                "aggregation": mapping.aggregation,
                "n": len(values),
                "source_fields": list(mapping.source_fields),
                "category_field": mapping.category_field,
                "category_values": list(mapping.category_values),
            },
            diagnostics={
                "benchmark_id": self.benchmark_id,
                "source_path": source_path,
                "status": "official_result_aggregated",
                "official_results_normalizer": OFFICIAL_RESULTS_NORMALIZER_SCHEMA_VERSION,
            },
        )

    def _blocked_metric(
        self,
        metric_id: str,
        reason: str,
        diagnostics: Mapping[str, JsonValue] | None = None,
    ) -> MetricResult:
        """
        Build a blocked metric result instead of fabricating a score.

        This is used when a metric cannot be successfully normalized or aggregated
        due to missing data, invalid format, or other issues.

        Args:
            metric_id: Requested metric id.
            reason: Stable blocked reason.
            diagnostics: Structured details for the blocked result.
        """
        return MetricResult(
            sample_id=f"{self.benchmark_id}:official_results",
            metric_id=metric_id,
            valid=False,
            coverage=0.0,
            skip_reason=reason,
            diagnostics={
                "status": "blocked",
                "benchmark_id": self.benchmark_id,
                "official_results_normalizer": OFFICIAL_RESULTS_NORMALIZER_SCHEMA_VERSION,
                **dict(diagnostics or {}),
            },
        )


def load_official_result_records(
    path: str | Path,
) -> tuple[tuple[Mapping[str, JsonValue], ...], str | None, Mapping[str, JsonValue]]:
    """
    Load JSON, JSONL, or CSV records with explicit preflight checks.

    This function attempts to detect the file type by its extension and
    parses its content into a sequence of dictionary-like records.
    It returns the records, a reason for blocking if any issues occurred,
    and diagnostic information.

    Args:
        path: Official result export path.
    """
    source_path = Path(path)
    if not source_path.is_file():
        return (), "official_results_file_missing", {"path": str(source_path)}
    if source_path.stat().st_size == 0:
        return (), "official_results_file_empty", {"path": str(source_path)}

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_records(source_path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl_records(source_path)
    if suffix == ".json":
        return _load_json_records(source_path)

    # Return blocked status if the file format is not supported.
    return (), "unsupported_official_results_format", {"path": str(source_path), "suffix": suffix}


def build_official_metric_mappings(entry: BenchmarkZooEntry) -> tuple[OfficialMetricMapping, ...]:
    """
    Build configurable metric mappings for all benchmark metrics in an entry.

    Args:
        entry: Benchmark-zoo entry with metric schemas.
    """
    return tuple(OfficialMetricMapping.from_metric_spec(metric) for metric in entry.metrics)


def _load_csv_records(path: Path) -> tuple[tuple[Mapping[str, JsonValue], ...], str | None, Mapping[str, JsonValue]]:
    """
    Load records from a CSV file.

    Performs basic validation like checking for a header and non-empty rows.

    Args:
        path: Path to the CSV file.

    Returns:
        A tuple containing the loaded records, a blocked reason (if any),
        and diagnostic information.
    """
    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    # Check if the CSV seems to have a header row by looking for a comma in the first line.
    if "," not in first_line:
        return (), "official_results_csv_header_missing", {"path": str(path)}
    # Use csv.DictReader to parse rows into dictionaries.
    rows = tuple(dict(row) for row in csv.DictReader(text.splitlines()))
    if not rows:
        return (), "official_results_no_records", {"path": str(path)}
    return rows, None, {"path": str(path), "format": "csv"}


def _load_jsonl_records(path: Path) -> tuple[tuple[Mapping[str, JsonValue], ...], str | None, Mapping[str, JsonValue]]:
    """
    Load records from a JSONL (JSON Lines) file.

    Validates each line as a potential JSON object and ensures overall parseability.

    Args:
        path: Path to the JSONL file.

    Returns:
        A tuple containing the loaded records, a blocked reason (if any),
        and diagnostic information.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Identify lines that do not visually resemble JSON objects.
    invalid_lines = [index + 1 for index, line in enumerate(lines) if not _looks_like_json_object(line.strip())]
    if invalid_lines:
        return (), "official_results_jsonl_record_invalid", {"path": str(path), "lines": invalid_lines}
    # Validate if the collection of lines can be parsed as a valid JSON array.
    if not _json_text_valid("[" + ",".join(lines) + "]"):
        return (), "official_results_jsonl_parse_failed", {"path": str(path)}
    # Load each valid JSON line into a Python object.
    rows = tuple(json.loads(line) for line in lines)
    # Filter for only mapping (dictionary) type records.
    mapping_rows = tuple(row for row in rows if isinstance(row, Mapping))
    if len(mapping_rows) != len(rows):
        return (), "official_results_jsonl_record_not_object", {"path": str(path)}
    if not mapping_rows:
        return (), "official_results_no_records", {"path": str(path)}
    return mapping_rows, None, {"path": str(path), "format": "jsonl"}


def _load_json_records(path: Path) -> tuple[tuple[Mapping[str, JsonValue], ...], str | None, Mapping[str, JsonValue]]:
    """
    Load records from a single JSON file.

    Handles both array and object root JSON structures, extracting records
    from common container keys if present.

    Args:
        path: Path to the JSON file.

    Returns:
        A tuple containing the loaded records, a blocked reason (if any),
        and diagnostic information.
    """
    text = path.read_text(encoding="utf-8").strip()
    # Perform a quick structural check to see if the text looks like a JSON object or array.
    if not _looks_like_json_value(text):
        return (), "official_results_json_invalid_shape", {"path": str(path)}
    # Perform a full JSON validity check using the json.tool module.
    if not _json_text_valid(text):
        return (), "official_results_json_parse_failed", {"path": str(path)}
    payload = json.loads(text)
    # Extract records from the parsed JSON payload, potentially from nested keys.
    rows = _records_from_json_payload(payload)
    if not rows:
        return (), "official_results_no_records", {"path": str(path)}
    return rows, None, {"path": str(path), "format": "json"}


def _records_from_json_payload(payload: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    """Extract result records from standard or nested fields in a JSON payload.

    If the payload is a list, it extracts all mapping items. If it's a mapping,
    it checks for common container keys (e.g., 'samples', 'results') and
    extracts lists of mappings from there. If no list is found, the mapping
    itself is considered a single record.

    Args:
        payload: Decoded JSON data.

    Returns:
        Tuple of mapped records.
    """
    if isinstance(payload, list):
        return tuple(item for item in payload if isinstance(item, Mapping))
    if not isinstance(payload, Mapping):
        return ()
    # Look for common keys that might contain a list of records.
    for key in RECORD_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return tuple(item for item in value if isinstance(item, Mapping))
        if isinstance(value, Mapping):
            # Handle cases where records are nested inside a mapping under a container key.
            return _mapping_records(value)
    # If no list of records found, treat the top-level mapping as a single record.
    return (payload,)


def _mapping_records(value: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    """Helper to convert key-mapped JSON metrics into a list of standardized records.

    This function handles a specific JSON structure where a dictionary's keys
    represent metric IDs and their values are dictionaries of metric components.
    It transforms this into a list of records, each including a 'metric_id' field.

    Args:
        value: Key-value mapping of metric items.

    Returns:
        Tuple of reconstructed dictionaries.
    """
    # If all values in the mapping are themselves mappings, assume keys are metric IDs.
    if all(isinstance(item, Mapping) for item in value.values()):
        return tuple(dict(item, metric_id=str(key)) for key, item in value.items())
    # Otherwise, treat the entire mapping as a single record.
    return (value,)


def _record_matches_metric(record: Mapping[str, JsonValue], mapping: OfficialMetricMapping) -> bool:
    """Evaluate if a given record matches a metric mapping's category criteria.

    This is primarily used for "short-format" records where a single record
    may contain data for multiple metrics, distinguished by a category field.

    Args:
        record: Standardized results record.
        mapping: Configured metric mapping spec.

    Returns:
        True if the record matches the metric mapping, False otherwise.
    """
    if mapping.category_field is None:
        return True  # No category filtering specified, so all records match.
    category = record.get(mapping.category_field)
    if category is None:
        return False  # Category field is defined but missing in record.
    # Determine accepted category values: either explicitly listed or fallback to metric ID/source fields.
    accepted = mapping.category_values or (mapping.metric_id, *mapping.source_fields)
    return _metric_key(category) in {_metric_key(item) for item in accepted}


def _missing_required_fields(
    records: Sequence[Mapping[str, JsonValue]], mapping: OfficialMetricMapping
) -> tuple[str, ...]:
    """Check if any of the mandatory fields required by a metric are missing across matching records.

    Args:
        records: Loaded results records.
        mapping: Configured metric mapping spec.

    Returns:
        Tuple of missing required field names.
    """
    fields = mapping.required_fields
    if not fields:
        return ()  # No required fields specified, so none are missing.

    # Optimization: if any "long-format" record directly supplies a value, assume requirements met.
    if any(
        _long_metric_row_matches(record, mapping) and _first_present_value_field(record, mapping) for record in records
    ):
        return ()

    # Filter records to only those relevant to this metric based on category.
    matched_records = tuple(record for record in records if _record_matches_metric(record, mapping))
    if not matched_records:
        return ()  # No records matched, so required fields can't be found.

    # Check each required field to ensure it is present and not null/empty in all matched records.
    return tuple(
        field
        for field in fields
        if not all(field in record and record[field] not in (None, "") for record in matched_records)
    )


def _first_present_field(record: Mapping[str, JsonValue], mapping: OfficialMetricMapping) -> str | None:
    """
    Find the first field from the mapping's source or value fields that is present and not empty in the record.

    Args:
        record: A single record (dictionary) from the official results.
        mapping: The OfficialMetricMapping for the current metric.

    Returns:
        The name of the first found field, or None if no such field exists.
    """
    fields = mapping.source_fields
    if mapping.value_field is not None:
        fields = (mapping.value_field, *fields)  # Prioritize explicit value_field if present.
    for field_name in fields:
        if field_name in record and record[field_name] not in (None, ""):
            return field_name
    return None


def _first_present_value_field(record: Mapping[str, JsonValue], mapping: OfficialMetricMapping) -> str | None:
    """
    Find the first field from the mapping's explicit value field or common VALUE_FIELDS
    that is present and not empty in the record.

    This is used when a record is identified as a "long-format" metric row, meaning
    it contains a specific metric's value rather than being categorized by a field.

    Args:
        record: A single record (dictionary) from the official results.
        mapping: The OfficialMetricMapping for the current metric.

    Returns:
        The name of the first found field, or None if no such field exists.
    """
    # Prioritize the mapping's explicit value_field, then common value field aliases.
    fields = ((mapping.value_field,) if mapping.value_field is not None else ()) + VALUE_FIELDS
    for field_name in fields:
        if field_name in record and record[field_name] not in (None, ""):
            return field_name
    return None


def _long_metric_row_matches(record: Mapping[str, JsonValue], mapping: OfficialMetricMapping) -> bool:
    """
    Determine if a record represents a "long-format" metric row for the given mapping.

    A "long-format" row explicitly identifies the metric within the record itself
    (e.g., a "metric_id" column whose value matches the metric's ID or source field).

    Args:
        record: A single record (dictionary) from the official results.
        mapping: The OfficialMetricMapping for the current metric.

    Returns:
        True if the record contains a field from `METRIC_ID_FIELDS` whose value
        matches the metric's ID or source fields, False otherwise.
    """
    # Create a set of normalized accepted metric identifiers (metric_id, source_fields).
    accepted = {mapping.metric_id, *mapping.source_fields}
    normalized = {_metric_key(item) for item in accepted if item}
    # Check if any of the common metric ID fields in the record have a value
    # that matches one of the accepted normalized metric identifiers.
    return any(
        field_name in record and _metric_key(record[field_name]) in normalized for field_name in METRIC_ID_FIELDS
    )


def _sample_id(record: Mapping[str, JsonValue], fields: Sequence[str]) -> str | None:
    """
    Extract a sample ID from a record using a list of potential field names.

    Args:
        record: A single record (dictionary) from the official results.
        fields: A sequence of field names to check for a sample ID.

    Returns:
        The string representation of the first non-empty sample ID found, or None.
    """
    for field_name in fields:
        value = record.get(field_name)
        if value not in (None, ""):
            return str(value)
    return None


def _numeric_value(value: JsonValue) -> float | None:
    """
    Convert a value to a float if possible, handling various types and string representations.

    Args:
        value: The value to convert.

    Returns:
        The float representation of the value, or None if conversion is not possible.
    """
    # Exclude booleans and empty/None values directly.
    if isinstance(value, bool) or value in (None, ""):
        return None
    # Directly convert integers and floats.
    if isinstance(value, (int, float)):
        return float(value)
    # If it's a string, attempt to parse it as a float.
    if isinstance(value, str):
        text = value.strip()
        if _looks_numeric(text):
            return float(text)
    return None


def _aggregate_values(values: Sequence[float], aggregation: str) -> float:
    """
    Aggregate a sequence of float values using the specified aggregation method.

    Args:
        values: A sequence of float numbers to aggregate.
        aggregation: The name of the aggregation method (e.g., "mean", "sum", "first").

    Returns:
        The aggregated float value. Defaults to mean if aggregation method is unknown.
    """
    kind = aggregation.strip().lower()
    if kind in {"first", "last"}:
        return values[0] if kind == "first" else values[-1]
    if kind == "sum":
        return float(sum(values))
    # All these terms typically imply an average.
    if kind in {"mean", "weighted_mean", "average", "pass_rate", "accuracy"}:
        return float(sum(values) / len(values))
    # Default to mean for unknown aggregation types.
    return float(sum(values) / len(values))


def _metric_raw_row(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """
    Convert a MetricResult dictionary into a format suitable for raw_metric_table.jsonl.

    Args:
        result: A dictionary representation of a MetricResult.

    Returns:
        A dictionary with 'available' status and 'reason' for unavailable metrics.
    """
    value = result.get("normalized_value")
    available = result.get("valid") is True and value is not None
    row = dict(result)
    row["available"] = available
    if not available:
        row["reason"] = row.get("skip_reason") or "metric_not_available"
    return row


def _metric_scorecard_entry(result: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """
    Convert a MetricResult dictionary into a format suitable for a scorecard entry.

    Args:
        result: A dictionary representation of a MetricResult.

    Returns:
        A dictionary representing the scorecard entry, including score,
        availability, and diagnostic information.
    """
    value = result.get("normalized_value")
    if result.get("valid") is True and value is not None:
        return {
            "available": True,
            "raw_score": result.get("raw_value"),
            "normalized_score": value,
            "coverage": result.get("coverage", 1.0),
            "components": result.get("components", {}),
            "diagnostics": result.get("diagnostics", {}),
        }
    return {
        "available": False,
        "reason": result.get("skip_reason") or "metric_not_available",
        "diagnostics": result.get("diagnostics", {}),
    }


def _tuple_of_str(value: JsonValue) -> tuple[str, ...]:
    """
    Convert a value or sequence of values into a tuple of strings.

    Args:
        value: The value to convert. Can be None, a string, or a sequence.

    Returns:
        A tuple of strings. Empty tuple if input is None, single-element
        tuple if input is a string, or converted elements if input is a sequence.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _dedupe_text(values: Sequence[JsonValue]) -> tuple[str, ...]:
    """
    Deduplicate and convert a sequence of values to a tuple of unique strings.

    Args:
        values: A sequence of values (can be any JSON type).

    Returns:
        A tuple containing unique string representations of the input values,
        excluding None or empty strings.
    """
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        if text not in result:
            result.append(text)
    return tuple(result)


def _optional_str(value: JsonValue) -> str | None:
    """
    Convert a value to a string, returning None if the value is None or empty.

    Args:
        value: The value to convert.

    Returns:
        A string or None.
    """
    return None if value in (None, "") else str(value)


def _looks_numeric(value: str) -> bool:
    """
    Heuristically determine if a string represents a numeric value.

    Args:
        value: The string to check.

    Returns:
        True if the string appears numeric, False otherwise.
    """
    text = value.strip()
    if not text:
        return False
    # Handle optional leading sign (+ or -).
    normalized = text[1:] if text[0] in "+-" else text
    # Split by decimal point to check integer and fractional parts.
    parts = normalized.split(".", 1)
    # Must have at most one decimal point, and all parts (if present) must be digits.
    # Also, at least one part must contain digits (e.g., "." alone is not numeric).
    return len(parts) <= 2 and all(part.isdigit() for part in parts if part) and any(part for part in parts)


def _looks_like_json_value(value: str) -> bool:
    """
    Quick check to see if a string structurally resembles a JSON object or array.

    Args:
        value: The string to check.

    Returns:
        True if the string starts and ends with '{}' or '[]', False otherwise.
    """
    return (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]"))


def _looks_like_json_object(value: str) -> bool:
    """
    Quick check to see if a string structurally resembles a JSON object.

    Args:
        value: The string to check.

    Returns:
        True if the string starts and ends with '{}', False otherwise.
    """
    return value.startswith("{") and value.endswith("}")


def _json_text_valid(value: str) -> bool:
    """
    Validate if a given string is valid JSON using Python's `json.tool` module.

    This method uses a subprocess call to leverage the robust JSON parsing and validation
    built into the standard library's `json.tool` module, providing a more reliable check
    than simple `try-except json.loads`.

    Args:
        value: The string containing potential JSON data.

    Returns:
        True if the string is valid JSON, False otherwise.
    """
    completed = subprocess.run(
        (sys.executable, "-m", "json.tool"),  # Execute `python -m json.tool`
        input=value,  # Pass the JSON string as stdin
        capture_output=True,
        text=True,
        check=False,  # Do not raise an exception for non-zero exit codes
    )
    return completed.returncode == 0  # JSON.tool returns 0 for valid JSON, non-zero for invalid.


def _metric_key(value: JsonValue) -> str:
    """
    Normalize a metric identifier string for comparison.

    Converts the value to a string, strips whitespace, converts to lowercase,
    and replaces underscores and spaces with hyphens.

    Args:
        value: The metric identifier.

    Returns:
        The normalized metric key string.
    """
    return str(value).strip().casefold().replace("_", "-").replace(" ", "-")
