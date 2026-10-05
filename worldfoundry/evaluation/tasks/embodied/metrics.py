"""Metric computation suite for VLA, Video Action, and World Action Model evaluations.

This module provides standard, robust evaluation metrics tailored to the four main
embodied AI tracks. It dynamically matches scalar metrics emitted by runner results
across various namespaces/metadata structures, coerces diverse types (including string success indicators),
and aggregates results over samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.api import (
    AggregateResult,
    GenerationRequest,
    GenerationResult,
    MetricResult,
    is_generation_result_successful,
)
from worldfoundry.evaluation.tasks.metrics.registry import default_metric_registry

# Default set of metrics that this module recognizes and supports mapping.
DEFAULT_METRIC_IDS = (
    "generation_success",
    "task_success",
    "success",
    "success_rate",
    "action_accuracy",
    "planning_success",
    "rollout_success",
    "world_state_consistency",
    "reward",
)

# Semantic sets representing positive/negative boolean conditions in raw runner logs.
_SUCCESS_TRUE = {"true", "yes", "success", "succeeded", "pass", "passed", "ok", "done"}
_SUCCESS_FALSE = {"false", "no", "failure", "failed", "fail", "error", "errored"}


def _coerce_number(value: Any) -> float | None:
    """Coerces arbitrary value types (booleans, strings, percentage strings, numbers) into a standard float.

    This function handles common representations of numerical values or success indicators,
    converting them into a `float` where possible, or `None` otherwise.

    Args:
        value: The input value to coerce. Can be a boolean, int, float, or string.

    Returns:
        The coerced float value, or `None` if coercion is not possible.

    Examples:
        - True -> 1.0, False -> 0.0
        - "passed" -> 1.0, "failed" -> 0.0
        - "85%" -> 0.85
        - "0.91" -> 0.91
        - "invalid" -> None
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        # Handle string representations of success/failure
        if lowered in _SUCCESS_TRUE:
            return 1.0
        if lowered in _SUCCESS_FALSE:
            return 0.0
        # Handle percentage strings (e.g., "85%")
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return None
        # Attempt direct float conversion for numeric strings
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _nested_mappings(result: GenerationResult) -> tuple[Mapping[str, Any], ...]:
    """Retrieves all potential mappings nested in a GenerationResult where metrics might reside.

    Since different model runners emit metrics in varying namespaces (e.g., standard metadata,
    specialized `vla_va_wam` block, or generic `extra` collections), this helper creates a prioritized
    list of dictionary views to search. Each potential mapping is checked to ensure it is actually
    a dictionary before being included.

    Args:
        result: The `GenerationResult` object containing metadata.

    Returns:
        A tuple of mappings, prioritized by likelihood of containing relevant metrics.
    """
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    vla_va_wam = metadata.get("vla_va_wam") if isinstance(metadata.get("vla_va_wam"), Mapping) else {}
    extra = metadata.get("extra") if isinstance(metadata.get("extra"), Mapping) else {}
    return (
        metadata,
        metadata.get("metrics") if isinstance(metadata.get("metrics"), Mapping) else {},
        metadata.get("scores") if isinstance(metadata.get("scores"), Mapping) else {},
        metadata.get("outputs") if isinstance(metadata.get("outputs"), Mapping) else {},
        vla_va_wam,
        vla_va_wam.get("metrics") if isinstance(vla_va_wam.get("metrics"), Mapping) else {},
        vla_va_wam.get("scores") if isinstance(vla_va_wam.get("scores"), Mapping) else {},
        vla_va_wam.get("outputs") if isinstance(vla_va_wam.get("outputs"), Mapping) else {},
        extra,
        extra.get("metrics") if isinstance(extra.get("metrics"), Mapping) else {},
        extra.get("scores") if isinstance(extra.get("scores"), Mapping) else {},
    )


def _is_failed(result: GenerationResult) -> bool:
    """Returns True if the basic generation call failed during model execution (e.g., OOM, crash).

    This checks the fundamental success status of the `GenerationResult` itself,
    independent of task-specific success metrics.

    Args:
        result: The `GenerationResult` to check.

    Returns:
        True if the generation result indicates a failure, False otherwise.
    """
    return not is_generation_result_successful(result)


def _lookup_result_value(result: GenerationResult, names: Sequence[str]) -> tuple[str | None, float | None]:
    """Searches across prioritized nested mappings for the first matched scalar metric.

    This function iterates through a sequence of potential field names (aliases) and
    a prioritized list of nested dictionaries within the `GenerationResult` to find
    and coerce a metric value.

    Args:
        result: The `GenerationResult` payload to search within.
        names: A sequence of alias keys to search for (e.g., `["task_success", "success", "success_rate"]`).
               The search order follows this sequence.

    Returns:
        A tuple of (`matched_field_name`, `coerced_float_value`).
        Returns `(None, None)` if no matching field with a coercible value is found.
    """
    for name in names:
        # Handle "generation_success" as a special case, directly derived from the result's top-level status.
        if name == "generation_success":
            return name, 0.0 if _is_failed(result) else 1.0
        # Iterate through all known potential locations for metrics within the result metadata.
        for mapping in _nested_mappings(result):
            if name not in mapping:
                continue
            # Attempt to coerce the found value into a float.
            value = _coerce_number(mapping.get(name))
            if value is not None:
                return name, value
    return None, None


@dataclass(frozen=True)
class ResultFieldMetric:
    """An abstract Metric implementation that reads scalar values directly from a GenerationResult.

    This class defines how to compute a single metric from a `GenerationResult`
    by looking up a specified field (or aliases) and how to aggregate these
    individual sample results across a collection.

    Attributes:
        name: The canonical name of the metric (e.g., "task_success").
        field_names: A tuple of field names (aliases) to search for in the result metadata,
                     in order of preference. If empty, `name` is used as the sole field name.
        version: The version of this metric definition (default: "1.0").
        required_artifacts: A tuple of artifact IDs that must be present for this metric to be valid.
                            (Currently not used in compute_sample, but available for future extensions).
        higher_is_better: A boolean indicating whether higher values of this metric are preferable.
                          `None` if not applicable or undefined.
    """

    name: str
    field_names: tuple[str, ...] = ()
    version: str = "1.0"
    required_artifacts: tuple[str, ...] = ()
    higher_is_better: bool | None = None

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {"field_names": self.field_names}

    def __post_init__(self) -> None:
        """Normalizes and sets attribute values after initialization.

        Ensures that `name`, `field_names`, and `required_artifacts` are of the correct
        type (str, tuple of str) and handles default values for `field_names`.
        Uses `object.__setattr__` because the dataclass is frozen.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "field_names", tuple(str(item) for item in (self.field_names or (self.name,))))
        object.__setattr__(self, "required_artifacts", tuple(str(item) for item in self.required_artifacts))

    def compute_sample(self, request: GenerationRequest, result: GenerationResult) -> MetricResult:
        """Computes a single sample's metric result by looking up configured fields.

        It attempts to find a numeric value for the metric within the `GenerationResult`
        using the defined `field_names` and returns a `MetricResult`.

        Args:
            request: The `GenerationRequest` associated with the sample (not directly used by this metric).
            result: The `GenerationResult` from which to extract the metric value.

        Returns:
            A `MetricResult` instance containing the computed value, or a skipped result
            if the value could not be found or coerced.
        """
        del request  # Not used by this metric implementation
        source_field, value = _lookup_result_value(result, self.field_names)
        if value is None:
            # If no valid, coercible value is found, mark the sample as invalid/skipped.
            return MetricResult(
                sample_id=result.sample_id,
                metric_id=self.name,
                valid=False,
                coverage=0.0,
                skip_reason="metric value not emitted by runner",
                diagnostics={"searched_fields": list(self.field_names)},
            )
        return MetricResult(
            sample_id=result.sample_id,
            metric_id=self.name,
            raw_value=value,
            normalized_value=value,
            diagnostics={"source_field": source_field or self.name},
        )

    def aggregate(self, results: Sequence[MetricResult]) -> AggregateResult:
        """Aggregates individual sample results across the entire dataset.

        Computes the mean of all valid normalized/raw values and provides counts
        of total, valid, and skipped samples.

        Args:
            results: A sequence of `MetricResult` objects, typically from `compute_sample`.

        Returns:
            An `AggregateResult` summarizing the performance for this metric across all samples.
        """
        values: list[float] = []
        for result in results:
            if not result.valid:
                continue
            # Coerce value again to ensure it's a float, handling cases where it might be stored as other types.
            value = _coerce_number(result.normalized_value if result.normalized_value is not None else result.raw_value)
            if value is not None:
                values.append(value)

        # Calculate the mean only if there are valid values to average.
        stats = {"mean": sum(values) / len(values)} if values else {}
        return AggregateResult(
            metric_id=self.name,
            n_total=len(results),
            n_valid=len(values),
            n_skipped=len(results) - len(values),
            raw_stats=stats,
            normalized_stats=stats,
            valid=bool(values),  # Aggregate result is valid if at least one sample contributed a value.
        )


def metric_from_id(metric_id: str) -> ResultFieldMetric:
    """Factory to build a `ResultFieldMetric` from standard ID with built-in fallback aliasing.

    Ensures that mismatched keys (e.g., "success" vs "task_success" vs "success_rate") from
    different environment configurations map cleanly to standard targets by providing a list
    of possible field names to search for.

    Args:
        metric_id: The canonical ID of the metric to create (e.g., "task_success").

    Returns:
        A `ResultFieldMetric` instance configured with appropriate field aliases.
    """
    # Define common aliases for metrics to handle variations in runner output field names.
    aliases = {
        "task_success": ("task_success", "success", "success_rate"),
        "success_rate": ("success_rate", "success", "task_success"),
        "rollout_success": ("rollout_success", "success", "task_success"),
        "episode_success": ("episode_success", "success", "success_rate", "task_success"),
        "sequence_success": ("sequence_success", "success", "success_rate", "task_success"),
        "goal_success": ("goal_success", "success", "success_rate", "task_success"),
        "planning_success": ("planning_success", "success", "task_success"),
        "action_accuracy": ("action_accuracy", "accuracy"),
        "normalized_return": ("normalized_return", "return", "reward"),
        "world_state_consistency": ("world_state_consistency", "state_consistency", "consistency"),
    }
    try:
        direction = default_metric_registry().get(metric_id).higher_is_better
    except KeyError:
        direction = True if metric_id in DEFAULT_METRIC_IDS or metric_id in aliases else None
    return ResultFieldMetric(
        name=metric_id, field_names=aliases.get(metric_id, (metric_id,)), higher_is_better=direction,
    )


def default_metric_ids_for_track(track: str) -> tuple[str, ...]:
    """Retrieves standard default evaluation metric IDs associated with a specific evaluation track.

    Args:
        track: The name of the evaluation track (e.g., "vla", "va", "wam"). Case-insensitive.

    Returns:
        A tuple of default metric IDs for the specified track. Returns only "generation_success"
        if the track is not recognized.
    """
    normalized = track.strip().lower()
    if normalized == "vla":
        return ("generation_success", "task_success", "action_accuracy")
    if normalized in {"va", "vam"}:
        return ("generation_success", "task_success", "planning_success", "action_accuracy")
    if normalized == "wam":
        return ("generation_success", "rollout_success", "world_state_consistency", "reward")
    return ("generation_success",)


def metric_suite(metric_ids: Sequence[str] | None = None, *, track: str | None = None) -> tuple[ResultFieldMetric, ...]:
    """Assembles a suite of evaluation metrics based on requested IDs or defaults for the given track.

    If `metric_ids` are provided, those specific metrics will be created. Otherwise,
    default metrics for the specified `track` will be used.

    Args:
        metric_ids: An optional sequence of metric IDs to include in the suite.
                    If `None` or empty, `track` will be used to determine defaults.
        track: An optional string indicating the evaluation track. Used to select
               default metrics if `metric_ids` is not provided.

    Returns:
        A tuple of `ResultFieldMetric` instances comprising the evaluation suite.
    """
    selected = tuple(str(item) for item in (metric_ids or ()))
    # If no specific metric IDs are provided, determine the default set based on the track.
    if not selected:
        selected = default_metric_ids_for_track(track or "")
    return tuple(metric_from_id(metric_id) for metric_id in selected)


__all__ = [
    "DEFAULT_METRIC_IDS",
    "ResultFieldMetric",
    "default_metric_ids_for_track",
    "metric_from_id",
    "metric_suite",
]
