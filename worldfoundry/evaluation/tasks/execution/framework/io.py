"""Input/output and score coercion utilities for WorldFoundry execution.

This module provides serialization mappings, environment variable path parsers,
and statistical math helper functions to parse, normalize, and coerce raw numerical
scores or metrics emitted by official benchmark runtimes into canonical forms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from worldfoundry.core.io.serialization import (
    append_jsonl,
    jsonable,
    read_json as load_json,
    read_json,
    read_json_object,
    read_json_or_jsonl,
    read_jsonl_objects,
    reset_jsonl,
    write_json,
    write_jsonl,
)
from worldfoundry.core.time import utc_now_iso


# Typed dictionary format for general JSON-like data mappings
JsonMapping = Mapping[str, Any]

# Standard upstream key names searched in descending order to locate numerical score evaluations
DEFAULT_SCORE_KEYS = ("score", "raw_score", "value", "mean", "average")


def env_path(*names: object, default: Path | str | None = None) -> Path | None:
    """Attempts to find and parse a file path from a prioritized list of OS environment variables.

    Iterates through the provided list of environment variable names. The first non-empty
    value located is wrapped as a Path. If none are found, returns the designated default path.

    Args:
        *names: Prioritized list of environment variable keys to scan. If the last item
               is not a string and no default parameter is supplied, it is treated as the default value.
        default: Fallback path to return if none of the environment variables are set.

    Returns:
        The resolved Path object, or None if no environment keys exist and no default was specified.
    """
    if names and not isinstance(names[-1], str):
        if default is None:
            default = names[-1]
        names = names[:-1]
    for name in names:
        value = os.environ.get(str(name))
        if value:
            return Path(value)
    return None if default is None else Path(default)


def mean_numeric(values: Iterable[float | int | None]) -> float | None:
    """Computes the arithmetic mean of numeric values while strictly ignoring None or booleans.

    Args:
        values: An iterable of optional integers, floats, or None.

    Returns:
        The computed float mean, or None if no valid numeric values were present in the iterable.
    """
    clean = [float(value) for value in values if value is not None and not isinstance(value, bool)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def optional_float(value: Any) -> float | None:
    """Convert an optional numeric value to ``float`` while preserving missing values."""
    return None if value in (None, "") else float(value)


def scalar_number(
    value: Any,
    *,
    dict_keys: Iterable[str] = DEFAULT_SCORE_KEYS,
    list_mode: str | None = None,
    allow_bool: bool = False,
    reject_negative: bool = False,
) -> float | None:
    """Extracts a singular normalized float score from various possible unstructured result types.

    This utility handles parsing float scalars, boolean indicators, string-encoded numbers,
    iterables/lists (supporting mean or first item reductions), and complex key-value dictionaries.

    Args:
        value: The raw input to parse (can be scalar, list, dict, str, etc.).
        dict_keys: An ordered sequence of keys to inspect sequentially when traversing a dictionary.
        list_mode: Dictates list resolution: "mean" to average nested numbers, "first" to return
                   the first valid numerical item, or None to ignore list types entirely.
        allow_bool: If True, converts booleans to float scores (True -> 1.0, False -> 0.0).
        reject_negative: If True, treats any value less than 0.0 as invalid/missing (returns None).

    Returns:
        The extracted and validated float score, or None if no numeric value could be resolved.
    """
    if isinstance(value, bool):
        if not allow_bool:
            return None
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return None if reject_negative and number < 0 else number
    if isinstance(value, str):
        try:
            return scalar_number(
                float(value.strip()),
                dict_keys=dict_keys,
                list_mode=list_mode,
                allow_bool=allow_bool,
                reject_negative=reject_negative,
            )
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        if list_mode not in {"mean", "first"}:
            return None
        numbers = [
            scalar_number(
                item,
                dict_keys=dict_keys,
                list_mode=list_mode,
                allow_bool=allow_bool,
                reject_negative=reject_negative,
            )
            for item in value
        ]
        if list_mode == "first":
            # Return the first encountered non-empty score
            return next((number for number in numbers if number is not None), None)
        return mean_numeric(numbers)
    if isinstance(value, Mapping):
        # Scan predefined dictionary keys in priority sequence
        for key in dict_keys:
            if key in value:
                number = scalar_number(
                    value[key],
                    dict_keys=dict_keys,
                    list_mode=list_mode,
                    allow_bool=allow_bool,
                    reject_negative=reject_negative,
                )
                if number is not None:
                    return number
    return None


def normalize_unit_score(raw_score: float | None) -> float | None:
    """Normalizes raw scores into the standard [0.0, 1.0] unit interval.

    Supports handling both 0..1 interval scores and 0..100 percentage scores by dividing
    the latter by 100.0. Clamps any out-of-bound float values directly into the interval.

    Args:
        raw_score: The raw input float score to normalize.

    Returns:
        The coerced unit interval score [0.0, 1.0], or None if the input was None.
    """
    if raw_score is None:
        return None
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    if 1.0 < raw_score <= 100.0:
        return raw_score / 100.0
    return max(0.0, min(1.0, raw_score))


def score_item(raw_score: float | None, source: str, sample_count: int | float | None = None) -> dict[str, Any]:
    """Constructs a standardized evaluation scorecard record capturing score provenance.

    Maintains tracing metadata indicating what the raw score was, which field/logic produced it,
    and how many evaluation samples were compiled to derive it.

    Args:
        raw_score: The final extracted raw float score.
        source: The name of the upstream field or rule responsible for compiling the score.
        sample_count: Optional count representing the size of the underlying sample dataset.

    Returns:
        A dictionary matching the WorldFoundry scorecard schema for score properties.
    """
    return {"raw_score": raw_score, "source": source, "sample_count": sample_count}


__all__ = [
    "DEFAULT_SCORE_KEYS",
    "JsonMapping",
    "append_jsonl",
    "env_path",
    "jsonable",
    "load_json",
    "mean_numeric",
    "normalize_unit_score",
    "optional_float",
    "read_json",
    "read_json_object",
    "read_json_or_jsonl",
    "read_jsonl_objects",
    "reset_jsonl",
    "scalar_number",
    "score_item",
    "utc_now_iso",
    "write_json",
    "write_jsonl",
]
