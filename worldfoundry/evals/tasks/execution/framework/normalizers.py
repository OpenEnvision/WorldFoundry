"""Metric normalizer utilities to convert raw scores to unit intervals.

This module provides parsing for normalizer specifications (e.g., identity, scale_max,
and vbench minmax) and applies them to map raw benchmark output scores onto [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class NormalizerSpecError(ValueError):
    """Raised when a benchmark-zoo metric normalizer string is malformed."""


@dataclass(frozen=True)
class NormalizerSpec:
    """Specification of a parsed normalizer kind and its configuring arguments.

    Attributes:
        kind: The identifier of the normalizer (e.g. 'scale_max').
        args: Optional float parameters for the normalizer calculation.
    """

    kind: str
    args: tuple[float, ...] = ()

    def apply(self, value: float | int) -> float:
        """Apply this normalization specification to a numeric score.

        Args:
            value: Input raw numeric score.

        Returns:
            Normalized float value.
        """
        return apply_normalizer(self, value)


def _parse_float(value: str, *, context: str) -> float:
    """Parse a string value to a finite float, raising a descriptive error on failure.

    Args:
        value: Input string.
        context: Context label for diagnostic error reporting.

    Returns:
        The parsed float value.

    Raises:
        NormalizerSpecError: If the parsing fails or the value is not finite.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise NormalizerSpecError(f"{context} must be numeric: {value!r}") from exc
    if not isfinite(parsed):
        raise NormalizerSpecError(f"{context} must be finite: {value!r}")
    return parsed


def _clip_unit(value: float) -> float:
    """Clip a float value to the [0.0, 1.0] unit interval.

    Args:
        value: Input float.

    Returns:
        Clipped float value.
    """
    return min(1.0, max(0.0, value))


def parse_normalizer(value: str | None) -> NormalizerSpec:
    """Parse a normalizer specification string into a NormalizerSpec object.

    Args:
        value: Normalizer specification string (e.g., 'scale_max:5.0').

    Returns:
        Parsed NormalizerSpec.

    Raises:
        NormalizerSpecError: If the specification syntax is invalid or unsupported.
    """
    if value is None or value == "":
        return NormalizerSpec("identity")

    parts = value.split(":")
    kind = parts[0].strip()
    raw_args = parts[1:]

    if kind == "identity":
        if len(raw_args) not in {0, 2}:
            raise NormalizerSpecError("identity normalizer accepts zero args or advisory min/max bounds")
        return NormalizerSpec(kind, tuple(_parse_float(item, context="identity bound") for item in raw_args))

    if kind == "scale_max":
        if len(raw_args) != 1:
            raise NormalizerSpecError("scale_max normalizer requires one max value")
        max_value = _parse_float(raw_args[0], context="scale_max max")
        if max_value <= 0:
            raise NormalizerSpecError("scale_max max value must be positive")
        return NormalizerSpec(kind, (max_value,))

    if kind == "percent_or_fraction_to_unit":
        if raw_args:
            raise NormalizerSpecError("percent_or_fraction_to_unit normalizer does not accept args")
        return NormalizerSpec(kind)

    if kind == "official_vbench_minmax":
        if len(raw_args) != 2:
            raise NormalizerSpecError("official_vbench_minmax normalizer requires min and max")
        lower = _parse_float(raw_args[0], context="official_vbench_minmax min")
        upper = _parse_float(raw_args[1], context="official_vbench_minmax max")
        if upper <= lower:
            raise NormalizerSpecError("official_vbench_minmax max must be greater than min")
        return NormalizerSpec(kind, (lower, upper))

    raise NormalizerSpecError(f"unknown benchmark-zoo normalizer: {value!r}")


def apply_normalizer(spec: str | NormalizerSpec | None, value: float | int) -> float:
    """Apply a normalizer specification to a numeric score.

    Args:
        spec: Normalizer spec string, object, or None (interpreted as identity).
        value: Input raw numeric score.

    Returns:
        Normalized score.

    Raises:
        NormalizerSpecError: If the score is not finite or normalizer kind is unsupported.
    """
    parsed = parse_normalizer(spec) if isinstance(spec, str) or spec is None else spec
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise NormalizerSpecError(f"metric value must be finite: {value!r}")

    if parsed.kind == "identity":
        return numeric_value
    if parsed.kind == "scale_max":
        return _clip_unit(numeric_value / parsed.args[0])
    if parsed.kind == "percent_or_fraction_to_unit":
        return _clip_unit(numeric_value / 100.0 if numeric_value > 1.0 else numeric_value)
    if parsed.kind == "official_vbench_minmax":
        lower, upper = parsed.args
        return _clip_unit((numeric_value - lower) / (upper - lower))

    raise NormalizerSpecError(f"unsupported benchmark-zoo normalizer kind: {parsed.kind!r}")
