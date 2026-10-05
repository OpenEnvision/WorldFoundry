"""Lightweight cross-layer contracts shared by WorldFoundry runtimes.

Studio, CLI, evaluation, and synthesis all need two JSON-safe bags:

- :class:`PipelineInvocation` — one generate request after path /
  prompt / media normalization. ``request`` stays ``Any`` because the
  concrete ``GenerationRequest`` type lives in the optional evaluation
  layer.
- :class:`OptimizationSnapshot` — requested vs effective acceleration
  flags (TeaCache, compile, FP8, …), fallbacks, and a quality tier.
  Extra keys go into ``extensions`` and must not collide with the
  reserved field names.

Values are coerced through a strict JSON subset (no NaN, no numpy
scalars) so a snapshot can be written to a manifest without a custom
encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

# ──────────────────────────────────────────────────────────────────────────
# JSON subset — reject NaN / numpy scalars so manifests need no encoder
# ──────────────────────────────────────────────────────────────────────────

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class PipelineInvocation:
    """Normalized inputs for one pipeline-backed generation request.

    ``request`` deliberately remains typed as :class:`Any`: the core contract
    is also used by pipelines and synthesis runtimes, while the concrete
    ``GenerationRequest`` type belongs to the optional evaluation layer.
    """

    request: Any
    prompt: str
    image: Any
    video: Any
    interactions: Any
    ref_image_path: Any
    output_path: Path
    operator_kwargs: Mapping[str, Any]
    pipeline_kwargs: Mapping[str, Any]


def _json_value(value: Any, *, path: str = "value") -> JsonValue:
    """Return a detached, strictly JSON-compatible representation."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _json_mapping(value: Mapping[str, Any] | None, *, path: str) -> dict[str, JsonValue]:
    """Coerce ``None`` to ``{}`` and require the result to be a JSON object."""
    normalized = _json_value({} if value is None else value, path=path)
    return cast(dict[str, JsonValue], normalized)


def _extensions(data: Mapping[str, Any], known: frozenset[str]) -> dict[str, JsonValue]:
    """Collect explicit ``extensions`` plus any keys outside *known* reserved names."""
    explicit = data.get("extensions")
    result = _json_mapping(explicit if isinstance(explicit, Mapping) else {}, path="extensions")
    for key, value in data.items():
        if key not in known:
            result[key] = _json_value(value, path=key)
    return result


def _with_extensions(payload: dict[str, JsonValue], extensions: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Merge extensions last; refuse keys that collide with reserved snapshot fields."""
    for key, value in extensions.items():
        if key in payload:
            raise ValueError(f"extension key conflicts with a manifest field: {key!r}")
        payload[key] = _json_value(value, path=f"extensions.{key}")
    return payload


# ──────────────────────────────────────────────────────────────────────────
# Optimization snapshot — requested vs effective flags, JSON-safe
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OptimizationSnapshot:
    """Requested and effective optimization state for one execution."""

    requested: Mapping[str, JsonValue] = field(default_factory=dict)
    effective: Mapping[str, JsonValue] = field(default_factory=dict)
    fallbacks: tuple[JsonValue, ...] = ()
    quality_tier: str = "exact"
    extensions: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze every field through the JSON subset so a later dump cannot fail."""
        object.__setattr__(self, "requested", _json_mapping(self.requested, path="requested"))
        object.__setattr__(self, "effective", _json_mapping(self.effective, path="effective"))
        object.__setattr__(
            self,
            "fallbacks",
            tuple(_json_value(value, path=f"fallbacks[{index}]") for index, value in enumerate(self.fallbacks)),
        )
        object.__setattr__(self, "extensions", _json_mapping(self.extensions, path="extensions"))

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize to a JSON-ready mapping, merging ``extensions`` last."""
        payload: dict[str, JsonValue] = {
            "requested": _json_mapping(self.requested, path="requested"),
            "effective": _json_mapping(self.effective, path="effective"),
            "fallbacks": [_json_value(value, path="fallbacks") for value in self.fallbacks],
            "quality_tier": self.quality_tier,
        }
        return _with_extensions(payload, self.extensions)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizationSnapshot":
        """Rebuild a snapshot; unknown keys become ``extensions``."""
        known = frozenset({"requested", "effective", "fallbacks", "quality_tier", "extensions"})
        requested = data.get("requested")
        effective = data.get("effective")
        fallbacks = data.get("fallbacks")
        return cls(
            requested=requested if isinstance(requested, Mapping) else {},
            effective=effective if isinstance(effective, Mapping) else {},
            fallbacks=(
                tuple(fallbacks)
                if isinstance(fallbacks, Sequence) and not isinstance(fallbacks, (str, bytes))
                else ()
            ),
            quality_tier=str(data.get("quality_tier") or "exact"),
            extensions=_extensions(data, known),
        )


__all__ = ["JsonScalar", "JsonValue", "OptimizationSnapshot", "PipelineInvocation"]
