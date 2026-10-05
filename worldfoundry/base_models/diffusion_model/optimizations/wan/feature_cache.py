"""Wan feature-cache selection and calibrated policy construction.

The public runtime accepts exactly one whole-stack or block-selective policy.
Selection happens while the component is built; an unsupported calibration
fails there instead of accepting a flag that later no-ops. Every CFG branch
receives an independent cache instance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from worldfoundry.core.acceleration.cache import (
    AdaCacheResidualCache,
    BlockTaylorSeerCache,
    CustomTaylorResidualCache,
    DualBlockFeatureCache,
    DynamicBlockFeatureCache,
    FirstBlockFeatureCache,
    MagCacheResidualCache,
    TaylorSeerResidualCache,
    TeaCacheResidualCache,
)

_TEACACHE_PRESETS: tuple[
    tuple[tuple[str, ...], tuple[float, ...], str, int], ...
] = (
    (
        ("wan2.1-t2v-1.3b", "wan2.1-fun-1.3b", "wan2.1-vace-1.3b"),
        (-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02),
        "timestep-modulation",
        5,
    ),
    (
        ("wan2.1-t2v-14b",),
        (-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01),
        "timestep-modulation",
        5,
    ),
    (
        ("wan2.1-i2v-14b-480p",),
        (2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01),
        "timestep-modulation",
        5,
    ),
    (
        (
            "wan2.1-i2v-14b-720p",
            "wan2.1-fun-14b",
            "wan2.2-fun",
            "wan2.2-i2v-a14b",
            "wan2.2-t2v-a14b",
            "wan2.1-vace",
        ),
        (8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02),
        "timestep-modulation",
        5,
    ),
    (
        ("wan2.2-ti2v-5b",),
        (1.57472669e05, -1.15702395e05, 3.10761669e04, -3.83116651e03, 2.21608777e02, -4.81179567),
        "time-embedding",
        1,
    ),
)

_CACHE_NAMES = (
    "teacache",
    "magcache",
    "adacache",
    "taylorseer",
    "blocktaylorseer",
    "custom",
    "firstblock",
    "dualblock",
    "dynamicblock",
)


def _normalize_algorithm(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "taylorseerblock": "blocktaylorseer",
        "lightx2vtaylorseer": "blocktaylorseer",
        "customcaching": "custom",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class WanFeatureCacheConfig:
    """Validated model-specific feature-cache configuration."""

    algorithm: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        algorithm = _normalize_algorithm(self.algorithm)
        if algorithm not in _CACHE_NAMES:
            raise ValueError(f"unsupported Wan feature cache algorithm: {self.algorithm!r}")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


def _scope_value(context: Any, name: str) -> Any:
    if name in context.component_options:
        return context.component_options[name]
    return context.policy.options.get(name)


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if value is True:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be true or a mapping")
    return {str(key): item for key, item in value.items()}


def _teacache_preset(model_id: str) -> tuple[tuple[float, ...], str, int] | None:
    normalized = str(model_id).lower()
    for aliases, coefficients, signal, warmup in _TEACACHE_PRESETS:
        if any(alias in normalized for alias in aliases):
            return coefficients, signal, warmup
    return None


def resolve_wan_feature_cache(context: Any) -> WanFeatureCacheConfig | None:
    """Resolve one feature-cache request or fail closed on ambiguity."""

    generic = _scope_value(context, "feature_cache")
    selected: list[tuple[str, Any]] = []
    if generic not in (None, False, "", "none", "disabled"):
        if isinstance(generic, str):
            selected.append((generic, True))
        elif isinstance(generic, Mapping):
            values = dict(generic)
            algorithm = values.pop("algorithm", values.pop("kind", None))
            if algorithm is None:
                raise ValueError("feature_cache mapping requires 'algorithm' or 'kind'")
            selected.append((str(algorithm), values))
        else:
            raise TypeError("feature_cache must be a string or mapping")

    for name in _CACHE_NAMES:
        value = _scope_value(context, name)
        if value not in (None, False, "", "none", "disabled"):
            selected.append((name, value))
    if not selected:
        return None
    normalized_names = [_normalize_algorithm(name) for name, _ in selected]
    if len(selected) != 1:
        raise ValueError(
            "Wan feature caches are mutually exclusive; requested "
            + ", ".join(normalized_names)
        )

    algorithm = normalized_names[0]
    raw_value = selected[0][1]
    if algorithm in {"teacache", "custom"}:
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            options: dict[str, Any] = {"threshold": float(raw_value)}
        else:
            options = _mapping(raw_value, owner=algorithm)
        legacy_threshold = _scope_value(context, "teacache_thresh")
        if legacy_threshold is not None:
            options["threshold"] = float(legacy_threshold)
        options.setdefault("threshold", 0.26)
        preset = _teacache_preset(context.model_id)
        if "coefficients" not in options:
            if preset is None:
                raise ValueError(
                    f"TeaCache has no published coefficients for {context.model_id!r}; "
                    "supply teacache={'coefficients': [...]} explicitly"
                )
            options["coefficients"], options["signal"], options["warmup_steps"] = preset
        options.setdefault("signal", "timestep-modulation")
        options.setdefault("warmup_steps", 5)
        allowed_options = {
            "coefficients",
            "dense_last",
            "eps",
            "signal",
            "threshold",
            "warmup_steps",
        }
        unknown_options = sorted(set(options) - allowed_options)
        if unknown_options:
            raise ValueError(f"unsupported {algorithm} options: {unknown_options}")
        threshold = float(options["threshold"])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f"{algorithm} threshold must be finite and non-negative")
        options["threshold"] = threshold
        options["warmup_steps"] = _nonnegative_integer(
            options["warmup_steps"],
            owner="warmup_steps",
        )
        if algorithm == "custom":
            # Pinned LightX2V doubles its global CFG call counter. Per-branch
            # state translates that exactly to five warmup steps/no dense tail
            # for timestep modulation, or one warmup/one dense tail for the
            # time-embedding (non-ret-step) configuration.
            options.setdefault(
                "dense_last",
                0 if int(options["warmup_steps"]) > 1 else 1,
            )
        options["dense_last"] = _nonnegative_integer(
            options.get("dense_last", 1),
            owner="dense_last",
        )
        if "eps" in options:
            eps = float(options["eps"])
            if not math.isfinite(eps) or eps <= 0:
                raise ValueError("feature-cache eps must be finite and positive")
            options["eps"] = eps
    elif algorithm == "magcache":
        options = _mapping(raw_value, owner="magcache")
        if "ratios" not in options:
            raise ValueError(
                "MagCache requires calibrated per-step 'ratios'; no Wan2.2 preset is "
                "published by LightX2V for this model"
            )
        options.setdefault("threshold", 0.24)
        options.setdefault("max_skip_steps", options.pop("K", 6))
        options.setdefault("retention_ratio", 0.2)
    elif algorithm == "adacache":
        options = _mapping(raw_value, owner="adacache")
    elif algorithm == "taylorseer":
        options = _mapping(raw_value, owner="taylorseer")
        options.setdefault("interval", 4)
    elif algorithm == "blocktaylorseer":
        options = _mapping(raw_value, owner="blocktaylorseer")
        allowed_options = {"dense_pattern", "dense_last"}
        unknown_options = sorted(set(options) - allowed_options)
        if unknown_options:
            raise ValueError(
                f"unsupported blocktaylorseer options: {unknown_options}"
            )
        pattern = options.get("dense_pattern", (True, False, False, False))
        if not isinstance(pattern, Sequence) or isinstance(pattern, (str, bytes)):
            raise TypeError("BlockTaylorSeer dense_pattern must be a bool sequence")
        parsed_pattern = tuple(pattern)
        if not parsed_pattern or any(not isinstance(value, bool) for value in parsed_pattern):
            raise TypeError("BlockTaylorSeer dense_pattern must be a non-empty bool sequence")
        if not any(parsed_pattern):
            raise ValueError("BlockTaylorSeer dense_pattern must contain a dense step")
        options["dense_pattern"] = parsed_pattern
        options["dense_last"] = _nonnegative_integer(
            options.get("dense_last", 0),
            owner="dense_last",
        )
    elif algorithm in {"firstblock", "dualblock", "dynamicblock"}:
        options = _mapping(raw_value, owner=algorithm)
        allowed_options = {
            "residual_diff_threshold",
            "downsample_factor",
            "dense_first",
            "dense_last",
            "eps",
        }
        if algorithm == "dualblock":
            allowed_options.update({"front_blocks", "back_blocks"})
        unknown_options = sorted(set(options) - allowed_options)
        if unknown_options:
            raise ValueError(
                f"unsupported {algorithm} options: {unknown_options}"
            )
        if "residual_diff_threshold" not in options:
            raise ValueError(
                f"{algorithm} requires an explicit 'residual_diff_threshold'; "
                "no quality-safe default is published"
            )
        threshold = options["residual_diff_threshold"]
        if isinstance(threshold, bool):
            raise TypeError("residual_diff_threshold must be a number")
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("residual_diff_threshold must be finite and non-negative")
        options["residual_diff_threshold"] = threshold
        options["downsample_factor"] = _positive_integer(
            options.get("downsample_factor", 1),
            owner="downsample_factor",
        )
        options["dense_first"] = _positive_integer(
            options.get("dense_first", 1),
            owner="dense_first",
        )
        options["dense_last"] = _nonnegative_integer(
            options.get("dense_last", 0),
            owner="dense_last",
        )
        if "eps" in options:
            eps = float(options["eps"])
            if not math.isfinite(eps) or eps <= 0:
                raise ValueError("feature-cache eps must be finite and positive")
            options["eps"] = eps
        if algorithm == "dualblock":
            options["front_blocks"] = _positive_integer(
                options.get("front_blocks", 5),
                owner="front_blocks",
            )
            options["back_blocks"] = _positive_integer(
                options.get("back_blocks", 5),
                owner="back_blocks",
            )
    else:
        raise ValueError(f"unsupported Wan feature cache algorithm: {algorithm!r}")
    return WanFeatureCacheConfig(algorithm, options)


def _float_sequence(value: Any, *, owner: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} must be a sequence of numbers")
    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"{owner} must not be empty")
    return values


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if value < 1:
        raise ValueError(f"{owner} must be positive")
    return value


def _nonnegative_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if value < 0:
        raise ValueError(f"{owner} must be non-negative")
    return value


def build_wan_feature_cache(
    config: WanFeatureCacheConfig,
    *,
    branch: str,
    total_steps: int,
) -> object:
    """Create one request/CFG-branch-local cache from a validated config."""

    options = dict(config.options)
    if config.algorithm == "teacache":
        cache = TeaCacheResidualCache(
            float(options["threshold"]),
            _float_sequence(options["coefficients"], owner="TeaCache coefficients"),
            warmup_steps=int(options.get("warmup_steps", 5)),
            dense_last=int(options.get("dense_last", 1)),
            total_steps=total_steps,
        )
        cache.signal_kind = str(options.get("signal", "timestep-modulation"))
        return cache
    if config.algorithm == "custom":
        cache = CustomTaylorResidualCache(
            float(options["threshold"]),
            _float_sequence(options["coefficients"], owner="Custom cache coefficients"),
            warmup_steps=int(options.get("warmup_steps", 5)),
            dense_last=int(options.get("dense_last", 0)),
            total_steps=total_steps,
            eps=float(options.get("eps", 1e-6)),
        )
        cache.signal_kind = str(options.get("signal", "timestep-modulation"))
        cache.branch = str(branch)
        return cache
    if config.algorithm == "magcache":
        ratios_value = options["ratios"]
        if (
            isinstance(ratios_value, Sequence)
            and not isinstance(ratios_value, (str, bytes))
            and ratios_value
            and isinstance(ratios_value[0], Sequence)
        ):
            branch_index = 1 if str(branch).startswith("negative") else 0
            try:
                ratios_value = ratios_value[branch_index]
            except IndexError as error:
                raise ValueError("MagCache branch ratios require positive and negative rows") from error
        return MagCacheResidualCache(
            _float_sequence(ratios_value, owner="MagCache ratios"),
            threshold=float(options.get("threshold", 0.24)),
            max_skip_steps=int(options.get("max_skip_steps", 6)),
            retention_ratio=float(options.get("retention_ratio", 0.2)),
            total_steps=total_steps,
        )
    if config.algorithm == "adacache":
        codebook = options.get("codebook")
        parsed_codebook = None
        if codebook is not None:
            if not isinstance(codebook, Mapping):
                raise TypeError("AdaCache codebook must be a threshold-to-rate mapping")
            parsed_codebook = tuple((float(limit), int(rate)) for limit, rate in codebook.items())
        return AdaCacheResidualCache(
            codebook=parsed_codebook,
            dense_last=int(options.get("dense_last", 1)),
            total_steps=total_steps,
        )
    if config.algorithm == "taylorseer":
        return TaylorSeerResidualCache(
            interval=int(options.get("interval", 4)),
            dense_last=int(options.get("dense_last", 1)),
            total_steps=total_steps,
        )
    if config.algorithm == "blocktaylorseer":
        cache = BlockTaylorSeerCache(
            dense_pattern=options.get(
                "dense_pattern",
                (True, False, False, False),
            ),
            dense_last=int(options.get("dense_last", 0)),
            total_steps=total_steps,
        )
        cache.branch = str(branch)
        return cache
    if config.algorithm == "firstblock":
        cache = FirstBlockFeatureCache(
            float(options["residual_diff_threshold"]),
            downsample_factor=int(options.get("downsample_factor", 1)),
            dense_first=int(options.get("dense_first", 1)),
            dense_last=int(options.get("dense_last", 0)),
            eps=float(options.get("eps", 1e-6)),
        )
        cache.branch = str(branch)
        return cache
    if config.algorithm == "dualblock":
        cache = DualBlockFeatureCache(
            float(options["residual_diff_threshold"]),
            downsample_factor=int(options.get("downsample_factor", 1)),
            dense_first=int(options.get("dense_first", 1)),
            dense_last=int(options.get("dense_last", 0)),
            front_blocks=int(options.get("front_blocks", 5)),
            back_blocks=int(options.get("back_blocks", 5)),
            eps=float(options.get("eps", 1e-6)),
        )
        cache.branch = str(branch)
        return cache
    if config.algorithm == "dynamicblock":
        cache = DynamicBlockFeatureCache(
            float(options["residual_diff_threshold"]),
            downsample_factor=int(options.get("downsample_factor", 1)),
            dense_first=int(options.get("dense_first", 1)),
            dense_last=int(options.get("dense_last", 0)),
            eps=float(options.get("eps", 1e-6)),
        )
        cache.branch = str(branch)
        return cache
    raise AssertionError(f"unreachable Wan feature cache: {config.algorithm}")


__all__ = [
    "WanFeatureCacheConfig",
    "build_wan_feature_cache",
    "resolve_wan_feature_cache",
]
