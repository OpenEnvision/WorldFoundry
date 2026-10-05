"""Portable performance manifests for WorldFoundry runtimes.

The manifest types in this module deliberately depend only on the Python
standard library.  Framework and source-control discovery is best-effort and
is performed only when :func:`capture_runtime_fingerprint` is called.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform as stdlib_platform
import random
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, cast

from worldfoundry.core.contracts import OptimizationSnapshot

PERFORMANCE_MANIFEST_SCHEMA_VERSION = "worldfoundry-performance-v1"

FINGERPRINT_COMPATIBILITY_FIELDS = (
    "platform",
    "vendor",
    "arch",
    "device",
    "memory_bytes",
    "driver_version",
    "runtime_version",
    "torch_version",
    "python_version",
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_value(value: Any, *, path: str = "value") -> JsonValue:
    """Return a detached, strictly JSON-compatible representation of *value*."""

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
    normalized = _json_value({} if value is None else value, path=path)
    return cast(dict[str, JsonValue], normalized)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _extensions(data: Mapping[str, Any], known: frozenset[str]) -> dict[str, JsonValue]:
    explicit = data.get("extensions")
    result = _json_mapping(explicit if isinstance(explicit, Mapping) else {}, path="extensions")
    for key, value in data.items():
        if key not in known:
            result[key] = _json_value(value, path=key)
    return result


def _with_extensions(payload: dict[str, JsonValue], extensions: Mapping[str, Any]) -> dict[str, JsonValue]:
    for key, value in extensions.items():
        if key in payload:
            raise ValueError(f"extension key conflicts with a manifest field: {key!r}")
        payload[key] = _json_value(value, path=f"extensions.{key}")
    return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    """Hardware and software identity associated with a measurement.

    All fields are optional so a manifest remains useful in minimal Python
    environments and on platforms for which no framework probe is available.
    Unknown fields read from a newer schema are retained in ``extensions``.
    """

    platform: str | None = None
    vendor: str | None = None
    arch: str | None = None
    device: str | None = None
    device_index: int | None = None
    memory_bytes: int | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    torch_version: str | None = None
    python_version: str | None = None
    worldfoundry_version: str | None = None
    worldfoundry_commit: str | None = None
    extensions: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", _json_mapping(self.extensions, path="extensions"))

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "platform": self.platform,
            "vendor": self.vendor,
            "arch": self.arch,
            "device": self.device,
            "device_index": self.device_index,
            "memory_bytes": self.memory_bytes,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "worldfoundry_version": self.worldfoundry_version,
            "worldfoundry_commit": self.worldfoundry_commit,
        }
        return _with_extensions(payload, self.extensions)

    def compatibility_mismatches(
        self,
        other: "RuntimeFingerprint",
        *,
        fields: Sequence[str] = FINGERPRINT_COMPATIBILITY_FIELDS,
    ) -> dict[str, tuple[JsonScalar, JsonScalar]]:
        """Return hardware/software fields that make paired results incomparable.

        Source revision is intentionally not part of the default set: a paired
        experiment may compare two revisions.  Callers that require identical
        source can include ``worldfoundry_commit`` explicitly.
        """

        mismatches: dict[str, tuple[JsonScalar, JsonScalar]] = {}
        for name in fields:
            if not hasattr(self, name) or not hasattr(other, name):
                raise ValueError(f"unknown runtime fingerprint field: {name}")
            baseline = cast(JsonScalar, getattr(self, name))
            candidate = cast(JsonScalar, getattr(other, name))
            if baseline != candidate:
                mismatches[name] = (baseline, candidate)
        return mismatches

    def is_compatible_with(
        self,
        other: "RuntimeFingerprint",
        *,
        fields: Sequence[str] = FINGERPRINT_COMPATIBILITY_FIELDS,
    ) -> bool:
        return not self.compatibility_mismatches(other, fields=fields)

    def assert_compatible_with(
        self,
        other: "RuntimeFingerprint",
        *,
        fields: Sequence[str] = FINGERPRINT_COMPATIBILITY_FIELDS,
    ) -> None:
        mismatches = self.compatibility_mismatches(other, fields=fields)
        if mismatches:
            details = ", ".join(
                f"{name}={baseline!r}/{candidate!r}"
                for name, (baseline, candidate) in mismatches.items()
            )
            raise ValueError(f"incompatible runtime fingerprints: {details}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeFingerprint":
        known = frozenset(
            {
                "platform",
                "vendor",
                "arch",
                "device",
                "device_index",
                "memory_bytes",
                "driver_version",
                "runtime_version",
                "torch_version",
                "python_version",
                "worldfoundry_version",
                "worldfoundry_commit",
                "extensions",
            }
        )
        return cls(
            platform=_optional_string(data.get("platform")),
            vendor=_optional_string(data.get("vendor")),
            arch=_optional_string(data.get("arch")),
            device=_optional_string(data.get("device")),
            device_index=_optional_int(data.get("device_index")),
            memory_bytes=_optional_int(data.get("memory_bytes")),
            driver_version=_optional_string(data.get("driver_version")),
            runtime_version=_optional_string(data.get("runtime_version")),
            torch_version=_optional_string(data.get("torch_version")),
            python_version=_optional_string(data.get("python_version")),
            worldfoundry_version=_optional_string(data.get("worldfoundry_version")),
            worldfoundry_commit=_optional_string(data.get("worldfoundry_commit")),
            extensions=_extensions(data, known),
        )


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Common performance values plus open-ended counters.

    Mapping fields are keyed by stage, unit, memory kind, or counter name.  For
    example, ``timings_ms`` may contain ``load_pipeline`` and ``denoise`` while
    ``throughput`` can contain ``frames_per_second``.  This avoids fixing a
    model-specific metric vocabulary in the core schema.
    """

    timings_ms: Mapping[str, JsonValue] = field(default_factory=dict)
    throughput: Mapping[str, JsonValue] = field(default_factory=dict)
    ttff_ms: float | None = None
    peak_memory_bytes: Mapping[str, JsonValue] = field(default_factory=dict)
    cache_counters: Mapping[str, JsonValue] = field(default_factory=dict)
    graph_counters: Mapping[str, JsonValue] = field(default_factory=dict)
    batch_counters: Mapping[str, JsonValue] = field(default_factory=dict)
    raw_samples_ms: Mapping[str, JsonValue] = field(default_factory=dict)
    extensions: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "timings_ms",
            "throughput",
            "peak_memory_bytes",
            "cache_counters",
            "graph_counters",
            "batch_counters",
            "raw_samples_ms",
            "extensions",
        ):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), path=name))
        if self.ttff_ms is not None:
            ttff_ms = _optional_float(self.ttff_ms)
            if ttff_ms is None:
                raise ValueError("ttff_ms must be a finite number")
            object.__setattr__(self, "ttff_ms", ttff_ms)

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "timings_ms": _json_mapping(self.timings_ms, path="timings_ms"),
            "throughput": _json_mapping(self.throughput, path="throughput"),
            "ttff_ms": self.ttff_ms,
            "peak_memory_bytes": _json_mapping(self.peak_memory_bytes, path="peak_memory_bytes"),
            "cache_counters": _json_mapping(self.cache_counters, path="cache_counters"),
            "graph_counters": _json_mapping(self.graph_counters, path="graph_counters"),
            "batch_counters": _json_mapping(self.batch_counters, path="batch_counters"),
            "raw_samples_ms": _json_mapping(self.raw_samples_ms, path="raw_samples_ms"),
        }
        return _with_extensions(payload, self.extensions)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerformanceMetrics":
        known = frozenset(
            {
                "timings_ms",
                "throughput",
                "ttff_ms",
                "peak_memory_bytes",
                "cache_counters",
                "graph_counters",
                "batch_counters",
                "raw_samples_ms",
                "extensions",
            }
        )

        def mapping(name: str) -> Mapping[str, Any]:
            value = data.get(name)
            return value if isinstance(value, Mapping) else {}

        return cls(
            timings_ms=mapping("timings_ms"),
            throughput=mapping("throughput"),
            ttff_ms=_optional_float(data.get("ttff_ms")),
            peak_memory_bytes=mapping("peak_memory_bytes"),
            cache_counters=mapping("cache_counters"),
            graph_counters=mapping("graph_counters"),
            batch_counters=mapping("batch_counters"),
            raw_samples_ms=mapping("raw_samples_ms"),
            extensions=_extensions(data, known),
        )


def _finite_samples(
    values: Sequence[float],
    *,
    name: str,
    require_positive: bool = False,
) -> tuple[float, ...]:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError(f"{name} must contain at least one sample")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError(f"{name} samples must be finite")
    if require_positive and any(value <= 0 for value in samples):
        raise ValueError(f"{name} samples must be finite and greater than zero")
    return samples


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class SampleDistribution:
    """Robust descriptive statistics for one timing sample series."""

    median: float
    p10: float
    p90: float
    mad: float
    mad_over_median: float | None

    @classmethod
    def from_samples(
        cls,
        values: Sequence[float],
        *,
        name: str = "samples",
        require_positive: bool = False,
    ) -> "SampleDistribution":
        samples = _finite_samples(values, name=name, require_positive=require_positive)
        median = statistics.median(samples)
        mad = statistics.median(abs(value - median) for value in samples)
        return cls(
            median=median,
            p10=_percentile(samples, 0.10),
            p90=_percentile(samples, 0.90),
            mad=mad,
            mad_over_median=mad / abs(median) if median else None,
        )

    def to_dict(self) -> dict[str, float | None]:
        return {
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "mad": self.mad,
            "mad_over_median": self.mad_over_median,
        }


@dataclass(frozen=True, slots=True)
class PairedPerformanceComparison:
    """A deterministic paired A/B timing comparison with raw evidence."""

    baseline_ms: tuple[float, ...]
    candidate_ms: tuple[float, ...]
    baseline: SampleDistribution
    candidate: SampleDistribution
    paired_speedup: SampleDistribution
    paired_delta_ms: SampleDistribution
    median_speedup_ci95: tuple[float, float]

    @property
    def sample_count(self) -> int:
        return len(self.baseline_ms)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "sample_count": self.sample_count,
            "baseline_ms": list(self.baseline_ms),
            "candidate_ms": list(self.candidate_ms),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "paired_speedup": self.paired_speedup.to_dict(),
            "paired_delta_ms": self.paired_delta_ms.to_dict(),
            "median_speedup_ci95": list(self.median_speedup_ci95),
        }


def compare_paired_samples(
    baseline_ms: Sequence[float],
    candidate_ms: Sequence[float],
    *,
    minimum_pairs: int = 5,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> PairedPerformanceComparison:
    """Compare interleaved A/B timings using paired ratios and bootstrap CI."""

    baseline = _finite_samples(baseline_ms, name="baseline_ms", require_positive=True)
    candidate = _finite_samples(candidate_ms, name="candidate_ms", require_positive=True)
    if len(baseline) != len(candidate):
        raise ValueError("baseline_ms and candidate_ms must contain the same number of paired samples")
    if len(baseline) < minimum_pairs:
        raise ValueError(f"paired comparison requires at least {minimum_pairs} valid pairs")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be greater than zero")

    speedups = tuple(before / after for before, after in zip(baseline, candidate, strict=True))
    deltas = tuple(before - after for before, after in zip(baseline, candidate, strict=True))
    generator = random.Random(bootstrap_seed)
    medians = []
    for _ in range(bootstrap_resamples):
        resample = [speedups[generator.randrange(len(speedups))] for _ in speedups]
        medians.append(statistics.median(resample))

    return PairedPerformanceComparison(
        baseline_ms=baseline,
        candidate_ms=candidate,
        baseline=SampleDistribution.from_samples(baseline, name="baseline_ms", require_positive=True),
        candidate=SampleDistribution.from_samples(candidate, name="candidate_ms", require_positive=True),
        paired_speedup=SampleDistribution.from_samples(speedups, name="paired_speedup", require_positive=True),
        paired_delta_ms=SampleDistribution.from_samples(deltas, name="paired_delta_ms"),
        median_speedup_ci95=(_percentile(medians, 0.025), _percentile(medians, 0.975)),
    )


@dataclass(frozen=True, slots=True)
class PerformanceManifest:
    """Self-contained record of one WorldFoundry performance measurement."""

    model: Mapping[str, JsonValue]
    workload: Mapping[str, JsonValue]
    fingerprint: RuntimeFingerprint = field(default_factory=RuntimeFingerprint)
    optimization: OptimizationSnapshot = field(default_factory=OptimizationSnapshot)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    timestamp: str = field(default_factory=_utc_now_iso)
    schema_version: str = PERFORMANCE_MANIFEST_SCHEMA_VERSION
    extensions: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _json_mapping(self.model, path="model"))
        object.__setattr__(self, "workload", _json_mapping(self.workload, path="workload"))
        object.__setattr__(self, "extensions", _json_mapping(self.extensions, path="extensions"))

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "model": _json_mapping(self.model, path="model"),
            "workload": _json_mapping(self.workload, path="workload"),
            "fingerprint": self.fingerprint.to_dict(),
            "optimization": self.optimization.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
        }
        return _with_extensions(payload, self.extensions)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerformanceManifest":
        known = frozenset(
            {
                "schema_version",
                "model",
                "workload",
                "fingerprint",
                "optimization",
                "metrics",
                "timestamp",
                "extensions",
            }
        )

        def mapping(name: str) -> Mapping[str, Any]:
            value = data.get(name)
            return value if isinstance(value, Mapping) else {}

        return cls(
            model=mapping("model"),
            workload=mapping("workload"),
            fingerprint=RuntimeFingerprint.from_dict(mapping("fingerprint")),
            optimization=OptimizationSnapshot.from_dict(mapping("optimization")),
            metrics=PerformanceMetrics.from_dict(mapping("metrics")),
            timestamp=str(data.get("timestamp") or _utc_now_iso()),
            schema_version=str(data.get("schema_version") or PERFORMANCE_MANIFEST_SCHEMA_VERSION),
            extensions=_extensions(data, known),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest to standards-compliant UTF-8 JSON text."""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "PerformanceManifest":
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise TypeError("performance manifest JSON must contain an object")
        return cls.from_dict(payload)

    @classmethod
    def read_json(cls, path: str | Path) -> "PerformanceManifest":
        return cls.from_json(Path(path).read_bytes())

    def write_json(self, path: str | Path) -> Path:
        """Atomically write this manifest through a sibling temporary file."""

        destination = Path(path)
        fd, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(self.to_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        return destination


def _safe_call(callable_: Any, *args: Any) -> Any | None:
    try:
        return callable_(*args)
    except Exception:
        return None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except Exception:
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    stdout = completed.stdout
    if isinstance(stdout, bytes):
        stdout = stdout.decode("ascii", errors="ignore")
    commit = str(stdout).strip()
    return commit or None


def capture_runtime_fingerprint(
    *,
    device_index: int = 0,
    repo_root: str | Path | None = None,
    probe_torch: bool = True,
    probe_git: bool = True,
) -> RuntimeFingerprint:
    """Best-effort capture of the current framework, device, and source tree.

    Torch is imported lazily.  A missing framework, unavailable accelerator,
    unsupported device API, or missing ``git`` executable simply leaves the
    corresponding optional fields empty.
    """

    detected_platform = "cpu"
    vendor: str | None = None
    arch: str | None = stdlib_platform.machine() or None
    device: str | None = stdlib_platform.processor() or None
    memory_bytes: int | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    torch_version: str | None = None

    torch: Any | None = None
    if probe_torch:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            torch = None

    if torch is not None:
        torch_version = _optional_string(getattr(torch, "__version__", None))
        version = getattr(torch, "version", None)
        hip_version = getattr(version, "hip", None)
        cuda_version = getattr(version, "cuda", None)
        xpu_version = getattr(version, "xpu", None)
        xpu = getattr(torch, "xpu", None)
        mps = getattr(getattr(torch, "backends", None), "mps", None)

        if hip_version:
            detected_platform, vendor, runtime_version = "rocm", "amd", str(hip_version)
        elif cuda_version:
            detected_platform, vendor, runtime_version = "cuda", "nvidia", str(cuda_version)
        elif xpu is not None and _safe_call(getattr(xpu, "is_available", lambda: False)):
            detected_platform, vendor = "xpu", "intel"
            runtime_version = _optional_string(xpu_version)
        elif mps is not None and _safe_call(getattr(mps, "is_available", lambda: False)):
            detected_platform, vendor, arch = "mps", "apple", "apple_silicon"

        if detected_platform in {"cuda", "rocm"}:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and _safe_call(getattr(cuda, "is_available", lambda: False)):
                properties = _safe_call(getattr(cuda, "get_device_properties", lambda _index: None), device_index)
                name = _safe_call(getattr(cuda, "get_device_name", lambda _index: None), device_index)
                device = _optional_string(name or getattr(properties, "name", None)) or device
                memory_bytes = _optional_int(getattr(properties, "total_memory", None))
                if detected_platform == "rocm":
                    arch = _optional_string(getattr(properties, "gcnArchName", None)) or arch
                else:
                    capability = _safe_call(getattr(cuda, "get_device_capability", lambda _index: None), device_index)
                    if isinstance(capability, Sequence) and len(capability) >= 2:
                        major = _optional_int(capability[0])
                        minor = _optional_int(capability[1])
                        if major is not None and minor is not None:
                            arch = f"sm_{major}{minor}"
                driver = _safe_call(getattr(cuda, "get_driver_version", lambda: None))
                driver_version = _optional_string(driver)
        elif detected_platform == "xpu" and xpu is not None:
            properties = _safe_call(getattr(xpu, "get_device_properties", lambda _index: None), device_index)
            name = _safe_call(getattr(xpu, "get_device_name", lambda _index: None), device_index)
            device = _optional_string(name or getattr(properties, "name", None)) or device
            memory_bytes = _optional_int(getattr(properties, "total_memory", None))

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    commit = _git_commit(root) if probe_git else None
    extensions: dict[str, JsonValue] = {
        "system": stdlib_platform.system() or None,
        "system_release": stdlib_platform.release() or None,
    }
    return RuntimeFingerprint(
        platform=detected_platform,
        vendor=vendor,
        arch=arch,
        device=device,
        device_index=device_index if detected_platform != "cpu" else None,
        memory_bytes=memory_bytes,
        driver_version=driver_version,
        runtime_version=runtime_version,
        torch_version=torch_version,
        python_version=stdlib_platform.python_version(),
        worldfoundry_version=_package_version("worldfoundry"),
        worldfoundry_commit=commit,
        extensions=extensions,
    )


__all__ = [
    "FINGERPRINT_COMPATIBILITY_FIELDS",
    "PERFORMANCE_MANIFEST_SCHEMA_VERSION",
    "OptimizationSnapshot",
    "PairedPerformanceComparison",
    "PerformanceManifest",
    "PerformanceMetrics",
    "RuntimeFingerprint",
    "SampleDistribution",
    "capture_runtime_fingerprint",
    "compare_paired_samples",
]
