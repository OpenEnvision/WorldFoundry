"""Paired A/B microbenchmark harness with reproducible, fingerprinted output.

Design (mirrors ``plan/inference_operator_optimization_plan.md`` §2.3, §5.2):

- **Paired & interleaved.** A and B are measured back to back within each round,
  and the measurement order alternates every round, so GPU clock ramp, cache
  warmth, and a single noisy launch do not bias one side.
- **Raw samples retained.** Every per-round timing is kept, not just an
  aggregate, so a later run can recompute statistics or a stricter test.
- **Robust statistics.** median / p10 / p90 / MAD-over-median for each side, plus
  a paired bootstrap 95% CI on the B-over-A speedup ratio. The bootstrap is
  seeded, so the same samples always yield the same interval.
- **Fingerprinted.** Reuses ``capture_runtime_fingerprint`` and emits a
  ``PerformanceManifest``; the paired result lives under the manifest's
  ``extensions`` so the core schema is not mutated.

CUDA-event timing is used when a CUDA tensor/device is involved; otherwise a
monotonic wall clock with an explicit sync hook.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry.runtime.performance import (
    OptimizationSnapshot,
    PerformanceManifest,
    PerformanceMetrics,
    capture_runtime_fingerprint,
)

_BOOTSTRAP_SEED = 20260804
_BOOTSTRAP_ROUNDS = 2000


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile ``q`` in [0, 1] over pre-sorted values."""

    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return float(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac)


@dataclass(frozen=True, slots=True)
class SampleStats:
    """Robust summary of one side's raw per-round timings (milliseconds)."""

    count: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    mad_over_median: float
    raw_ms: tuple[float, ...]

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "SampleStats":
        if not samples:
            raise ValueError("need at least one sample")
        ordered = sorted(float(s) for s in samples)
        median = statistics.median(ordered)
        abs_dev = [abs(s - median) for s in ordered]
        mad = statistics.median(abs_dev)
        return cls(
            count=len(ordered),
            median_ms=median,
            p10_ms=_percentile(ordered, 0.10),
            p90_ms=_percentile(ordered, 0.90),
            mad_over_median=(mad / median) if median > 0 else 0.0,
            raw_ms=tuple(float(s) for s in samples),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "median_ms": self.median_ms,
            "p10_ms": self.p10_ms,
            "p90_ms": self.p90_ms,
            "mad_over_median": self.mad_over_median,
            "raw_ms": list(self.raw_ms),
        }


@dataclass(frozen=True, slots=True)
class PairedABResult:
    """One paired A/B comparison: baseline (A) vs candidate (B)."""

    name: str
    label_a: str
    label_b: str
    stats_a: SampleStats
    stats_b: SampleStats
    speedup_median: float
    speedup_ci_low: float
    speedup_ci_high: float
    workload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "workload": self.workload,
            "baseline": self.stats_a.to_dict(),
            "candidate": self.stats_b.to_dict(),
            "speedup_median": self.speedup_median,
            "speedup_ci95": [self.speedup_ci_low, self.speedup_ci_high],
        }


def _paired_bootstrap_ci(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    seed: int = _BOOTSTRAP_SEED,
    rounds: int = _BOOTSTRAP_ROUNDS,
) -> tuple[float, float]:
    """Seeded paired bootstrap 95% CI for the A-over-B speedup ratio.

    A and B are paired by round index (they were measured together), so pairs
    are resampled jointly to preserve their correlation.
    """

    n = min(len(samples_a), len(samples_b))
    if n < 2:
        med_a = statistics.median(samples_a)
        med_b = statistics.median(samples_b)
        ratio = (med_a / med_b) if med_b > 0 else 0.0
        return ratio, ratio
    rng = random.Random(seed)
    ratios: list[float] = []
    indices = range(n)
    for _ in range(rounds):
        picks = [rng.randrange(n) for _ in indices]
        med_a = statistics.median([samples_a[i] for i in picks])
        med_b = statistics.median([samples_b[i] for i in picks])
        if med_b > 0:
            ratios.append(med_a / med_b)
    if not ratios:
        return 0.0, 0.0
    ratios.sort()
    return _percentile(ratios, 0.025), _percentile(ratios, 0.975)


def _make_timer(device: Any | None) -> Callable[[Callable[[], Any], int], float]:
    """Return a ``(fn, iters) -> mean_ms`` timer for the given device.

    Uses CUDA events on a CUDA device (accurate GPU timing), otherwise a
    monotonic wall clock. The returned function times ``iters`` back-to-back
    calls and returns the mean milliseconds per call.
    """

    is_cuda = device is not None and getattr(device, "type", None) == "cuda"
    if is_cuda:
        import torch

        def cuda_timer(fn: Callable[[], Any], iters: int) -> float:
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / iters

        return cuda_timer

    def wall_timer(fn: Callable[[], Any], iters: int) -> float:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) * 1000.0 / iters

    return wall_timer


def bench_paired(
    name: str,
    fn_a: Callable[[], Any],
    fn_b: Callable[[], Any],
    *,
    label_a: str = "baseline",
    label_b: str = "candidate",
    device: Any | None = None,
    warmup: int = 15,
    iters: int = 50,
    rounds: int = 8,
    workload: dict[str, Any] | None = None,
) -> PairedABResult:
    """Interleaved paired A/B measurement of two zero-arg callables.

    Each round times ``iters`` back-to-back calls of A and of B; the order
    alternates every round. Returns per-side robust statistics and a paired
    bootstrap CI on the A-over-B speedup (``label_a`` time / ``label_b`` time,
    so > 1 means B is faster).
    """

    timer = _make_timer(device)
    for _ in range(warmup):
        fn_a()
        fn_b()
    samples_a: list[float] = []
    samples_b: list[float] = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            samples_a.append(timer(fn_a, iters))
            samples_b.append(timer(fn_b, iters))
        else:
            samples_b.append(timer(fn_b, iters))
            samples_a.append(timer(fn_a, iters))
    stats_a = SampleStats.from_samples(samples_a)
    stats_b = SampleStats.from_samples(samples_b)
    speedup = stats_a.median_ms / stats_b.median_ms if stats_b.median_ms > 0 else 0.0
    ci_low, ci_high = _paired_bootstrap_ci(samples_a, samples_b)
    return PairedABResult(
        name=name,
        label_a=label_a,
        label_b=label_b,
        stats_a=stats_a,
        stats_b=stats_b,
        speedup_median=speedup,
        speedup_ci_low=ci_low,
        speedup_ci_high=ci_high,
        workload=dict(workload or {}),
    )


def build_manifest(
    results: Sequence[PairedABResult],
    *,
    suite: str,
    model: dict[str, Any] | None = None,
    device_index: int = 0,
    extra: dict[str, Any] | None = None,
    optimization: OptimizationSnapshot | None = None,
) -> PerformanceManifest:
    """Wrap paired A/B results in a fingerprinted ``PerformanceManifest``.

    The paired comparisons live under ``extensions.paired_ab`` so the core
    schema stays untouched; ``metrics.timings_ms`` carries the median timings
    for quick inspection by existing manifest tooling. ``optimization`` records
    the requested/effective/fallback backends actually exercised, so a result
    can never claim a speedup without disclosing which optimizations produced
    it (or silently fell back).
    """

    fingerprint = capture_runtime_fingerprint(device_index=device_index)
    timings: dict[str, Any] = {}
    for result in results:
        timings[f"{result.name}::{result.label_a}"] = result.stats_a.median_ms
        timings[f"{result.name}::{result.label_b}"] = result.stats_b.median_ms
    extensions: dict[str, Any] = {
        "suite": suite,
        "paired_ab": [result.to_dict() for result in results],
    }
    if extra:
        extensions.update(extra)
    return PerformanceManifest(
        model=dict(model or {"name": suite}),
        workload={"suite": suite, "comparisons": len(results)},
        fingerprint=fingerprint,
        optimization=optimization or OptimizationSnapshot(),
        metrics=PerformanceMetrics(timings_ms=timings),
        extensions=extensions,
    )


def summarize_markdown(results: Sequence[PairedABResult], *, suite: str) -> str:
    """Human-readable Markdown table of the paired A/B results."""

    lines = [
        f"# Benchmark: {suite}",
        "",
        "| workload | baseline (ms) | candidate (ms) | speedup | 95% CI | MAD/med (cand) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for r in results:
        ci = f"[{r.speedup_ci_low:.2f}, {r.speedup_ci_high:.2f}]"
        lines.append(
            f"| {r.name} | {r.stats_a.median_ms:.1f} | {r.stats_b.median_ms:.1f} "
            f"| {r.speedup_median:.2f}x | {ci} | {r.stats_b.mad_over_median * 100:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def write_result(
    results: Sequence[PairedABResult],
    *,
    suite: str,
    out_dir: str | Path,
    model: dict[str, Any] | None = None,
    device_index: int = 0,
    extra: dict[str, Any] | None = None,
    optimization: OptimizationSnapshot | None = None,
) -> tuple[Path, Path]:
    """Write ``<suite>.json`` (manifest) and ``<suite>.md`` (summary) atomically.

    Returns the two written paths. The JSON round-trips through
    ``PerformanceManifest.read_json``.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        results, suite=suite, model=model, device_index=device_index, extra=extra, optimization=optimization
    )
    json_path = manifest.write_json(directory / f"{suite}.json")
    md_path = directory / f"{suite}.md"
    md_text = summarize_markdown(results, suite=suite)
    md_tmp = md_path.with_suffix(".md.tmp")
    md_tmp.write_text(md_text, encoding="utf-8")
    md_tmp.replace(md_path)
    return json_path, md_path
