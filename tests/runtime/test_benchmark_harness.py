"""CPU-only regression tests for the F0 paired A/B benchmark harness.

These guard the statistics and artifact round-trip without a GPU, so
performance governance starts at P0 as the plan requires.
"""

from __future__ import annotations

import time

from benchmarks.harness import (
    PairedABResult,
    SampleStats,
    _paired_bootstrap_ci,
    _percentile,
    bench_paired,
    build_manifest,
    summarize_markdown,
    write_result,
)
from worldfoundry.runtime.performance import PerformanceManifest


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 4.0
    assert _percentile(values, 0.5) == 2.5


def test_sample_stats_basic() -> None:
    stats = SampleStats.from_samples([10.0, 12.0, 11.0, 13.0, 9.0])
    assert stats.count == 5
    assert stats.median_ms == 11.0
    assert stats.p10_ms <= stats.median_ms <= stats.p90_ms
    assert 0.0 <= stats.mad_over_median < 1.0
    assert len(stats.raw_ms) == 5


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    a = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    b = [5.0, 5.2, 4.8, 5.1, 4.9, 5.05]
    low1, high1 = _paired_bootstrap_ci(a, b)
    low2, high2 = _paired_bootstrap_ci(a, b)
    assert (low1, high1) == (low2, high2)  # seeded -> reproducible
    assert low1 <= high1
    # a is ~2x b, so the speedup CI should bracket ~2.0
    assert 1.7 < low1 <= high1 < 2.3


def test_bench_paired_measures_relative_cost() -> None:
    # B sleeps half as long as A -> speedup ~2x on a wall clock.
    def slow() -> None:
        time.sleep(0.0020)

    def fast() -> None:
        time.sleep(0.0010)

    result = bench_paired(
        "sleep", slow, fast, label_a="slow", label_b="fast", device=None, warmup=2, iters=1, rounds=6
    )
    assert isinstance(result, PairedABResult)
    assert result.stats_a.median_ms > result.stats_b.median_ms
    assert result.speedup_median > 1.3  # generous margin for scheduler jitter
    assert result.speedup_ci_low <= result.speedup_median <= result.speedup_ci_high


def test_manifest_round_trips(tmp_path) -> None:
    def a() -> None:
        time.sleep(0.001)

    def b() -> None:
        time.sleep(0.0006)

    results = [
        bench_paired("case", a, b, device=None, warmup=1, iters=1, rounds=4, workload={"tag": "x"})
    ]
    manifest = build_manifest(results, suite="unit")
    restored = PerformanceManifest.from_json(manifest.to_json())
    assert restored.to_dict() == manifest.to_dict()
    assert restored.extensions["suite"] == "unit"
    assert len(restored.extensions["paired_ab"]) == 1

    json_path, md_path = write_result(results, suite="unit", out_dir=tmp_path)
    assert json_path.exists() and md_path.exists()
    reloaded = PerformanceManifest.read_json(json_path)
    assert reloaded.extensions["suite"] == "unit"
    assert "speedup" in summarize_markdown(results, suite="unit")
