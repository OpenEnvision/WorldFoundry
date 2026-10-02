from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

import worldfoundry.runtime.performance as performance
from worldfoundry.runtime.performance import (
    OptimizationSnapshot,
    PerformanceManifest,
    PerformanceMetrics,
    RuntimeFingerprint,
    compare_paired_samples,
)


def _manifest() -> PerformanceManifest:
    return PerformanceManifest(
        model={"id": "example/dit", "revision": "abc123"},
        workload={"height": 720, "width": 1280, "frames": 49, "steps": 20},
        fingerprint=RuntimeFingerprint(
            platform="cuda",
            vendor="nvidia",
            arch="sm_80",
            device="A100-SXM4-80GB",
            memory_bytes=80 * 1024**3,
            torch_version="2.7.0",
        ),
        optimization=OptimizationSnapshot(
            requested={"attention": "fa3"},
            effective={"attention": "torch_sdpa"},
            fallbacks=({"component": "attention", "reason": "unsupported_arch"},),
            quality_tier="exact",
        ),
        metrics=PerformanceMetrics(
            timings_ms={"load_pipeline": 10.5, "denoise": 42.0},
            throughput={"frames_per_second": 23.3},
            ttff_ms=12.75,
            peak_memory_bytes={"allocated": 1024, "reserved": 2048},
            cache_counters={"hits": 4, "misses": 1},
            raw_samples_ms={"denoise": [42.0, 41.5, 42.5]},
        ),
        timestamp="2026-07-12T00:00:00Z",
    )


def test_performance_manifest_round_trips_and_preserves_future_fields() -> None:
    payload = _manifest().to_dict()
    payload["scheduler"] = {"queue": "interactive"}
    restored = PerformanceManifest.from_json(PerformanceManifest.from_dict(payload).to_json())

    assert restored.to_dict()["scheduler"] == {"queue": "interactive"}
    assert restored.metrics.timings_ms["denoise"] == 42.0


def test_performance_manifest_writes_atomically_and_rejects_nan(tmp_path: Path) -> None:
    destination = tmp_path / "performance.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = os.replace
    observations: list[Path] = []

    def observed_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        observations.append(source_path)
        assert source_path.parent == Path(target).parent
        real_replace(source, target)

    with mock.patch("worldfoundry.runtime.performance.os.replace", side_effect=observed_replace):
        _manifest().write_json(destination)

    assert observations
    assert PerformanceManifest.read_json(destination) == _manifest()
    with pytest.raises(ValueError, match="NaN"):
        PerformanceMetrics(timings_ms={"denoise": float("nan")})


def test_runtime_fingerprint_degrades_without_torch_or_git(tmp_path: Path) -> None:
    def missing_import(name: str):
        if name == "torch":
            raise ImportError("torch is not installed")
        raise AssertionError(f"unexpected import: {name}")

    with (
        mock.patch.object(performance.importlib, "import_module", side_effect=missing_import),
        mock.patch.object(performance.subprocess, "run", side_effect=FileNotFoundError("git missing")),
    ):
        fingerprint = performance.capture_runtime_fingerprint(repo_root=tmp_path)

    assert fingerprint.platform == "cpu"
    assert fingerprint.torch_version is None
    assert fingerprint.worldfoundry_commit is None


def test_runtime_fingerprint_comparison_rejects_incompatible_hardware() -> None:
    baseline = RuntimeFingerprint(
        platform="cuda",
        vendor="nvidia",
        arch="sm_90",
        device="NVIDIA H100 80GB HBM3",
        torch_version="2.5.1+cu121",
    )
    same_runtime_new_source = RuntimeFingerprint(
        platform="cuda",
        vendor="nvidia",
        arch="sm_90",
        device="NVIDIA H100 80GB HBM3",
        torch_version="2.5.1+cu121",
        worldfoundry_commit="candidate",
    )
    incompatible = RuntimeFingerprint(
        platform="cuda",
        vendor="nvidia",
        arch="sm_80",
        device="NVIDIA A100-SXM4-80GB",
        torch_version="2.5.1+cu121",
    )

    assert baseline.is_compatible_with(same_runtime_new_source)
    assert not baseline.is_compatible_with(incompatible)
    with pytest.raises(ValueError, match="arch='sm_90'/'sm_80'"):
        baseline.assert_compatible_with(incompatible)


def test_paired_comparison_preserves_raw_samples_and_is_deterministic() -> None:
    baseline = [100.0, 102.0, 98.0, 101.0, 99.0]
    candidate = [80.0, 81.0, 79.0, 80.5, 79.5]

    first = compare_paired_samples(baseline, candidate, bootstrap_resamples=1_000, bootstrap_seed=7)
    second = compare_paired_samples(baseline, candidate, bootstrap_resamples=1_000, bootstrap_seed=7)

    assert first == second
    assert first.sample_count == 5
    assert first.baseline_ms == tuple(baseline)
    assert first.candidate_ms == tuple(candidate)
    assert first.paired_speedup.median == pytest.approx(1.25, rel=0.02)
    assert first.median_speedup_ci95[0] > 1.0
    assert first.to_dict()["sample_count"] == 5


def test_paired_comparison_rejects_unpaired_or_insufficient_samples() -> None:
    with pytest.raises(ValueError, match="same number"):
        compare_paired_samples([1, 2, 3, 4, 5], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="at least 5"):
        compare_paired_samples([1, 2, 3, 4], [1, 2, 3, 4])
