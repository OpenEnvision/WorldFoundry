from __future__ import annotations

import pytest

from worldfoundry.core.observability.realtime_timing import (
    RealtimeChunkTiming,
    RealtimeTimingWindow,
    TimingDistribution,
)


def _timing(
    chunk_index: int,
    *,
    started_at_s: float,
    completed_at_s: float,
    generation_ms: float,
    output_frames: int = 2,
    dropped_frames: int = 0,
    warmup: bool = False,
) -> RealtimeChunkTiming:
    return RealtimeChunkTiming(
        session_id="session-a",
        chunk_index=chunk_index,
        transport="webrtc",
        started_at_s=started_at_s,
        completed_at_s=completed_at_s,
        output_frames=output_frames,
        queue_depth=1,
        dropped_frames=dropped_frames,
        warmup=warmup,
        stage_ms={"generation_ms": generation_ms, "decode_ms": generation_ms / 5.0},
        gauges={"cache_frames": float(chunk_index * 2)},
    )


def test_timing_distribution_uses_interpolated_percentiles() -> None:
    summary = TimingDistribution.from_values([30.0, 10.0, 20.0])

    assert summary.count == 3
    assert summary.mean_ms == pytest.approx(20.0)
    assert summary.p50_ms == pytest.approx(20.0)
    assert summary.p90_ms == pytest.approx(28.0)


def test_realtime_timing_window_excludes_warmup_and_tracks_drop_delta() -> None:
    window = RealtimeTimingWindow()
    window.record(
        _timing(
            1,
            started_at_s=0.0,
            completed_at_s=0.1,
            generation_ms=100.0,
            dropped_frames=2,
            warmup=True,
        )
    )
    window.record(
        _timing(
            2,
            started_at_s=1.0,
            completed_at_s=1.1,
            generation_ms=10.0,
            dropped_frames=2,
        )
    )
    window.record(
        _timing(
            3,
            started_at_s=2.0,
            completed_at_s=2.2,
            generation_ms=30.0,
            dropped_frames=3,
        )
    )

    summary = window.summary()

    assert summary is not None
    assert window.observed_chunks == 3
    assert window.measured_chunks == 2
    assert summary.chunk_count == 2
    assert summary.frame_count == 4
    assert summary.interval_s == pytest.approx(1.2)
    assert summary.throughput_fps == pytest.approx(4.0 / 1.2)
    assert summary.dropped_frames == 1
    assert summary.stages["generation_ms"].p50_ms == pytest.approx(20.0)
    assert summary.stages["generation_ms"].p90_ms == pytest.approx(28.0)
    assert summary.gauges == {"cache_frames": 6.0}


def test_realtime_timing_window_reset_retains_cumulative_drop_baseline() -> None:
    window = RealtimeTimingWindow()
    window.record(
        _timing(
            1,
            started_at_s=1.0,
            completed_at_s=1.1,
            generation_ms=10.0,
            dropped_frames=4,
        )
    )
    window.reset_interval()

    assert window.summary() is None
    assert window.observed_chunks == 0
    window.record(
        _timing(
            2,
            started_at_s=2.0,
            completed_at_s=2.1,
            generation_ms=20.0,
            dropped_frames=5,
        )
    )

    summary = window.summary()
    assert summary is not None
    assert summary.dropped_frames == 1


def test_chunk_timing_payload_uses_duration_instead_of_monotonic_timestamps() -> None:
    payload = _timing(
        4,
        started_at_s=12.0,
        completed_at_s=12.25,
        generation_ms=200.0,
    ).to_payload()

    assert payload["server_chunk_ms"] == pytest.approx(250.0)
    assert "started_at_s" not in payload
    assert "completed_at_s" not in payload
