"""PhyFPS-Bench-Gen metric formulas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

METRIC_ORDER = (
    "avg_error_fps",
    "pct_error",
    "inter_video_cv",
    "intra_video_cv",
    "phyfps_bench_gen_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "avg_error_fps": {
        "name": "Avg. Error (FPS)",
        "group": "alignment",
        "higher_is_better": False,
        "description": "Mean absolute gap between Meta FPS and average PhyFPS.",
    },
    "pct_error": {
        "name": "Pct. Error (%)",
        "group": "alignment",
        "higher_is_better": False,
        "description": "Mean relative Meta-vs-PhyFPS error percentage.",
    },
    "inter_video_cv": {
        "name": "Inter-video CV",
        "group": "consistency",
        "higher_is_better": False,
        "description": "Cross-video coefficient of variation over average PhyFPS.",
    },
    "intra_video_cv": {
        "name": "Intra-video CV",
        "group": "consistency",
        "higher_is_better": False,
        "description": "Mean within-video coefficient of variation over sliding-window clips.",
    },
    "phyfps_bench_gen_average": {
        "name": "PhyFPS-Bench-Gen Average",
        "group": "aggregate",
        "higher_is_better": False,
        "description": "Mean over available PhyFPS-Bench-Gen score families.",
        "primary": True,
    },
}


@dataclass(frozen=True)
class VideoPhyFPSRecord:
    video: str
    avg_phyfps: float
    segment_phyfps: tuple[float, ...]


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values]
    if len(clean) < 2:
        return 0.0 if clean else None
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return variance**0.5


def compute_phyfps_metrics(
    *,
    video_records: Sequence[VideoPhyFPSRecord],
    meta_fps_by_video: Mapping[str, float],
) -> dict[str, Any]:
    """Compute official PhyFPS-Bench-Gen metrics from per-video PhyFPS records."""
    avg_phyfps_values = [record.avg_phyfps for record in video_records]
    avg_errors: list[float] = []
    pct_errors: list[float] = []
    intra_cvs: list[float] = []

    for record in video_records:
        meta_fps = meta_fps_by_video.get(record.video)
        if meta_fps is not None and meta_fps > 0:
            gap = abs(record.avg_phyfps - float(meta_fps))
            avg_errors.append(gap)
            pct_errors.append(gap / float(meta_fps) * 100.0)
        if record.segment_phyfps:
            segment_mean = _mean(record.segment_phyfps)
            segment_std = _std(record.segment_phyfps)
            if segment_mean not in (None, 0.0) and segment_std is not None:
                intra_cvs.append(segment_std / segment_mean)

    inter_mean = _mean(avg_phyfps_values)
    inter_std = _std(avg_phyfps_values)
    inter_video_cv = (inter_std / inter_mean) if inter_mean not in (None, 0.0) and inter_std is not None else None

    direct_metrics = {
        "avg_error_fps": _mean(avg_errors),
        "pct_error": _mean(pct_errors),
        "inter_video_cv": inter_video_cv,
        "intra_video_cv": _mean(intra_cvs),
    }
    available = [value for value in direct_metrics.values() if value is not None]
    direct_metrics["phyfps_bench_gen_average"] = _mean(available) if available else None

    return {
        "metrics": direct_metrics,
        "components": {
            "video_count": len(video_records),
            "meta_fps_coverage_count": sum(1 for record in video_records if record.video in meta_fps_by_video),
            "avg_phyfps_values": avg_phyfps_values,
            "avg_errors": avg_errors,
            "pct_errors": pct_errors,
            "intra_video_cvs": intra_cvs,
        },
    }
