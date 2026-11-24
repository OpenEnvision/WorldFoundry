"""Visual Chronometer standalone PhyFPS metric formulas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from worldfoundry.evals.tasks.execution.runners.phyfps_bench_gen.phyfps_metrics import VideoPhyFPSRecord

METRIC_ORDER = (
    "mean_phyfps",
    "inter_video_cv",
    "intra_video_cv",
    "visual_chronometer_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "mean_phyfps": {
        "name": "Mean PhyFPS",
        "group": "prediction",
        "higher_is_better": False,
        "description": "Mean average PhyFPS across evaluated videos.",
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
    "visual_chronometer_average": {
        "name": "Visual Chronometer Average",
        "group": "aggregate",
        "higher_is_better": False,
        "description": "Mean over available Visual Chronometer consistency score families.",
        "primary": True,
    },
}


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


def compute_visual_chronometer_metrics(
    *,
    video_records: Sequence[VideoPhyFPSRecord],
) -> dict[str, Any]:
    """Compute standalone Visual Chronometer metrics from per-video PhyFPS records."""
    avg_phyfps_values = [record.avg_phyfps for record in video_records]
    intra_cvs: list[float] = []

    for record in video_records:
        if record.segment_phyfps:
            segment_mean = _mean(record.segment_phyfps)
            segment_std = _std(record.segment_phyfps)
            if segment_mean not in (None, 0.0) and segment_std is not None:
                intra_cvs.append(segment_std / segment_mean)

    inter_mean = _mean(avg_phyfps_values)
    inter_std = _std(avg_phyfps_values)
    inter_video_cv = (inter_std / inter_mean) if inter_mean not in (None, 0.0) and inter_std is not None else None

    direct_metrics = {
        "mean_phyfps": inter_mean,
        "inter_video_cv": inter_video_cv,
        "intra_video_cv": _mean(intra_cvs),
    }
    consistency_metrics = [value for key, value in direct_metrics.items() if key != "mean_phyfps" and value is not None]
    direct_metrics["visual_chronometer_average"] = _mean(consistency_metrics) if consistency_metrics else None

    return {
        "metrics": direct_metrics,
        "components": {
            "video_count": len(video_records),
            "avg_phyfps_values": avg_phyfps_values,
            "intra_video_cvs": intra_cvs,
        },
    }


def metric_rows_from_computed(
    computed: Mapping[str, Any],
    *,
    source_path: Path,
    official_runtime_executed: bool,
) -> list[dict[str, Any]]:
    direct_metrics = (
        cast(Mapping[str, Any], computed.get("metrics")) if isinstance(computed.get("metrics"), Mapping) else {}
    )
    components = (
        cast(Mapping[str, Any], computed.get("components")) if isinstance(computed.get("components"), Mapping) else {}
    )
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        score = direct_metrics.get(metric_id)
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": score is not None,
                "raw_score": score,
                "normalized_score": score,
                "score": score,
                "higher_is_better": spec["higher_is_better"],
                "group": spec["group"],
                "source": "visual_chronometer_official_runtime"
                if official_runtime_executed
                else "visual_chronometer_results_csv",
                "source_path": str(source_path),
                "components": components,
                "reason": None if score is not None else "score_not_available_in_visual_chronometer_results",
            }
        )
    return rows
