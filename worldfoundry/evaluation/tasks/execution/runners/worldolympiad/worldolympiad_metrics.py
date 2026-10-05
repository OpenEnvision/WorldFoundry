"""Normalize WorldOlympiad triathlon judge outputs into WorldFoundry metric rows.

The official evaluator writes one ``<prefix>_judge_<case_id>.json`` file per
case and pipeline. This module accepts any of the artifacts that layout
produces:

* a single judge JSON file,
* a directory tree such as ``outputs_batch/`` holding many judge JSON files,
* a ``batch_scheduler`` ``summary.jsonl`` whose rows point at judge JSON files,
* an aggregate JSON produced by ``batch_test/summarize_scores.py``.

Per-case score fields are averaged across cases, matching the upstream
summarizer. Physical question buckets are accumulated corpus-wide because the
upstream summarizer aggregates those as counts rather than per-case means.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric, scalar_number

TRACKS = ("physical", "geometry", "interaction", "aggregate")

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "combined_score": {
        "name": "Combined Triathlon Score",
        "track": "aggregate",
        "scale": "0-1",
        "description": "Mean of the physical, interaction, and geometry track scores on the unit interval.",
    },
    "physical_score": {
        "name": "Physical Faithfulness",
        "track": "physical",
        "scale": "0-1",
        "description": "Confidence-weighted compliance over physics questions the MLLM judge marked related.",
    },
    "physical_mechanics": {
        "name": "Physical Mechanics",
        "track": "physical",
        "scale": "0-1",
        "description": "Physical score restricted to mechanics questions.",
    },
    "physical_thermotics": {
        "name": "Physical Thermodynamics",
        "track": "physical",
        "scale": "0-1",
        "description": "Physical score restricted to thermodynamics questions.",
    },
    "physical_material": {
        "name": "Physical Material Properties",
        "track": "physical",
        "scale": "0-1",
        "description": "Physical score restricted to material-property questions.",
    },
    "physical_compliance_rate": {
        "name": "Physical Compliance Rate",
        "track": "physical",
        "scale": "0-1",
        "aggregator": "ratio",
        "description": "Corpus-wide compliant questions divided by related questions.",
    },
    "three_d_score": {
        "name": "Geometry Score",
        "track": "geometry",
        "scale": "0-1",
        "description": "DA3 geometry score normalized from the native 0-3 range.",
    },
    "three_d_raw": {
        "name": "Geometry Score (raw)",
        "track": "geometry",
        "scale": "0-3",
        "description": "Sum of the Gaussian-splatting, meta-view, and camera-motion components.",
    },
    "gs_score": {
        "name": "Gaussian Splatting Reconstruction",
        "track": "geometry",
        "scale": "0-1",
        "description": "DA3 Gaussian-splatting reconstruction quality component.",
    },
    "meta_view_score": {
        "name": "Meta-View Consistency",
        "track": "geometry",
        "scale": "0-1",
        "description": "Diagnostic meta-view consistency component.",
    },
    "camera_motion_score": {
        "name": "Camera Trajectory Alignment",
        "track": "geometry",
        "scale": "0-1",
        "description": "Alignment between the recovered and reference camera trajectories.",
    },
    "interaction_score": {
        "name": "Interaction Score",
        "track": "interaction",
        "scale": "0-1",
        "description": "VLM interaction score normalized from the native 0-5 range.",
    },
    "interaction_raw": {
        "name": "Interaction Score (raw)",
        "track": "interaction",
        "scale": "0-5",
        "description": "Mean of chunk, transition, and global interaction judgements.",
    },
    "chunk_instruction_following": {
        "name": "Chunk Instruction Following",
        "track": "interaction",
        "scale": "0-5",
        "description": "Per-chunk prompt adherence judged against the chunk timestamps.",
    },
    "transition_smoothness": {
        "name": "Transition Smoothness",
        "track": "interaction",
        "scale": "0-5",
        "description": "Smoothness across adjacent generated chunk boundaries.",
    },
    "global_consistency": {
        "name": "Long-Range Consistency",
        "track": "interaction",
        "scale": "0-5",
        "description": "Whole-video consistency and global text alignment.",
    },
    "clip_semantic_adherence": {
        "name": "CLIP Semantic Adherence",
        "track": "interaction",
        "scale": "0-1",
        "description": "CLIP similarity between sampled chunk frames and their captions.",
    },
}
METRIC_ORDER = tuple(METRIC_SPECS)
PRIMARY_METRIC = "combined_score"

TRACK_METRICS: dict[str, tuple[str, ...]] = {
    track: tuple(metric for metric, spec in METRIC_SPECS.items() if spec["track"] == track) for track in TRACKS
}

# Per-case judge JSON path for each metric that is read directly from a case file.
CASE_SCORE_PATHS: dict[str, tuple[str, ...]] = {
    "combined_score": ("combined_score",),
    "physical_score": ("physical", "score"),
    "physical_mechanics": ("physical", "dimension_scores", "mechanics"),
    "physical_thermotics": ("physical", "dimension_scores", "thermotics"),
    "physical_material": ("physical", "dimension_scores", "material"),
    "three_d_score": ("three_d", "final_score_normalized"),
    "three_d_raw": ("three_d", "final_score_raw"),
    "gs_score": ("three_d", "gs_score"),
    "meta_view_score": ("three_d", "meta_score"),
    "camera_motion_score": ("three_d", "camera_motion_score"),
    "interaction_score": ("interaction", "score"),
    "interaction_raw": ("interaction", "overall_raw"),
    "chunk_instruction_following": ("interaction", "summary", "chunk_mean"),
    "transition_smoothness": ("interaction", "summary", "transition_mean"),
    "global_consistency": ("interaction", "summary", "global_score"),
    "clip_semantic_adherence": ("clip_interaction", "summary", "semantic_adherence"),
}

# Metric path inside a ``summarize_scores.py`` aggregate document.
SUMMARY_SCORE_PATHS: dict[str, tuple[str, ...]] = {
    "combined_score": ("scores", "combined", "mean"),
    "three_d_score": ("scores", "three_d", "mean"),
    "interaction_score": ("scores", "interaction", "mean"),
    "clip_semantic_adherence": ("scores", "clip_interaction", "mean"),
}

JUDGE_GLOB = "*judge*.json"


class WorldOlympiadResultError(ValueError):
    """Raised when a results path holds no readable WorldOlympiad judge output."""


def _nested(payload: Any, path: Sequence[str]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | None:
    number = scalar_number(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_judge_payload(payload: Any) -> bool:
    """Return whether a decoded JSON document looks like a per-case judge file."""

    if not isinstance(payload, Mapping):
        return False
    if "combined_score" in payload:
        return True
    return any(isinstance(payload.get(key), Mapping) for key in ("physical", "three_d", "interaction"))


def is_summary_payload(payload: Any) -> bool:
    """Return whether a decoded JSON document is a ``summarize_scores`` aggregate."""

    if not isinstance(payload, Mapping):
        return False
    scores = payload.get("scores")
    return isinstance(scores, Mapping) and any(isinstance(scores.get(key), Mapping) for key in ("combined", "three_d"))


def _judge_paths_from_summary_jsonl(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        output = row.get("output") if isinstance(row, Mapping) else None
        if not output:
            continue
        candidate = Path(str(output))
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        if candidate.is_file():
            paths.append(candidate)
    return paths


def discover_judge_files(results_path: Path) -> list[Path]:
    """Collect per-case judge JSON files reachable from ``results_path``."""

    results_path = results_path.expanduser()
    if results_path.is_dir():
        return sorted(
            candidate
            for candidate in results_path.rglob(JUDGE_GLOB)
            if candidate.is_file() and is_judge_payload(_safe_read(candidate))
        )
    if not results_path.is_file():
        raise FileNotFoundError(f"WorldOlympiad results path does not exist: {results_path}")
    if results_path.suffix.lower() == ".jsonl":
        return _judge_paths_from_summary_jsonl(results_path)
    payload = _safe_read(results_path)
    return [results_path] if is_judge_payload(payload) else []


def _safe_read(path: Path) -> Any:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def physical_question_counts(physical: Mapping[str, Any]) -> dict[str, int]:
    """Bucket per-question judge verdicts the way the upstream summarizer does."""

    counts = {"total": 0, "related": 0, "compliant": 0, "incompliant": 0, "irrelevant": 0, "unknown": 0}
    results = physical.get("results")
    if not isinstance(results, list):
        return counts
    for item in results:
        if not isinstance(item, Mapping):
            counts["unknown"] += 1
            continue
        counts["total"] += 1
        related = item.get("related")
        compliant = item.get("compliant")
        if related is False:
            counts["irrelevant"] += 1
        elif related is True and compliant is True:
            counts["related"] += 1
            counts["compliant"] += 1
        elif related is True and compliant is False:
            counts["related"] += 1
            counts["incompliant"] += 1
        else:
            counts["unknown"] += 1
    return counts


def case_record(payload: Mapping[str, Any], *, source: Path | None = None) -> dict[str, Any]:
    """Flatten one judge JSON document into metric values plus provenance."""

    scores = {metric: _number(_nested(payload, path)) for metric, path in CASE_SCORE_PATHS.items()}
    physical = payload.get("physical")
    counts = physical_question_counts(physical) if isinstance(physical, Mapping) else physical_question_counts({})
    if counts["related"]:
        scores["physical_compliance_rate"] = counts["compliant"] / counts["related"]
    else:
        scores["physical_compliance_rate"] = None
    video = payload.get("video")
    return {
        "case_id": _case_id(payload, source),
        "video": None if video is None else str(video),
        "source_path": None if source is None else str(source),
        "physical_question_counts": counts,
        "tracks_available": sorted(
            track
            for track in TRACKS
            if any(scores.get(metric) is not None for metric in TRACK_METRICS[track])
        ),
        "scores": scores,
    }


def _case_id(payload: Mapping[str, Any], source: Path | None) -> str | None:
    video = payload.get("video")
    if isinstance(video, str) and video:
        return Path(video).stem
    if source is not None:
        return source.stem
    return None


def load_case_records(results_path: Path) -> list[dict[str, Any]]:
    """Read every judge JSON reachable from ``results_path`` into case records."""

    records: list[dict[str, Any]] = []
    for path in discover_judge_files(results_path):
        payload = _safe_read(path)
        if is_judge_payload(payload):
            records.append(case_record(payload, source=path))
    return records


def aggregate_case_records(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Average per-case scores, accumulating physical question buckets corpus-wide."""

    rows = list(records)
    aggregated: dict[str, float] = {}
    for metric in METRIC_ORDER:
        if metric == "physical_compliance_rate":
            continue
        value = mean_numeric(row.get("scores", {}).get(metric) for row in rows)
        if value is not None:
            aggregated[metric] = value
    related = sum(int(row.get("physical_question_counts", {}).get("related") or 0) for row in rows)
    compliant = sum(int(row.get("physical_question_counts", {}).get("compliant") or 0) for row in rows)
    if related:
        aggregated["physical_compliance_rate"] = compliant / related
    return aggregated


def scores_from_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    """Read metric means straight out of a ``summarize_scores.py`` aggregate."""

    aggregated: dict[str, float] = {}
    for metric, path in SUMMARY_SCORE_PATHS.items():
        value = _number(_nested(payload, path))
        if value is not None:
            aggregated[metric] = value
    overall = _nested(payload, ("physical_questions", "overall"))
    if isinstance(overall, Mapping):
        correct = _number(overall.get("correct")) or 0.0
        incorrect = _number(overall.get("incorrect")) or 0.0
        if correct + incorrect > 0:
            aggregated["physical_compliance_rate"] = correct / (correct + incorrect)
    return aggregated


def normalize_results(results_path: Path) -> dict[str, Any]:
    """Normalize any supported WorldOlympiad artifact into aggregated metrics."""

    results_path = Path(results_path).expanduser()
    payload = _safe_read(results_path) if results_path.is_file() else None
    if is_summary_payload(payload):
        assert isinstance(payload, Mapping)
        scores = scores_from_summary(payload)
        if not scores:
            raise WorldOlympiadResultError(f"no WorldOlympiad scores found in summary: {results_path}")
        return {
            "kind": "summarize_scores_aggregate",
            "scores": scores,
            "case_records": [],
            "case_count": int(_number(_nested(payload, ("scores", "combined", "count"))) or 0),
            "source_path": str(results_path),
        }
    records = load_case_records(results_path)
    if not records:
        raise WorldOlympiadResultError(
            f"no WorldOlympiad judge JSON files found under {results_path}; expected files named "
            f"like '<prefix>_judge_<case_id>.json' or a summarize_scores aggregate"
        )
    return {
        "kind": "per_case_judge_files",
        "scores": aggregate_case_records(records),
        "case_records": records,
        "case_count": len(records),
        "source_path": str(results_path),
    }


def metric_rows(scores: Mapping[str, float], *, source: str, source_path: str) -> list[dict[str, Any]]:
    """Build the canonical per-metric table rows for the scorecard."""

    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        value = scores.get(metric_id)
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "track": spec["track"],
                "scale": spec["scale"],
                "description": spec["description"],
                "available": value is not None,
                "raw_score": value,
                "normalized_score": None if value is None else _to_unit(metric_id, value),
                "score": value,
                "higher_is_better": True,
                "aggregator": spec.get("aggregator", "mean"),
                "source": source,
                "source_path": source_path,
                "reason": None if value is not None else "metric_not_present_in_worldolympiad_results",
            }
        )
    return rows


def _to_unit(metric_id: str, value: float) -> float:
    """Map a metric onto the unit interval using its declared native scale."""

    scale = METRIC_SPECS[metric_id]["scale"]
    if scale == "0-3":
        value = value / 3.0
    elif scale == "0-5":
        value = value / 5.0
    return max(0.0, min(1.0, value))


def track_summary(scores: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Summarize which triathlon tracks the supplied scores actually cover."""

    summary: dict[str, dict[str, Any]] = {}
    for track, metrics in TRACK_METRICS.items():
        available = [metric for metric in metrics if scores.get(metric) is not None]
        summary[track] = {
            "available_metric_count": len(available),
            "declared_metric_count": len(metrics),
            "available": bool(available),
            "metrics": available,
        }
    return summary


def iter_case_metric_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield one flat row per case for ``per_case_metrics.jsonl``."""

    for record in records:
        row = {
            "case_id": record.get("case_id"),
            "video": record.get("video"),
            "source_path": record.get("source_path"),
            "tracks_available": record.get("tracks_available"),
        }
        row.update({metric: record.get("scores", {}).get(metric) for metric in METRIC_ORDER})
        row["physical_question_counts"] = record.get("physical_question_counts")
        yield row
