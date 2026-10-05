"""MIND metric specification, discovery, extraction, and aggregation.

MIND (``https://github.com/CSU-JPG/MIND``) evaluates memory consistency and
action control in interactive world models. Its official entry point
``src/process.py`` writes a single JSON document per run:

```json
{
  "video_max_time": 97,
  "data": [
    {
      "path": "scene_0001", "perspective": "1st_data", "test_type": "mem_test",
      "error": null, "mark_time": 120, "total_time": 240, "sample_frames": 120,
      "lcm":            {"mse": [...], "avg_mse": ..., "psnr": [...], "avg_psnr": ...,
                         "ssim": [...], "avg_ssim": ..., "lpips": [...], "avg_lpips": ...},
      "visual_quality": {"imaging": [...], "avg_imaging": ..., "aesthetic": [...], "avg_aesthetic": ...},
      "dino":           {"dino_mse": [...], "avg_dino_mse": ...},
      "action":         {"__overall__": {"count": ..., "rpe_trans_mean": ..., "rpe_trans_median": ...,
                                          "rpe_rot_mean_deg": ..., "rpe_rot_median_deg": ...},
                         "translation": {...}, "rotation": {...}, "other": {...}, "act:forward": {...}},
      "video_results":  [{"video_name": "path-1.mp4", "gsc": {"avg_mse": ..., "avg_psnr": ...,
                                                              "avg_ssim": ..., "avg_lpips": ...}}]
    }
  ]
}
```

``mem_test`` and ``action_space_test`` samples carry ``lcm``/``visual_quality``/
``dino``/``action``; ``mirror_test`` samples carry ``video_results[*].gsc``
(general scene consistency between a trajectory and its mirrored replay).

This module turns those raw upstream shapes into WorldFoundry metric rows with
normalized scores in ``[0, 1]``. The normalizations are declared per metric in
:data:`METRIC_SPECS` so nothing is implicit: MIND publishes no official
composite score, so ``mind_average`` is a WorldFoundry-derived aggregate and is
never a leaderboard number.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MindResultError(ValueError):
    """Raised when a MIND results path holds nothing this module can read."""


# ---------------------------------------------------------------------------
# Metric specification
# ---------------------------------------------------------------------------

#: Upper bound of the mean squared error between two L2-normalized DINOv3
#: patch-token vectors. ``dino.py`` normalizes each 768-dim token to unit norm,
#: so the squared distance per token is at most 4 and the per-element mean is at
#: most ``4 / 768``.
DINO_MSE_MAX = 4.0 / 768.0

#: PSNR (dB) treated as visually lossless by the MIND pixel metrics, which run
#: with ``data_range=1.0``.
PSNR_SATURATION_DB = 50.0

METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "memory_consistency": ("lcm_psnr", "lcm_ssim", "lcm_lpips", "lcm_mse", "dino_mse"),
    "visual_quality": ("imaging_quality", "aesthetic_quality"),
    "action_control": ("action_rpe_trans_mean", "action_rpe_rot_mean_deg"),
    "scene_consistency": ("gsc_psnr", "gsc_ssim", "gsc_lpips", "gsc_mse"),
}

GROUP_SCORE_IDS = tuple(f"{group}_score" for group in METRIC_GROUPS)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "lcm_psnr": {
        "name": "LCM PSNR",
        "group": "memory_consistency",
        "higher_is_better": True,
        "native_scale": "dB (data_range=1.0)",
        "normalization": "psnr_db",
        "description": (
            "Long-context-memory PSNR between the predicted continuation and the ground-truth "
            "frames after the memory mark time."
        ),
    },
    "lcm_ssim": {
        "name": "LCM SSIM",
        "group": "memory_consistency",
        "higher_is_better": True,
        "native_scale": "0..1",
        "normalization": "unit",
        "description": "Long-context-memory SSIM against the ground-truth continuation.",
    },
    "lcm_lpips": {
        "name": "LCM LPIPS",
        "group": "memory_consistency",
        "higher_is_better": False,
        "native_scale": "0..1 (lower is better)",
        "normalization": "unit_inverse",
        "description": "Long-context-memory LPIPS (AlexNet) against the ground-truth continuation.",
    },
    "lcm_mse": {
        "name": "LCM MSE",
        "group": "memory_consistency",
        "higher_is_better": False,
        "native_scale": "0..1 (lower is better)",
        "normalization": "unit_inverse",
        "description": "Long-context-memory pixel MSE against the ground-truth continuation.",
    },
    "dino_mse": {
        "name": "DINOv3 Feature MSE",
        "group": "memory_consistency",
        "higher_is_better": False,
        "native_scale": f"0..{DINO_MSE_MAX:.6f} (lower is better)",
        "normalization": "dino_feature_mse",
        "description": (
            "Mean squared error between L2-normalized DINOv3 patch tokens of the predicted and "
            "ground-truth continuation frames."
        ),
    },
    "imaging_quality": {
        "name": "Imaging Quality",
        "group": "visual_quality",
        "higher_is_better": True,
        "native_scale": "0..1 (MUSIQ-SPAQ / 100)",
        "normalization": "unit",
        "description": "MUSIQ-SPAQ imaging quality of the generated frames, divided by 100 upstream.",
    },
    "aesthetic_quality": {
        "name": "Aesthetic Quality",
        "group": "visual_quality",
        "higher_is_better": True,
        "native_scale": "0..1 (LAION aesthetic / 10)",
        "normalization": "unit",
        "description": "LAION CLIP ViT-L/14 aesthetic predictor score, divided by 10 upstream.",
    },
    "action_rpe_trans_mean": {
        "name": "Action RPE Translation (mean)",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "Sim(3)-aligned translation units (lower is better)",
        "normalization": "inverse_offset",
        "description": (
            "Mean relative pose error in translation between the ViPE trajectory of the generated "
            "video and the ground-truth trajectory, over all valid action steps."
        ),
    },
    "action_rpe_rot_mean_deg": {
        "name": "Action RPE Rotation (mean)",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "degrees, 0..180 (lower is better)",
        "normalization": "degrees",
        "description": (
            "Mean relative pose error in rotation between the ViPE trajectory of the generated "
            "video and the ground-truth trajectory, over all valid action steps."
        ),
    },
    "action_rpe_trans_median": {
        "name": "Action RPE Translation (median)",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "Sim(3)-aligned translation units (lower is better)",
        "normalization": "inverse_offset",
        "aggregate_component": False,
        "description": "Median relative pose error in translation over all valid action steps.",
    },
    "action_rpe_rot_median_deg": {
        "name": "Action RPE Rotation (median)",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "degrees, 0..180 (lower is better)",
        "normalization": "degrees",
        "aggregate_component": False,
        "description": "Median relative pose error in rotation over all valid action steps.",
    },
    "action_translation_rpe_trans_mean": {
        "name": "Translation-Action RPE Translation",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "Sim(3)-aligned translation units (lower is better)",
        "normalization": "inverse_offset",
        "aggregate_component": False,
        "description": (
            "Mean translation RPE restricted to pure translation actions "
            "(forward/backward/left/right)."
        ),
    },
    "action_rotation_rpe_rot_mean_deg": {
        "name": "Rotation-Action RPE Rotation",
        "group": "action_control",
        "higher_is_better": False,
        "native_scale": "degrees, 0..180 (lower is better)",
        "normalization": "degrees",
        "aggregate_component": False,
        "description": "Mean rotation RPE restricted to pure camera-rotation actions.",
    },
    "gsc_psnr": {
        "name": "GSC PSNR",
        "group": "scene_consistency",
        "higher_is_better": True,
        "native_scale": "dB (data_range=1.0)",
        "normalization": "psnr_db",
        "description": (
            "General scene consistency PSNR between the first half of a mirror-test rollout and "
            "its time-reversed second half."
        ),
    },
    "gsc_ssim": {
        "name": "GSC SSIM",
        "group": "scene_consistency",
        "higher_is_better": True,
        "native_scale": "0..1",
        "normalization": "unit",
        "description": "General scene consistency SSIM on the mirror-test rollout.",
    },
    "gsc_lpips": {
        "name": "GSC LPIPS",
        "group": "scene_consistency",
        "higher_is_better": False,
        "native_scale": "0..1 (lower is better)",
        "normalization": "unit_inverse",
        "description": "General scene consistency LPIPS on the mirror-test rollout.",
    },
    "gsc_mse": {
        "name": "GSC MSE",
        "group": "scene_consistency",
        "higher_is_better": False,
        "native_scale": "0..1 (lower is better)",
        "normalization": "unit_inverse",
        "description": "General scene consistency pixel MSE on the mirror-test rollout.",
    },
}

METRIC_SPECS.update(
    {
        f"{group}_score": {
            "name": f"{group.replace('_', ' ').title()} Score",
            "group": "aggregate",
            "higher_is_better": True,
            "native_scale": "0..1",
            "normalization": "unit",
            "aggregate_component": False,
            "description": (
                f"WorldFoundry-derived mean of the normalized {group.replace('_', ' ')} component metrics."
            ),
        }
        for group in METRIC_GROUPS
    }
)
METRIC_SPECS["mind_average"] = {
    "name": "MIND Average",
    "group": "aggregate",
    "higher_is_better": True,
    "native_scale": "0..1",
    "normalization": "unit",
    "aggregate_component": False,
    "primary": True,
    "description": (
        "WorldFoundry-derived mean of the available MIND ability-group scores. MIND publishes no "
        "official composite score, so this is a WorldFoundry aggregate and is never a leaderboard number."
    ),
}

METRIC_ORDER: tuple[str, ...] = (
    "lcm_psnr",
    "lcm_ssim",
    "lcm_lpips",
    "lcm_mse",
    "dino_mse",
    "imaging_quality",
    "aesthetic_quality",
    "action_rpe_trans_mean",
    "action_rpe_rot_mean_deg",
    "action_rpe_trans_median",
    "action_rpe_rot_median_deg",
    "action_translation_rpe_trans_mean",
    "action_rotation_rpe_rot_mean_deg",
    "gsc_psnr",
    "gsc_ssim",
    "gsc_lpips",
    "gsc_mse",
    *GROUP_SCORE_IDS,
    "mind_average",
)

PRIMARY_METRIC_ID = "mind_average"

#: Component metrics that must all be present for a MIND run to be comparable.
COMPONENT_METRIC_IDS: tuple[str, ...] = tuple(
    metric_id for metric_ids in METRIC_GROUPS.values() for metric_id in metric_ids
)

MIND_PERSPECTIVES = ("1st_data", "3rd_data")
MIND_TEST_TYPES = ("mem_test", "action_space_test", "mirror_test")


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """Coerce an upstream JSON value into a finite float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            return _number(float(text))
        except ValueError:
            return None
    return None


def _clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_score(metric_id: str, raw_score: float | None) -> float | None:
    """Map a raw MIND score onto ``[0, 1]`` where higher always means better."""
    if raw_score is None:
        return None
    spec = METRIC_SPECS.get(metric_id)
    if spec is None:
        return None
    value = float(raw_score)
    rule = spec["normalization"]
    if rule == "unit":
        return _clamp01(value)
    if rule == "unit_inverse":
        return _clamp01(1.0 - value)
    if rule == "psnr_db":
        return _clamp01(value / PSNR_SATURATION_DB)
    if rule == "dino_feature_mse":
        return _clamp01(1.0 - value / DINO_MSE_MAX)
    if rule == "degrees":
        return _clamp01(1.0 - value / 180.0)
    if rule == "inverse_offset":
        return _clamp01(1.0 / (1.0 + max(0.0, value)))
    return None


# ---------------------------------------------------------------------------
# Result discovery and loading
# ---------------------------------------------------------------------------


def _looks_like_mind_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and isinstance(payload.get("data"), list)


def discover_result_files(path: Path) -> list[Path]:
    """Return the MIND result JSON documents reachable from ``path``.

    ``src/process.py`` writes one JSON file (``--output``, default
    ``result_<test_root>_<timestamp>.json``). A directory is searched for those
    files, preferring ``result*.json`` before falling back to every JSON file
    whose payload has a ``data`` list.
    """
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise MindResultError(f"MIND results path does not exist: {path}")
    preferred = sorted(candidate for candidate in path.rglob("result*.json") if candidate.is_file())
    if preferred:
        return preferred
    fallback = [
        candidate
        for candidate in sorted(path.rglob("*.json"))
        if candidate.is_file() and _looks_like_mind_payload(_safe_load(candidate))
    ]
    if not fallback:
        raise MindResultError(f"no MIND result JSON found under {path}")
    return fallback


def _safe_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def load_result_payloads(paths: Sequence[Path]) -> list[tuple[Path, Mapping[str, Any]]]:
    """Load every readable MIND payload out of ``paths``."""
    payloads: list[tuple[Path, Mapping[str, Any]]] = []
    for path in paths:
        payload = _safe_load(Path(path))
        if _looks_like_mind_payload(payload):
            payloads.append((Path(path), payload))
    if not payloads:
        listed = ", ".join(str(path) for path in paths) or "<empty>"
        raise MindResultError(f"no readable MIND result payload in: {listed}")
    return payloads


# ---------------------------------------------------------------------------
# Per-sample record extraction
# ---------------------------------------------------------------------------


def _lcm_values(block: Any, prefix: str) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    values: dict[str, float] = {}
    for suffix, key, fallback in (
        ("mse", "avg_mse", "mse"),
        ("psnr", "avg_psnr", "psnr"),
        ("ssim", "avg_ssim", "ssim"),
        ("lpips", "avg_lpips", "lpips"),
    ):
        number = _number(block.get(key))
        if number is None:
            number = mean_numeric(_number(item) for item in block.get(fallback) or ())
        if number is not None:
            values[f"{prefix}_{suffix}"] = number
    return values


def _visual_quality_values(block: Any) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    values: dict[str, float] = {}
    imaging = _number(block.get("avg_imaging"))
    if imaging is None:
        imaging = mean_numeric(_number(item) for item in block.get("imaging") or ())
    if imaging is not None:
        values["imaging_quality"] = imaging
    # The upstream README mislabels the aesthetic average as ``avg_imaging``;
    # ``visual_quality.py`` actually writes ``avg_aesthetic``. Read the real key
    # first and fall back to the per-frame list.
    aesthetic = _number(block.get("avg_aesthetic"))
    if aesthetic is None:
        aesthetic = mean_numeric(_number(item) for item in block.get("aesthetic") or ())
    if aesthetic is not None:
        values["aesthetic_quality"] = aesthetic
    return values


def _dino_values(block: Any) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    number = _number(block.get("avg_dino_mse"))
    if number is None:
        number = mean_numeric(_number(item) for item in block.get("dino_mse") or ())
    return {} if number is None else {"dino_mse": number}


def _action_values(block: Any) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    values: dict[str, float] = {}
    overall = block.get("__overall__")
    if isinstance(overall, Mapping):
        for metric_id, key in (
            ("action_rpe_trans_mean", "rpe_trans_mean"),
            ("action_rpe_trans_median", "rpe_trans_median"),
            ("action_rpe_rot_mean_deg", "rpe_rot_mean_deg"),
            ("action_rpe_rot_median_deg", "rpe_rot_median_deg"),
        ):
            number = _number(overall.get(key))
            if number is not None:
                values[metric_id] = number
    translation = block.get("translation")
    if isinstance(translation, Mapping):
        number = _number(translation.get("rpe_trans_mean"))
        if number is not None:
            values["action_translation_rpe_trans_mean"] = number
    rotation = block.get("rotation")
    if isinstance(rotation, Mapping):
        number = _number(rotation.get("rpe_rot_mean_deg"))
        if number is not None:
            values["action_rotation_rpe_rot_mean_deg"] = number
    return values


def _gsc_values(video_results: Any) -> dict[str, float]:
    if not isinstance(video_results, list):
        return {}
    buckets: dict[str, list[float]] = {}
    for entry in video_results:
        if not isinstance(entry, Mapping):
            continue
        for metric_id, value in _lcm_values(entry.get("gsc"), "gsc").items():
            buckets.setdefault(metric_id, []).append(value)
    return {metric_id: mean for metric_id, values in buckets.items() if (mean := mean_numeric(values)) is not None}


def extract_sample_record(entry: Mapping[str, Any], *, source_path: Path | str | None = None) -> dict[str, Any]:
    """Turn one entry of the upstream ``data`` list into a flat WorldFoundry record."""
    metrics: dict[str, float] = {}
    metrics.update(_lcm_values(entry.get("lcm"), "lcm"))
    metrics.update(_visual_quality_values(entry.get("visual_quality")))
    metrics.update(_dino_values(entry.get("dino")))
    metrics.update(_action_values(entry.get("action")))
    metrics.update(_gsc_values(entry.get("video_results")))
    video_results = entry.get("video_results")
    return {
        "sample_id": str(entry.get("path") or ""),
        "perspective": entry.get("perspective"),
        "test_type": entry.get("test_type"),
        "error": entry.get("error"),
        "mark_time": entry.get("mark_time"),
        "total_time": entry.get("total_time"),
        "sample_frames": entry.get("sample_frames"),
        "video_count": len(video_results) if isinstance(video_results, list) else None,
        "metrics": metrics,
        "normalized_metrics": {
            metric_id: normalized
            for metric_id, value in metrics.items()
            if (normalized := normalize_score(metric_id, value)) is not None
        },
        "source_path": None if source_path is None else str(source_path),
    }


def sample_records(payloads: Iterable[tuple[Path, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Extract per-sample records across payloads, de-duplicating repeated samples.

    MIND rewrites its output file incrementally, so the same
    ``(perspective, test_type, path)`` triple can appear in several files. The
    last record wins, matching upstream's own overwrite behaviour.
    """
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_path, payload in payloads:
        for entry in payload.get("data") or ():
            if not isinstance(entry, Mapping):
                continue
            record = extract_sample_record(entry, source_path=source_path)
            key = (
                str(record["perspective"] or ""),
                str(record["test_type"] or ""),
                record["sample_id"],
            )
            merged[key] = record
    if not merged:
        raise MindResultError("MIND result payload contains no sample records")
    return list(merged.values())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Average per-sample MIND metrics and derive the group and average scores."""
    buckets: dict[str, list[float]] = {}
    sources: dict[str, set[str]] = {}
    for record in records:
        source_path = record.get("source_path")
        for metric_id, value in (record.get("metrics") or {}).items():
            if metric_id not in METRIC_SPECS:
                continue
            buckets.setdefault(metric_id, []).append(float(value))
            if source_path:
                sources.setdefault(metric_id, set()).add(str(source_path))

    metrics: dict[str, dict[str, Any]] = {}
    for metric_id, values in buckets.items():
        raw_mean = mean_numeric(values)
        normalized = normalize_score(metric_id, raw_mean)
        metrics[metric_id] = {
            "metric_id": metric_id,
            "raw_score": raw_mean,
            "normalized_score": normalized,
            "sample_count": len(values),
            "source_paths": sorted(sources.get(metric_id, ())),
        }

    for group, metric_ids in METRIC_GROUPS.items():
        components = [
            metrics[metric_id]["normalized_score"]
            for metric_id in metric_ids
            if metric_id in metrics and metrics[metric_id]["normalized_score"] is not None
        ]
        group_score = mean_numeric(components)
        if group_score is None:
            continue
        metrics[f"{group}_score"] = {
            "metric_id": f"{group}_score",
            "raw_score": group_score,
            "normalized_score": group_score,
            "sample_count": len(components),
            "source_paths": [],
            "component_metric_ids": list(metric_ids),
            "available_component_count": len(components),
        }

    group_scores = [
        metrics[group_id]["normalized_score"]
        for group_id in GROUP_SCORE_IDS
        if group_id in metrics and metrics[group_id]["normalized_score"] is not None
    ]
    average = mean_numeric(group_scores)
    if average is not None:
        metrics["mind_average"] = {
            "metric_id": "mind_average",
            "raw_score": average,
            "normalized_score": average,
            "sample_count": len(group_scores),
            "source_paths": [],
            "component_metric_ids": [
                group_id for group_id in GROUP_SCORE_IDS if group_id in metrics
            ],
        }
    return metrics


def missing_component_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Return the declared component metrics that the results did not provide."""
    return [
        metric_id
        for metric_id in COMPONENT_METRIC_IDS
        if metrics.get(metric_id, {}).get("normalized_score") is None
    ]


def sample_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize sample coverage and upstream per-sample errors."""
    failed = [record for record in records if record.get("error")]
    by_test_type = {test_type: 0 for test_type in MIND_TEST_TYPES}
    by_perspective = {perspective: 0 for perspective in MIND_PERSPECTIVES}
    for record in records:
        test_type = str(record.get("test_type") or "")
        perspective = str(record.get("perspective") or "")
        if test_type in by_test_type:
            by_test_type[test_type] += 1
        if perspective in by_perspective:
            by_perspective[perspective] += 1
    return {
        "sample_count": len(records),
        "failed_sample_count": len(failed),
        "failed_sample_ids": sorted({str(record.get("sample_id") or "") for record in failed}),
        "samples_by_test_type": by_test_type,
        "samples_by_perspective": by_perspective,
    }


# ---------------------------------------------------------------------------
# Metric rows
# ---------------------------------------------------------------------------


def metric_rows(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    source: str,
    source_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Render the canonical WorldFoundry metric table for MIND."""
    rows: list[dict[str, Any]] = []
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        item = metrics.get(metric_id) or {}
        normalized = item.get("normalized_score")
        row_source_paths = item.get("source_paths") or []
        row_source_path = (
            ";".join(str(path) for path in row_source_paths[:5])
            if row_source_paths
            else (None if source_path is None else str(source_path))
        )
        rows.append(
            {
                "metric_id": metric_id,
                "name": spec["name"],
                "available": normalized is not None,
                "raw_score": item.get("raw_score"),
                "normalized_score": normalized,
                "score": normalized,
                "higher_is_better": spec["higher_is_better"],
                "source": source,
                "source_path": row_source_path,
                "reason": None if normalized is not None else "score_not_available_in_mind_results",
                "group": spec["group"],
                "native_scale": spec["native_scale"],
                "normalization": spec["normalization"],
                "description": spec["description"],
                "sample_count": item.get("sample_count"),
                "primary": bool(spec.get("primary", False)),
            }
        )
    return rows


__all__ = [
    "COMPONENT_METRIC_IDS",
    "DINO_MSE_MAX",
    "GROUP_SCORE_IDS",
    "METRIC_GROUPS",
    "METRIC_ORDER",
    "METRIC_SPECS",
    "MIND_PERSPECTIVES",
    "MIND_TEST_TYPES",
    "MindResultError",
    "PRIMARY_METRIC_ID",
    "PSNR_SATURATION_DB",
    "aggregate_metrics",
    "discover_result_files",
    "extract_sample_record",
    "load_result_payloads",
    "metric_rows",
    "missing_component_metrics",
    "normalize_score",
    "sample_records",
    "sample_summary",
]
