"""IPV-Bench metric normalization from official exports and per-sample annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.execution.framework.io import mean_numeric as _mean
from worldfoundry.evaluation.tasks.execution.framework.io import try_float as _to_float

METRIC_ORDER = (
    "visual_quality",
    "prompt_following",
    "impossible_video_score",
    "judgement_accuracy",
    "mcqa_accuracy",
    "open_qa_score",
    "ipv_bench_average",
)

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "visual_quality": {
        "name": "Visual Quality",
        "group": "generation",
        "higher_is_better": True,
    },
    "prompt_following": {
        "name": "Prompt Following",
        "group": "generation",
        "higher_is_better": True,
    },
    "impossible_video_score": {
        "name": "Impossible Video Score",
        "group": "generation",
        "higher_is_better": True,
    },
    "judgement_accuracy": {
        "name": "Judgement Accuracy",
        "group": "understanding",
        "higher_is_better": True,
    },
    "mcqa_accuracy": {
        "name": "MCQA Accuracy",
        "group": "understanding",
        "higher_is_better": True,
    },
    "open_qa_score": {
        "name": "Open QA Score",
        "group": "understanding",
        "higher_is_better": True,
    },
    "ipv_bench_average": {
        "name": "IPV-Bench Average",
        "group": "aggregate",
        "higher_is_better": True,
        "primary": True,
    },
}

UPSTREAM_METRIC_ALIASES = {
    "visual_quality": "visual_quality",
    "visualquality": "visual_quality",
    "vq": "visual_quality",
    "prompt_following": "prompt_following",
    "promptfollowing": "prompt_following",
    "pf": "prompt_following",
    "impossible_video_score": "impossible_video_score",
    "ipv_score": "impossible_video_score",
    "ipvscore": "impossible_video_score",
    "judgement_accuracy": "judgement_accuracy",
    "judgment_accuracy": "judgement_accuracy",
    "judgement": "judgement_accuracy",
    "mcqa_accuracy": "mcqa_accuracy",
    "mcqa": "mcqa_accuracy",
    "open_qa_score": "open_qa_score",
    "openqa_score": "open_qa_score",
    "open_qa": "open_qa_score",
    "ipv_bench_average": "ipv_bench_average",
    "ipvbenchaverage": "ipv_bench_average",
    "overall": "ipv_bench_average",
    "average": "ipv_bench_average",
}

IPV_QUALITY_THRESHOLD = 4.0
IPV_FOLLOWING_THRESHOLD = 4.0


def _canonical_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum() or ch == "_")


def _metric_id_from_key(value: Any) -> str | None:
    key = _canonical_key(value)
    return UPSTREAM_METRIC_ALIASES.get(key)


def _extract_generation_sample_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    quality_values: list[float] = []
    following_values: list[float] = []
    passing = 0
    total = 0
    for row in rows:
        quality = _to_float(row.get("visual_quality") or row.get("Visual Quality") or row.get("vq"))
        following = _to_float(row.get("prompt_following") or row.get("Prompt Following") or row.get("pf"))
        if quality is None and following is None:
            continue
        total += 1
        if quality is not None:
            quality_values.append(quality)
        if following is not None:
            following_values.append(following)
        if quality is not None and following is not None:
            if quality >= IPV_QUALITY_THRESHOLD and following >= IPV_FOLLOWING_THRESHOLD:
                passing += 1
    metrics: dict[str, float] = {}
    if quality_values:
        metrics["visual_quality"] = float(_mean(quality_values))
    if following_values:
        metrics["prompt_following"] = float(_mean(following_values))
    if total > 0:
        metrics["impossible_video_score"] = passing / total
    return metrics


def _extract_understanding_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_id: str,
    label_key: str,
    pred_key: str,
) -> float | None:
    correct = 0
    total = 0
    for row in rows:
        label = row.get(label_key) or row.get("answer") or row.get("gt") or row.get("label")
        pred = row.get(pred_key) or row.get("prediction") or row.get("pred") or row.get("model_answer")
        if label is None or pred is None:
            continue
        total += 1
        if str(label).strip().lower() == str(pred).strip().lower():
            correct += 1
    return None if total == 0 else correct / total


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        metric_id = _metric_id_from_key(row.get("metric_id") or row.get("metric") or row.get("name"))
        score = _to_float(row.get("score") if "score" in row else row.get("value"))
        if metric_id is None or score is None:
            continue
        metrics[metric_id] = score
    return metrics


def compute_ipv_bench_metrics(
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    results_path: Path | None = None,
) -> dict[str, Any]:
    parsed_rows = list(rows or [])
    if results_path is not None and not parsed_rows:
        parsed_rows = load_results_rows(results_path)

    metrics = _summary_rows(parsed_rows)
    if isinstance(parsed_rows, list) and parsed_rows and all(isinstance(row, Mapping) for row in parsed_rows):
        metrics.update(_extract_generation_sample_rows(parsed_rows))
        for metric_id, label_key, pred_key in (
            ("judgement_accuracy", "answer", "pred"),
            ("mcqa_accuracy", "mcqa_answer", "mcqa_pred"),
            ("open_qa_score", "open_qa_score", "open_qa_pred"),
        ):
            if metric_id in metrics:
                continue
            score = _extract_understanding_accuracy(
                parsed_rows,
                metric_id=metric_id,
                label_key=label_key,
                pred_key=pred_key,
            )
            if score is not None:
                metrics[metric_id] = score

    if metrics.get("ipv_bench_average") is None:
        component_values = [value for key, value in metrics.items() if key != "ipv_bench_average" and value is not None]
        if component_values:
            metrics["ipv_bench_average"] = sum(component_values) / len(component_values)

    return {
        "metrics": metrics,
        "components": {
            "row_count": len(parsed_rows),
            "quality_threshold": IPV_QUALITY_THRESHOLD,
            "following_threshold": IPV_FOLLOWING_THRESHOLD,
        },
    }


def load_results_rows(results_path: Path) -> list[dict[str, Any]]:
    suffix = results_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "rows", "records", "metrics"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
            return [dict(payload)]
        raise ValueError(f"Unsupported IPV JSON results shape: {results_path}")
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
