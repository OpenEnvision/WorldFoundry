#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime import materialize_artifact_scores
from worldfoundry.evaluation.tasks.execution.framework.io import env_path, utc_now_iso, write_json, write_jsonl
from worldfoundry.evaluation.utils import REPO_ROOT

SCORECARD_SCHEMA_VERSION = "worldfoundry-scorecard"
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
METRIC_ORDER = (
    "foreground_miou",
    "foreground_dice",
    "background_rmse",
    "text_based_accuracy",
    "multiple_choice_accuracy",
    "binary_accuracy",
)
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "foreground_miou": {
        "name": "Foreground mIoU",
        "group": "video_based",
        "description": "Conventional per-object foreground intersection-over-union.",
        "higher_is_better": True,
    },
    "foreground_dice": {
        "name": "Foreground Dice (release compatibility)",
        "group": "diagnostic",
        "description": "Dice overlap computed by the public release while naming it mIoU.",
        "higher_is_better": True,
    },
    "background_rmse": {
        "name": "Background RMSE",
        "group": "video_based",
        "description": "Normalized-pixel RMSE on ground-truth background pixels.",
        "higher_is_better": False,
    },
    "text_based_accuracy": {
        "name": "Text-Based Accuracy",
        "group": "text_based",
        "description": "Accuracy on WorldBench text-question tasks.",
        "higher_is_better": True,
    },
    "multiple_choice_accuracy": {
        "name": "Multiple-Choice Accuracy",
        "group": "text_based",
        "description": "Accuracy on WorldBench multiple-choice text questions.",
        "higher_is_better": True,
    },
    "binary_accuracy": {
        "name": "Binary Accuracy",
        "group": "text_based",
        "description": "Accuracy on WorldBench binary text questions.",
        "higher_is_better": True,
    },
}

SCORE_KEYS = ("score", "raw_score", "value", "mean", "average", "avg", "overall", "accuracy", "acc")
NORMALIZED_SCORE_KEYS = ("normalized_score", "score_normalized", "normalized", "norm_score")
GENERIC_SCORE_KEYS = {"score", "accuracy", "acc", "average", "mean", "overall", "overall_score", "overall_accuracy"}
SUMMARY_KEYS = ("summary", "metrics", "scores", "leaderboard", "leaderboard_metrics", "aggregate", "aggregates")
SAMPLE_CONTAINER_KEYS = (
    "per_sample_scores",
    "per_sample_metrics",
    "sample_scores",
    "sample_results",
    "samples",
    "predictions",
    "preds",
    "answers",
    "records",
    "rows",
    "results",
)
ID_KEYS = ("sample_id", "id", "uid", "scene_id", "scene", "video_id", "video_name", "question_id", "qid")
QUESTION_TYPE_KEYS = ("question_type", "question_kind", "answer_type", "type")
COMPONENT_KEYS = ("component", "split", "subset", "task", "benchmark", "evaluation_type")
ROW_HINT_KEYS = (
    *ID_KEYS,
    *QUESTION_TYPE_KEYS,
    *COMPONENT_KEYS,
    "video_path",
    "image_path",
    "question",
    "prompt",
    "prediction",
    "pred",
    "answer",
    "target",
    "label",
    "gold",
    "ground_truth",
    "correct",
    "is_correct",
    "score",
    "accuracy",
)
METRIC_ID_KEYS = ("metric_id", "metric", "metric_name", "name", "key", "category")


def canonical_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def parse_number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return parse_number(value)
    if isinstance(value, (list, tuple)):
        values = [scalar(item) for item in value]
        return mean(values)
    if isinstance(value, dict):
        for key in (*SCORE_KEYS, *NORMALIZED_SCORE_KEYS):
            if key in value:
                number = scalar(value[key])
                if number is not None:
                    return number
    return None


def normalized_only_score(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    if any(key in value for key in SCORE_KEYS):
        return None
    for key in NORMALIZED_SCORE_KEYS:
        if key in value:
            return scalar(value[key])
    return None


def score_scale(value: Any, raw_score: float | None, normalized_score: float | None = None) -> str:
    if normalized_score is not None and raw_score is None:
        return "normalized"
    if isinstance(value, str) and value.strip().endswith("%"):
        return "percent"
    if raw_score is None:
        return "unknown"
    if 0.0 <= raw_score <= 1.0:
        return "fraction"
    if 1.0 < raw_score <= 100.0:
        return "percent"
    return "raw"


def normalize_accuracy(raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    if 1.0 < raw_score <= 100.0:
        return raw_score / 100.0
    return raw_score


METRIC_ALIASES = {
    "foreground_miou": "foreground_miou",
    "foreground_iou": "foreground_miou",
    "mean_iou": "foreground_miou",
    "video_based": "foreground_miou",
    "video_based_accuracy": "foreground_miou",
    "video_accuracy": "foreground_miou",
    "v2v_accuracy": "foreground_miou",
    "foreground_dice": "foreground_dice",
    "dice": "foreground_dice",
    "dice_score": "foreground_dice",
    # The public release's value named mIoU is mathematically Dice.
    "miou": "foreground_dice",
    "m_iou": "foreground_dice",
    "background_rmse": "background_rmse",
    "bg_rmse": "background_rmse",
    "background_error": "background_rmse",
    "text_based": "text_based_accuracy",
    "text_based_accuracy": "text_based_accuracy",
    "text_accuracy": "text_based_accuracy",
    "qa_accuracy": "text_based_accuracy",
    "vqa_accuracy": "text_based_accuracy",
    "question_accuracy": "text_based_accuracy",
    "language_accuracy": "text_based_accuracy",
    "multiple_choice": "multiple_choice_accuracy",
    "multiple_choice_accuracy": "multiple_choice_accuracy",
    "multi_choice_accuracy": "multiple_choice_accuracy",
    "mc_accuracy": "multiple_choice_accuracy",
    "choice_accuracy": "multiple_choice_accuracy",
    "binary": "binary_accuracy",
    "binary_accuracy": "binary_accuracy",
    "yes_no_accuracy": "binary_accuracy",
    "true_false_accuracy": "binary_accuracy",
}


def context_metric_id(context_key: Any) -> str | None:
    key = canonical_key(context_key)
    if not key:
        return None
    if "multiple_choice" in key or key.startswith("mc") or "_mc" in key:
        return "multiple_choice_accuracy"
    if "binary" in key or "yes_no" in key or "true_false" in key:
        return "binary_accuracy"
    if "video" in key or "v2v" in key or "physical_prediction" in key:
        return "foreground_miou"
    if "text" in key or "question" in key or "vqa" in key or "qa" in key:
        return "text_based_accuracy"
    return None


def metric_id_for_key(raw_key: Any, context_key: Any = None) -> str | None:
    key = canonical_key(raw_key)
    metric_id = METRIC_ALIASES.get(key)
    if metric_id:
        return metric_id
    if key in GENERIC_SCORE_KEYS:
        return context_metric_id(context_key)
    return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    for raw_key, value in row.items():
        if canonical_key(raw_key) in {canonical_key(key) for key in keys}:
            return value
    return None


def first_score_value(row: dict[str, Any]) -> tuple[Any, bool]:
    for key in (*SCORE_KEYS, *NORMALIZED_SCORE_KEYS):
        if key in row:
            return row[key], key in NORMALIZED_SCORE_KEYS
    canonical_score_keys = {canonical_key(key) for key in (*SCORE_KEYS, *NORMALIZED_SCORE_KEYS)}
    canonical_normalized_keys = {canonical_key(key) for key in NORMALIZED_SCORE_KEYS}
    for raw_key, value in row.items():
        key = canonical_key(raw_key)
        if key in canonical_score_keys:
            return value, key in canonical_normalized_keys
    return None, False


def metric_item(metric_id: str, raw_value: Any, source: str, sample_count: int | None = None) -> dict[str, Any]:
    normalized_score = normalized_only_score(raw_value)
    raw_score = None if normalized_score is not None else scalar(raw_value)
    if normalized_score is None:
        normalized_score = normalize_accuracy(raw_score)
    item: dict[str, Any] = {
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "source": source,
        "score_scale": score_scale(raw_value, raw_score, normalized_score if raw_score is None else None),
    }
    if sample_count is not None:
        item["sample_count"] = sample_count
    return item


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_file(path: Path) -> list[Any]:
    rows: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_table_file(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def load_result_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = load_json_file(path)
        source_format = "json"
    elif suffix == ".jsonl":
        payload = load_jsonl_file(path)
        source_format = "jsonl"
    elif suffix in {".csv", ".tsv"}:
        payload = load_table_file(path)
        source_format = suffix.removeprefix(".")
    else:
        raise ValueError(f"unsupported WorldBench result file suffix: {path}")
    return {
        "path": path,
        "format": source_format,
        "payload": payload,
    }


def load_upstream_results(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return [load_result_file(path)]
    if path.is_dir():
        files = [
            item for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not files:
            raise FileNotFoundError(f"no WorldBench result files found under: {path}")
        return [load_result_file(item) for item in files]
    raise FileNotFoundError(f"WorldBench results path not found: {path}")


def looks_like_sample_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    keys = {canonical_key(key) for key in row}
    return bool(keys & {canonical_key(key) for key in ROW_HINT_KEYS})


def looks_like_metric_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    keys = {canonical_key(key) for key in row}
    return bool(keys & {canonical_key(key) for key in METRIC_ID_KEYS}) and bool(
        keys & {canonical_key(key) for key in (*SCORE_KEYS, *NORMALIZED_SCORE_KEYS)}
    )


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        key = canonical_key(value)
        if key in {"true", "yes", "y", "correct", "right", "pass", "passed", "1"}:
            return True
        if key in {"false", "no", "n", "incorrect", "wrong", "fail", "failed", "0"}:
            return False
    return None


def normalize_question_type(value: Any) -> str | None:
    key = canonical_key(value)
    if not key:
        return None
    if "multiple_choice" in key or key in {"mc", "choice", "multi_choice"}:
        return "multiple_choice"
    if "binary" in key or "yes_no" in key or "true_false" in key or key in {"yn", "tf"}:
        return "binary"
    return key


def infer_question_type(row: dict[str, Any], context_key: Any = None) -> str | None:
    for key in QUESTION_TYPE_KEYS:
        value = first_present(row, (key,))
        question_type = normalize_question_type(value)
        if question_type:
            return question_type
    return normalize_question_type(context_key)


def infer_component(row: dict[str, Any], context_key: Any = None, question_type: str | None = None) -> str | None:
    candidates = [context_key]
    for key in COMPONENT_KEYS:
        candidates.append(first_present(row, (key,)))
    for candidate in candidates:
        key = canonical_key(candidate)
        if not key:
            continue
        if "video" in key or "v2v" in key or "physical_prediction" in key:
            return "video_based"
        if "text" in key or "question" in key or "vqa" in key or "qa" in key:
            return "text_based"
        if (
            "multiple_choice" in key
            or key.startswith("mc")
            or "binary" in key
            or "yes_no" in key
            or "true_false" in key
        ):
            return "text_based"
    if question_type in {"multiple_choice", "binary"}:
        return "text_based"
    return None


def text_equal(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    if isinstance(right, (list, tuple, set)):
        return any(text_equal(left, item) is True for item in right)
    return str(left).strip().lower() == str(right).strip().lower()


def normalize_sample_row(
    row: dict[str, Any],
    *,
    source_path: Path,
    index: int,
    context_key: Any = None,
    sample_id: Any = None,
) -> dict[str, Any]:
    raw_sample_id = sample_id
    if raw_sample_id is None:
        raw_sample_id = first_present(row, ID_KEYS)
    if raw_sample_id is None:
        raw_sample_id = f"{source_path.stem}:{index}"

    question_type = infer_question_type(row, context_key)
    component = infer_component(row, context_key, question_type)

    correct = parse_bool(first_present(row, ("correct", "is_correct", "hit", "success")))
    if correct is None:
        prediction = first_present(row, ("prediction", "pred", "model_answer", "output"))
        answer = first_present(row, ("answer", "target", "label", "gold", "ground_truth", "gt"))
        correct = text_equal(prediction, answer)

    score_value, _ = first_score_value(row)
    raw_score = 1.0 if correct is True else 0.0 if correct is False else scalar(score_value)
    normalized_score = normalize_accuracy(raw_score)

    return {
        "sample_id": str(raw_sample_id),
        "component": component,
        "question_type": question_type,
        "category": first_present(row, ("category", "domain", "skill", "subtask")),
        "prompt": first_present(row, ("prompt", "question", "instruction")),
        "prediction": first_present(row, ("prediction", "pred", "model_answer", "output")),
        "answer": first_present(row, ("answer", "target", "label", "gold", "ground_truth", "gt")),
        "correct": correct,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "available": normalized_score is not None,
        "source": str(source_path.resolve()),
    }


def rows_from_container(value: Any, *, source_path: Path, context_key: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if looks_like_metric_row(item):
                continue
            if isinstance(item, dict) and looks_like_sample_row(item):
                rows.append(normalize_sample_row(item, source_path=source_path, index=index, context_key=context_key))
        return rows
    if isinstance(value, dict):
        if value and all(metric_id_for_key(key, context_key) is not None for key in value):
            return rows
        for index, (sample_id, item) in enumerate(sorted(value.items(), key=lambda pair: str(pair[0]))):
            if isinstance(item, dict):
                candidate = dict(item)
            else:
                candidate = {"raw_response": item}
            if looks_like_sample_row(candidate) or not isinstance(item, dict):
                rows.append(
                    normalize_sample_row(
                        candidate,
                        source_path=source_path,
                        index=index,
                        context_key=context_key,
                        sample_id=sample_id,
                    )
                )
        return rows
    return rows


def extract_sample_rows(loaded_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loaded in loaded_results:
        source_path = loaded["path"]
        payload = loaded["payload"]
        if isinstance(payload, list):
            rows.extend(rows_from_container(payload, source_path=source_path, context_key=source_path.stem))
            continue
        if not isinstance(payload, dict):
            continue
        for key in SAMPLE_CONTAINER_KEYS:
            if key not in payload:
                continue
            rows.extend(rows_from_container(payload[key], source_path=source_path, context_key=key))
        for key, value in payload.items():
            if key in SUMMARY_KEYS or key in SAMPLE_CONTAINER_KEYS:
                continue
            context_metric = context_metric_id(key)
            if context_metric and isinstance(value, (list, dict)):
                rows.extend(rows_from_container(value, source_path=source_path, context_key=key))
    return rows


def candidate_metric_maps(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], Any]]:
    maps: list[tuple[str, dict[str, Any], Any]] = [("root", payload, None)]
    for key in SUMMARY_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            maps.append((key, value, key))
    for key, value in payload.items():
        if isinstance(value, dict) and not looks_like_sample_row(value):
            maps.append((str(key), value, key))
    return maps


def extract_scores_from_metric_row(row: dict[str, Any], source: str) -> tuple[str, dict[str, Any]] | None:
    raw_metric_key = first_present(row, METRIC_ID_KEYS)
    metric_id = metric_id_for_key(raw_metric_key)
    if metric_id is None:
        return None
    value, normalized_only = first_score_value(row)
    if value is None:
        return None
    if normalized_only:
        item = {
            "raw_score": None,
            "normalized_score": scalar(value),
            "source": source,
            "score_scale": "normalized",
        }
    else:
        item = metric_item(metric_id, value, source)
    return metric_id, item


def extract_scores_from_payloads(loaded_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for loaded in loaded_results:
        source_path = loaded["path"]
        payload = loaded["payload"]
        source_prefix = str(source_path.resolve())

        if isinstance(payload, list):
            for index, row in enumerate(payload):
                if not looks_like_metric_row(row):
                    continue
                result = extract_scores_from_metric_row(row, f"{source_prefix}#{index}")
                if result is not None and result[0] not in extracted:
                    extracted[result[0]] = result[1]
            continue

        if not isinstance(payload, dict):
            continue

        for source_name, metric_map, context_key in candidate_metric_maps(payload):
            for raw_key, raw_value in metric_map.items():
                metric_id = metric_id_for_key(raw_key, context_key)
                if metric_id is None or metric_id in extracted:
                    continue
                item = metric_item(metric_id, raw_value, f"{source_prefix}.{source_name}.{raw_key}")
                if item["raw_score"] is not None or item["normalized_score"] is not None:
                    extracted[metric_id] = item
    return extracted


def sample_scores(sample_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float | None]] = {
        "foreground_miou": [],
        "text_based_accuracy": [],
        "multiple_choice_accuracy": [],
        "binary_accuracy": [],
    }
    for row in sample_rows:
        score = row.get("normalized_score")
        if not isinstance(score, (int, float)):
            continue
        component = row.get("component")
        question_type = row.get("question_type")
        if component == "video_based":
            groups["foreground_miou"].append(float(score))
        if component == "text_based" or question_type in {"multiple_choice", "binary"}:
            groups["text_based_accuracy"].append(float(score))
        if question_type == "multiple_choice":
            groups["multiple_choice_accuracy"].append(float(score))
        if question_type == "binary":
            groups["binary_accuracy"].append(float(score))

    extracted: dict[str, dict[str, Any]] = {}
    for metric_id, values in groups.items():
        score = mean(values)
        if score is None:
            continue
        extracted[metric_id] = {
            "raw_score": score,
            "normalized_score": score,
            "source": "computed_from_per_sample_scores",
            "score_scale": "normalized",
            "sample_count": len(values),
        }
    return extracted


def normalized_metric_score(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    value = item.get("normalized_score")
    if isinstance(value, (int, float)):
        return float(value)
    return normalize_accuracy(item.get("raw_score"))


def add_computed_scores(extracted: dict[str, dict[str, Any]]) -> None:
    if "text_based_accuracy" not in extracted:
        text_score = mean(
            [
                normalized_metric_score(extracted.get("multiple_choice_accuracy")),
                normalized_metric_score(extracted.get("binary_accuracy")),
            ]
        )
        if text_score is not None:
            extracted["text_based_accuracy"] = {
                "raw_score": text_score,
                "normalized_score": text_score,
                "source": "computed_from_text_question_type_scores",
                "score_scale": "normalized",
            }


def normalize_worldbench_results(
    loaded_results: list[dict[str, Any]],
    *,
    benchmark_id: str,
    output_dir: Path,
    results_path: Path,
    artifact_score_imported: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_scores_path = output_dir / "per_sample_scores.jsonl"

    per_sample_rows = extract_sample_rows(loaded_results)
    extracted_scores = extract_scores_from_payloads(loaded_results)
    for metric_id, item in sample_scores(per_sample_rows).items():
        extracted_scores.setdefault(metric_id, item)
    add_computed_scores(extracted_scores)

    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    for metric_id in METRIC_ORDER:
        spec = METRIC_SPECS[metric_id]
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        normalized_score = normalized_metric_score(item)
        row = {
            "metric_id": metric_id,
            "name": spec["name"],
            "available": normalized_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "higher_is_better": spec.get("higher_is_better", True),
            "normalizer": "identity" if metric_id == "background_rmse" else "percent_or_fraction_to_unit",
            "source": item.get("source"),
            "score_scale": item.get("score_scale"),
            "sample_count": item.get("sample_count"),
            "group": spec["group"],
        }
        if normalized_score is None:
            row["reason"] = "score_not_found_in_worldbench_results"
        else:
            leaderboard[metric_id] = normalized_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    available_count = sum(1 for row in metric_rows if row["available"])
    official_result_shape = {
        "checked": True,
        "ok": available_count > 0,
        "input_path": str(results_path),
        "file_count": len(loaded_results),
        "formats": sorted({str(item["format"]) for item in loaded_results}),
        "sample_rows_detected": len(per_sample_rows),
        "available_metric_count": available_count,
        "issues": [] if available_count > 0 else [{"reason": "no_scores_detected"}],
    }

    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_scores_path, per_sample_rows)

    normalization_ok = available_count > 0
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": (
                "artifact_score_normalization"
                if artifact_score_imported and normalization_ok
                else "official_result_normalization"
                if normalization_ok
                else "failed"
            ),
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_worldbench_official_runner",
            "command": command,
            "returncode": 0 if normalization_ok else 1,
            "duration_seconds": None,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "WorldBench",
            "contract_only": False,
            "requires_upstream_runtime": False,
            "official_runtime_available": True,
        },
        "dataset": {
            "upstream_results": str(results_path.resolve()),
            "result_file_count": len(loaded_results),
            "sample_count": len(per_sample_rows),
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "WorldBench scorecard is not leaderboard-valid until complete official task coverage and submission protocol are audited",
            ],
        },
        "generation": {
            "successful": len([row for row in per_sample_rows if row["available"]]),
            "failed": len([row for row in per_sample_rows if not row["available"]]),
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "video_based": ["foreground_miou", "background_rmse"],
                "text_based": ["text_based_accuracy", "multiple_choice_accuracy", "binary_accuracy"],
                "diagnostic": ["foreground_dice"],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(per_sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": (
                "worldfoundry_artifact_score_normalizer"
                if artifact_score_imported
                else "official_worldbench_result_normalizer"
            ),
            "upstream_results": str(results_path.resolve()),
            "num_results": len(per_sample_rows),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "validation": {
            "normalizer_only": True,
            "official_runtime_executed": False,
            "artifact_score_imported": artifact_score_imported,
            "official_result_shape": official_result_shape,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_scores_path.resolve()),
            "upstream_results": str(results_path.resolve()),
            "upstream_stdout": None,
            "upstream_stderr": None,
        },
        # This branch imports existing results. Raw-artifact evaluation is
        # available through the same CLI with --run-official, but importing a
        # score file alone is intentionally not integration evidence.
        "official_benchmark_verified": False,
        "integration_evidence": False,
        "normalization_ok": normalization_ok,
        "official_results_imported": (not artifact_score_imported) and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the in-tree WorldBench IntuitivePhysics evaluator on generated artifacts, "
            "or normalize an existing result file."
        )
    )
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "worldbench"))
    parser.add_argument(
        "--official-results-path",
        "--results-path",
        dest="results_path",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_RESULTS_PATH"),
    )
    parser.add_argument("--from-upstream-results", dest="results_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="Evaluate raw generated continuations and/or answer predictions with the in-tree runtime.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT"),
        help="WorldBench IntuitivePhysics dataset root containing scenes/ and textual_questions/.",
    )
    parser.add_argument(
        "--generated-video-dir",
        "--generated-artifact-dir",
        dest="generated_video_dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_GENERATED_ARTIFACT_DIR"),
        help="Generated continuation videos arranged by WorldBench sample ID.",
    )
    parser.add_argument(
        "--video-manifest",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_VIDEO_MANIFEST"),
        help="Optional JSON/JSONL/CSV/TSV sample_id-to-video mapping.",
    )
    parser.add_argument(
        "--answer-manifest",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_ANSWER_MANIFEST"),
        help="Optional WorldBench question predictions in JSON/JSONL/CSV/TSV format.",
    )
    parser.add_argument(
        "--predicted-mask-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_PREDICTED_MASK_DIR"),
        help="Use precomputed label masks instead of running SAM2 (testing/reproducibility path).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "worldfoundry/data/benchmarks/assets/worldbench/evaluator.yaml",
        help="In-tree WorldBench evaluator configuration.",
    )
    parser.add_argument("--sample-id", action="append", default=[], help="Evaluate one scene ID; repeatable.")
    parser.add_argument("--limit", type=int, help="Bound the number of matched video scenes.")
    parser.add_argument("--max-frames", type=int, help="Maximum continuation frames per video; 0 means all.")
    parser.add_argument("--ground-truth-start-frame", type=int, help="Ground-truth frame aligned to generated frame 0.")
    parser.add_argument("--generated-skip-frames", type=int, help="Conditioning frames to skip in each artifact.")
    parser.add_argument(
        "--sam2-model-id",
        default=os.environ.get("WORLDFOUNDRY_WORLDBENCH_SAM2_MODEL_ID"),
        help="SAM2 Hugging Face model ID (defaults to the data config).",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_SAM2_CKPT"),
    )
    parser.add_argument("--sam2-config", default=os.environ.get("WORLDFOUNDRY_WORLDBENCH_SAM2_CONFIG"))
    parser.add_argument("--device", default=os.environ.get("WORLDFOUNDRY_DEVICE", "auto"))
    component_group = parser.add_mutually_exclusive_group()
    component_group.add_argument("--video-only", action="store_true")
    component_group.add_argument("--text-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-staged-frames", action="store_true")
    parser.add_argument("--save-predicted-masks", type=Path)
    parser.add_argument(
        "--from-artifact-scores",
        action="store_true",
        help="Normalize existing WorldFoundry metric artifacts (diagnostic; not an official benchmark run).",
    )
    parser.add_argument(
        "--artifact-score-dir",
        type=Path,
        default=env_path("WORLDFOUNDRY_WORLDBENCH_ARTIFACT_SCORE_DIR"),
        help="Directory containing WorldBench metric artifacts produced by WorldFoundry evaluators.",
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    if args.results_path is None and not args.from_artifact_scores and not args.run_official:
        print(
            "error: --official-results-path, WORLDFOUNDRY_WORLDBENCH_RESULTS_PATH, "
            "or --from-artifact-scores is required",
            file=sys.stderr,
        )
        return 2

    try:
        if args.run_official:
            if args.dataset_root is None:
                raise ValueError(
                    "--dataset-root or WORLDFOUNDRY_WORLDBENCH_DATASET_ROOT is required for --run-official"
                )
            if not any((args.generated_video_dir, args.video_manifest, args.answer_manifest)):
                raise ValueError("--run-official needs --generated-video-dir/--video-manifest and/or --answer-manifest")
            if args.text_only and args.answer_manifest is None:
                raise ValueError("--text-only requires --answer-manifest")
            from worldfoundry.evaluation.tasks.execution.runners.worldbench.runtime.worldbench import (
                WorldBenchEvaluationRequest,
                evaluate_worldbench,
            )
            from worldfoundry.evaluation.tasks.execution.runners.worldbench.runtime.worldbench.reporting import (
                write_evaluation_scorecard,
            )

            evaluation = evaluate_worldbench(
                WorldBenchEvaluationRequest(
                    dataset_root=args.dataset_root,
                    work_dir=args.output_dir / "work",
                    generated_video_dir=args.generated_video_dir,
                    video_manifest=args.video_manifest,
                    answer_manifest=args.answer_manifest,
                    predicted_mask_dir=args.predicted_mask_dir,
                    config_path=args.config,
                    sample_ids=tuple(args.sample_id),
                    limit=args.limit,
                    max_frames=args.max_frames,
                    ground_truth_start_frame=args.ground_truth_start_frame,
                    generated_skip_frames=args.generated_skip_frames,
                    sam2_model_id=args.sam2_model_id,
                    sam2_checkpoint=args.sam2_checkpoint,
                    sam2_config=args.sam2_config,
                    device=args.device,
                    evaluate_video=not args.text_only,
                    evaluate_text=not args.video_only,
                    continue_on_error=args.continue_on_error,
                    keep_staged_frames=args.keep_staged_frames,
                    save_predicted_masks=args.save_predicted_masks,
                )
            )
            scorecard = write_evaluation_scorecard(
                evaluation,
                benchmark_id=args.benchmark_id,
                output_dir=args.output_dir,
                command=[sys.executable, *sys.argv],
            )
            result = {
                "ok": bool(scorecard["integration_evidence"]),
                "benchmark_id": args.benchmark_id,
                "output_dir": str(args.output_dir),
                "scorecard": scorecard["artifacts"]["scorecard"],
                "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
                "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
                "worldbench_evaluation": scorecard["artifacts"]["worldbench_evaluation"],
                "official_benchmark_verified": scorecard["official_benchmark_verified"],
                "integration_evidence": scorecard["integration_evidence"],
                "normalization_ok": scorecard["normalization_ok"],
                "official_results_imported": False,
            }
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                print(f"{args.benchmark_id}: in-tree WorldBench evaluation completed")
                print(f"scorecard: {result['scorecard']}")
            return 0 if result["ok"] else 1
        command = None
        artifact_score_imported = False
        if args.from_artifact_scores:
            score_dir = args.artifact_score_dir
            if score_dir is None:
                raise ValueError("--artifact-score-dir or WORLDFOUNDRY_WORLDBENCH_ARTIFACT_SCORE_DIR is required")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            args.results_path = args.output_dir / "upstream" / "worldbench_results.json"
            materialize_artifact_scores(
                benchmark_id=args.benchmark_id,
                score_dir=score_dir,
                generated_video_dir=None,
                output_path=args.results_path,
            )
            artifact_score_imported = True
            command = [
                "worldfoundry.evaluation.tasks.execution.framework.artifact_score_runtime",
                "--benchmark-id",
                args.benchmark_id,
                "--score-dir",
                str(score_dir),
                "--output-path",
                str(args.results_path),
            ]
        loaded_results = load_upstream_results(args.results_path)
        scorecard = normalize_worldbench_results(
            loaded_results,
            benchmark_id=args.benchmark_id,
            output_dir=args.output_dir,
            results_path=args.results_path,
            artifact_score_imported=artifact_score_imported,
            command=command,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = "normalized" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official WorldBench result normalization {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
