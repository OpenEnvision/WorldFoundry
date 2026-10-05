"""Shared video-quality contract evaluation engine."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.evaluation.tasks.contracts.external import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.video_quality_specs import (
    supported_video_quality_benchmark_ids,
    video_quality_spec,
)


JsonValue = Any
LOCAL_METRIC_IDS = (
    "local_metadata_coverage",
    "local_prompt_video_pair_completeness",
)
OFFICIAL_RESULT_KEYS = (
    "official_results",
    "official_scores",
    "results",
    "scores",
    "raw_metric_table",
    "pairwise_preferences",
    "human_preference_data",
    "preferences",
)
OFFICIAL_RESULT_PATH_KEYS = (
    "official_results_path",
    "results_path",
    "raw_metric_table_path",
    "human_preference_data_path",
    "pairwise_preferences_path",
)
NUMERIC_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$")


def supports_video_quality_benchmark(benchmark_id: str) -> bool:
    """Return whether this local evaluator covers the benchmark.

    Args:
        benchmark_id: Benchmark identifier from the benchmark-zoo catalog.
    """

    return benchmark_id.strip().lower() in supported_video_quality_benchmark_ids()


def load_artifact_manifest(path: str | Path) -> JsonValue:
    """Load a JSON artifact manifest.

    Parameters:
        path: JSON file describing prompt/video artifact records.
    """

    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_video_quality_contract(
    benchmark_id: str,
    artifact_manifest: JsonValue,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    """Evaluate local video-quality contract checks and block official metrics.

    Parameters:
        benchmark_id: External benchmark id registered in benchmark_zoo contracts.
        artifact_manifest: Manifest rows or mapping containing generated video records.
        artifact_root: Optional root used to resolve relative artifact paths.
    """

    contract = get_external_benchmark_contract(benchmark_id)
    rows = _manifest_rows(artifact_manifest)
    root = None if artifact_root is None else Path(artifact_root)
    resolved_rows = [_resolved_row(row, root) for row in rows]
    official_rows = _official_rows(artifact_manifest, root)
    local_results = _local_metric_results(resolved_rows)
    official_results = _official_metric_results(
        contract.benchmark_id,
        contract.metric_ids,
        official_rows,
    )
    results = [*local_results, *official_results]
    return {
        "schema_version": "worldfoundry-video-quality-local-evaluator",
        "benchmark_id": contract.benchmark_id,
        "status": "completed",
        "local_metric_ids": list(LOCAL_METRIC_IDS),
        "official_metric_ids": list(contract.metric_ids),
        "blocked_metric_ids": [item["metric_id"] for item in official_results if item["status"] == "blocked"],
        "results": results,
        "summary": {
            "sample_count": len(resolved_rows),
            "local_passed": sum(1 for item in local_results if item["status"] == "passed"),
            "local_failed": sum(1 for item in local_results if item["status"] == "failed"),
            "official_scored": sum(1 for item in official_results if item["status"] == "scored"),
            "blocked": sum(1 for item in official_results if item["status"] == "blocked"),
        },
    }


def evaluate_video_quality_contract_file(
    benchmark_id: str,
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    """Evaluate a JSON artifact manifest file.

    Parameters:
        benchmark_id: External benchmark id registered in benchmark_zoo contracts.
        artifact_manifest_path: Path to the JSON manifest to evaluate.
        artifact_root: Optional root used to resolve relative artifact paths.
    """

    return evaluate_video_quality_contract(
        benchmark_id,
        load_artifact_manifest(artifact_manifest_path),
        artifact_root=artifact_root,
    )


def evaluate_aigcbench_video_quality_contract_file(
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    return evaluate_video_quality_contract_file("aigcbench", artifact_manifest_path, artifact_root=artifact_root)


def evaluate_fetv_video_quality_contract_file(
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    return evaluate_video_quality_contract_file("fetv", artifact_manifest_path, artifact_root=artifact_root)


def evaluate_genai_bench_video_quality_contract_file(
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    return evaluate_video_quality_contract_file("genai-bench", artifact_manifest_path, artifact_root=artifact_root)


def evaluate_ipv_bench_video_quality_contract_file(
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    return evaluate_video_quality_contract_file("ipv-bench", artifact_manifest_path, artifact_root=artifact_root)


def evaluate_mirabench_video_quality_contract_file(
    artifact_manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, JsonValue]:
    return evaluate_video_quality_contract_file("mirabench", artifact_manifest_path, artifact_root=artifact_root)


def _manifest_rows(manifest: JsonValue) -> list[Mapping[str, JsonValue]]:
    """Extract sample mapping records from any nested or raw manifest structure.

    Args:
        manifest: Raw manifest data (either a list or nested dict).

    Returns:
        List of matching sample mapping dictionaries.
    """
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, bytearray)):
        return [item for item in manifest if isinstance(item, Mapping)]
    if not isinstance(manifest, Mapping):
        return []
    for key in ("samples", "records", "items", "artifacts", "generated_artifacts"):
        value = manifest.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [item for item in value if isinstance(item, Mapping)]
    return [manifest]


def _first_text(row: Mapping[str, JsonValue], keys: tuple[str, ...]) -> str | None:
    """Retrieve the first non-empty text value found for a given set of key aliases.

    Args:
        row: Key-value mapping.
        keys: Collection of alias keys to try.

    Returns:
        The matched string value, or None if not found.
    """
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _video_path(row: Mapping[str, JsonValue]) -> str | None:
    """Resolve the video path string from a given manifest row or nested artifacts.

    Args:
        row: Sample manifest row.

    Returns:
        Resolved video path string, or None if missing.
    """
    direct = _first_text(row, ("generated_video", "video", "video_path", "path", "artifact_path", "uri"))
    if direct:
        return direct
    artifacts = row.get("artifacts")
    if isinstance(artifacts, Mapping):
        item = artifacts.get("generated_video") or artifacts.get("video")
        if isinstance(item, str):
            return item
        if isinstance(item, Mapping):
            return _first_text(item, ("uri", "path", "artifact_path"))
    return None


def _category(row: Mapping[str, JsonValue]) -> str | None:
    """Resolve a classification category or prompt type from a manifest row.

    Args:
        row: Sample manifest row.

    Returns:
        The category string, or None if missing.
    """
    value = _first_text(row, ("category", "prompt_type", "task", "domain", "dimension"))
    if value is not None:
        return value
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return _first_text(metadata, ("category", "prompt_type", "task", "domain", "dimension"))
    return None


def _resolved_row(row: Mapping[str, JsonValue], root: Path | None) -> dict[str, JsonValue]:
    """Parse a raw manifest row into a canonical schema with resolved paths.

    Args:
        row: Raw manifest row mapping.
        root: Optional root directory to resolve relative paths against.

    Returns:
        A dictionary with standardized keys: 'prompt_id', 'prompt', 'video_path', 'category', 'metadata'.
    """
    raw_path = _video_path(row)
    path = None if raw_path is None else Path(raw_path)
    resolved = None
    if path is not None:
        resolved_path = path if path.is_absolute() or root is None else root / path
        resolved = str(resolved_path)
    return {
        "prompt_id": _first_text(row, ("prompt_id", "id", "sample_id")),
        "prompt": _first_text(row, ("prompt", "text", "caption")),
        "video_path": resolved,
        "category": _category(row),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
    }


def _metadata_complete(row: Mapping[str, JsonValue]) -> bool:
    """Check if all prompt metadata fields (id, prompt, and path) are populated.

    Args:
        row: Canonical sample row.

    Returns:
        True if all required fields are present and non-empty.
    """
    return all(isinstance(row.get(key), str) and bool(str(row[key]).strip()) for key in ("prompt_id", "prompt", "video_path"))


def _pair_complete(row: Mapping[str, JsonValue]) -> bool:
    """Check if prompt and video path fields are populated, ignoring prompt ID.

    Args:
        row: Canonical sample row.

    Returns:
        True if prompt and video_path are present and non-empty.
    """
    return all(isinstance(row.get(key), str) and bool(str(row[key]).strip()) for key in ("prompt", "video_path"))


def _fraction(rows: Sequence[Mapping[str, JsonValue]], predicate: str) -> float | None:
    """Calculate the ratio of rows satisfying a given predicate check.

    Args:
        rows: Sequence of sample rows.
        predicate: Name of check ('metadata' or 'pair').

    Returns:
        Computed fraction float in [0.0, 1.0], or None if list is empty.
    """
    if not rows:
        return None
    checks = {
        "metadata": _metadata_complete,
        "pair": _pair_complete,
    }
    passed = sum(1 for row in rows if checks[predicate](row))
    return passed / len(rows)


def _local_result(metric_id: str, score: float | None, evidence: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    status = "blocked" if score is None else "passed" if score == 1.0 else "failed"
    blocked_reason = "asset_required" if score is None else None
    return {
        "metric_id": metric_id,
        "score": score,
        "value": score,
        "status": status,
        "evidence": dict(evidence),
        "blocked_reason": blocked_reason,
    }


def _local_metric_results(rows: Sequence[Mapping[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    sample_count = len(rows)
    scores = {
        "local_metadata_coverage": _fraction(rows, "metadata"),
        "local_prompt_video_pair_completeness": _fraction(rows, "pair"),
    }
    return [
        _local_result(
            metric_id,
            score,
            {
                "sample_count": sample_count,
                "protocol": "local_manifest_metadata_contract",
                "note": "artifact path presence is audited as manifest completeness, not as an official quality score",
            },
        )
        for metric_id, score in scores.items()
    ]


def _official_rows(manifest: JsonValue, root: Path | None) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    if isinstance(manifest, Mapping):
        for key in OFFICIAL_RESULT_KEYS:
            value = manifest.get(key)
            if value is not None:
                rows.extend(_normalize_payload_rows(value))
        for key in OFFICIAL_RESULT_PATH_KEYS:
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value)
                resolved_path = path if path.is_absolute() or root is None else root / path
                rows.extend(_load_result_rows(resolved_path))
        if not rows:
            rows.extend(_normalize_payload_rows(manifest))
    elif isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, bytearray)):
        rows.extend(_normalize_payload_rows(manifest))
    return rows


def _load_result_rows(path: Path) -> list[dict[str, JsonValue]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [item for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for item in _normalize_payload_rows(json.loads(line))]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    return _normalize_payload_rows(json.loads(path.read_text(encoding="utf-8")))


def _normalize_payload_rows(payload: JsonValue) -> list[dict[str, JsonValue]]:
    if isinstance(payload, Mapping):
        for key in ("results", "rows", "items", "samples", "records", "raw_metric_table", "pairwise_preferences", "preferences"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping):
            per_metric = metrics.get("per_metric")
            if isinstance(per_metric, Mapping):
                return _rows_from_metric_mapping(per_metric)
            return _rows_from_metric_mapping(metrics)
        if any(key in payload for key in ("metric_id", "metric", "name", "score", "value", "prediction", "human_label", "human_preference")):
            return [dict(payload)]
        return _rows_from_metric_mapping(payload)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    return []


def _rows_from_metric_mapping(metrics: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for metric_id, value in metrics.items():
        if isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("metric_id", metric_id)
            rows.append(row)
        elif _as_float(value) is not None:
            rows.append({"metric_id": metric_id, "score": value})
    return rows


def _official_metric_results(
    benchmark_id: str,
    metric_ids: Sequence[str],
    rows: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    if benchmark_id == "genai-bench":
        return _genai_metric_results(benchmark_id, metric_ids, rows)
    normalized_rows = _normalized_score_rows(benchmark_id, metric_ids, rows)
    aggregate_rows = _aggregate_score_rows(benchmark_id, metric_ids, normalized_rows)
    all_rows = [*normalized_rows, *aggregate_rows]
    return [
        _score_result(metric_id, benchmark_id, all_rows)
        if any(row["metric_id"] == metric_id for row in all_rows)
        else _blocked_result(metric_id, benchmark_id)
        for metric_id in metric_ids
    ]


def _normalized_score_rows(
    benchmark_id: str,
    metric_ids: Sequence[str],
    rows: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    metric_set = set(metric_ids)
    normalized: list[dict[str, JsonValue]] = []
    for row in rows:
        metric_id = _canonical_metric_id(_first_metric_text(row), benchmark_id)
        score = _score_from_row(row)
        if metric_id in metric_set and score is not None:
            normalized.append(_normalized_score_row(metric_id, score, row))
        if metric_id is None:
            for key, value in row.items():
                wide_metric_id = _canonical_metric_id(str(key), benchmark_id)
                wide_score = _as_float(value)
                if wide_metric_id in metric_set and wide_score is not None:
                    normalized.append(_normalized_score_row(wide_metric_id, wide_score, row))
    return normalized


def _first_metric_text(row: Mapping[str, JsonValue]) -> str | None:
    return _first_text(row, ("metric_id", "metric", "metric_name", "name", "dimension", "leaderboard_key"))


def _canonical_metric_id(value: str | None, benchmark_id: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", stripped).strip("_").lower()
    compact = snake.replace("_", "")
    aliases = video_quality_spec(benchmark_id).get("metric_aliases", {})
    return aliases.get(snake) or aliases.get(compact) or snake


def _score_from_row(row: Mapping[str, JsonValue]) -> float | None:
    for key in ("score", "value", "normalized_value", "mean", "accuracy", "metric_value"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _as_float(value: JsonValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if NUMERIC_TEXT.match(text):
            return float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    return None


def _normalized_score_row(metric_id: str, score: float, row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "metric_id": metric_id,
        "score": score,
        "sample_id": _first_text(row, ("sample_id", "prompt_id", "id")),
        "category": _first_text(row, ("category", "prompt_type", "task", "domain", "dimension")),
        "source": dict(row),
    }


def _aggregate_score_rows(
    benchmark_id: str,
    metric_ids: Sequence[str],
    rows: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    metric_set = set(metric_ids)
    components = video_quality_spec(benchmark_id).get("aggregate_components", {})
    aggregate_rows: list[dict[str, JsonValue]] = []
    available_scores = _mean_by_metric(rows)
    for metric_id, component_ids in components.items():
        if metric_id in available_scores or metric_id not in metric_set:
            continue
        if not all(item in available_scores for item in component_ids):
            continue
        component_scores = [available_scores[item] for item in component_ids if item in available_scores]
        if len(component_scores) == len(component_ids):
            aggregate_rows.append(
                {
                    "metric_id": metric_id,
                    "score": sum(component_scores) / len(component_scores),
                    "sample_id": None,
                    "category": None,
                    "source": {
                        "aggregation": "mean",
                        "components": list(component_ids),
                    },
                }
            )
            available_scores[metric_id] = float(aggregate_rows[-1]["score"])
    return aggregate_rows


def _mean_by_metric(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["metric_id"]), []).append(float(row["score"]))
    return {
        metric_id: sum(scores) / len(scores)
        for metric_id, scores in grouped.items()
    }


def _score_result(
    metric_id: str,
    benchmark_id: str,
    rows: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    metric_rows = [row for row in rows if row["metric_id"] == metric_id]
    score = sum(float(row["score"]) for row in metric_rows) / len(metric_rows)
    return {
        "metric_id": metric_id,
        "score": score,
        "value": score,
        "status": "scored",
        "evidence": {
            "benchmark_id": benchmark_id,
            "protocol": "official_result_normalization",
            "provenance": "normalized from caller-provided official CSV/JSON outputs; no upstream code executed",
            "aggregation": "mean",
            "sample_count": len(metric_rows),
            "category_scores": _category_scores(metric_rows),
        },
        "blocked_reason": None,
    }


def _category_scores(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        category = row.get("category")
        if isinstance(category, str) and category.strip():
            grouped.setdefault(category.strip(), []).append(float(row["score"]))
    return {
        category: sum(values) / len(values)
        for category, values in sorted(grouped.items())
    }


def _genai_metric_results(
    benchmark_id: str,
    metric_ids: Sequence[str],
    rows: Sequence[Mapping[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    pairs = _preference_pairs(rows)
    if not pairs:
        normalized_rows = _normalized_score_rows(benchmark_id, metric_ids, rows)
        aggregate_rows = _aggregate_score_rows(benchmark_id, metric_ids, normalized_rows)
        all_rows = [*normalized_rows, *aggregate_rows]
        return [
            _score_result(metric_id, benchmark_id, all_rows)
            if any(row["metric_id"] == metric_id for row in all_rows)
            else _blocked_result(metric_id, benchmark_id)
            for metric_id in metric_ids
        ]
    computed = _genai_computed_scores(pairs)
    return [
        _preference_result(metric_id, computed[metric_id], pairs)
        if metric_id in computed
        else _blocked_result(metric_id, benchmark_id)
        for metric_id in metric_ids
    ]


def _preference_pairs(rows: Sequence[Mapping[str, JsonValue]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in rows:
        label = _normalized_preference_label(_first_text(row, ("human_label", "human_preference", "label", "preference", "winner", "answer", "ground_truth")))
        prediction = _normalized_preference_label(_first_text(row, ("prediction", "model_prediction", "judge_prediction", "predicted_label", "output", "response")))
        task = _normalized_task(_first_text(row, ("task", "task_name", "split", "modality", "category")))
        if label is not None and prediction is not None:
            pairs.append({"label": label, "prediction": prediction, "task": task or "unknown"})
    return pairs


def _normalized_preference_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", "", value.strip().lower())
    label_map = {
        "a>b": "a>b",
        "left>right": "a>b",
        "leftisbetter": "a>b",
        "a": "a>b",
        "left": "a>b",
        "b>a": "b>a",
        "right>left": "b>a",
        "rightisbetter": "b>a",
        "b": "b>a",
        "right": "b>a",
        "a=b=good": "a=b=good",
        "tiegood": "a=b=good",
        "bothgood": "a=b=good",
        "a=b=bad": "a=b=bad",
        "tiebad": "a=b=bad",
        "bothbad": "a=b=bad",
        "tie": "tie",
        "equal": "tie",
    }
    return label_map.get(text)


def _normalized_task(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").lower()
    task_map = {
        "image_generation": "image_generation",
        "text_to_image": "image_generation",
        "image": "image_generation",
        "image_editing": "image_editing",
        "image_edition": "image_editing",
        "editing": "image_editing",
        "video_generation": "video_generation",
        "text_to_video": "video_generation",
        "video": "video_generation",
    }
    return task_map.get(text, text or None)


def _genai_computed_scores(pairs: Sequence[Mapping[str, str]]) -> dict[str, float]:
    scores = {"pairwise_accuracy": _pair_accuracy(pairs)}
    task_scores: list[float] = []
    genai_task_metrics = video_quality_spec("genai-bench").get("genai_task_metrics", {})
    for task, metric_id in genai_task_metrics.items():
        task_pairs = [row for row in pairs if row["task"] == task]
        if task_pairs:
            score = _pair_accuracy(task_pairs)
            scores[metric_id] = score
            task_scores.append(score)
    if task_scores:
        scores["genai_bench_average"] = sum(task_scores) / len(task_scores)
    return scores


def _pair_accuracy(pairs: Sequence[Mapping[str, str]]) -> float:
    return sum(1 for row in pairs if row["label"] == row["prediction"]) / len(pairs)


def _preference_result(metric_id: str, score: float, pairs: Sequence[Mapping[str, str]]) -> dict[str, JsonValue]:
    task_counts: dict[str, int] = {}
    for row in pairs:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
    return {
        "metric_id": metric_id,
        "score": score,
        "value": score,
        "status": "scored",
        "evidence": {
            "benchmark_id": "genai-bench",
            "protocol": "pairwise_preference_accuracy",
            "provenance": "GenAI-Bench compares model predictions against official human preference labels",
            "sample_count": len(pairs),
            "task_counts": task_counts,
            "label_set": ["A>B", "B>A", "A=B=Good", "A=B=Bad"],
        },
        "blocked_reason": None,
    }


def _blocked_result(metric_id: str, benchmark_id: str) -> dict[str, JsonValue]:
    config = video_quality_spec(benchmark_id)
    blocked_reason = config["blocked_reason"]
    return {
        "metric_id": metric_id,
        "score": None,
        "value": None,
        "status": "blocked",
        "evidence": {
            "benchmark_id": benchmark_id,
            "reason_detail": config["blocked_detail"],
            "required_input_fields": list(config["required_inputs"]),
        },
        "blocked_reason": blocked_reason,
    }
