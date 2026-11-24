"""GenAI-Bench metric aggregation helpers.

The official t2v_metrics repository contains dataset downloads and VQAScore
model inference scripts. WorldFoundry keeps the reusable metric aggregation for
already computed scores or preference rows in-tree.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .correlation import calc_metric, calc_pearson


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalized_preference_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", "", value.strip().lower())
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    text = text.replace(">>", ">")
    return {
        "a>b": "a>b",
        "leftvote": "a>b",
        "left>right": "a>b",
        "leftisbetter": "a>b",
        "a": "a>b",
        "left": "a>b",
        "b>a": "b>a",
        "rightvote": "b>a",
        "right>left": "b>a",
        "rightisbetter": "b>a",
        "b": "b>a",
        "right": "b>a",
        "a=b=good": "a=b=good",
        "tievote": "a=b=good",
        "tiegood": "a=b=good",
        "bothgood": "a=b=good",
        "a=b=bad": "a=b=bad",
        "bothbad_vote": "a=b=bad",
        "tiebad": "a=b=bad",
        "bothbad": "a=b=bad",
        "a=b": "tie",
        "tie": "tie",
        "equal": "tie",
    }.get(text)


def _normalized_task(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").lower()
    return {
        "image_generation": "image_generation",
        "text_to_image": "image_generation",
        "image": "image_generation",
        "image_editing": "image_editing",
        "image_edition": "image_editing",
        "editing": "image_editing",
        "video_generation": "video_generation",
        "text_to_video": "video_generation",
        "video": "video_generation",
    }.get(text, text or None)


def evaluate_genai_preference_rows(
    rows: list[dict[str, Any]],
    *,
    default_task: str | None = None,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        label = _normalized_preference_label(
            _first_text(
                row,
                (
                    "human_label",
                    "human_preference",
                    "human_vote",
                    "label",
                    "preference",
                    "winner",
                    "answer",
                    "ground_truth",
                ),
            )
        )
        prediction = _normalized_preference_label(
            _first_text(
                row,
                (
                    "prediction",
                    "model_prediction",
                    "model_vote",
                    "judge_prediction",
                    "predicted_label",
                    "output",
                    "response",
                ),
            )
        )
        task = _normalized_task(
            _first_text(row, ("task", "task_name", "split", "modality", "category")) or default_task
        )
        if label is not None and prediction is not None:
            official_correct = row.get("correct")
            is_correct = (
                official_correct
                if isinstance(official_correct, bool)
                else label == prediction or (prediction == "tie" and label in {"a=b=good", "a=b=bad"})
            )
            pairs.append(
                {
                    "label": label,
                    "prediction": prediction,
                    "task": task or "unknown",
                    "correct": is_correct,
                }
            )
    correct = sum(1 for row in pairs if row["correct"])
    per_task: dict[str, dict[str, float | int]] = {}
    for task in sorted({row["task"] for row in pairs}):
        task_rows = [row for row in pairs if row["task"] == task]
        task_correct = sum(1 for row in task_rows if row["correct"])
        per_task[task] = {
            "accuracy": task_correct / len(task_rows) if task_rows else 0.0,
            "num_correct": task_correct,
            "num_total": len(task_rows),
        }
    return {
        "pairwise_accuracy": correct / len(pairs) if pairs else 0.0,
        "num_correct": correct,
        "num_total": len(pairs),
        "per_task": per_task,
    }


def _as_numpy(scores: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(scores, torch.Tensor):
            return scores.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(scores)


def _mean_i2t_scores(scores: Any, item_count: int) -> list[float]:
    arr = _as_numpy(scores)
    if arr.shape[0] != item_count:
        raise ValueError(f"score count mismatch: expected {item_count}, got {arr.shape[0]}")
    if arr.ndim == 1:
        return [float(value) for value in arr]
    reduced = arr.mean(axis=1)
    if reduced.ndim > 1:
        reduced = reduced.reshape(reduced.shape[0], -1).mean(axis=1)
    return [float(value) for value in reduced]


def _human_alignment_from_items(items: list[dict[str, Any]], key: str = "human_alignment") -> list[float]:
    values: list[float] = []
    for item in items:
        human = item.get(key, item.get("human_score"))
        if isinstance(human, list):
            values.append(float(np.asarray(human, dtype=float).mean()))
        else:
            values.append(float(human))
    return values


def _correlation_payload(our_scores: list[float], human_scores: list[float]) -> dict[str, Any]:
    pairwise = calc_metric(human_scores, our_scores, variant="pairwise_acc_with_tie_optimization")
    return {
        "pearson": calc_pearson(human_scores, our_scores),
        "kendall_b": calc_metric(human_scores, our_scores, variant="tau_b"),
        "pairwise_acc": pairwise,
    }


def evaluate_genai_alignment_scores(
    scores: Any,
    items: list[dict[str, Any]],
    *,
    tags: dict[str, list[int]] | None = None,
    prompt_to_items: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate GenAI image/video alignment score tensors against human ratings."""

    our_scores = _mean_i2t_scores(scores, len(items))
    human_scores = _human_alignment_from_items(items)
    result: dict[str, Any] = {"alignment": _correlation_payload(our_scores, human_scores)}

    if tags is not None and prompt_to_items is not None:
        tag_results: dict[str, Any] = {}
        for tag, prompt_indices in tags.items():
            item_indices: list[int] = []
            for prompt_idx in prompt_indices:
                item_indices.extend(
                    prompt_to_items.get(f"{int(prompt_idx):05d}", prompt_to_items.get(str(prompt_idx), []))
                )
            if not item_indices:
                continue
            tag_results[tag] = {
                "alignment": _correlation_payload(
                    [our_scores[index] for index in item_indices], [human_scores[index] for index in item_indices]
                )
            }
        result["per_skill"] = tag_results
    return result
