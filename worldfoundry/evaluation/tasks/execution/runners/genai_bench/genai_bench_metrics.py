"""GenAI-Bench metric aggregation helpers.

The official t2v_metrics repository contains dataset downloads and VQAScore
model inference scripts. WorldFoundry keeps the reusable metric aggregation for
already computed scores or preference rows in-tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
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


def _load_scores(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".pt":
        import torch

        return _as_numpy(torch.load(path, map_location="cpu"))
    raise ValueError(f"unsupported GenAI-Bench score format: {path}")


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
                item_indices.extend(prompt_to_items.get(f"{int(prompt_idx):05d}", prompt_to_items.get(str(prompt_idx), [])))
            if not item_indices:
                continue
            tag_results[tag] = {
                "alignment": _correlation_payload([our_scores[index] for index in item_indices], [human_scores[index] for index in item_indices])
            }
        result["per_skill"] = tag_results
    return result


def evaluate_genai_alignment_score_file(
    score_path: Path,
    metadata_path: Path,
    *,
    tags_path: Path | None = None,
    prompt_to_items_path: Path | None = None,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError(f"GenAI metadata must be a list of item records: {metadata_path}")
    tags = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path is not None and tags_path.is_file() else None
    prompt_to_items = (
        json.loads(prompt_to_items_path.read_text(encoding="utf-8"))
        if prompt_to_items_path is not None and prompt_to_items_path.is_file()
        else None
    )
    return evaluate_genai_alignment_scores(_load_scores(score_path), metadata, tags=tags, prompt_to_items=prompt_to_items)


def evaluate_genai_ranking_scores(
    scores: Any,
    items: list[dict[str, Any]],
    *,
    images_to_prompt_idx: list[int] | None = None,
    tags: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Evaluate GenAI image-ranking score tensors with the official 9-image layout."""

    our_scores = np.asarray(_mean_i2t_scores(scores, len(items)), dtype=float)
    human_scores = np.asarray(_human_alignment_from_items(items, key="human_score"), dtype=float)
    if len(items) % 9 != 0:
        raise ValueError("GenAI ranking evaluation expects 9 images per prompt")
    prompt_count = len(items) // 9
    our_per_prompt = our_scores.reshape(prompt_count, 9)
    human_per_prompt = human_scores.reshape(prompt_count, 9)
    argmax_human = np.argmax(human_per_prompt, axis=1)
    argmin_human = np.argmin(human_per_prompt, axis=1)
    ranking_accuracy = our_per_prompt[np.arange(prompt_count), argmax_human] > our_per_prompt[np.arange(prompt_count), argmin_human]

    result: dict[str, Any] = {
        "pearson": calc_pearson(human_scores, our_scores),
        "kendall_b": calc_metric(human_scores, our_scores, variant="tau_b"),
        "ranking_accuracy": float(ranking_accuracy.mean()) if ranking_accuracy.size else 0.0,
        "num_prompts": int(prompt_count),
    }
    if tags is not None and images_to_prompt_idx is not None:
        per_skill: dict[str, Any] = {}
        for tag, prompt_indices in tags.items():
            selected = [index for index, prompt_idx in enumerate(images_to_prompt_idx[:prompt_count]) if prompt_idx in prompt_indices]
            if selected:
                per_skill[tag] = float(ranking_accuracy[selected].mean())
        per_skill["all"] = result["ranking_accuracy"]
        result["per_skill"] = per_skill
    return result
