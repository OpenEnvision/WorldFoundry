"""CameraBench metric aggregation in WorldFoundry.

This module ports the metric aggregation behavior from
``t2v_metrics/camerabench`` while keeping only the metric aggregation needed by
the in-tree runner.
"""

from __future__ import annotations

import json
import math
import os
import string
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners._benchmark_metrics.formulas import (
    camera_binary_classification_metrics,
    camera_retrieval_metrics,
    camera_vqa_metrics,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def find_binary_score_files(score_dir: Path) -> list[Path]:
    return sorted(set(score_dir.glob("classification_scores_*.json"))) if score_dir.exists() else []


def find_vqa_retrieval_score_files(score_dir: Path) -> list[Path]:
    return sorted(set(score_dir.glob("vqa_retrieval_scores_*.json"))) if score_dir.exists() else []


def find_caption_score_files(score_dir: Path) -> list[Path]:
    return sorted(set(score_dir.glob("caption_results_*.json"))) if score_dir.exists() else []


def _score_file_identity(score_file: Path, score_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = score_data.get("metadata") if isinstance(score_data.get("metadata"), dict) else {}
    model_name = str(metadata.get("model_name") or "Unknown_Model")
    checkpoint = str(metadata.get("checkpoint") or "")
    split_name = str(metadata.get("split_name") or metadata.get("skill_name") or score_file.stem)
    if checkpoint:
        clean_checkpoint = checkpoint.split("/")[-1]
        unique_id = f"{model_name}_{clean_checkpoint}_{split_name}"
    else:
        unique_id = f"{model_name}_{split_name}"
    return unique_id, {"metadata": metadata, "model_name": model_name, "checkpoint": checkpoint, "split_name": split_name, "unique_id": unique_id}


def evaluate_binary_score_file(score_file: Path) -> tuple[str, dict[str, Any] | None]:
    score_data = _read_json(score_file)
    scores = score_data.get("scores")
    if not isinstance(scores, list):
        raise ValueError(f"missing list field 'scores' in {score_file}")
    unique_id, identity = _score_file_identity(score_file, score_data)
    metrics = camera_binary_classification_metrics([item for item in scores if isinstance(item, dict)])
    if int(metrics["num_samples"]) == 0:
        return unique_id, None
    metrics.update(identity)
    return unique_id, metrics


def evaluate_camerabench_binary(
    score_files: list[Path] | None = None,
    *,
    score_dir: Path | None = None,
    output_file: Path | None = None,
) -> dict[str, Any]:
    files = score_files or (find_binary_score_files(score_dir) if score_dir is not None else [])
    results: dict[str, Any] = {}
    for score_file in files:
        if not score_file.is_file():
            continue
        split_name, metrics = evaluate_binary_score_file(score_file)
        if metrics is not None:
            results[split_name] = metrics

    average_precisions = [float(result["average_precision"]) for result in results.values() if result is not None]
    aucs = [float(result["roc_auc"]) for result in results.values() if result is not None]
    overall = None
    if average_precisions:
        overall = {
            "mean_average_precision": float(np.mean(average_precisions)),
            "std_average_precision": float(np.std(average_precisions)) if len(average_precisions) > 1 else 0.0,
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            "evaluated_splits": len(average_precisions),
        }
    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_average_precision": None if overall is None else overall["mean_average_precision"],
        "overall_roc_auc": None if overall is None else overall["mean_roc_auc"],
        "total_splits": len(results),
        "evaluated_splits": len(average_precisions),
        "overall_statistics": overall,
        "results_by_split": results,
    }
    if output_file is not None:
        _write_json(output_file, summary)
    return summary


def evaluate_vqa_retrieval_score_file(score_file: Path, *, mode: str = "both") -> tuple[str, dict[str, Any] | None]:
    if mode not in {"vqa", "retrieval", "both"}:
        raise ValueError("mode must be one of: vqa, retrieval, both")
    score_data = _read_json(score_file)
    scores = score_data.get("scores")
    if not isinstance(scores, list):
        raise ValueError(f"missing list field 'scores' in {score_file}")
    records = [item for item in scores if isinstance(item, dict)]
    unique_id, identity = _score_file_identity(score_file, score_data)
    if not records:
        return unique_id, None

    result = dict(identity)
    if mode in {"vqa", "both"}:
        result["vqa"] = camera_vqa_metrics(records)
    if mode in {"retrieval", "both"}:
        result["retrieval"] = camera_retrieval_metrics(records)
    if "vqa" not in result and "retrieval" not in result:
        return unique_id, None
    return unique_id, result


def evaluate_camerabench_vqa_retrieval(
    score_files: list[Path] | None = None,
    *,
    score_dir: Path | None = None,
    mode: str = "both",
    output_file: Path | None = None,
) -> dict[str, Any]:
    files = score_files or (find_vqa_retrieval_score_files(score_dir) if score_dir is not None else [])
    results: dict[str, Any] = {}
    for score_file in files:
        if not score_file.is_file():
            continue
        split_name, metrics = evaluate_vqa_retrieval_score_file(score_file, mode=mode)
        if metrics is not None:
            results[split_name] = metrics

    overall_stats: dict[str, Any] = {}
    valid_results = [result for result in results.values() if isinstance(result, dict)]
    if mode in {"vqa", "both"}:
        vqa_rows = [result["vqa"] for result in valid_results if "vqa" in result]
        if vqa_rows:
            binary = [float(row["binary_acc"]) for row in vqa_rows]
            question = [float(row["question_acc"]) for row in vqa_rows]
            overall_stats["vqa"] = {
                "mean_binary_acc": float(np.mean(binary)),
                "std_binary_acc": float(np.std(binary)) if len(binary) > 1 else 0.0,
                "mean_question_acc": float(np.mean(question)),
                "std_question_acc": float(np.std(question)) if len(question) > 1 else 0.0,
                "evaluated_splits": len(vqa_rows),
            }
    if mode in {"retrieval", "both"}:
        retrieval_rows = [result["retrieval"] for result in valid_results if "retrieval" in result]
        if retrieval_rows:
            text_scores = [float(row["text"]) for row in retrieval_rows]
            image_scores = [float(row["image"]) for row in retrieval_rows]
            group_scores = [float(row["group"]) for row in retrieval_rows]
            overall_stats["retrieval"] = {
                "mean_text": float(np.mean(text_scores)),
                "std_text": float(np.std(text_scores)) if len(text_scores) > 1 else 0.0,
                "mean_image": float(np.mean(image_scores)),
                "std_image": float(np.std(image_scores)) if len(image_scores) > 1 else 0.0,
                "mean_group": float(np.mean(group_scores)),
                "std_group": float(np.std(group_scores)) if len(group_scores) > 1 else 0.0,
                "evaluated_splits": len(retrieval_rows),
            }

    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": mode,
        "total_splits": len(results),
        "evaluated_splits": len(valid_results),
        "overall_statistics": overall_stats,
        "results_by_split": results,
    }
    if "vqa" in overall_stats:
        summary["overall_binary_acc"] = overall_stats["vqa"]["mean_binary_acc"]
        summary["overall_question_acc"] = overall_stats["vqa"]["mean_question_acc"]
    if "retrieval" in overall_stats:
        summary["overall_retrieval_text"] = overall_stats["retrieval"]["mean_text"]
        summary["overall_retrieval_image"] = overall_stats["retrieval"]["mean_image"]
        summary["overall_retrieval_group"] = overall_stats["retrieval"]["mean_group"]
    if output_file is not None:
        _write_json(output_file, summary)
    return summary


def _preprocess_text(text: Any) -> list[str]:
    if not text:
        return []
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def calculate_spice_score(reference: Any, candidate: Any) -> float:
    ref_words = set(_preprocess_text(reference))
    cand_words = set(_preprocess_text(candidate))
    if not cand_words:
        return 0.0
    intersection = ref_words.intersection(cand_words)
    precision = len(intersection) / len(cand_words)
    recall = len(intersection) / len(ref_words) if ref_words else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def calculate_cider_score(reference: Any, candidate: Any) -> float:
    ref_counts = Counter(_preprocess_text(reference))
    cand_counts = Counter(_preprocess_text(candidate))
    all_words = set(ref_counts) | set(cand_counts)
    if not all_words:
        return 0.0
    dot = sum(ref_counts[word] * cand_counts[word] for word in all_words)
    ref_mag = math.sqrt(sum(value**2 for value in ref_counts.values()))
    cand_mag = math.sqrt(sum(value**2 for value in cand_counts.values()))
    return 0.0 if ref_mag == 0 or cand_mag == 0 else dot / (ref_mag * cand_mag)


def calculate_bleu2_score(reference: Any, candidate: Any) -> float:
    if not reference or not candidate:
        return 0.0
    ref_tokens = _preprocess_text(reference)
    cand_tokens = _preprocess_text(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return float(sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5), smoothing_function=SmoothingFunction().method1))
    except Exception:
        unigram_precision = len(set(ref_tokens) & set(cand_tokens)) / len(cand_tokens)
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:], strict=False))
        cand_bigrams = set(zip(cand_tokens[:-1], cand_tokens[1:], strict=False))
        bigram_precision = len(ref_bigrams & cand_bigrams) / len(cand_bigrams) if cand_bigrams else 0.0
        return math.sqrt(max(unigram_precision, 0.0) * max(bigram_precision, 0.0))


def _lcs_length(left: list[str], right: list[str]) -> int:
    prev = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(prev[index - 1] + 1 if token == other else max(prev[index], current[-1]))
        prev = current
    return prev[-1]


def calculate_rouge_l_score(reference: Any, candidate: Any) -> float:
    if not reference or not candidate:
        return 0.0
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(str(reference), str(candidate))["rougeL"].fmeasure)
    except Exception:
        ref_tokens = _preprocess_text(reference)
        cand_tokens = _preprocess_text(candidate)
        if not ref_tokens or not cand_tokens:
            return 0.0
        lcs = _lcs_length(ref_tokens, cand_tokens)
        precision = lcs / len(cand_tokens)
        recall = lcs / len(ref_tokens)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def calculate_meteor_score(reference: Any, candidate: Any) -> float:
    ref_tokens = _preprocess_text(reference)
    cand_tokens = _preprocess_text(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_unigrams = set(ref_tokens)
    cand_unigrams = set(cand_tokens)
    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:], strict=False))
    cand_bigrams = set(zip(cand_tokens[:-1], cand_tokens[1:], strict=False))
    unigram_matches = len(ref_unigrams & cand_unigrams)
    bigram_matches = len(ref_bigrams & cand_bigrams)
    unigram_precision = unigram_matches / len(cand_unigrams) if cand_unigrams else 0.0
    unigram_recall = unigram_matches / len(ref_unigrams) if ref_unigrams else 0.0
    bigram_precision = bigram_matches / len(cand_bigrams) if cand_bigrams else 0.0
    bigram_recall = bigram_matches / len(ref_bigrams) if ref_bigrams else 0.0
    precision = 0.8 * unigram_precision + 0.2 * bigram_precision
    recall = 0.8 * unigram_recall + 0.2 * bigram_recall
    return 0.0 if precision + recall == 0 else (10 * precision * recall) / (recall + 9 * precision)


def calculate_generative_match(reference: Any, candidate: Any, *, api_key: str | None = None, retries: int = 3, delay: float = 2.0) -> float | None:
    if not reference or not candidate:
        return 0.0
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    client = OpenAI(api_key=api_key)
    prompt = f"Reference caption: '{reference}'\nCandidate caption: '{candidate}'\n\nDoes the candidate caption match the reference caption? Answer Yes or No."
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=5,
            )
            content = (response.choices[0].message.content or "").strip().lower()
            if content.startswith("yes"):
                return 1.0
            if content.startswith("no"):
                return 0.0
            logprobs = response.choices[0].logprobs
            if logprobs and logprobs.content:
                for token_info in logprobs.content[0].top_logprobs:
                    if token_info.token.strip().lower() == "yes":
                        return float(np.exp(token_info.logprob))
            return 0.1
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return 0.5


def evaluate_caption_file(score_file: Path, *, api_key: str | None = None) -> dict[str, Any] | None:
    data = _read_json(score_file)
    captions = data.get("captions")
    if not isinstance(captions, list):
        raise ValueError(f"missing list field 'captions' in {score_file}")
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    spice_scores: list[float] = []
    cider_scores: list[float] = []
    bleu2_scores: list[float] = []
    rouge_l_scores: list[float] = []
    meteor_scores: list[float] = []
    gen_match_scores: list[float] = []
    valid_samples = 0

    for item in captions:
        if not isinstance(item, dict) or item.get("error") is not None:
            continue
        reference = item.get("reference_answer") or item.get("reference") or item.get("answer")
        candidate = item.get("generated_caption") or item.get("candidate") or item.get("caption")
        if not reference or not candidate:
            continue
        valid_samples += 1
        spice_scores.append(calculate_spice_score(reference, candidate))
        cider_scores.append(calculate_cider_score(reference, candidate))
        bleu2_scores.append(calculate_bleu2_score(reference, candidate))
        rouge_l_scores.append(calculate_rouge_l_score(reference, candidate))
        meteor_scores.append(calculate_meteor_score(reference, candidate))
        gen_match = calculate_generative_match(reference, candidate, api_key=api_key) if api_key else None
        if gen_match is not None:
            gen_match_scores.append(gen_match)

    if valid_samples == 0:
        return None
    return {
        "model": metadata.get("model_name", "unknown"),
        "checkpoint": metadata.get("checkpoint", ""),
        "file_path": str(score_file),
        "total_samples": len(captions),
        "valid_samples": valid_samples,
        "spice": float(np.mean(spice_scores)) if spice_scores else 0.0,
        "cider": float(np.mean(cider_scores)) if cider_scores else 0.0,
        "bleu2": float(np.mean(bleu2_scores)) if bleu2_scores else 0.0,
        "rouge_l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
        "gen_match": float(np.mean(gen_match_scores)) if gen_match_scores else None,
    }


def evaluate_camerabench_caption(
    score_files: list[Path] | None = None,
    *,
    score_dir: Path | None = None,
    output_file: Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    files = score_files or (find_caption_score_files(score_dir) if score_dir is not None else [])
    results = [result for score_file in files if score_file.is_file() for result in [evaluate_caption_file(score_file, api_key=api_key)] if result]
    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_files": len(files),
        "total_models": len(results),
        "gpt_judge_enabled": bool(api_key),
        "results": results,
    }
    if output_file is not None:
        _write_json(output_file, summary)
    return summary


def evaluate_camerabench_from_score_dir(
    score_dir: Path,
    *,
    output_dir: Path,
    task: str = "all",
    mode: str = "both",
    no_gpt: bool = True,
    openai_api_key: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if task not in {"binary", "vqa_retrieval", "caption", "all"}:
        raise ValueError(f"unknown CameraBench task: {task}")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_results: dict[str, dict[str, Any]] = {}
    commands: list[str] = []
    selected = ("binary", "vqa_retrieval", "caption") if task == "all" else (task,)
    for item in selected:
        output_file = output_dir / f"camerabench_{item}_results.json"
        if item == "binary":
            task_results[item] = evaluate_camerabench_binary(score_dir=score_dir, output_file=output_file)
        elif item == "vqa_retrieval":
            task_results[item] = evaluate_camerabench_vqa_retrieval(score_dir=score_dir, mode=mode, output_file=output_file)
        elif item == "caption":
            api_key = None if no_gpt else openai_api_key
            task_results[item] = evaluate_camerabench_caption(score_dir=score_dir, output_file=output_file, api_key=api_key)
        commands.append(f"worldfoundry.evaluation.tasks.execution.runners.camerabench.camerabench_metrics:{item}")
    return task_results, commands
