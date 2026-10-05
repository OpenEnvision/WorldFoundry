"""
This module provides utilities for running and normalizing results from the TIGER-Lab VideoScore benchmark.

It handles setting up the environment, executing the official VideoScore evaluation script
(or a bounded, in-process version), and transforming its raw output into a standardized
WorldFoundry scorecard format. This includes extracting scores for various aspects
(e.g., visual quality, temporal consistency, text-to-video alignment) and calculating
an overall average, along with detailed metadata about the run and artifacts.
"""
from __future__ import annotations



from worldfoundry.evaluation.tasks.execution.framework.runner_common import SCORECARD_SCHEMA_VERSION, VIDEO_SUFFIXES

import argparse
import ast
import importlib
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

from worldfoundry.core.execution.process import run_logged_subprocess
from worldfoundry.evaluation.tasks.execution.framework.benchmark_data import (
    build_generated_video_manifest,
    build_local_dataset_manifest,
    discover_metadata_records,
    expected_stems_from_records,
)
from worldfoundry.evaluation.tasks.execution.framework.io import (
    env_path,
    load_json,
    scalar_number,
    utc_now_iso,
    write_json,
    write_jsonl,
)

# Import utility functions from worldfoundry.
from worldfoundry.evaluation.utils import REPO_ROOT, worldfoundry_hfd_dataset_root

# Define various constant paths and IDs used throughout the script.
IN_TREE_VIDEOSCORE_ROOT = (
    REPO_ROOT
    / "worldfoundry"
    / "evaluation"
    / "tasks"
    / "execution"
    / "runners"
    / "videoscore"
    / "runtime"
    / "videoscore"
)
DEFAULT_VIDEOSCORE_ROOT = env_path("WORLDFOUNDRY_VIDEOSCORE_ROOT", IN_TREE_VIDEOSCORE_ROOT)
LOCAL_VIDEOSCORE_BENCH_ROOT = worldfoundry_hfd_dataset_root() / "TIGER-Lab__VideoScore-Bench"
HF_DATASET_ID = "TIGER-Lab/VideoFeedback"
HF_DATASET_CONFIG = "real"
HF_DATASET_SPLIT = "train"
HF_DATASET_EXPECTED_ROWS = 4000
ASPECTS = (
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
)
ASPECT_LABELS = (
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency",
)
METRIC_ORDER = (*ASPECTS, "videoscore_average")
METRIC_ALIASES = {
    "visual_quality": "visual_quality",
    "visual quality": "visual_quality",
    "temporal_consistency": "temporal_consistency",
    "temporal consistency": "temporal_consistency",
    "dynamic_degree": "dynamic_degree",
    "dynamic degree": "dynamic_degree",
    "text_to_video_alignment": "text_to_video_alignment",
    "text-to-video alignment": "text_to_video_alignment",
    "t2v_alignment": "text_to_video_alignment",
    "factual_consistency": "factual_consistency",
    "factual consistency": "factual_consistency",
    "videoscore_average": "videoscore_average",
    "average": "videoscore_average",
    "overall": "videoscore_average",
}


def default_videoscore_root() -> Path | None:
    """Return the in-tree VideoScore runtime or an explicit caller override."""
    if DEFAULT_VIDEOSCORE_ROOT is not None and DEFAULT_VIDEOSCORE_ROOT.is_dir():
        return DEFAULT_VIDEOSCORE_ROOT
    if IN_TREE_VIDEOSCORE_ROOT.is_dir():
        return IN_TREE_VIDEOSCORE_ROOT
    return DEFAULT_VIDEOSCORE_ROOT


def default_model_repo_name() -> str:
    """
    Determines the default model repository name for VideoScore.

    Checks for a locally available model, otherwise falls back to the Hugging Face Hub ID.

    Returns:
        str: The determined model repository name (local path or Hugging Face ID).
    """
    return "TIGER-Lab/VideoScore-v1.1"


def default_bench_data_root() -> Path | None:
    """
    Determines the default root directory for VideoScore benchmark data.

    Checks for a locally available benchmark data root.

    Returns:
        Path | None: The determined benchmark data root directory, or None if not found locally.
    """
    if LOCAL_VIDEOSCORE_BENCH_ROOT.is_dir():
        return LOCAL_VIDEOSCORE_BENCH_ROOT
    return None


# Configure a partial function for scalar number extraction with common keys and list handling.
scalar = partial(
    scalar_number,
    dict_keys=("score", "raw_score", "value", "mean", "average", "avg", "overall"),
    list_mode="mean",
)


def parse_score_list(value: Any) -> list[float]:
    """
    Parses a given value into a list of floating-point scores.

    Handles string representations of lists/tuples, and extracts numeric values
    from iterables. Non-numeric items are ignored.

    Args:
        value (Any): The input value, potentially a string, list, tuple, or other.

    Returns:
        list[float]: A list of extracted numeric scores.
    """
    # Attempt to literal_eval strings to convert them into Python objects (e.g., list, tuple).
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    # If the value is not a list or tuple after potential conversion, return an empty list.
    if not isinstance(value, list) and not isinstance(value, tuple):
        return []
    scores: list[float] = []
    # Iterate through items and use the scalar utility to extract numbers.
    for item in value:
        number = scalar(item)
        if number is not None:
            scores.append(number)
    return scores


def normalize_videoscore_value(raw_score: float | None) -> float | None:
    """
    Normalizes a raw VideoScore score from the [1.0, 4.0] range to [0.0, 1.0].

    The normalization formula is (raw_score - 1.0) / 3.0. Scores are clamped
    to ensure they stay within [0.0, 1.0].

    Args:
        raw_score (float | None): The raw score to normalize, or None.

    Returns:
        float | None: The normalized score between 0.0 and 1.0, or None if input was None.
    """
    if raw_score is None:
        return None
    # Apply the normalization formula and clamp the result.
    return max(0.0, min(1.0, (raw_score - 1.0) / 3.0))


def result_rows(raw_results: Any) -> list[dict[str, Any]]:
    """
    Extracts a list of result dictionaries from the raw VideoScore output.

    Handles cases where the raw results are a list of dicts, or a dict containing
    a list of results under common keys like "results", "samples", etc.

    Args:
        raw_results (Any): The raw output from the VideoScore evaluation.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, each representing a sample's results.
    """
    if isinstance(raw_results, list):
        return [row for row in raw_results if isinstance(row, dict)]
    # If raw_results is a dictionary, check for common keys holding a list of results.
    if isinstance(raw_results, dict):
        for key in ("results", "samples", "per_sample_scores", "per_sample_metrics"):
            value = raw_results.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def extract_scores_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Extracts aggregated scores for each aspect from a list of sample result rows.

    Calculates the mean raw score for each aspect based on "ans", "scores", or
    "aspect_scores" fields in the rows. Also computes a "videoscore_average".

    Args:
        rows (list[dict[str, Any]]): A list of dictionaries, each representing a sample's results.

    Returns:
        dict[str, dict[str, Any]]: A dictionary where keys are metric IDs (aspects
                                    or 'videoscore_average') and values are dicts
                                    containing 'raw_score', 'source', and 'sample_count'.
    """
    # Initialize a dictionary to store lists of scores for each aspect.
    aspect_values: dict[str, list[float]] = {aspect: [] for aspect in ASPECTS}
    for row in rows:
        # Extract scores from common fields like "ans", "scores", "aspect_scores".
        scores = parse_score_list(row.get("ans") or row.get("scores") or row.get("aspect_scores"))
        # Assign extracted scores to their respective aspects, limited by the number of ASPECTS.
        for index, score in enumerate(scores[: len(ASPECTS)]):
            aspect_values[ASPECTS[index]].append(score)

    extracted: dict[str, dict[str, Any]] = {}
    # Calculate the mean raw score for each aspect.
    for aspect, values in aspect_values.items():
        if values:
            extracted[aspect] = {
                "raw_score": sum(values) / len(values),
                "source": "mean_official_ans_scores",
                "sample_count": len(values),
            }
    # Calculate the overall VideoScore average from the component aspect scores.
    component_scores = [item["raw_score"] for item in extracted.values() if item["raw_score"] is not None]
    if component_scores:
        # Compute the average of the available component scores.
        extracted["videoscore_average"] = {
            "raw_score": sum(component_scores) / len(component_scores),
            "source": "computed_from_videoscore_aspects",
            "sample_count": min((item.get("sample_count", 0) for item in extracted.values()), default=0),
        }
    return extracted


def extract_scores_from_mapping(raw_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Extracts scores directly from a dictionary mapping metric names to values.

    This function is used when the raw results are already structured as a dictionary
    where keys might be metric names (or aliases) and values are the scores.
    It also computes a "videoscore_average" if aspect scores are present.

    Args:
        raw_results (dict[str, Any]): A dictionary containing raw scores.

    Returns:
        dict[str, dict[str, Any]]: A dictionary where keys are normalized metric IDs
                                    and values are dicts containing 'raw_score',
                                    'source' (original key), and 'sample_count'.
    """
    # Prioritize a nested "scores" dictionary if present, otherwise use the top-level dict.
    payload = raw_results.get("scores") if isinstance(raw_results.get("scores"), dict) else raw_results
    extracted: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in payload.items():
        # Map raw keys to standardized metric IDs using METRIC_ALIASES.
        metric_id = METRIC_ALIASES.get(str(raw_key))
        # Skip if no alias found or if metric_id is already processed.
        if not metric_id or metric_id in extracted:
            continue
        raw_score = scalar(raw_value)
        extracted[metric_id] = {
            "raw_score": raw_score,
            "source": raw_key,
            "sample_count": None,  # Sample count is not typically available for aggregated scores.
        }
    # Compute the overall VideoScore average from component aspect scores if not already present.
    component_scores = [
        item["raw_score"]
        for metric_id, item in extracted.items()
        if metric_id in ASPECTS and item["raw_score"] is not None
    ]
    if component_scores and "videoscore_average" not in extracted:
        extracted["videoscore_average"] = {
            "raw_score": sum(component_scores) / len(component_scores),
            "source": "computed_from_videoscore_aspects",
            "sample_count": None,
        }
    return extracted


def extract_scores(raw_results: Any) -> dict[str, dict[str, Any]]:
    """
    High-level function to extract scores from various raw VideoScore output formats.

    Tries to extract scores from a list of rows first, then from a dictionary mapping.

    Args:
        raw_results (Any): The raw output from the VideoScore evaluation.

    Returns:
        dict[str, dict[str, Any]]: A dictionary of extracted metric scores,
                                    or an empty dictionary if no scores could be extracted.
    """
    rows = result_rows(raw_results)
    if rows:
        return extract_scores_from_rows(rows)
    # If no rows were found, try to extract from a direct dictionary mapping.
    if isinstance(raw_results, dict):
        return extract_scores_from_mapping(raw_results)
    return {}


def per_sample_rows(raw_results: Any) -> list[dict[str, Any]]:
    """
    Extracts and normalizes per-sample result data from raw VideoScore output.

    For each sample, it extracts ID, prompt, reference scores, and maps aspect
    scores to their respective names.

    Args:
        raw_results (Any): The raw output from the VideoScore evaluation.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, each representing normalized
                              per-sample data including sample_id, prompt,
                              reference_scores, aspect_scores, and the original raw row.
    """
    rows = result_rows(raw_results)
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        # Extract scores from common fields.
        scores = parse_score_list(row.get("ans") or row.get("scores") or row.get("aspect_scores"))
        # Construct a normalized per-sample row.
        normalized_rows.append(
            {
                "sample_id": row.get("id") or row.get("sample_id"),
                "prompt": row.get("text") or row.get("prompt"),
                "reference_scores": row.get("ref"),
                # Map extracted scores to their corresponding aspect names.
                "aspect_scores": {ASPECTS[index]: score for index, score in enumerate(scores[: len(ASPECTS)])},
                "raw": row,
            }
        )
    return normalized_rows


def normalize_videoscore_results(
    raw_results: Any,
    *,
    benchmark_id: str,
    output_dir: Path,
    upstream_results_path: Path,
    dataset_root: Path | None,
    frames_dir: Path | None,
    command: list[str] | None,
    duration_seconds: float | None,
    returncode: int,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    """
    Normalizes raw VideoScore results into a WorldFoundry scorecard format.

    This function processes the raw output, extracts and normalizes scores,
    generates manifests for generated videos and datasets, and compiles
    all information into a comprehensive scorecard dictionary.

    Args:
        raw_results (Any): The raw output from the VideoScore evaluation.
        benchmark_id (str): Identifier for the benchmark run.
        output_dir (Path): Directory where output artifacts should be saved.
        upstream_results_path (Path): Path to the original raw results file.
        dataset_root (Path | None): Root directory of the local dataset used.
        frames_dir (Path | None): Directory containing generated video frames.
        command (list[str] | None): The command used to run the upstream evaluation, if any.
        duration_seconds (float | None): Duration of the upstream evaluation run in seconds.
        returncode (int): The return code of the upstream process.
        stdout_path (Path | None): Path to the captured stdout of the upstream process.
        stderr_path (Path | None): Path to the captured stderr of the upstream process.

    Returns:
        dict[str, Any]: The generated WorldFoundry scorecard dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define paths for all output artifacts.
    scorecard_path = output_dir / "scorecard.json"
    raw_metric_table_path = output_dir / "raw_metric_table.jsonl"
    per_sample_scores_path = output_dir / "per_sample_scores.jsonl"
    generated_video_manifest_path = output_dir / "generated_video_manifest.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"

    # Extract aggregated and per-sample scores.
    extracted_scores = extract_scores(raw_results)
    sample_rows = per_sample_rows(raw_results)
    # Discover metadata and build manifests for dataset and generated videos.
    dataset_records = discover_metadata_records(dataset_root)
    expected_stems = expected_stems_from_records(dataset_records, ("id", "video_id", "video link", "video", "path"))
    generated_video_manifest = build_generated_video_manifest(
        frames_dir,
        expected_count=HF_DATASET_EXPECTED_ROWS,
        expected_stems=expected_stems,
    )
    dataset_manifest = build_local_dataset_manifest(
        dataset_root,
        dataset_id=HF_DATASET_ID,
        config=HF_DATASET_CONFIG,
        split=HF_DATASET_SPLIT,
        expected_rows=HF_DATASET_EXPECTED_ROWS,
        media_extensions=(".mp4", ".webm", ".mov", ".mkv", ".avi", ".jpg", ".jpeg", ".png", ".webp"),
    )
    metric_rows: list[dict[str, Any]] = []
    per_metric: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}

    # Process each metric according to a predefined order.
    for metric_id in METRIC_ORDER:
        item = extracted_scores.get(metric_id, {})
        raw_score = item.get("raw_score")
        normalized_score = normalize_videoscore_value(raw_score)
        row = {
            "metric_id": metric_id,
            "available": raw_score is not None,
            "raw_score": raw_score,
            "normalized_score": normalized_score,
            "raw_score_range": [1.0, 4.0],  # VideoScore's official range
            "source": item.get("source"),
            "sample_count": item.get("sample_count"),
        }
        if raw_score is None:
            row["reason"] = "score_not_found_in_videoscore_results"
        else:
            leaderboard[metric_id] = raw_score
        metric_rows.append(row)
        per_metric[metric_id] = row

    # Write all intermediate and final artifacts.
    write_jsonl(raw_metric_table_path, metric_rows)
    write_jsonl(per_sample_scores_path, sample_rows)
    write_json(generated_video_manifest_path, generated_video_manifest)
    write_json(dataset_manifest_path, dataset_manifest)
    available_count = sum(1 for row in metric_rows if row["available"])

    # Determine various status flags for the scorecard.
    normalization_ok = returncode == 0 and available_count > 0
    normalizer_only = command is None
    # eval_videoscore.py produces per-video aspect scores. The official
    # VideoScore-Bench SPCC/pairwise protocols are separate evaluation steps.
    component_verified = command is not None and normalization_ok

    # Assemble the final scorecard dictionary.
    run_status = (
        "official_component_verified"
        if component_verified
        else "official_results_imported"
        if normalizer_only and normalization_ok
        else "failed"
    )
    scorecard = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "run": {
            "status": run_status,
            "started_at": utc_now_iso(),
            "runner": "benchmark_zoo_videoscore_official_runner",
            "command": command,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
        },
        "benchmark": {
            "benchmark_id": benchmark_id,
            "name": "VideoScore",
            "contract_only": False,
            "requires_upstream_runtime": True,
            "requires_model_weights": True,
        },
        "dataset": {
            "hf_dataset_id": HF_DATASET_ID,
            "hf_config": HF_DATASET_CONFIG,
            "hf_split": HF_DATASET_SPLIT,
            "expected_rows": HF_DATASET_EXPECTED_ROWS,
            "sample_count": len(sample_rows),
            "local_dataset": dataset_manifest,
            "generated_videos": generated_video_manifest,
        },
        "eligibility": {
            "leaderboard_valid": False,
            "reasons": [
                "official VideoScore runtime validation; benchmark-level SPCC/pairwise evidence requires the upstream evaluation protocol",
            ],
        },
        "generation": {
            "successful": len(sample_rows),
            "failed": 0,
        },
        "metrics": {
            "leaderboard": leaderboard,
            "groups": {
                "videoscore_aspects": [metric for metric in ASPECTS if metric in leaderboard],
            },
            "per_metric": per_metric,
            "summary": {
                "sample_count": len(sample_rows),
                "metric_count": len(metric_rows),
                "available_metrics": available_count,
                "failed_metrics": len(metric_rows) - available_count,
            },
        },
        "evaluation": {
            "available": normalization_ok,
            "kind": "official_videoscore_component" if component_verified else "videoscore_results_imported",
            "upstream_results": str(upstream_results_path),
            "dataset_root": None if dataset_root is None else str(dataset_root.resolve()),
            "frames_dir": None if frames_dir is None else str(frames_dir.resolve()),
            "num_results": len(sample_rows),
            "leaderboard_metrics": leaderboard,
            "skip_count": len(metric_rows) - available_count,
        },
        "artifacts": {
            "scorecard": str(scorecard_path.resolve()),
            "raw_metric_table": str(raw_metric_table_path.resolve()),
            "per_sample_scores": str(per_sample_scores_path.resolve()),
            "generated_video_manifest": str(generated_video_manifest_path.resolve()),
            "dataset_manifest": str(dataset_manifest_path.resolve()),
            "upstream_results": str(upstream_results_path.resolve()),
            "upstream_stdout": None if stdout_path is None else str(stdout_path.resolve()),
            "upstream_stderr": None if stderr_path is None else str(stderr_path.resolve()),
        },
        "validation": {
            "normalizer_only": normalizer_only,
            "official_runtime_executed": command is not None,
            "official_component_verified": component_verified,
            "benchmark_protocol_executed": False,
            "official_results_imported": normalizer_only and normalization_ok,
        },
        "official_component_verified": component_verified,
        "official_benchmark_verified": False,
        "integration_evidence": component_verified,
        "normalization_ok": normalization_ok,
        "official_results_imported": normalizer_only and normalization_ok,
    }
    write_json(scorecard_path, scorecard)
    return scorecard


def build_name_postfix_arg(bench_name: str) -> str:
    """
    Builds the `--name_postfixs` argument for the official VideoScore script.

    Args:
        bench_name (str): The name of the benchmark.

    Returns:
        str: The formatted name postfix string.
    """
    return f"[{bench_name}]"


def build_official_command(args: argparse.Namespace, result_file: Path) -> list[str]:
    """
    Builds the command list to execute the official VideoScore evaluation script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        result_file (Path): The path where the upstream script should save its results.

    Returns:
        list[str]: A list of strings representing the command and its arguments.
    """
    return [
        args.python,
        str(args.videoscore_root / "benchmark" / "eval_videoscore.py"),
        "--model_repo_name",
        args.model_repo_name,
        "--data_repo_name",
        args.data_repo_name,
        "--frames_dir",
        str(args.frames_dir),
        "--name_postfixs",
        build_name_postfix_arg(args.bench_name),
        "--result_file",
        str(result_file),
        "--bench_name",
        args.bench_name,
    ]


def _as_list(value: Any) -> list[Any]:
    """
    Converts a single value or iterable into a list.

    Args:
        value (Any): The input value. Can be None, a list, a tuple, or an object
                     with a 'tolist' method (e.g., NumPy array).

    Returns:
        list[Any]: A list representation of the input value.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _conversation_value(row: dict[str, Any], index: int) -> str:
    """
    Extracts a specific conversation value from a row's 'conversations' list.

    Args:
        row (dict[str, Any]): A dictionary representing a data row.
        index (int): The index of the conversation item to extract.

    Returns:
        str: The extracted conversation value, or an empty string if not found.
    """
    conversations = _as_list(row.get("conversations"))
    if index >= len(conversations):
        return ""
    item = conversations[index]
    if isinstance(item, dict):
        return str(item.get("value") or "")
    return ""


def video_prompt_from_row(row: dict[str, Any]) -> str:
    """
    Extracts the video prompt text from a data row.

    Prioritizes the "text prompt" field, then attempts to parse it from the
    first conversation turn if available.

    Args:
        row (dict[str, Any]): A dictionary representing a data row.

    Returns:
        str: The extracted video prompt.
    """
    if row.get("text prompt"):
        return str(row["text prompt"])
    human_text = _conversation_value(row, 0)
    # Attempt to parse prompt from a specific marker in the conversation text.
    marker = 'text prompt is "'
    if marker in human_text:
        return human_text.split(marker, 1)[1].split('",\n', 1)[0]
    return human_text.strip()


def reference_scores_from_row(row: dict[str, Any], bench_name: str) -> list[float]:
    """
    Extracts reference scores from a data row based on the benchmark name.

    For "video_feedback", it looks for scores based on `ASPECT_LABELS`.
    Otherwise, it looks for a "score_list" field.

    Args:
        row (dict[str, Any]): A dictionary representing a data row.
        bench_name (str): The name of the benchmark.

    Returns:
        list[float]: A list of extracted reference scores.
    """
    if bench_name == "video_feedback":
        # For 'video_feedback', collect scores using predefined aspect labels.
        scores = [scalar(row.get(label)) for label in ASPECT_LABELS]
        if all(score is not None for score in scores):
            return [float(score) for score in scores if score is not None]
    values = _as_list(row.get("score_list"))
    scores = [scalar(value) for value in values]
    return [float(score) for score in scores if score is not None]


def bench_split_root(bench_data_root: Path, bench_name: str) -> Path:
    """
    Determines the root directory for a specific benchmark split.

    Checks for common file patterns indicating the split root, and if not
    found, assumes a sub-directory structure.

    Args:
        bench_data_root (Path): The general root directory for benchmark data.
        bench_name (str): The name of the benchmark.

    Returns:
        Path: The determined root directory for the benchmark split.
    """
    # Check if key files exist directly in bench_data_root, indicating it's already the split root.
    if (bench_data_root / f"frames_{bench_name}_test.zip").is_file() or (bench_data_root / "test-00000-of-00001.parquet").is_file():
        return bench_data_root
    # Otherwise, assume the split data is in a subdirectory named after the benchmark.
    return bench_data_root / bench_name


def load_bounded_rows(bench_data_root: Path, bench_name: str, sample_count: int) -> tuple[Path, list[dict[str, Any]]]:
    """
    Loads a bounded number of sample rows from a parquet file for the benchmark.

    This is used for the "bounded" run mode, where only a subset of data is processed.
    Requires pandas.

    Args:
        bench_data_root (Path): The general root directory for benchmark data.
        bench_name (str): The name of the benchmark.
        sample_count (int): The maximum number of samples to load.

    Returns:
        tuple[Path, list[dict[str, Any]]]: A tuple containing the split root path
                                           and a list of dictionaries, each representing a sample row.

    Raises:
        FileNotFoundError: If the parquet file is not found.
        RuntimeError: If pandas is not installed.
    """
    split_root = bench_split_root(bench_data_root, bench_name)
    parquet_path = split_root / "test-00000-of-00001.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"VideoScore-Bench parquet not found: {parquet_path}")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("bounded VideoScore scoring requires pandas and pyarrow") from exc

    # Load parquet data and convert a slice to a list of dictionaries.
    frame = pd.read_parquet(parquet_path)
    rows = frame.head(sample_count).to_dict(orient="records")
    return split_root, rows


def materialize_bounded_frames(
    *,
    split_root: Path,
    bench_name: str,
    row: dict[str, Any],
    frames_root: Path,
) -> list[str]:
    """
    Extracts and materializes video frames for a single sample from a zip archive.

    Used in the "bounded" run mode to make frames available on disk for the model.

    Args:
        split_root (Path): The root directory for the benchmark data split.
        bench_name (str): The name of the benchmark.
        row (dict[str, Any]): A dictionary representing a single sample row.
        frames_root (Path): The root directory where frames should be materialized.

    Returns:
        list[str]: A list of file paths to the materialized frames for the sample.

    Raises:
        FileNotFoundError: If the frames zip file or a specific frame within it is not found.
    """
    import zipfile

    sample_id = str(row["id"])
    zip_path = split_root / f"frames_{bench_name}_test.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"VideoScore-Bench frames zip not found: {zip_path}")
    target_dir = frames_root / f"frames_{bench_name}" / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        # Iterate through image names specified in the row to extract them from the zip.
        for image_name in _as_list(row.get("images")):
            image_name = str(image_name)
            member = f"{sample_id}/{image_name}"
            if member not in names:
                raise FileNotFoundError(f"frame {member!r} missing from {zip_path}")
            destination = target_dir / image_name
            # Only extract if the file does not already exist.
            if not destination.is_file():
                destination.write_bytes(archive.read(member))
            frame_paths.append(str(destination))
    return frame_paths


def load_official_videoscore_module(videoscore_root: Path) -> Any:
    """
    Dynamically loads the official VideoScore evaluation module.

    Adjusts `sys.path` to ensure the module can be imported correctly.

    Args:
        videoscore_root (Path): The root directory of the VideoScore repository.

    Returns:
        Any: The loaded `eval_videoscore` module object.

    Raises:
        FileNotFoundError: If the `eval_videoscore.py` script is not found.
    """
    benchmark_dir = videoscore_root / "benchmark"
    eval_path = benchmark_dir / "eval_videoscore.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"VideoScore eval_videoscore.py not found under: {videoscore_root}")
    # Prepend benchmark_dir and videoscore_root only for the duration of the
    # import so the vendored top-level module names cannot shadow imports made
    # later in this process (workspace/in-tree callers share the interpreter).
    inserted = [item for item in (str(benchmark_dir), str(videoscore_root)) if item not in sys.path]
    for item in reversed(inserted):
        sys.path.insert(0, item)
    try:
        return importlib.import_module("eval_videoscore")
    finally:
        for item in inserted:
            try:
                sys.path.remove(item)
            except ValueError:
                pass


def patch_transformers_dynamic_cache_api() -> Callable[[], None] | None:
    """
    Patches `transformers.cache_utils.DynamicCache` for compatibility.

    This addresses a potential API change in the Hugging Face Transformers library
    where `get_usable_length` might be missing but `get_seq_length` exists.

    Returns an undo callable when a patch was applied (so in-process callers can
    restore the class after the bounded run instead of leaking the monkeypatch
    to unrelated runners sharing the interpreter), or None when nothing changed.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:  # pragma: no cover - runtime dependency
        return None

    # Check if the patch is needed (if get_usable_length is missing but get_seq_length exists).
    if hasattr(DynamicCache, "get_usable_length") or not hasattr(DynamicCache, "get_seq_length"):
        return None

    # Define the compatibility method.
    def get_usable_length(self: Any, _new_seq_length: int | None = None, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    # Apply the patch.
    DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]

    def undo_patch() -> None:
        if getattr(DynamicCache, "get_usable_length", None) is get_usable_length:
            del DynamicCache.get_usable_length  # type: ignore[attr-defined]

    return undo_patch


def run_bounded_videoscore(args: argparse.Namespace, result_file: Path) -> tuple[list[dict[str, Any]], Path]:
    """
    Runs the VideoScore evaluation for a bounded number of samples in-process.

    This mode bypasses the external subprocess call and runs the core evaluation
    logic directly, making it suitable for quick testing or CI. It requires
    local benchmark data and materializes frames on demand.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        result_file (Path): The path where the results of this bounded run should be saved.

    Returns:
        tuple[list[dict[str, Any]], Path]: A tuple containing the list of raw
                                           results (dictionaries) and the path
                                           to the directory where frames were materialized.

    Raises:
        ValueError: If `bench_data_root` is not provided or `bounded_sample_count` is invalid.
    """
    if args.bench_data_root is None:
        raise ValueError("--bench-data-root or WORLDFOUNDRY_VIDEOSCORE_BENCH_ROOT is required for --bounded-sample-count")
    if args.bounded_sample_count < 1:
        raise ValueError("--bounded-sample-count must be positive")

    # Load bounded data rows and prepare frames root.
    split_root, rows = load_bounded_rows(args.bench_data_root, args.bench_name, args.bounded_sample_count)
    frames_root = args.output_dir / "bounded_frames"
    # Apply compatibility patch for transformers library; undone after the run
    # so the class-level monkeypatch does not leak to other in-process runners.
    undo_transformers_patch = patch_transformers_dynamic_cache_api()
    try:
        official = load_official_videoscore_module(args.videoscore_root)

        # Initialize model and processor.
        processor = official.AutoProcessor.from_pretrained(args.model_repo_name, torch_dtype=official.torch.bfloat16)
        model = official.Idefics2ForSequenceClassification.from_pretrained(
            args.model_repo_name,
            torch_dtype=official.torch.bfloat16,
        ).eval()
        device = official.torch.device("cuda" if official.torch.cuda.is_available() else "cpu")
        model.to(device)

        result_rows: list[dict[str, Any]] = []
        # Process each row, materialize frames, run model inference, and store results.
        for row in rows:
            sample_id = str(row["id"])
            frame_paths = materialize_bounded_frames(
                split_root=split_root,
                bench_name=args.bench_name,
                row=row,
                frames_root=frames_root,
            )
            prompt = video_prompt_from_row(row)
            # Call the official model output function directly.
            scores = official._model_output(model, processor, prompt, frame_paths)
            result_rows.append(
                {
                    "id": sample_id,
                    "text": prompt,
                    "ref": str(reference_scores_from_row(row, args.bench_name)),
                    "ans": str(scores),
                }
            )

        write_json(result_file, result_rows)
        return result_rows, frames_root / f"frames_{args.bench_name}"
    finally:
        if undo_transformers_patch is not None:
            undo_transformers_patch()


def run_official_videoscore(args: argparse.Namespace) -> dict[str, Any]:
    """
    Executes the VideoScore benchmark, either by calling the official script
    as a subprocess, running a bounded in-process version, or by normalizing
    pre-existing results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict[str, Any]: The generated WorldFoundry scorecard dictionary.

    Raises:
        ValueError: If required arguments like `frames_dir` or `bench_data_root` are missing.
        FileNotFoundError: If the official VideoScore script is not found.
    """
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define paths for stdout/stderr logs and upstream output directory.
    stdout_path = output_dir / "upstream_stdout.log"
    stderr_path = output_dir / "upstream_stderr.log"
    upstream_output_dir = output_dir / "upstream"
    upstream_output_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: Normalize pre-existing results.
    if args.from_upstream_results:
        raw_results = load_json(args.from_upstream_results)
        return normalize_videoscore_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=args.from_upstream_results,
            dataset_root=args.dataset_root,
            frames_dir=args.frames_dir,
            command=None,  # No command executed if loading from existing results.
            duration_seconds=None,
            returncode=0, # Assume success if results are provided.
        )

    # Case 2: Run bounded in-process evaluation.
    if args.bounded_sample_count is not None:
        result_file = upstream_output_dir / f"eval_{args.bench_name}_videoscore.json"
        # Construct the command for recording in the scorecard (if this were run externally).
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "--bounded-sample-count",
            str(args.bounded_sample_count),
            "--videoscore-root",
            str(args.videoscore_root),
            "--model-repo-name",
            args.model_repo_name,
            "--bench-data-root",
            str(args.bench_data_root),
            "--bench-name",
            args.bench_name,
            "--output-dir",
            str(args.output_dir),
            "--json",
        ]
        start = time.monotonic()
        raw_results, bounded_frames_dir = run_bounded_videoscore(args, result_file)
        duration_seconds = time.monotonic() - start
        # For in-process run, stdout/stderr logs are empty as they are not captured from a subprocess.
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return normalize_videoscore_results(
            raw_results,
            benchmark_id=args.benchmark_id,
            output_dir=output_dir,
            upstream_results_path=result_file,
            dataset_root=bench_split_root(args.bench_data_root, args.bench_name),
            frames_dir=bounded_frames_dir,
            command=command,
            duration_seconds=duration_seconds,
            returncode=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    # Case 3: Run official VideoScore script as a subprocess.
    if args.frames_dir is None:
        raise ValueError("--frames-dir or WORLDFOUNDRY_VIDEOSCORE_FRAMES_DIR is required unless --official-results-path is used")
    if not (args.videoscore_root / "benchmark" / "eval_videoscore.py").is_file():
        raise FileNotFoundError(f"VideoScore eval_videoscore.py not found under: {args.videoscore_root}")

    result_file = upstream_output_dir / f"eval_{args.bench_name}_videoscore.json"
    command = build_official_command(args, result_file)
    env = os.environ.copy()
    # Modify PYTHONPATH to include VideoScore's benchmark and root directories.
    env["PYTHONPATH"] = f"{args.videoscore_root / 'benchmark'}{os.pathsep}{args.videoscore_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    start = time.monotonic()
    completed = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=args.videoscore_root / "benchmark", # Run from the benchmark directory as expected by VideoScore.
        env=env,
        timeout=args.timeout,
        start_new_session=False,
    )
    duration_seconds = time.monotonic() - start

    # Load raw results from the file generated by the subprocess, or create an error result if file is missing.
    if result_file.is_file():
        raw_results = load_json(result_file)
    else:
        write_json(result_file, {"error": "missing upstream VideoScore eval result JSON"})
        raw_results = load_json(result_file)

    return normalize_videoscore_results(
        raw_results,
        benchmark_id=args.benchmark_id,
        output_dir=output_dir,
        upstream_results_path=result_file,
        dataset_root=args.dataset_root,
        frames_dir=args.frames_dir,
        command=command,
        duration_seconds=duration_seconds,
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Builds and configures the argument parser for the script.

    Defines all command-line arguments, their types, defaults, and help messages.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run official VideoScore and normalize its output to a WorldFoundry scorecard.")
    parser.add_argument("--benchmark-id", default=os.environ.get("WORLDFOUNDRY_BENCHMARK_ID", "videoscore"))
    parser.add_argument("--videoscore-root", type=Path, default=env_path("WORLDFOUNDRY_VIDEOSCORE_ROOT", default_videoscore_root()))
    parser.add_argument("--model-repo-name", default=os.environ.get("WORLDFOUNDRY_VIDEOSCORE_MODEL_REPO", default_model_repo_name()))
    parser.add_argument("--data-repo-name", default=os.environ.get("WORLDFOUNDRY_VIDEOSCORE_DATA_REPO", "TIGER-Lab/VideoScore-Bench"))
    parser.add_argument("--bench-name", default=os.environ.get("WORLDFOUNDRY_VIDEOSCORE_BENCH_NAME", "video_feedback"))
    parser.add_argument("--frames-dir", type=Path, default=env_path("WORLDFOUNDRY_VIDEOSCORE_FRAMES_DIR"))
    parser.add_argument("--bench-data-root", type=Path, default=env_path("WORLDFOUNDRY_VIDEOSCORE_BENCH_ROOT", default_bench_data_root()))
    parser.add_argument("--bounded-sample-count", type=int)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=env_path("WORLDFOUNDRY_VIDEOFEEDBACK_DATASET_ROOT"),
        help="Local TIGER-Lab/VideoFeedback dataset root for real-data discovery evidence.",
    )
    parser.add_argument("--output-dir", type=Path, default=env_path("WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("WORLDFOUNDRY_VIDEOSCORE_TIMEOUT", "7200")))
    parser.add_argument("--official-results-path", dest="from_upstream_results", type=Path)
    parser.add_argument("--from-upstream-results", dest="from_upstream_results", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the script.

    Parses arguments, runs the VideoScore evaluation, and prints a summary
    or the full scorecard in JSON format.

    Args:
        argv (list[str] | None): List of command-line arguments, defaults to `sys.argv[1:]`.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        print("error: --output-dir or WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR is required", file=sys.stderr)
        return 2

    try:
        scorecard = run_official_videoscore(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Extract key results from the generated scorecard for summary output.
    result = {
        "ok": scorecard["normalization_ok"],
        "benchmark_id": args.benchmark_id,
        "output_dir": str(args.output_dir),
        "scorecard": scorecard["artifacts"]["scorecard"],
        "raw_metric_table": scorecard["artifacts"]["raw_metric_table"],
        "per_sample_scores": scorecard["artifacts"]["per_sample_scores"],
        "generated_video_manifest": scorecard["artifacts"]["generated_video_manifest"],
        "dataset_manifest": scorecard["artifacts"]["dataset_manifest"],
        "upstream_results": scorecard["artifacts"]["upstream_results"],
        "official_benchmark_verified": scorecard["official_benchmark_verified"],
        "integration_evidence": scorecard["integration_evidence"],
        "normalization_ok": scorecard["normalization_ok"],
        "official_results_imported": scorecard["official_results_imported"],
    }
    if args.json:
        # Print full JSON result if --json flag is set.
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        # Print human-readable summary.
        status = "ok" if result["ok"] else "failed"
        print(f"{args.benchmark_id}: official VideoScore validation {status}")
        print(f"scorecard: {result['scorecard']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
