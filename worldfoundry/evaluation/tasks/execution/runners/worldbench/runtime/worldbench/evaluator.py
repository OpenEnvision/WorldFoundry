"""End-to-end in-tree evaluator for WorldBench IntuitivePhysics artifacts."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

from .io import (
    ArtifactIndex,
    canonical_sample_id,
    discover_questions,
    discover_scenes,
    load_answer_predictions,
    load_frames,
    load_label_frames,
    load_path_manifest,
    load_predicted_masks,
    load_rgba_frames,
    prediction_for_question,
)
from .metrics import (
    answer_is_correct,
    background_rmse_by_frame,
    first_visible_object_boxes,
    mean_available,
    overlap_by_frame,
    question_type,
)
from .sam2_adapter import SAM2MaskTracker, stage_video_frames

DEFAULT_CONFIG_PATH = bundled_benchmark_asset("worldbench", "evaluator.yaml")


@dataclass(frozen=True)
class WorldBenchEvaluationRequest:
    dataset_root: Path
    work_dir: Path
    generated_video_dir: Path | None = None
    video_manifest: Path | None = None
    answer_manifest: Path | None = None
    predicted_mask_dir: Path | None = None
    config_path: Path = DEFAULT_CONFIG_PATH
    sample_ids: tuple[str, ...] = ()
    limit: int | None = None
    max_frames: int | None = None
    ground_truth_start_frame: int | None = None
    generated_skip_frames: int | None = None
    sam2_model_id: str | None = None
    sam2_checkpoint: Path | None = None
    sam2_config: str | None = None
    device: str = "auto"
    evaluate_video: bool = True
    evaluate_text: bool = True
    continue_on_error: bool = False
    keep_staged_frames: bool = False
    save_predicted_masks: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"WorldBench evaluator config not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"WorldBench evaluator config must be a mapping: {resolved}")
    return payload


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    integer = int(value)
    minimum = 0 if allow_zero else 1
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


def _metric(
    value: float | None,
    *,
    sample_count: int,
    source: str,
    higher_is_better: bool,
    description: str,
) -> dict[str, Any]:
    return {
        "available": value is not None,
        "raw_score": value,
        "normalized_score": value,
        "sample_count": sample_count,
        "source": source,
        "higher_is_better": higher_is_better,
        "description": description,
    }


def _write_label_sequence(root: Path, sample_id: str, frames: list[np.ndarray]) -> Path:
    output = root.expanduser().resolve() / canonical_sample_id(sample_id)
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        maximum = int(np.max(frame)) if frame.size else 0
        dtype = np.uint8 if maximum <= 255 else np.uint16
        Image.fromarray(np.asarray(frame, dtype=dtype)).save(output / f"{index:05d}.png")
    return output


def _aggregate_video_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    available = [row for row in rows if row.get("available")]
    miou = mean_available([row.get("foreground_miou") for row in available])
    dice = mean_available([row.get("foreground_dice") for row in available])
    background = mean_available([row.get("background_rmse") for row in available])
    return {
        "foreground_miou": _metric(
            miou,
            sample_count=sum(row.get("foreground_miou") is not None for row in available),
            source="computed_from_generated_video_and_ground_truth_segmentation",
            higher_is_better=True,
            description="Mean per-object intersection-over-union across aligned continuation frames.",
        ),
        "foreground_dice": _metric(
            dice,
            sample_count=sum(row.get("foreground_dice") is not None for row in available),
            source="worldbench_public_release_compatibility",
            higher_is_better=True,
            description="Dice overlap computed by the public release while labeling the value mIoU.",
        ),
        "background_rmse": _metric(
            background,
            sample_count=sum(row.get("background_rmse") is not None for row in available),
            source="computed_on_ground_truth_background_pixels",
            higher_is_better=False,
            description="RGBA root-mean-square error over ground-truth background pixels.",
        ),
    }


def _evaluate_text(
    dataset_root: Path,
    answer_manifest: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    questions = discover_questions(dataset_root)
    predictions = load_answer_predictions(answer_manifest, questions)
    rows: list[dict[str, Any]] = []
    for question in questions:
        prediction = prediction_for_question(predictions, question)
        kind = question_type(question.question)
        available = prediction is not None
        correct = answer_is_correct(prediction, question.answer, question.question) if available else None
        rows.append(
            {
                "sample_id": question.question_id,
                "component": "text_based",
                "question_type": kind,
                "video_name": question.video_name,
                "question": question.question,
                "prediction": prediction,
                "target": question.answer,
                "correct": correct,
                "available": available,
                "raw_score": float(correct) if correct is not None else None,
                "normalized_score": float(correct) if correct is not None else None,
                "source": f"{question.source}#{question.row_index}",
            }
        )

    available_rows = [row for row in rows if row["available"]]
    multiple_choice = [row for row in available_rows if row["question_type"] == "multiple_choice"]
    binary = [row for row in available_rows if row["question_type"] == "binary"]

    def accuracy(group: list[dict[str, Any]]) -> float | None:
        return mean_available([row["normalized_score"] for row in group])

    metrics = {
        "text_based_accuracy": _metric(
            accuracy(available_rows),
            sample_count=len(available_rows),
            source="exact_worldbench_answer_matching",
            higher_is_better=True,
            description="Accuracy across available WorldBench textual questions.",
        ),
        "multiple_choice_accuracy": _metric(
            accuracy(multiple_choice),
            sample_count=len(multiple_choice),
            source="exact_worldbench_answer_matching",
            higher_is_better=True,
            description="Accuracy across available multiple-choice questions.",
        ),
        "binary_accuracy": _metric(
            accuracy(binary),
            sample_count=len(binary),
            source="exact_worldbench_answer_matching",
            higher_is_better=True,
            description="Accuracy across available yes/no questions.",
        ),
    }
    coverage = {
        "discovered_questions": len(questions),
        "scored_questions": len(available_rows),
        "missing_predictions": len(questions) - len(available_rows),
        "complete": bool(questions) and len(available_rows) == len(questions),
    }
    return rows, metrics, coverage


def evaluate_worldbench(request: WorldBenchEvaluationRequest) -> dict[str, Any]:
    """Evaluate generated continuations and/or question answers in-tree."""

    config = _load_config(request.config_path)
    video_config = config.get("video") or {}
    sam2_config = config.get("sam2") or {}
    target_size_raw = video_config.get("target_size", [640, 1024])
    if not isinstance(target_size_raw, list) or len(target_size_raw) != 2:
        raise ValueError("video.target_size must be [height, width]")
    target_size = (
        _positive_int(target_size_raw[0], name="target height"),
        _positive_int(target_size_raw[1], name="target width"),
    )
    gt_start = _positive_int(
        request.ground_truth_start_frame
        if request.ground_truth_start_frame is not None
        else video_config.get("ground_truth_start_frame", 9),
        name="ground_truth_start_frame",
        allow_zero=True,
    )
    generated_skip = _positive_int(
        request.generated_skip_frames
        if request.generated_skip_frames is not None
        else video_config.get("generated_skip_frames", 0),
        name="generated_skip_frames",
        allow_zero=True,
    )
    max_frames_value = request.max_frames if request.max_frames is not None else video_config.get("max_frames", 24)
    max_frames = None if max_frames_value in {None, 0, "all"} else _positive_int(max_frames_value, name="max_frames")
    background_label = _positive_int(
        video_config.get("background_label", 1),
        name="background_label",
        allow_zero=True,
    )

    request.work_dir.mkdir(parents=True, exist_ok=True)
    scenes = discover_scenes(request.dataset_root) if request.evaluate_video else []
    scene_by_id = {scene.sample_id: scene for scene in scenes}
    manifest = load_path_manifest(request.video_manifest, base_dir=request.generated_video_dir)
    artifact_index = ArtifactIndex(request.generated_video_dir, manifest)
    selected: list[tuple[Any, Path]] = []
    missing_requested: list[str] = []

    requested_ids = [canonical_sample_id(item) for item in request.sample_ids]
    candidate_scenes = [scene_by_id[item] for item in requested_ids if item in scene_by_id] if requested_ids else scenes
    unknown_ids = sorted(set(requested_ids) - set(scene_by_id))
    if unknown_ids:
        raise ValueError(f"unknown WorldBench sample IDs: {', '.join(unknown_ids)}")
    for scene in candidate_scenes:
        artifact = artifact_index.resolve(scene.sample_id)
        if artifact is None:
            if requested_ids:
                missing_requested.append(scene.sample_id)
            continue
        selected.append((scene, artifact))
    if missing_requested:
        raise FileNotFoundError(f"generated artifacts missing for: {', '.join(missing_requested)}")
    if request.limit is not None:
        selected = selected[: _positive_int(request.limit, name="limit")]

    tracker: SAM2MaskTracker | None = None
    if request.evaluate_video and selected and request.predicted_mask_dir is None:
        tracker = SAM2MaskTracker(
            model_id=request.sam2_model_id or str(sam2_config.get("model_id", "facebook/sam2.1-hiera-large")),
            checkpoint=request.sam2_checkpoint,
            config_name=request.sam2_config or sam2_config.get("config_name"),
            device=request.device,
            offload_video_to_cpu=bool(sam2_config.get("offload_video_to_cpu", True)),
            offload_state_to_cpu=bool(sam2_config.get("offload_state_to_cpu", False)),
        )

    video_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    staged_root = request.work_dir / "staged_frames"
    if staged_root.exists() and not request.keep_staged_frames:
        shutil.rmtree(staged_root)

    for scene, artifact in selected:
        try:
            decode_limit = None if max_frames is None else generated_skip + max_frames
            decoded = load_frames(artifact, limit=decode_limit)
            generated = decoded[generated_skip:]
            available_gt = min(len(scene.segmentation_paths), len(scene.rgba_paths)) - gt_start
            frame_count = min(len(generated), max(available_gt, 0))
            if max_frames is not None:
                frame_count = min(frame_count, max_frames)
            if frame_count <= 0:
                raise ValueError(
                    f"no aligned frames (generated={len(generated)}, ground_truth_after_offset={available_gt})"
                )
            generated = generated[:frame_count]

            if request.keep_staged_frames:
                frames_dir = staged_root / scene.sample_id
                staged_generated = stage_video_frames(generated, frames_dir, size=target_size)
            else:
                temporary = tempfile.TemporaryDirectory(prefix="worldbench-", dir=request.work_dir)
                try:
                    frames_dir = Path(temporary.name) / "frames"
                    staged_generated = stage_video_frames(generated, frames_dir, size=target_size)
                    target_labels = load_label_frames(
                        scene.segmentation_paths,
                        start=gt_start,
                        count=frame_count,
                        size=target_size,
                        background_label=background_label,
                    )
                    target_rgba = load_rgba_frames(scene.rgba_paths, start=gt_start, count=frame_count)
                    prompts = first_visible_object_boxes(target_labels)
                    object_ids = sorted(prompts)
                    predicted = (
                        load_predicted_masks(
                            request.predicted_mask_dir,
                            scene.sample_id,
                            count=frame_count,
                            size=target_size,
                        )
                        if request.predicted_mask_dir is not None
                        else tracker.track(
                            frames_dir,
                            prompts,
                            frame_count=frame_count,
                            size=target_size,
                        )
                    )
                finally:
                    temporary.cleanup()

            if request.keep_staged_frames:
                target_labels = load_label_frames(
                    scene.segmentation_paths,
                    start=gt_start,
                    count=frame_count,
                    size=target_size,
                    background_label=background_label,
                )
                target_rgba = load_rgba_frames(scene.rgba_paths, start=gt_start, count=frame_count)
                prompts = first_visible_object_boxes(target_labels)
                object_ids = sorted(prompts)
                predicted = (
                    load_predicted_masks(
                        request.predicted_mask_dir,
                        scene.sample_id,
                        count=frame_count,
                        size=target_size,
                    )
                    if request.predicted_mask_dir is not None
                    else tracker.track(frames_dir, prompts, frame_count=frame_count, size=target_size)
                )

            overlaps = overlap_by_frame(predicted, target_labels, object_ids=object_ids)
            background_values = background_rmse_by_frame(staged_generated, target_rgba, target_labels)
            foreground_miou = mean_available([row["foreground_miou"] for row in overlaps])
            foreground_dice = mean_available([row["foreground_dice"] for row in overlaps])
            background_rmse = mean_available(background_values)
            if request.save_predicted_masks is not None:
                _write_label_sequence(request.save_predicted_masks, scene.sample_id, predicted)

            for row, rmse in zip(overlaps, background_values, strict=True):
                row["background_rmse"] = rmse
                row["ground_truth_frame_index"] = gt_start + int(row["frame_index"])
                row["generated_frame_index"] = generated_skip + int(row["frame_index"])
            video_rows.append(
                {
                    "sample_id": scene.sample_id,
                    "component": "video_based",
                    "available": True,
                    "generated_artifact": str(artifact),
                    "ground_truth": str(scene.root),
                    "frame_count": frame_count,
                    "object_ids": object_ids,
                    "foreground_miou": foreground_miou,
                    "foreground_dice": foreground_dice,
                    "background_rmse": background_rmse,
                    "raw_score": foreground_miou,
                    "normalized_score": foreground_miou,
                    "frame_metrics": overlaps,
                }
            )
        except Exception as exc:
            failure = {
                "sample_id": scene.sample_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            video_rows.append(
                {
                    "sample_id": scene.sample_id,
                    "component": "video_based",
                    "available": False,
                    "generated_artifact": str(artifact),
                    "ground_truth": str(scene.root),
                    "reason": "evaluation_failed",
                    **failure,
                }
            )
            if not request.continue_on_error:
                raise RuntimeError(f"WorldBench sample {scene.sample_id} failed: {exc}") from exc

    if request.evaluate_video and (request.generated_video_dir is not None or request.video_manifest is not None):
        if not selected:
            raise ValueError("no generated artifacts matched WorldBench scene IDs")
        if not any(row.get("available") for row in video_rows):
            raise RuntimeError("all selected WorldBench video samples failed")

    metrics: dict[str, dict[str, Any]] = {}
    if request.evaluate_video:
        metrics.update(_aggregate_video_metrics(video_rows))

    text_rows: list[dict[str, Any]] = []
    text_coverage = {
        "discovered_questions": 0,
        "scored_questions": 0,
        "missing_predictions": 0,
        "complete": False,
    }
    if request.evaluate_text:
        text_rows, text_metrics, text_coverage = _evaluate_text(request.dataset_root, request.answer_manifest)
        metrics.update(text_metrics)

    video_coverage = {
        "discovered_scenes": len(scenes),
        "matched_scenes": len(selected),
        "scored_scenes": sum(row.get("available") is True for row in video_rows),
        "failed_scenes": len(failures),
        "complete": bool(scenes) and len(selected) == len(scenes) and not failures,
    }
    return {
        "schema_version": "worldfoundry-worldbench-evaluation-v1",
        "config": {
            "path": str(request.config_path.expanduser().resolve()),
            "target_size": list(target_size),
            "ground_truth_start_frame": gt_start,
            "generated_skip_frames": generated_skip,
            "max_frames": max_frames,
            "background_label": background_label,
        },
        "dataset": {
            "root": str(request.dataset_root.expanduser().resolve()),
            "video_coverage": video_coverage,
            "text_coverage": text_coverage,
        },
        "model": tracker.provenance
        if tracker is not None
        else {
            "implementation": "precomputed_label_masks" if request.predicted_mask_dir else None,
            "predicted_mask_dir": str(request.predicted_mask_dir) if request.predicted_mask_dir else None,
        },
        "metrics": metrics,
        "per_sample_scores": [*video_rows, *text_rows],
        "failures": failures,
        "metadata": dict(request.metadata),
    }
