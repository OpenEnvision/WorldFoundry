"""In-tree PAI-Bench-C transfer-control and diversity evaluation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import (
    VIDEO_SUFFIXES,
    find_sidecar,
    find_video,
    load_array,
    load_json,
    load_records,
    read_video,
    resolve_dataset_path,
)
from .metrics import (
    align_video,
    bilateral_blur,
    blur_ssim,
    canny_edges,
    canny_scores,
    depth_si_rmse,
    segmentation_scores,
)

CONDITIONAL_METRICS = (
    "dover_tech_score",
    "blur_ssim",
    "canny_f1_score",
    "canny_precision",
    "canny_recall",
    "depth_si_rmse",
    "seg_m_iou",
    "seg_recall",
    "lpips_diversity",
)
CANNY_METRICS = {"canny_f1_score", "canny_precision", "canny_recall"}
SEGMENTATION_METRICS = {"seg_m_iou", "seg_recall"}


@dataclass(frozen=True)
class ConditionalGenerationRequest:
    dataset_root: Path
    generated_video_dir: Path
    output_dir: Path
    metadata_path: Path | None = None
    metrics: tuple[str, ...] = CONDITIONAL_METRICS
    pred_depth_dir: Path | None = None
    pred_segmentation_dir: Path | None = None
    depth_checkpoint: Path | None = None
    grounding_checkpoint: Path | None = None
    sam2_checkpoint: Path | None = None
    dover_checkpoint: Path | None = None
    allow_trusted_pickle: bool = False
    max_frames: int = 121
    limit: int | None = None


def _metadata_rows(request: ConditionalGenerationRequest) -> list[dict[str, Any]]:
    path = request.metadata_path or request.dataset_root / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"PAI-Bench-C metadata not found: {path}")
    rows = load_records(path)
    return rows[: request.limit] if request.limit is not None else rows


def _file_name(row: Mapping[str, Any], position: int) -> str:
    value = row.get("file_name") or row.get("video") or row.get("id") or f"task_{position + 1:04d}.mp4"
    return Path(str(value)).name


def _prediction_videos(root: Path, name: str) -> list[Path]:
    """Find the base prediction and all official ``__seed`` variants for one task."""

    stem = Path(name).stem
    preferred_root = root / "videos" if (root / "videos").is_dir() else root

    def accepted(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and (path.stem == stem or path.stem.startswith(f"{stem}__"))
        )

    direct = sorted(path for path in preferred_root.iterdir() if accepted(path))
    if direct:
        return direct
    matches: dict[Path, Path] = {}
    for path in preferred_root.rglob("*"):
        if accepted(path):
            matches[path.resolve()] = path
    if not matches:
        raise FileNotFoundError(f"generated video not found for {name!r} below {root}")
    return sorted(matches.values(), key=lambda path: path.as_posix())


def _reference_video(root: Path, row: Mapping[str, Any], column: str, folder: str, name: str) -> Path:
    value = row.get(column)
    fallback_name = name
    if value not in (None, "", "nan") and Path(str(value)).suffix:
        fallback_name = Path(str(value)).name
    path = resolve_dataset_path(root, value, folder, fallback_name)
    if not path.is_file():
        raise FileNotFoundError(f"PAI-Bench-C {column} reference not found: {path}")
    return path


def _ground_truth_video(request: ConditionalGenerationRequest, row: Mapping[str, Any], name: str) -> np.ndarray:
    path = _reference_video(request.dataset_root, row, "video", "videos", name)
    return read_video(path, max_frames=request.max_frames, rgb=True)


def _depth_reference(request: ConditionalGenerationRequest, row: Mapping[str, Any], name: str) -> np.ndarray:
    value = row.get("depth")
    candidates: list[Path] = []
    if value not in (None, "", "nan"):
        candidate = Path(str(value))
        if candidate.suffix.lower() in {".npy", ".npz"}:
            candidates.append(candidate if candidate.is_absolute() else request.dataset_root / candidate)
    stem = Path(name).stem
    candidates.extend(
        [
            request.dataset_root / "depth_npzs" / f"{stem}.npz",
            request.dataset_root / "depth_npzs" / f"{stem}.npy",
            request.dataset_root / "depth" / f"{stem}.npy",
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"PAI-Bench-C depth reference not found for {name}")
    return load_array(path)[: request.max_frames]


class _DepthPredictor:
    def __init__(self, checkpoint: Path | None) -> None:
        self.checkpoint = checkpoint
        self._model: Any = None

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if self._model is None:
            from worldfoundry.base_models.three_dimensions.depth.videodepthanything import (
                VideoDepthAnythingDepthModel,
            )

            self._model = VideoDepthAnythingDepthModel(
                model="vits",
                weights_path=str(self.checkpoint) if self.checkpoint is not None else None,
            )
        return np.asarray(
            self._model.model.infer_video_depth(
                [frame for frame in frames],
                input_size=518,
                fp32=True,
            )
        )


def _normalize_segmentation_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("records", "detections", "segments", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("segmentation sidecar must contain a list")
    records: list[dict[str, Any]] = []
    for value in payload:
        if isinstance(value, dict):
            records.append(value)
        elif hasattr(value, "to_dict"):
            records.append(value.to_dict())
        elif hasattr(value, "__dict__"):
            records.append(dict(vars(value)))
        else:
            raise TypeError(f"unsupported segmentation record: {type(value).__name__}")
    return records


def _load_segmentation(path: Path, *, allow_pickle: bool) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _normalize_segmentation_records(load_json(path))
    if path.suffix.lower() in {".pkl", ".pickle"}:
        if not allow_pickle:
            raise ValueError(
                f"refusing to load pickle {path}; pass --allow-trusted-pickle only for the trusted official dataset"
            )
        with path.open("rb") as handle:
            return _normalize_segmentation_records(pickle.load(handle))  # noqa: S301 - explicit trusted opt-in
    raise ValueError(f"unsupported segmentation sidecar: {path}")


def _segmentation_reference(request: ConditionalGenerationRequest, row: Mapping[str, Any], name: str) -> Path:
    value = row.get("sam2")
    stem = Path(name).stem
    candidates = []
    if value not in (None, "", "nan"):
        candidate = Path(str(value))
        if candidate.suffix.lower() in {".json", ".pkl", ".pickle"}:
            candidates.append(candidate if candidate.is_absolute() else request.dataset_root / candidate)
    candidates.extend(
        [
            request.dataset_root / "sam2_pkls" / f"{stem}.pkl",
            request.dataset_root / "sam2" / f"{stem}.pkl",
            request.dataset_root / "sam2_pkls" / f"{stem}.json",
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"PAI-Bench-C segmentation reference not found for {name}")
    return path


def _caption_phrases(request: ConditionalGenerationRequest, row: Mapping[str, Any], name: str) -> list[str]:
    value = row.get("caption_text")
    paths: list[Path] = []
    if isinstance(value, str) and len(value) < 1024 and value.strip().lower().endswith(".json"):
        candidate = Path(value.strip()).expanduser()
        paths.append(candidate if candidate.is_absolute() else request.dataset_root / candidate)
    paths.append(request.dataset_root / "captions" / f"{Path(name).stem}.json")
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is not None:
        payload = load_json(path)
        if isinstance(payload, dict):
            for key in ("phrases", "objects", "entities", "captions"):
                if isinstance(payload.get(key), list):
                    return [str(item.get("phrase", item) if isinstance(item, dict) else item) for item in payload[key]]
            for key in ("caption", "text", "prompt"):
                if isinstance(payload.get(key), str):
                    return [payload[key]]
            if len(payload) == 1:
                only_value = next(iter(payload.values()))
                if isinstance(only_value, str) and only_value.strip():
                    return [only_value.strip()]
        if isinstance(payload, list):
            return [str(item.get("phrase", item) if isinstance(item, dict) else item) for item in payload]
        if isinstance(payload, str) and payload.strip():
            return [payload.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _mean_summary(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if isinstance(row.get(metric), (int, float))]
        result[metric] = float(np.mean(values)) if values else None
    return result


def _lpips_diversity(
    request: ConditionalGenerationRequest, rows: list[dict[str, Any]]
) -> tuple[float | None, list[dict[str, Any]]]:
    from worldfoundry.base_models.perception_core.video_quality.lpips import LPIPSMetric

    scorer = LPIPSMetric(net="vgg")
    details: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        name = _file_name(row, position)
        stem = Path(name).stem
        names = [f"{stem}.mp4", *[f"{stem}_caption{index}.mp4" for index in range(1, 6)]]
        try:
            videos = [
                read_video(find_video(request.generated_video_dir, item), max_frames=request.max_frames)
                for item in names
            ]
        except FileNotFoundError:
            continue
        scores: list[float] = []
        for left in range(len(videos)):
            for right in range(len(videos)):
                if left == right:
                    continue
                video_a, video_b = align_video(videos[left], videos[right], nearest=False)
                scores.extend(float(value) for value in scorer(video_a, video_b))
        if scores:
            details.append({"sample_id": stem, "lpips_diversity": float(np.mean(scores))})
    values = [row["lpips_diversity"] for row in details]
    return (float(np.mean(values)) if values else None), details


def evaluate_conditional_generation(request: ConditionalGenerationRequest) -> dict[str, Any]:
    requested = list(dict.fromkeys(request.metrics))
    unknown = sorted(set(requested) - set(CONDITIONAL_METRICS))
    if unknown:
        raise ValueError(f"unknown PAI-Bench-C metrics: {', '.join(unknown)}")
    rows = _metadata_rows(request)
    sample_metrics = [metric for metric in requested if metric != "lpips_diversity"]
    results: list[dict[str, Any]] = []
    dover = None
    depth_predictor = _DepthPredictor(request.depth_checkpoint)
    segmenter = None
    for position, row in enumerate(rows if sample_metrics else []):
        name = _file_name(row, position)
        stem = Path(name).stem
        gt_depth: np.ndarray | None = None
        gt_records: list[dict[str, Any]] | None = None
        captions: list[str] | None = None
        try:
            pred_paths = _prediction_videos(request.generated_video_dir, name)
        except FileNotFoundError:
            continue
        for pred_path in pred_paths:
            pred_stem = pred_path.stem
            frames = read_video(pred_path, max_frames=request.max_frames, rgb=True)
            aligned_frames: np.ndarray | None = None
            gt_frames: np.ndarray | None = None

            def aligned_video_frames() -> tuple[np.ndarray, np.ndarray]:
                nonlocal aligned_frames, gt_frames
                if aligned_frames is None or gt_frames is None:
                    gt_frames = _ground_truth_video(request, row, name)
                    aligned_frames, gt_frames = align_video(frames, gt_frames, nearest=False)
                return aligned_frames, gt_frames

            result: dict[str, Any] = {
                "sample_id": pred_stem,
                "reference_id": stem,
                "video_path": str(pred_path),
            }
            if "dover_tech_score" in requested:
                if dover is None:
                    from worldfoundry.base_models.perception_core.video_quality.dover import DOVERTechnicalScorer

                    dover = DOVERTechnicalScorer(checkpoint=request.dover_checkpoint)
                result["dover_tech_score"] = dover(pred_path)
            if "blur_ssim" in requested:
                pred_rgb, gt_rgb = aligned_video_frames()
                result["blur_ssim"] = blur_ssim(bilateral_blur(pred_rgb), bilateral_blur(gt_rgb))
            if set(requested) & CANNY_METRICS:
                pred_rgb, gt_rgb = aligned_video_frames()
                scores = canny_scores(canny_edges(pred_rgb), canny_edges(gt_rgb))
                result.update({key: value for key, value in scores.items() if key in requested})
            if "depth_si_rmse" in requested:
                if gt_depth is None:
                    gt_depth = _depth_reference(request, row, name)
                pred_path_depth = find_sidecar(request.pred_depth_dir, pred_stem, (".npy", ".npz"))
                pred_depth = (
                    load_array(pred_path_depth)
                    if pred_path_depth is not None
                    else depth_predictor(aligned_video_frames()[0])
                )
                result["depth_si_rmse"] = depth_si_rmse(pred_depth[: request.max_frames], gt_depth)
            if set(requested) & SEGMENTATION_METRICS:
                if gt_records is None:
                    gt_path = _segmentation_reference(request, row, name)
                    gt_records = _load_segmentation(gt_path, allow_pickle=request.allow_trusted_pickle)
                pred_seg_path = find_sidecar(
                    request.pred_segmentation_dir,
                    pred_stem,
                    (".json", ".pkl", ".pickle"),
                )
                if pred_seg_path is not None:
                    pred_records = _load_segmentation(pred_seg_path, allow_pickle=request.allow_trusted_pickle)
                else:
                    if segmenter is None:
                        from worldfoundry.base_models.perception_core.segment.grounded_segment_anything import (
                            GroundedSAM2VideoSegmenter,
                        )

                        segmenter = GroundedSAM2VideoSegmenter(
                            grounding_checkpoint=request.grounding_checkpoint,
                            sam2_checkpoint=request.sam2_checkpoint,
                        )
                    if captions is None:
                        captions = _caption_phrases(request, row, name)
                    pred_records = segmenter.segment(aligned_video_frames()[0], captions)
                scores = segmentation_scores(gt_records, pred_records)
                result.update({key: value for key, value in scores.items() if key in requested})
            results.append(result)
    if sample_metrics and not results:
        raise FileNotFoundError(
            f"no generated PAI-Bench-C transfer videos matched metadata below {request.generated_video_dir}"
        )
    summary = _mean_summary(results, sample_metrics)
    diversity_rows: list[dict[str, Any]] = []
    if "lpips_diversity" in requested:
        summary["lpips_diversity"], diversity_rows = _lpips_diversity(request, rows)
    return {
        "track": "conditional-generation",
        "summary": summary,
        "samples": results,
        "diversity_samples": diversity_rows,
        "model_reuse": {
            "dover": "worldfoundry.base_models.perception_core.video_quality.dover",
            "lpips": "worldfoundry.base_models.perception_core.video_quality.lpips",
            "depth": "worldfoundry.base_models.three_dimensions.depth.videodepthanything",
            "segmentation": "worldfoundry.base_models.perception_core.segment.grounded_segment_anything",
        },
    }
