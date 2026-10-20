# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Match annotated source targets to generated video frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

from ..revisit_protocol import validate_expected_revisits


class ManifestBuildError(RuntimeError):
    """A preparation failure that must block formal evaluation."""


def _box(value: Sequence[float], *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ManifestBuildError(f"{field} must be [x1,y1,x2,y2]")
    try:
        box = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError) as error:
        raise ManifestBuildError(f"{field} contains a non-numeric value") from error
    if not np.isfinite(box).all():
        raise ManifestBuildError(f"{field} contains a non-finite value")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ManifestBuildError(f"{field} is empty or inverted")
    return box


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def normalized_embedding(value: Any) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise ManifestBuildError("InsightFace returned an invalid embedding")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ManifestBuildError("InsightFace returned a zero-norm embedding")
    return embedding / norm


def clip_box(box: Sequence[float], *, width: int, height: int) -> list[float]:
    clipped = [
        max(0.0, min(float(width), float(box[0]))),
        max(0.0, min(float(height), float(box[1]))),
        max(0.0, min(float(width), float(box[2]))),
        max(0.0, min(float(height), float(box[3]))),
    ]
    return _box(clipped, field="generated InsightFace bbox")


def load_case(
    samples_dir: str,
    case_id: str,
    *,
    description_overrides: dict[str, list[str]],
) -> dict[str, Any]:
    case_dir = Path(samples_dir) / case_id
    prompt_path = case_dir / "prompt.json"
    if not prompt_path.is_file():
        raise ManifestBuildError(f"canonical prompt not found: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as handle:
        data = json.load(handle)
    annotations = data.get("annotations") or {}
    raw_boxes = annotations.get("target_face_bboxes_first_frame") or []
    if not raw_boxes:
        raise ManifestBuildError(f"{case_id}: canonical target bbox list is empty")
    boxes = [
        _box(value, field=f"{case_id}.target_face_bboxes_first_frame[{index}]") for index, value in enumerate(raw_boxes)
    ]
    descriptions = description_overrides.get(
        case_id,
        annotations.get("target_descriptions") or [],
    )
    descriptions = [str(value).strip() for value in descriptions]
    if len(descriptions) != len(boxes) or any(not value for value in descriptions):
        raise ManifestBuildError(f"{case_id}: target_descriptions must contain {len(boxes)} non-blank values")
    size = data.get("frame_size") or {}
    try:
        source_width, source_height = int(size["w"]), int(size["h"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestBuildError(f"{case_id}: invalid frame_size") from error
    for target_id, box in enumerate(boxes):
        if box[0] < 0.0 or box[1] < 0.0 or box[2] > source_width or box[3] > source_height:
            raise ManifestBuildError(f"{case_id}: canonical target_{target_id:03d} bbox lies outside frame_size")
    source_image = case_dir / str(data.get("first_frame") or "first_frame.jpg")
    if not source_image.is_file():
        raise ManifestBuildError(f"{case_id}: source first frame not found: {source_image}")
    return {
        "case_id": case_id,
        "category": data.get("category"),
        "boxes": boxes,
        "descriptions": descriptions,
        "source_width": source_width,
        "source_height": source_height,
        "source_image": source_image,
    }


def detect_faces(app, frame_bgr: np.ndarray, *, min_det_score: float) -> list[dict]:
    detections: list[dict] = []
    for face in app.get(frame_bgr):
        raw_box = getattr(face, "bbox", None)
        raw_embedding = getattr(face, "normed_embedding", None)
        confidence = float(getattr(face, "det_score", 0.0))
        if raw_box is None or raw_embedding is None or confidence < min_det_score:
            continue
        detections.append(
            {
                "bbox": _box(np.asarray(raw_box).tolist(), field="InsightFace bbox"),
                "embedding": normalized_embedding(raw_embedding),
                "det_score": confidence,
            }
        )
    return detections


def _hungarian_pairs(scores: np.ndarray) -> list[tuple[int, int]]:
    if scores.ndim != 2:
        raise ValueError("assignment scores must be two-dimensional")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(scores, maximize=True)
    return [(int(row), int(column)) for row, column in zip(rows, columns)]


def source_anchors(
    canonical_boxes: Sequence[Sequence[float]],
    detections: Sequence[dict],
    *,
    iou_threshold: float,
) -> tuple[list[np.ndarray], list[dict]]:
    if len(detections) < len(canonical_boxes):
        raise ManifestBuildError(
            f"source first frame has {len(detections)} detected faces but {len(canonical_boxes)} canonical targets"
        )
    scores = np.asarray(
        [[iou(box, detection["bbox"]) for detection in detections] for box in canonical_boxes],
        dtype=np.float64,
    )
    pairs = _hungarian_pairs(scores)
    by_target = {target_id: detection_id for target_id, detection_id in pairs}
    if len(by_target) != len(canonical_boxes):
        raise ManifestBuildError("source first-frame target assignment is incomplete")
    anchors: list[np.ndarray] = []
    audit: list[dict] = []
    for target_id, canonical_box in enumerate(canonical_boxes):
        detection_id = by_target[target_id]
        overlap = float(scores[target_id, detection_id])
        if overlap < iou_threshold:
            raise ManifestBuildError(
                f"canonical target_{target_id:03d} has source detection IoU {overlap:.4f} below {iou_threshold:.4f}"
            )
        detection = detections[detection_id]
        anchors.append(detection["embedding"])
        audit.append(
            {
                "target_id": target_id,
                "canonical_bbox": list(canonical_box),
                "source_detection_bbox": detection["bbox"],
                "source_detection_score": detection["det_score"],
                "source_iou": overlap,
            }
        )
    return anchors, audit


def generated_target_boxes(
    anchors: Sequence[np.ndarray],
    detections: Sequence[dict],
    *,
    similarity_threshold: float,
    width: int,
    height: int,
) -> tuple[list[list[float]], list[dict], list[list[float]]]:
    if len(detections) < len(anchors):
        raise ManifestBuildError(
            f"generated first frame has {len(detections)} detected faces but {len(anchors)} canonical targets"
        )
    scores = np.asarray(
        [[float(np.dot(anchor, detection["embedding"])) for detection in detections] for anchor in anchors],
        dtype=np.float64,
    )
    pairs = _hungarian_pairs(scores)
    by_target = {target_id: detection_id for target_id, detection_id in pairs}
    if len(by_target) != len(anchors):
        raise ManifestBuildError("generated first-frame identity assignment is incomplete")
    boxes: list[list[float]] = []
    audit: list[dict] = []
    used: set[int] = set()
    for target_id in range(len(anchors)):
        detection_id = by_target[target_id]
        similarity = float(scores[target_id, detection_id])
        if similarity < similarity_threshold:
            raise ManifestBuildError(
                f"target_{target_id:03d} ArcFace similarity {similarity:.4f} below {similarity_threshold:.4f}"
            )
        detection = detections[detection_id]
        box = clip_box(detection["bbox"], width=width, height=height)
        boxes.append(box)
        audit.append(
            {
                "target_id": target_id,
                "generated_bbox": box,
                "generated_detection_score": detection["det_score"],
                "arcface_similarity": similarity,
            }
        )
        used.add(detection_id)
    extras = [
        clip_box(detection["bbox"], width=width, height=height)
        for detection_id, detection in enumerate(detections)
        if detection_id not in used
    ]
    return boxes, audit, extras


def scaled_canonical_boxes(
    case: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[list[float]]:
    scale_x = float(width) / float(case["source_width"])
    scale_y = float(height) / float(case["source_height"])
    return [
        clip_box(
            [
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ],
            width=width,
            height=height,
        )
        for box in case["boxes"]
    ]


def probe_first_frame(video_path: str) -> tuple[int, int, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ManifestBuildError(f"video is unreadable: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or width <= 0 or height <= 0:
        raise ManifestBuildError(f"failed to decode frame 0: {video_path}")
    return width, height, frame


def load_source_frame(case: dict[str, Any]) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(case["source_image"]))
    if frame is None:
        raise ManifestBuildError(f"failed to read source image: {case['source_image']}")
    height, width = frame.shape[:2]
    if (width, height) != (case["source_width"], case["source_height"]):
        raise ManifestBuildError(
            f"{case['case_id']}: prompt frame_size "
            f"{case['source_width']}x{case['source_height']} does not match source image "
            f"{width}x{height}"
        )
    return frame


def prepare_samples(
    samples: list[dict],
    dataset: Path,
    *,
    expected_revisits: int | None,
    app,
    audit_rows: list[dict],
) -> list[dict]:
    """Match each generated first frame to frozen source targets, in source order."""
    validate_expected_revisits(expected_revisits)
    # Compare identical case sets across models within each trajectory.
    matrix: dict[str, dict[str, set[str]]] = {}
    for sample in samples:
        matrix.setdefault(sample["condition_id"], {}).setdefault(sample["model_id"], set()).add(sample["sample_id"])
    for trajectory, model_cases in matrix.items():
        reference = next(iter(model_cases.values()))
        if any(cases != reference for cases in model_cases.values()):
            raise ManifestBuildError(f"model case sets differ for trajectory {trajectory}")
    cache = {}
    prepared = []
    for sample in samples:
        case_id = sample["sample_id"]
        audit = {"item_id": sample["item_id"], "status": "error"}
        try:
            if case_id not in cache:
                case = load_case(str(dataset), case_id, description_overrides={})
                detections = detect_faces(app, load_source_frame(case), min_det_score=0.0)
                anchors, source_audit = source_anchors(case["boxes"], detections, iou_threshold=0.1)
                cache[case_id] = case, anchors, source_audit
            case, anchors, source_audit = cache[case_id]
            width, height, frame = probe_first_frame(sample["video"])
            detections = detect_faces(app, frame, min_det_score=0.0)
            match_status, match_error = "ok", None
            try:
                boxes, generated_audit, extras = generated_target_boxes(
                    anchors,
                    detections,
                    similarity_threshold=0.35,
                    width=width,
                    height=height,
                )
                target_audit = [{**source, **generated} for source, generated in zip(source_audit, generated_audit)]
            except ManifestBuildError as error:
                # The reference gives the whole model/case/trajectory zero,
                # retaining the canonical target count and explicit reason.
                match_status, match_error = "failed", str(error)
                boxes = scaled_canonical_boxes(case, width=width, height=height)
                extras = [clip_box(detection["bbox"], width=width, height=height) for detection in detections]
                target_audit = [
                    {
                        **source,
                        "generated_bbox": boxes[index],
                        "assignment_method": "first_frame_match_failed",
                        "first_frame_match_error": match_error,
                    }
                    for index, source in enumerate(source_audit)
                ]
            prepared.append(
                {
                    **sample,
                    "annotations": {
                        "target_face_bboxes_first_frame": boxes,
                        "target_descriptions": list(case["descriptions"]),
                        "expected_revisits": expected_revisits,
                        "first_frame_match_status": match_status,
                        "first_frame_match_error": match_error,
                    },
                }
            )
            audit.update(
                status="ok",
                first_frame_match_status=match_status,
                first_frame_match_error=match_error,
                target_matches=target_audit,
                extra_first_frame_face_bboxes=extras,
            )
            print(f"{sample['item_id']}: first-frame match {match_status}", flush=True)
        except Exception as error:
            audit["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            audit_rows.append(audit)
    return prepared


def prepare_for_evaluation(
    samples: list[dict], context, *, expected_revisits: int | None, weight_metric: str = "human_identity"
) -> None:
    """Annotate standard-layout Human inputs before publishing the run manifest."""
    from .evaluator import _face_backend

    for sample in samples:
        if sample.get("category") not in (None, "human"):
            continue
        if "first_frame_match_status" in (sample.get("annotations") or {}):
            if expected_revisits is not None:
                sample["annotations"]["expected_revisits"] = validate_expected_revisits(expected_revisits)
            continue
        if sample.get("category") != "human":
            continue
        sample.pop("human_preparation_error", None)
        try:
            selected_revisits = validate_expected_revisits(
                expected_revisits
                if expected_revisits is not None
                else (sample.get("annotations") or {}).get("expected_revisits")
            )
            video = Path(sample["video"])
            if video.name != "video.mp4" or video.parent.name != sample.get("sample_id"):
                raise DataInvalidError("automatic Human preparation requires video.mp4 in the case folder")
            # Source image/annotations and generated video share the case folder.
            source_root = video.parents[1]
            try:
                case = load_case(str(source_root), sample["sample_id"], description_overrides={})
                load_source_frame(case)
            except Exception as error:
                raise DataInvalidError(str(error)) from error
            prepared = prepare_samples(
                [sample],
                source_root,
                expected_revisits=selected_revisits,
                app=_face_backend(context, weight_metric).app,
                audit_rows=[],
            )
            sample.update(prepared[0])
        except Exception as error:
            # Persist preparation failures as input provenance and let each
            # Human metric report them through the usual per-case error path.
            sample["human_preparation_error"] = {
                "status": "data_invalid"
                if isinstance(error, (DataInvalidError, ManifestBuildError))
                else "eval_failed",
                "message": f"{type(error).__name__}: {error}",
            }
