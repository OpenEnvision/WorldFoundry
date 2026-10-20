# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Face inference, seeded phase tracking, and frozen-target identity scoring."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import (
    DataInvalidError,
    EvaluationError,
)

from ..revisit_protocol import (
    MAX_SHORT_MISSED_FRAMES,
    REVISIT_BOUNDARY_BUFFER_FRAMES,
    infer_target_phase,
)
from .schemas import (
    BoundingBox,
    FaceObservation,
    FaceTrackAnalysis,
    HumanEntityProtocol,
    IdentityAnalysis,
    IdentityTargetResult,
    TargetPhase,
)
from .target_association import cosine_matrix, hungarian_matches, normalized_embedding

TRACK_ASSOCIATION_THRESHOLD = 0.4


class FaceIdentityBackend(Protocol):
    def embed_box(self, frame_rgb: np.ndarray, bbox: BoundingBox) -> np.ndarray: ...

    def detect(self, frame_rgb: np.ndarray) -> list[FaceObservation]: ...


class InsightFaceBackend:
    """SCRFD detection and ArcFace embeddings from InsightFace buffalo_l."""

    def __init__(
        self,
        *,
        device: str,
        model_name: str = "buffalo_l",
        model_root: str | Path,
        detection_size: int = 640,
    ) -> None:
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise EvaluationError(
                "InsightFace backend requires insightface and onnxruntime; "
                "install the WorldMemBench metric requirements"
            ) from error

        root = Path(model_root).expanduser().resolve()
        model_dir = root / "models" / model_name
        if not all((model_dir / name).is_file() for name in ("det_10g.onnx", "w600k_r50.onnx")):
            raise EvaluationError(
                f"InsightFace model directory not found: {model_dir}; "
                "WorldMemBench never downloads evaluator weights implicitly"
            )
        if device.startswith("cuda"):
            # Changing LD_LIBRARY_PATH after Python starts does not reliably
            # affect dlopen. ORT's loader explicitly preloads CUDA/cuDNN from
            # the active environment's nvidia packages. If the caller already
            # supplied a cuDNN path, avoid loading the same libraries twice.
            preload = getattr(ort, "preload_dlls", None)
            if preload is not None and "cudnn" not in os.environ.get("LD_LIBRARY_PATH", ""):
                preload(msvc=False, directory="")
            raw_device = device.partition(":")[2]
            device_id = int(raw_device) if raw_device else 0
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            device_id = -1
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=model_name,
            root=str(root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
            # Enable TF32 for CUDA face inference.
            provider_options=([{"device_id": device_id, "use_tf32": 1}, {}] if device_id >= 0 else [{}]),
        )
        self.app.prepare(
            ctx_id=device_id,
            det_size=(detection_size, detection_size),
        )
        if device_id >= 0:
            for model in self.app.models.values():
                if "CUDAExecutionProvider" not in model.session.get_providers():
                    raise EvaluationError(
                        "InsightFace could not activate CUDAExecutionProvider; check ONNX Runtime CUDA/cuDNN libraries"
                    )
        self._cached_frame: np.ndarray | None = None
        self._cached_observations: list[FaceObservation] = []

    def _observations(self, frame_rgb: np.ndarray) -> list[FaceObservation]:
        if frame_rgb is self._cached_frame:
            return list(self._cached_observations)
        import cv2

        faces = self.app.get(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        observations: list[FaceObservation] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            raw_box = getattr(face, "bbox", None)
            if embedding is None or raw_box is None:
                continue
            try:
                bbox = BoundingBox.from_value(
                    np.asarray(raw_box).tolist(),
                    field_name="InsightFace bbox",
                )
                normalized = normalized_embedding(embedding)
            except (ValueError, EvaluationError):
                continue
            observations.append(FaceObservation(bbox=bbox, embedding=normalized))
        self._cached_frame = frame_rgb
        self._cached_observations = observations
        return list(observations)

    @staticmethod
    def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = first.area + second.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def embed_box(self, frame_rgb: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        observations = self._observations(frame_rgb)
        if not observations:
            raise DataInvalidError("InsightFace found no face in the first frame")
        match = max(
            observations,
            key=lambda observation: self._box_iou(observation.bbox, bbox),
        )
        if self._box_iou(match.bbox, bbox) < 0.1:
            raise DataInvalidError("target face bbox does not overlap an InsightFace detection")
        return match.embedding

    def detect(self, frame_rgb: np.ndarray) -> list[FaceObservation]:
        return self._observations(frame_rgb)


def _frame_matches(
    reference_sets: Sequence[Sequence[np.ndarray]],
    candidates: Sequence[FaceObservation],
    *,
    threshold: float,
    target_ids: Sequence[int] | None = None,
) -> dict[int, FaceObservation]:
    if target_ids is None:
        target_ids = tuple(range(len(reference_sets)))
    if len(target_ids) != len(reference_sets):
        raise ValueError("target_ids and reference_sets must have the same length")
    if not candidates or not reference_sets:
        return {}
    candidate_embeddings = [candidate.embedding for candidate in candidates]
    score_rows: list[np.ndarray] = []
    for references in reference_sets:
        similarities = cosine_matrix(references, candidate_embeddings)
        score_rows.append(similarities.max(axis=0))
    scores = np.stack(score_rows, axis=0)
    matches = hungarian_matches(scores, threshold=threshold)
    return {target_ids[row_index]: candidates[candidate_index] for row_index, candidate_index in matches.items()}


def _seeded_track_associations(
    *,
    anchors: Sequence[np.ndarray],
    seeded_association: dict[int, FaceObservation],
    candidates_by_frame: dict[int, list[FaceObservation]],
    total_frames: int,
    threshold: float,
) -> dict[int, dict[int, FaceObservation]]:
    """MBench-style rolling tracks seeded only by the frozen frame-0 faces."""

    track_embeddings: list[list[np.ndarray]] = [[anchor] for anchor in anchors]
    missed_frames = [0 for _ in anchors]
    references_frozen = [False for _ in anchors]
    associations: dict[int, dict[int, FaceObservation]] = {0: seeded_association}
    for frame_index in range(1, total_frames):
        reference_sets = [[embeddings[-1], np.mean(embeddings, axis=0)] for embeddings in track_embeddings]
        matches = _frame_matches(
            reference_sets,
            candidates_by_frame[frame_index],
            threshold=threshold,
        )
        associations[frame_index] = matches
        for target_id in range(len(anchors)):
            observation = matches.get(target_id)
            if observation is None:
                if not references_frozen[target_id]:
                    missed_frames[target_id] += 1
                    if missed_frames[target_id] > MAX_SHORT_MISSED_FRAMES:
                        references_frozen[target_id] = True
                continue
            missed_frames[target_id] = 0
            if references_frozen[target_id]:
                continue
            track_embeddings[target_id].append(normalized_embedding(observation.embedding))
    return associations


def prepare_face_tracks(
    *,
    protocol: HumanEntityProtocol,
    decoded_frames: dict[int, np.ndarray],
    backend: FaceIdentityBackend,
    track_association_threshold: float = TRACK_ASSOCIATION_THRESHOLD,
) -> FaceTrackAnalysis:
    if not 0.0 <= track_association_threshold <= 1.0:
        raise ValueError("track_association_threshold must be in [0, 1]")
    first_frame = decoded_frames.get(0)
    if first_frame is None:
        raise EvaluationError("the decoded frame set does not contain frame 0")
    anchors = tuple(normalized_embedding(backend.embed_box(first_frame, bbox)) for bbox in protocol.target_face_boxes)
    if len({anchor.shape for anchor in anchors}) != 1:
        raise EvaluationError("ArcFace anchors have inconsistent dimensions")
    seeded_association = {
        target_id: FaceObservation(box, anchor)
        for target_id, (box, anchor) in enumerate(zip(protocol.target_face_boxes, anchors))
    }
    candidates_by_frame: dict[int, list[FaceObservation]] = {}
    for frame_index in range(1, protocol.total_frames):
        try:
            frame = decoded_frames[frame_index]
        except KeyError as error:
            raise EvaluationError(f"the decoded frame set does not contain frame {frame_index}") from error
        candidates_by_frame[frame_index] = backend.detect(frame)
    track_associations = _seeded_track_associations(
        anchors=anchors,
        seeded_association=seeded_association,
        candidates_by_frame=candidates_by_frame,
        total_frames=protocol.total_frames,
        threshold=track_association_threshold,
    )
    return FaceTrackAnalysis(
        anchors=anchors,
        seeded_association=seeded_association,
        candidates_by_frame=candidates_by_frame,
        track_associations=track_associations,
    )


def build_frozen_reference_libraries(
    face_tracks: FaceTrackAnalysis,
    target_phases: Sequence[TargetPhase],
) -> tuple[tuple[np.ndarray, ...], ...]:
    references: list[list[np.ndarray]] = [[anchor] for anchor in face_tracks.anchors]
    for phase in target_phases:
        for frame_index in phase.visible_indices:
            if frame_index == 0:
                continue
            observation = face_tracks.track_associations[frame_index].get(phase.target_id)
            if observation is not None:
                references[phase.target_id].append(normalized_embedding(observation.embedding))
    return tuple(tuple(values) for values in references)


def _face_phases_from_associations(
    *,
    protocol: HumanEntityProtocol,
    associations: dict[int, dict[int, FaceObservation]],
    max_revisit_gap_frames: int,
    revisit_boundary_buffer_frames: int,
) -> tuple[TargetPhase, ...]:
    return tuple(
        infer_target_phase(
            target_id=target_id,
            total_frames=protocol.total_frames,
            matched_frame_indices={
                frame_index for frame_index, matches in associations.items() if target_id in matches
            },
            max_revisit_gap_frames=max_revisit_gap_frames,
            expected_revisits=protocol.expected_revisits,
            revisit_boundary_buffer_frames=revisit_boundary_buffer_frames,
        )
        for target_id in range(protocol.n_targets)
    )


def analyze_identity(
    *,
    protocol: HumanEntityProtocol,
    face_tracks: FaceTrackAnalysis | None = None,
    decoded_frames: dict[int, np.ndarray],
    backend: FaceIdentityBackend,
    video_path: str,
    target_phases: tuple[TargetPhase, ...],
    body_assignments: dict[int, dict[int, BoundingBox]],
    track_association_threshold: float = TRACK_ASSOCIATION_THRESHOLD,
    max_revisit_gap_frames: int = MAX_SHORT_MISSED_FRAMES,
    revisit_boundary_buffer_frames: int = REVISIT_BOUNDARY_BUFFER_FRAMES,
) -> IdentityAnalysis:
    if not 0.0 <= track_association_threshold <= 1.0:
        raise ValueError("track_association_threshold must be in [0, 1]")
    if face_tracks is None:
        face_tracks = prepare_face_tracks(
            protocol=protocol,
            decoded_frames=decoded_frames,
            backend=backend,
            track_association_threshold=track_association_threshold,
        )
    appearance_target_phases = target_phases
    target_phases = _face_phases_from_associations(
        protocol=protocol,
        associations=face_tracks.track_associations,
        max_revisit_gap_frames=max_revisit_gap_frames,
        revisit_boundary_buffer_frames=revisit_boundary_buffer_frames,
    )
    # Identity and appearance infer D independently.  Only the face-derived V
    # may extend the frozen identity library; return observations never do.
    frozen_references = build_frozen_reference_libraries(
        face_tracks,
        target_phases,
    )
    associations = face_tracks.track_associations

    results: list[IdentityTargetResult] = []
    for phase in target_phases:
        target_id = phase.target_id

        return_indices = phase.return_indices
        frame_scores = {}
        returned_embeddings: list[np.ndarray] = []
        returned_by_frame: dict[int, np.ndarray] = {}
        for frame_index in return_indices:
            observation = associations.get(frame_index, {}).get(target_id)
            if observation is None:
                frame_scores[frame_index] = 0.0
                continue
            embedding = normalized_embedding(observation.embedding)
            similarities = cosine_matrix(
                frozen_references[target_id],
                [embedding],
            )[:, 0]
            frame_scores[frame_index] = float(np.clip(similarities, 0.0, 1.0).max())
            returned_embeddings.append(embedding)
            returned_by_frame[frame_index] = embedding
        coverage = len(returned_embeddings) / len(return_indices) if return_indices else 0.0
        if returned_embeddings and phase.is_scorable:
            cross = cosine_matrix(frozen_references[target_id], returned_embeddings)
            best_by_embedding = np.clip(cross, 0.0, 1.0).max(axis=0)
            similarity = float(best_by_embedding.mean())
        else:
            similarity = 0.0
        revisit_scores: list[float] = []
        revisit_coverages: list[float] = []
        n_revisit_embeddings: list[int] = []
        for revisit_indices in phase.revisit_indices:
            embeddings = [returned_by_frame[index] for index in revisit_indices if index in returned_by_frame]
            n_revisit_embeddings.append(len(embeddings))
            revisit_coverages.append(len(embeddings) / len(revisit_indices) if revisit_indices else 0.0)
            if embeddings and phase.is_scorable:
                cross = cosine_matrix(frozen_references[target_id], embeddings)
                best_by_embedding = np.clip(cross, 0.0, 1.0).max(axis=0)
                revisit_scores.append(float(best_by_embedding.mean()))
            else:
                revisit_scores.append(0.0)
        results.append(
            IdentityTargetResult(
                target_id=target_id,
                phase=phase,
                frame_scores=frame_scores,
                similarity=similarity,
                coverage=coverage,
                n_reference_embeddings=len(frozen_references[target_id]),
                n_return_embeddings=len(returned_embeddings),
                revisit_scores=tuple(revisit_scores),
                revisit_coverages=tuple(revisit_coverages),
                n_revisit_embeddings=tuple(n_revisit_embeddings),
            )
        )
    return IdentityAnalysis(
        protocol=protocol,
        target_phases=target_phases,
        appearance_target_phases=appearance_target_phases,
        body_assignments=body_assignments,
        target_results=tuple(results),
        # Keep the cross-metric cache lightweight. Appearance re-decodes the
        # full item and attaches frames to a transient analysis copy.
        decoded_frames={},
        video_path=video_path,
        max_revisit_gap_frames=max_revisit_gap_frames,
        revisit_boundary_buffer_frames=revisit_boundary_buffer_frames,
    )
