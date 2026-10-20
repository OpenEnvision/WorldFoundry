# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""One-to-one target association and cosine helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .schemas import BoundingBox


def normalized_embedding(value: np.ndarray | Sequence[float]) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise EvaluationError("embedding must be finite and non-empty")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise EvaluationError("embedding has zero or non-finite norm")
    return embedding / norm


def cosine_matrix(
    references: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
) -> np.ndarray:
    if not references or not candidates:
        return np.empty((len(references), len(candidates)), dtype=np.float32)
    reference_matrix = np.stack([normalized_embedding(value) for value in references], axis=0)
    candidate_matrix = np.stack([normalized_embedding(value) for value in candidates], axis=0)
    return reference_matrix @ candidate_matrix.T


def hungarian_matches(
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[int, int]:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2D matrix")
    if matrix.size == 0:
        return {}
    if not np.isfinite(matrix).all():
        raise EvaluationError("association matrix contains a non-finite value")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise EvaluationError(
            "Human Entity association requires scipy; install the WorldMemBench metric requirements"
        ) from error
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return {int(row): int(column) for row, column in zip(rows, columns) if matrix[row, column] >= threshold}


def assign_faces_to_person_boxes(
    face_boxes: dict[int, BoundingBox],
    person_boxes: Sequence[tuple[BoundingBox, float]],
) -> dict[int, BoundingBox]:
    """Assign distinct containing person boxes while maximizing confidence."""

    if not face_boxes or not person_boxes:
        return {}
    target_ids = sorted(face_boxes)
    scores = np.full((len(target_ids), len(person_boxes)), -1e9, dtype=np.float64)
    valid = np.zeros_like(scores, dtype=bool)
    for row, target_id in enumerate(target_ids):
        for column, (person_box, confidence) in enumerate(person_boxes):
            if person_box.contains(face_boxes[target_id].center):
                scores[row, column] = float(confidence)
                valid[row, column] = True
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise EvaluationError(
            "Human Appearance person assignment requires scipy; install the WorldMemBench metric requirements"
        ) from error
    rows, columns = linear_sum_assignment(scores, maximize=True)
    return {
        target_ids[int(row)]: person_boxes[int(column)][0] for row, column in zip(rows, columns) if valid[row, column]
    }
