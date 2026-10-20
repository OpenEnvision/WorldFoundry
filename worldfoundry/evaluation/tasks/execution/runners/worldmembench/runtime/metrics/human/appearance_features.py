# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Phase-local body tracking and masked DINOv2 appearance scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

from .person_mask_tracker import MaskPropagator
from .schemas import (
    AppearanceAnalysis,
    AppearanceTargetResult,
    BoundingBox,
    IdentityAnalysis,
)
from .target_association import (
    cosine_matrix,
    normalized_embedding,
)


class AppearanceBackend(Protocol):
    def extract(
        self,
        identity: IdentityAnalysis,
    ) -> dict[int, dict[int, np.ndarray]]: ...


class AppearanceFeatureExtractor(Protocol):
    def embed_batch(
        self,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray | None]: ...


class YoloSamDinoAppearanceBackend:
    def __init__(
        self,
        *,
        mask_propagator: MaskPropagator,
        feature_extractor: AppearanceFeatureExtractor,
    ) -> None:
        self.mask_propagator = mask_propagator
        self.feature_extractor = feature_extractor

    def _target_phase_features(
        self,
        identity: IdentityAnalysis,
        target_id: int,
        phase_indices: Sequence[int],
        person_assignments: dict[int, dict[int, BoundingBox]],
    ) -> dict[int, dict[int, np.ndarray]]:
        if not phase_indices:
            return {}
        frames = [identity.decoded_frames[index] for index in phase_indices]
        target_prompts: list[tuple[int, BoundingBox]] = []
        for local_index, frame_index in enumerate(phase_indices):
            person_box = person_assignments.get(frame_index, {}).get(target_id)
            if person_box is not None:
                target_prompts.append((local_index, person_box))
        prompts = {target_id: target_prompts} if target_prompts else {}
        # Appearance keeps only frames with a direct body-box prompt below.
        # Propagated masks cannot replace those direct masks and are discarded.
        local_masks = self.mask_propagator.track(
            frames,
            prompts,
            propagate=False,
        )
        features: dict[int, dict[int, np.ndarray]] = {}
        sample_keys: list[tuple[int, int]] = []
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for tracked_target_id, target_masks in sorted(local_masks.items()):
            for local_index, mask in sorted(target_masks.items()):
                if local_index < 0 or local_index >= len(phase_indices):
                    raise EvaluationError("SAM2 returned an out-of-range phase frame")
                frame_index = phase_indices[local_index]
                # A propagated mask is valid only while the body association
                # confirms that the tracked mask still overlaps a person.
                if tracked_target_id not in person_assignments.get(frame_index, {}):
                    continue
                sample_keys.append((tracked_target_id, frame_index))
                samples.append(
                    (
                        identity.decoded_frames[frame_index],
                        np.asarray(mask, dtype=bool),
                    )
                )
        embeddings = self.feature_extractor.embed_batch(samples)
        if len(embeddings) != len(sample_keys):
            raise EvaluationError("DINO batch output count does not match its input count")
        for (tracked_target_id, frame_index), embedding in zip(
            sample_keys,
            embeddings,
        ):
            if embedding is not None:
                features.setdefault(tracked_target_id, {})[frame_index] = normalized_embedding(embedding)
        return features

    def extract(self, identity: IdentityAnalysis) -> dict[int, dict[int, np.ndarray]]:
        person_assignments = identity.body_assignments
        merged: dict[int, dict[int, np.ndarray]] = {}
        for phase in identity.appearance_target_phases:
            if not phase.is_scorable:
                continue
            visible = self._target_phase_features(
                identity,
                phase.target_id,
                phase.visible_indices,
                person_assignments,
            )
            returned = self._target_phase_features(
                identity,
                phase.target_id,
                phase.return_indices,
                person_assignments,
            )
            for source in (visible, returned):
                for target_id, frame_features in source.items():
                    merged.setdefault(target_id, {}).update(frame_features)
        return merged


def analyze_appearance(
    identity: IdentityAnalysis,
    backend: AppearanceBackend,
) -> AppearanceAnalysis:
    features = backend.extract(identity)
    results: list[AppearanceTargetResult] = []
    for phase in identity.appearance_target_phases:
        target_id = phase.target_id
        target_features = features.get(target_id, {})
        references = [target_features[index] for index in phase.visible_indices if index in target_features]
        frame_scores: dict[int, float] = {}
        n_return_features = 0
        for frame_index in phase.return_indices:
            candidate = target_features.get(frame_index)
            if candidate is None:
                frame_scores[frame_index] = 0.0
                continue
            n_return_features += 1
            if not references or not phase.is_scorable:
                frame_scores[frame_index] = 0.0
                continue
            similarities = cosine_matrix(references, [candidate])[:, 0]
            frame_scores[frame_index] = float(np.clip(similarities.max(), 0.0, 1.0))
        return_frame_count = len(phase.return_indices)
        score = sum(frame_scores.values()) / n_return_features if n_return_features else 0.0
        coverage = n_return_features / return_frame_count if return_frame_count else 0.0
        if not phase.is_scorable:
            score = 0.0
        revisit_scores: list[float] = []
        revisit_coverages: list[float] = []
        n_revisit_features: list[int] = []
        for revisit_indices in phase.revisit_indices:
            feature_indices = [index for index in revisit_indices if index in target_features]
            n_features = len(feature_indices)
            n_revisit_features.append(n_features)
            revisit_coverages.append(n_features / len(revisit_indices) if revisit_indices else 0.0)
            if n_features and phase.is_scorable:
                revisit_scores.append(sum(frame_scores[index] for index in feature_indices) / n_features)
            else:
                revisit_scores.append(0.0)
        results.append(
            AppearanceTargetResult(
                target_id=target_id,
                phase=phase,
                frame_scores=frame_scores,
                score=score,
                reappearance_coverage=coverage,
                n_reference_features=len(references),
                n_return_features=n_return_features,
                revisit_scores=tuple(revisit_scores),
                revisit_coverages=tuple(revisit_coverages),
                n_revisit_features=tuple(n_revisit_features),
            )
        )
    return AppearanceAnalysis(target_results=tuple(results))
