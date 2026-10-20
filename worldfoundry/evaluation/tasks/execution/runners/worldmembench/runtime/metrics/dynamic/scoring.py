# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Calculate Dynamic Memory scores from structured answers, without model inference."""

from statistics import mean

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import UnitResult

from .checklist import CATEGORIES


def score_answers(checklist, response, frame_indices) -> list[UnitResult]:
    """Stage two: compare each answer with its target and apply the event prerequisite."""
    event_started = response.event_started.answer == "yes"
    units = []
    for question, answer in zip(checklist.questions, response.answers, strict=True):
        ungated = float(answer.answer == question.expected_answer)
        units.append(
            UnitResult(
                question.id,
                unit_type="question",
                scores={"score": ungated if event_started else 0.0},
                metadata={
                    "category": question.category,
                    "question": question.question,
                    "expected_answer": question.expected_answer,
                    "predicted_answer": answer.answer,
                    "evidence_window": question.evidence_window,
                    "reason": answer.reason,
                    "evidence_image_numbers": list(answer.evidence_image_numbers),
                    "evidence_frame_indices": [frame_indices[n - 1] for n in answer.evidence_image_numbers],
                    "ungated_score": ungated,
                },
            )
        )
    return units


def aggregate_scores(units) -> dict[str, float]:
    if len(units) != 8:
        raise EvaluationError("Dynamic Memory requires 8 question units")
    scores = {"score": mean(unit.scores["score"] for unit in units)}
    for category in CATEGORIES:
        scores[f"{category}_score"] = mean(
            unit.scores["score"] for unit in units if unit.metadata["category"] == category
        )
    return scores
