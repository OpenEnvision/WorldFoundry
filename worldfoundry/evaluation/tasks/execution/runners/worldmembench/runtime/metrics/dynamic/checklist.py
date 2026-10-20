# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Validate one frozen eight-question Dynamic Memory checklist."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import DataInvalidError

CATEGORIES = (
    "return_progress",
    "persistence_reset",
    "final_state",
    "causal_physics",
)
EXPECTED_IDS = tuple(f"{category}_{index:02d}" for category in CATEGORIES for index in (1, 2))
EVIDENCE_WINDOWS = {
    "returned_view",
    "returned_hold_sequence",
    "returned_and_later_visible_sequence",
    "final_stationary_view",
    "ordered_visible_sequence",
}


@dataclass(frozen=True)
class ChecklistQuestion:
    id: str
    category: str
    question: str
    expected_answer: Literal["yes", "no"]
    evidence_window: str


@dataclass(frozen=True)
class CaseChecklist:
    case_id: str
    questions: tuple[ChecklistQuestion, ...]


def parse_checklist(value: Any) -> CaseChecklist:
    if not isinstance(value, dict):
        raise DataInvalidError("checklist case must be an object")
    case_id = str(value.get("case_id", ""))
    if not re.fullmatch(r"case_[0-9]{3}", case_id):
        raise DataInvalidError(f"invalid checklist case_id: {case_id!r}")
    raw_questions = value.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) != 8:
        raise DataInvalidError(f"{case_id}: checklist must contain 8 questions")
    questions: list[ChecklistQuestion] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise DataInvalidError(f"{case_id}: question {index} is not an object")
        question_id = str(raw.get("id", ""))
        expected_id = EXPECTED_IDS[index]
        category = str(raw.get("category", ""))
        if question_id != expected_id or category != expected_id.rsplit("_", 1)[0]:
            raise DataInvalidError(f"{case_id}: invalid question layout at {index}")
        text = str(raw.get("question", "")).strip()
        if not text.endswith("?"):
            raise DataInvalidError(f"{case_id}/{question_id}: invalid question text")
        expected_answer = str(raw.get("expected_answer", "")).lower()
        if expected_answer not in {"yes", "no"}:
            raise DataInvalidError(f"{case_id}/{question_id}: expected_answer must be yes or no")
        evidence_window = str(raw.get("evidence_window", "")).strip()
        if evidence_window not in EVIDENCE_WINDOWS:
            raise DataInvalidError(f"{case_id}/{question_id}: invalid evidence_window {evidence_window!r}")
        questions.append(
            ChecklistQuestion(
                id=question_id,
                category=category,
                question=text,
                expected_answer=expected_answer,  # type: ignore[arg-type]
                evidence_window=evidence_window,
            )
        )
    return CaseChecklist(case_id=case_id, questions=tuple(questions))
