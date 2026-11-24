# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Unit, item, and summary results shared by all metrics."""

import json
import shlex
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ResultStatus(str, Enum):
    OK = "ok"
    MODEL_FAILURE = "model_failure"
    DATA_INVALID = "data_invalid"
    EVAL_FAILED = "eval_failed"
    NOT_APPLICABLE = "not_applicable"

    @property
    def is_scored(self) -> bool:
        return self in {ResultStatus.OK, ResultStatus.MODEL_FAILURE}


@dataclass
class UnitResult:
    unit_id: str
    unit_type: str = "item"
    scores: dict[str, float] = field(default_factory=dict)
    status: ResultStatus = ResultStatus.OK
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemResult:
    item_id: str
    model_id: str | None = None
    sample_id: str | None = None
    condition_id: str | None = None
    category: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    num_units: int = 0
    status: ResultStatus = ResultStatus.OK
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryResult:
    num_items: int
    status_counts: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    group_by: str = "model_id"
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    groups: list[dict[str, Any]] = field(default_factory=list)


class ResultWriter:
    def __init__(self, output: str | Path) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def write_command(self, command: list[str] | None) -> None:
        invocation = shlex.join(command) if command is not None else "Python API call (no CLI command supplied)"
        (self.output / "command.txt").write_text(
            f"Working directory: {Path.cwd()}\nPython: {sys.executable}\n\n{invocation}\n",
            encoding="utf-8",
        )

    def write_metric(self, name: str, items: list[ItemResult], summary: SummaryResult) -> None:
        directory = self.output / "metrics" / name
        directory.mkdir(parents=True)
        with (directory / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(self._json(asdict(item)) + "\n")
        (directory / "summary.json").write_text(
            json.dumps(summary_payload(summary), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )


def summary_payload(summary: SummaryResult) -> dict:
    """Keep successful summaries concise, but retain non-OK coverage details."""

    def compact_group(value):
        value = dict(value)
        metadata = dict(value.pop("metadata", {}))
        if "n_total" in metadata:
            value["num_cases"] = metadata.pop("n_total")
        scored = metadata.pop("n_scored", None)
        if scored is not None and scored != value.get("num_cases"):
            value["num_scored"] = scored
        counts = {status.value: metadata.pop(f"n_{status.value}", 0) for status in ResultStatus}
        if any(count for status, count in counts.items() if status != "ok"):
            value["status_counts"] = {status: count for status, count in counts.items() if count}
        if metadata:
            value["metadata"] = metadata
        return value

    payload = {"num_cases": summary.num_items}
    if any(count for status, count in summary.status_counts.items() if status != "ok"):
        payload["status_counts"] = {status: count for status, count in summary.status_counts.items() if count}
    if summary.scores:
        payload["scores"] = summary.scores
    if summary.metadata:
        payload["metadata"] = summary.metadata
    if summary.results:
        payload["results"] = {key: compact_group(value) for key, value in summary.results.items()}
    if summary.groups:
        payload["groups"] = [compact_group(value) for value in summary.groups]
    return payload
