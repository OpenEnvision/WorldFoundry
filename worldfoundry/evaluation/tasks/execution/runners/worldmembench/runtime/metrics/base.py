# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Common interface for metric evaluation and aggregation."""

from abc import ABC, abstractmethod
from collections import Counter

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.context import EvaluationContext
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.results import (
    ItemResult,
    SummaryResult,
    UnitResult,
)


class Metric(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def evaluate(self, sample: dict, context: EvaluationContext) -> list[UnitResult]:
        """Evaluate one sample and return its fine-grained units."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_item(self, sample: dict, units: list[UnitResult]) -> ItemResult:
        """Apply this metric's unit-to-item scoring rule."""
        raise NotImplementedError

    def aggregate_summary(self, items: list[ItemResult]) -> SummaryResult:
        """Count statuses; metrics add their explicitly defined score aggregation."""
        return SummaryResult(
            num_items=len(items),
            status_counts=dict(Counter(item.status.value for item in items)),
        )
