# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Explicit name-based registration of metric implementations."""

import re

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        # Metric names are also output-directory components.
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", metric.name):
            raise ValueError("metric names must match [a-z][a-z0-9_-]*")
        if metric.name in self._metrics:
            raise ValueError(f"metric already registered: {metric.name}")
        self._metrics[metric.name] = metric

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def get(self, name: str) -> Metric:
        if name not in self._metrics:
            raise ValueError(f"unknown metric: {name}; available: {', '.join(self.names())}")
        return self._metrics[name]
