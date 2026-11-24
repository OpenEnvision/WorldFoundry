# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Per-run configuration, resource lifecycle, and generic value cache."""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


class EvaluationContext:
    def __init__(
        self,
        *,
        device: str,
        cache_dir: str | Path,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = device
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})
        self.logger = logger or logging.getLogger("worldmembench")
        self._resources: dict[Hashable, Any] = {}
        self._cache: dict[Hashable, Any] = {}

    def metric_config(self, metric_name: str) -> dict[str, Any]:
        metrics = self.config.get("metrics") or {}
        value = metrics.get(metric_name) or {}
        if not isinstance(value, dict):
            raise TypeError(f"config.metrics.{metric_name} must be a mapping")
        return dict(value)

    def get_resource(self, key: Hashable, factory: Callable[[], T]) -> T:
        if key not in self._resources:
            self._resources[key] = factory()
        return cast(T, self._resources[key])

    def get_cached(self, key: Hashable, factory: Callable[[], T]) -> T:
        if key not in self._cache:
            self._cache[key] = factory()
        return cast(T, self._cache[key])

    def close(self) -> None:
        for resource in reversed(list(self._resources.values())):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self._resources.clear()
        self._cache.clear()

    def clear_frame_cache(self) -> None:
        """Release decoded frames after all metrics have evaluated one sample."""
        self._cache.clear()
