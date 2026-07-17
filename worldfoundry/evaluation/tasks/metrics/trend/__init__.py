"""TREND distribution metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_trend = lazy_export(f"{__name__}.wrapper", "compute_trend", owner=__name__)
compute_trend_jsd = lazy_export(f"{__name__}.wrapper", "compute_trend_jsd", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = ["compute_trend", "compute_trend_jsd", "package_root"]
