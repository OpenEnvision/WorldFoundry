"""DreamSim perceptual image similarity metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_dreamsim = lazy_export(f"{__name__}.wrapper", "compute_dreamsim", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = [
    "compute_dreamsim",
    "package_root",
]
