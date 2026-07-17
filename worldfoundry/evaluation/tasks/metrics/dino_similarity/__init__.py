"""DINO similarity metric for condition consistency evaluation."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_dino_similarity = lazy_export(f"{__name__}.wrapper", "compute_dino_similarity", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = [
    "compute_dino_similarity",
    "package_root",
]
