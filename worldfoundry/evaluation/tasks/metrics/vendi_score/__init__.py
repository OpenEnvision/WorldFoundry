"""Vendi Score diversity metric for generative model evaluation."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_vendi_score = lazy_export(f"{__name__}.wrapper", "compute_vendi_score", owner=__name__)
compute_vendi_score_from_features = lazy_export(
    f"{__name__}.wrapper", "compute_vendi_score_from_features", owner=__name__
)
package_root = package_root_export(__file__, owner=__name__)


__all__ = [
    "compute_vendi_score",
    "compute_vendi_score_from_features",
    "package_root",
]
