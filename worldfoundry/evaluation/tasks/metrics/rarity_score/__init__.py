"""Rarity Score metric for evaluating uncommonness of synthesized images."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_rarity_scores = lazy_export(f"{__name__}.wrapper", "compute_rarity_scores", owner=__name__)
compute_mean_rarity_score = lazy_export(
    f"{__name__}.wrapper", "compute_mean_rarity_score", owner=__name__
)
package_root = package_root_export(__file__, owner=__name__)


__all__ = [
    "compute_mean_rarity_score",
    "compute_rarity_scores",
    "package_root",
]
