"""CMMD (CLIP Maximum Mean Discrepancy) metric for image generation evaluation."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_cmmd = lazy_export(f"{__name__}.wrapper", "compute_cmmd", owner=__name__)
compute_cmmd_from_embeddings = lazy_export(
    f"{__name__}.wrapper", "compute_cmmd_from_embeddings", owner=__name__
)
package_root = package_root_export(__file__, owner=__name__)


__all__ = [
    "compute_cmmd",
    "compute_cmmd_from_embeddings",
    "package_root",
]
