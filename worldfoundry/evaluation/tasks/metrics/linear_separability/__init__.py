"""StyleGAN Linear Separability metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_linear_separability = lazy_export(
    f"{__name__}.wrapper", "compute_linear_separability", owner=__name__
)
package_root = package_root_export(__file__, owner=__name__)


__all__ = ["compute_linear_separability", "package_root"]
