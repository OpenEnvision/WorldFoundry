"""Feature Likelihood Divergence (FLD) metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

compute_fld = lazy_export(f"{__name__}.wrapper", "compute_fld", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = ["compute_fld", "package_root"]
