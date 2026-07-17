"""ArtScore artness evaluation metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

load_artscore_model = lazy_export(f"{__name__}.wrapper", "load_artscore_model", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = ["load_artscore_model", "package_root"]
