"""FaceScore face quality metric."""

from __future__ import annotations

from worldfoundry.evaluation.tasks.metrics._shared.lazy import lazy_export, package_root_export

FaceScoreModel = lazy_export(f"{__name__}.wrapper", "FaceScoreModel", owner=__name__)
compute_facescore = lazy_export(f"{__name__}.wrapper", "compute_facescore", owner=__name__)
package_root = package_root_export(__file__, owner=__name__)


__all__ = ["FaceScoreModel", "compute_facescore", "package_root"]
