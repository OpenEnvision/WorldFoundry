"""In-tree VideoJEDi metric package for WorldFoundry video distribution scoring."""

from __future__ import annotations

from worldfoundry.evals.tasks.metrics.distribution.jedi.wrapper import (
    JEDiMetric,
    bundled_config_path,
    compute_jedi_from_features,
    compute_mock_jedi,
    mmd_poly,
    package_root,
    resolve_config_path,
    resolve_feature_path,
    resolve_model_dir,
)

__all__ = [
    "JEDiMetric",
    "bundled_config_path",
    "compute_jedi_from_features",
    "compute_mock_jedi",
    "mmd_poly",
    "package_root",
    "resolve_config_path",
    "resolve_feature_path",
    "resolve_model_dir",
]
