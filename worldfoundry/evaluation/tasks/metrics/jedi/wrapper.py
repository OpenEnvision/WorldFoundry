"""WorldFoundry facade over the vendored VideoJEDi package."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIM = 1280
DEFAULT_NUM_SAMPLES = 5000


def package_root() -> Path:
    return PACKAGE_ROOT


def bundled_config_path() -> Path:
    from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset

    return bundled_benchmark_asset("jedi", "vith16_ssv2_16x2x3.yaml")


def resolve_model_dir(explicit: str | Path | None = None) -> str:
    if explicit is not None:
        return str(Path(explicit).expanduser())
    for name in ("WORLDFOUNDRY_JEDI_MODEL_DIR", "WORLDFOUNDRY_JEDI_VJEPA_DIR", "WORLDFOUNDRY_VJEPA_MODEL_DIR"):
        value = os.environ.get(name)
        if value:
            return str(Path(value).expanduser())
    return str(Path(os.environ.get("WORLDFOUNDRY_HFD_ROOT", "cache/hfd")) / "jedi-vjepa")


def resolve_feature_path(explicit: str | Path | None = None) -> str | None:
    if explicit is not None:
        return str(Path(explicit).expanduser())
    value = os.environ.get("WORLDFOUNDRY_JEDI_FEATURE_PATH")
    return str(Path(value).expanduser()) if value else None


def resolve_config_path(explicit: str | Path | None = None) -> str:
    if explicit is not None:
        return str(Path(explicit).expanduser())
    env_value = os.environ.get("WORLDFOUNDRY_JEDI_CONFIG_PATH")
    if env_value:
        return str(Path(env_value).expanduser())
    bundled = bundled_config_path()
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError(
        "JEDi V-JEPA config not found. Set WORLDFOUNDRY_JEDI_CONFIG_PATH or bundle "
        "worldfoundry/data/benchmarks/assets/jedi/vith16_ssv2_16x2x3.yaml."
    )


@lru_cache(maxsize=1)
def _load_jedi_metric() -> Any:
    from worldfoundry.evaluation.tasks.metrics.jedi.JEDi import JEDiMetric as _JEDiMetric

    return _JEDiMetric


def mmd_poly(train_features: np.ndarray, test_features: np.ndarray) -> float:
    from worldfoundry.evaluation.tasks.metrics.jedi.mmd_polynomial import mmd_poly as _mmd_poly

    return float(_mmd_poly(train_features, test_features, degree=2, coef0=0) * 100.0)


def compute_jedi_from_features(train_features: np.ndarray, test_features: np.ndarray) -> float:
    return mmd_poly(np.asarray(train_features), np.asarray(test_features))


def JEDiMetric(
    *,
    feature_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Any:
    cls = _load_jedi_metric()
    return cls(
        feature_path=resolve_feature_path(feature_path),
        model_dir=resolve_model_dir(model_dir),
        config_path=resolve_config_path(config_path) if config_path is None else str(Path(config_path).expanduser()),
    )


def deterministic_feature_matrix(*, seed: str, num_samples: int, feature_dim: int) -> np.ndarray:
    rng = np.random.default_rng(list(seed.encode("utf-8")))
    return rng.random((num_samples, feature_dim), dtype=np.float64)


def compute_mock_jedi(
    *,
    train_seed: str = "jedi-train",
    test_seed: str = "jedi-test",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> float:
    train_features = deterministic_feature_matrix(seed=train_seed, num_samples=num_samples, feature_dim=feature_dim)
    test_features = deterministic_feature_matrix(seed=test_seed, num_samples=num_samples, feature_dim=feature_dim)
    return compute_jedi_from_features(train_features, test_features)
