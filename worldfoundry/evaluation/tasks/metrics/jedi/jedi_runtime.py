"""JEDi scorer runtime with mock and in-tree VideoJEDi backends."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics.jedi.wrapper import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_NUM_SAMPLES,
    compute_jedi_from_features,
    compute_mock_jedi,
    deterministic_feature_matrix,
    resolve_feature_path,
    resolve_model_dir,
)


@dataclass(frozen=True)
class JEDiScorerConfig:
    backend: str
    num_samples: int = DEFAULT_NUM_SAMPLES
    feature_dim: int = DEFAULT_FEATURE_DIM
    train_seed: str = "jedi-train"
    test_seed: str = "jedi-test"
    model_dir: str | None = None
    feature_path: str | None = None


def scorer_config_from_env() -> JEDiScorerConfig:
    backend = (os.environ.get("WORLDFOUNDRY_JEDI_BACKEND") or "mock").strip().lower()
    return JEDiScorerConfig(
        backend=backend,
        num_samples=int(os.environ.get("WORLDFOUNDRY_JEDI_NUM_SAMPLES", DEFAULT_NUM_SAMPLES)),
        feature_dim=int(os.environ.get("WORLDFOUNDRY_JEDI_FEATURE_DIM", DEFAULT_FEATURE_DIM)),
        train_seed=os.environ.get("WORLDFOUNDRY_JEDI_TRAIN_SEED", "jedi-train"),
        test_seed=os.environ.get("WORLDFOUNDRY_JEDI_TEST_SEED", "jedi-test"),
        model_dir=resolve_model_dir(),
        feature_path=resolve_feature_path(),
    )


def _load_feature_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    path = Path(str(value))
    if path.is_file() and path.suffix == ".npy":
        return np.load(path)
    return None


def _feature_arrays_from_payload(payload: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(payload, Mapping):
        return None
    train = _load_feature_array(payload.get("train_features"))
    if train is None:
        train = _load_feature_array(payload.get("reference_features"))
    test = _load_feature_array(payload.get("test_features"))
    if test is None:
        test = _load_feature_array(payload.get("generated_features"))
    if train is None or test is None:
        train_path = payload.get("train_features_path") or payload.get("reference_features_path")
        test_path = payload.get("test_features_path") or payload.get("generated_features_path")
        if train_path:
            train = _load_feature_array(train_path)
        if test_path:
            test = _load_feature_array(test_path)
    if train is None or test is None:
        return None
    return train, test


def score_from_feature_payload(payload: Mapping[str, Any] | None) -> float | None:
    features = _feature_arrays_from_payload(payload)
    return compute_jedi_from_features(*features) if features is not None else None


def run_jedi_scorer(
    *,
    output_dir: Path,
    config: JEDiScorerConfig | None = None,
    feature_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or scorer_config_from_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _feature_arrays_from_payload(feature_payload)
    if features is not None:
        train, test = features
        score = compute_jedi_from_features(train, test)
        backend = "precomputed_features"
        reference_num_samples, generated_num_samples = len(train), len(test)
        feature_dim = train.shape[1]
    elif cfg.backend == "mock":
        score = compute_mock_jedi(
            train_seed=cfg.train_seed,
            test_seed=cfg.test_seed,
            num_samples=cfg.num_samples,
            feature_dim=cfg.feature_dim,
        )
        backend = "mock"
        reference_num_samples = generated_num_samples = cfg.num_samples
        feature_dim = cfg.feature_dim
    elif cfg.backend in {"official", "videojedi", "vjepa"}:
        raise NotImplementedError(
            "Official JEDi feature extraction requires V-JEPA dataloaders and GPU weights. "
            "Provide precomputed train/test feature arrays or use backend=mock for CI."
        )
    else:
        raise ValueError(f"unsupported JEDi backend: {cfg.backend}")

    result = {
        "metric_id": "jedi_score",
        "score": score,
        "backend": backend,
        "num_samples": reference_num_samples if reference_num_samples == generated_num_samples else None,
        "reference_num_samples": reference_num_samples,
        "generated_num_samples": generated_num_samples,
        "feature_dim": feature_dim,
    }
    results_path = output_dir / "jedi_score.json"
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["results_path"] = str(results_path.resolve())
    return result


def mock_feature_payload(*, train_seed: str, test_seed: str, num_samples: int, feature_dim: int) -> dict[str, Any]:
    return {
        "train_features": deterministic_feature_matrix(
            seed=train_seed,
            num_samples=num_samples,
            feature_dim=feature_dim,
        ).tolist(),
        "test_features": deterministic_feature_matrix(
            seed=test_seed,
            num_samples=num_samples,
            feature_dim=feature_dim,
        ).tolist(),
    }
