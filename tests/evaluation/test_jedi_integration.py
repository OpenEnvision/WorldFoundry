"""Smoke tests for in-tree VideoJEDi metric integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


class JEDiIntegrationSmokeTests(unittest.TestCase):
    def test_jedi_wrapper_compute_from_features(self) -> None:
        from worldfoundry.evaluation.tasks.metrics.jedi import compute_jedi_from_features, compute_mock_jedi

        score = compute_mock_jedi(num_samples=128, feature_dim=64)
        self.assertGreaterEqual(score, 0.0)
        train = np.random.rand(128, 64)
        test = np.random.rand(128, 64)
        self.assertGreaterEqual(compute_jedi_from_features(train, test), 0.0)

    def test_bundled_jedi_assets_resolve_without_env(self) -> None:
        from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset
        from worldfoundry.evaluation.tasks.metrics.jedi import bundled_config_path, resolve_config_path

        metadata = json.loads(bundled_benchmark_asset("jedi", "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["metric_id"], "jedi_score")
        config_path = resolve_config_path()
        self.assertTrue(Path(config_path).is_file())
        self.assertEqual(Path(config_path), bundled_config_path())

    def test_mock_runtime_writes_score(self) -> None:
        from worldfoundry.evaluation.tasks.metrics.jedi.jedi_runtime import JEDiScorerConfig, run_jedi_scorer

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_jedi_scorer(
                output_dir=Path(tmp),
                config=JEDiScorerConfig(backend="mock", num_samples=64, feature_dim=32),
            )
            self.assertEqual(summary["backend"], "mock")
            self.assertIn("results_path", summary)
            payload = json.loads(Path(summary["results_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["metric_id"], "jedi_score")

    def test_local_jedi_evaluator_from_payload(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners._benchmark_metrics import evaluate_external_metric
        from worldfoundry.evaluation.tasks.metrics.jedi.jedi_runtime import mock_feature_payload

        payload = mock_feature_payload(
            train_seed="train-fixture",
            test_seed="test-fixture",
            num_samples=32,
            feature_dim=16,
        )
        result = evaluate_external_metric(
            "fetv",
            "jedi_score",
            generated_artifact_manifest=payload,
        )
        self.assertTrue(result.valid)
        self.assertGreaterEqual(float(result.normalized_value), 0.0)


if __name__ == "__main__":
    unittest.main()
