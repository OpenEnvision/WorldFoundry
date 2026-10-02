"""Integration tests for the WorldReasonBench in-tree protocol adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from worldfoundry.evaluation.tasks.contracts import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.tasks.execution.runners.worldreasonbench.worldreasonbench_metrics import (
    normalize_pairwise,
    normalize_pointwise,
    normalize_qa,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "worldfoundry/data/benchmarks/assets/worldreasonbench"
RUNNER = (
    REPO_ROOT
    / "worldfoundry/evaluation/tasks/execution/runners/worldreasonbench/run_worldreasonbench_official_runner.py"
)


class WorldReasonBenchIntegrationTests(unittest.TestCase):
    def test_all_protocol_metrics(self) -> None:
        qa = normalize_qa(ASSETS / "summary.json")
        pointwise = normalize_pointwise(
            ASSETS / "pointwise_eval.jsonl",
            ASSETS / "pointwise_eval.induced_pairs.jsonl",
        )
        pairwise = normalize_pairwise(ASSETS / "pairwise_eval.jsonl")

        self.assertAlmostEqual(qa["metrics"]["score_pr"], 0.75**0.8 * 0.5**0.2)
        self.assertEqual(qa["metrics"]["dynamic_reasoning_score"], 0.5)
        self.assertAlmostEqual(qa["metrics"]["reasoning_gap"], 0.25)
        self.assertEqual(pointwise["metrics"]["pointwise_score"], 3.5)
        self.assertEqual(pointwise["metrics"]["pointwise_spearman"], 1.0)
        self.assertEqual(pointwise["metrics"]["induced_pairwise_accuracy"], 1.0)
        self.assertEqual(pairwise["metrics"]["pairwise_accuracy_with_ties"], 0.75)
        self.assertEqual(pairwise["metrics"]["pairwise_accuracy_without_ties"], 0.5)

    def test_fixture_runner_writes_complete_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pythonpath = [str(REPO_ROOT), *(path for path in sys.path if path)]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(pythonpath))}
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--run-fixture", "--output-dir", tmp, "--strict", "--json"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertTrue(scorecard["evaluation"]["full_protocol_set"])
            self.assertEqual(scorecard["metrics"]["summary"]["available_metric_count"], 17)
            self.assertFalse(scorecard["official_benchmark_verified"])

    def test_contract_and_specialized_runner_are_registered(self) -> None:
        contract = get_external_benchmark_contract("worldreasonbench")
        self.assertEqual(contract.display_name, "WorldReasonBench")
        self.assertEqual(len(contract.metric_ids), 17)
        self.assertIn("worldreasonbench", VIDEO_RUNNER_REGISTRY)
        self.assertTrue((REPO_ROOT / VIDEO_RUNNER_REGISTRY["worldreasonbench"].script).is_file())


if __name__ == "__main__":
    unittest.main()
