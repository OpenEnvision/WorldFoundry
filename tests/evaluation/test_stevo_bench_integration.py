"""Integration tests for the STEVO-Bench in-tree runner and its wiring."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from worldfoundry.evaluation.tasks.contracts import get_external_benchmark_contract
from worldfoundry.evaluation.tasks.execution.framework.integration import BENCHMARK_INTEGRATION_REGISTRY
from worldfoundry.evaluation.tasks.execution.framework.runner_registry import VIDEO_RUNNER_REGISTRY
from worldfoundry.evaluation.tasks.execution.runners.stevo_bench.stevo_bench_metrics import (
    METRIC_ORDER,
    aggregate_metrics,
    load_sample_records,
)
from worldfoundry.evaluation.tasks.execution.runners.workspace_registry import CLI_RUNNERS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "worldfoundry/data/benchmarks/assets/stevo-bench/sample_run"
RUNNER = REPO_ROOT / "worldfoundry/evaluation/tasks/execution/runners/stevo_bench/run_stevo_bench_official_runner.py"


class StevoBenchIntegrationTests(unittest.TestCase):
    def test_fixture_normalization_reproduces_official_aggregation(self) -> None:
        records = load_sample_records(FIXTURE)
        self.assertEqual(len(records), 4)
        by_task = {record["task_id"]: record for record in records}
        # per_task/candle_burn_01 fills the verdict summary.json left as null.
        self.assertTrue(by_task["candle_burn_01"]["state_evol"])
        self.assertTrue(by_task["candle_burn_01"]["task_success"])
        self.assertTrue(by_task["ice_cube_melt_00"]["baseline"])

        metrics = aggregate_metrics(records)
        self.assertAlmostEqual(metrics["task_success"]["raw_score"], 0.75)
        self.assertAlmostEqual(metrics["state_evol_success"]["raw_score"], 0.75)
        self.assertAlmostEqual(metrics["physical_inaccuracy"]["raw_score"], 0.25)
        self.assertAlmostEqual(metrics["physical_inaccuracy"]["normalized_score"], 0.75)
        self.assertAlmostEqual(metrics["occlusion_done"]["raw_score"], 0.75)
        self.assertAlmostEqual(metrics["trigger_applied"]["raw_score"], 1.0)
        self.assertAlmostEqual(metrics["control_success"]["raw_score"], 0.75)

    def test_fixture_runner_writes_complete_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pythonpath = [str(REPO_ROOT), *(path for path in sys.path if path)]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(pythonpath))}
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--run-fixture", "--output-dir", tmp, "--json"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertTrue(scorecard["normalizer_only"])
            # A fixture normalization must never claim official verification.
            self.assertFalse(scorecard["official_benchmark_verified"])
            self.assertFalse(scorecard["leaderboard_valid"])
            self.assertEqual(scorecard["metrics"]["summary"]["available_metrics"], len(METRIC_ORDER))
            self.assertEqual(scorecard["metrics"]["summary"]["sample_count"], 4)
            leaderboard = scorecard["metrics"]["leaderboard"]
            self.assertAlmostEqual(leaderboard["task_success"], 0.75)
            self.assertTrue((Path(tmp) / "raw_metric_table.jsonl").is_file())
            self.assertTrue((Path(tmp) / "per_sample_scores.jsonl").is_file())
            self.assertTrue((Path(tmp) / "benchmark_contract.json").is_file())

    def test_contract_and_registries_are_wired(self) -> None:
        contract = get_external_benchmark_contract("stevo-bench")
        self.assertEqual(contract.display_name, "STEVO-Bench")
        self.assertEqual(len(contract.metric_ids), len(METRIC_ORDER))
        self.assertIn("stevo-bench", VIDEO_RUNNER_REGISTRY)
        self.assertTrue((REPO_ROOT / VIDEO_RUNNER_REGISTRY["stevo-bench"].script).is_file())
        self.assertIn("stevo-bench", BENCHMARK_INTEGRATION_REGISTRY)
        workspace_spec = CLI_RUNNERS["stevo-bench"]
        self.assertTrue(workspace_spec.supports_fixture)
        self.assertTrue(workspace_spec.supports_official_runtime)
        expected_module = VIDEO_RUNNER_REGISTRY["stevo-bench"].script.removesuffix(".py").replace("/", ".")
        self.assertEqual(workspace_spec.module, expected_module)

    def test_catalog_and_runtime_profile_exist(self) -> None:
        catalog = REPO_ROOT / "worldfoundry/data/benchmarks/catalog/video/stevo-bench.yaml"
        profile = REPO_ROOT / "worldfoundry/data/benchmarks/runtime_profiles/official/stevo-bench.yaml"
        task_yaml = REPO_ROOT / "worldfoundry/data/benchmarks/tasks/external/stevo-bench.yaml"
        for path in (catalog, profile, task_yaml):
            self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
