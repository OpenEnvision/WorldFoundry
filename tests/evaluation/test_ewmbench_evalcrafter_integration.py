"""Smoke tests for in-tree EWMBench and EvalCrafter integrations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners"


class BenchmarkInTreeSmokeTests(unittest.TestCase):
    def _run_runner(self, script: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("WORLDFOUNDRY_") and "BACKEND" not in key:
                if key.endswith("_ROOT") or key.endswith("_RESULTS_PATH") or key.endswith("_CONFIG_PATH"):
                    env.pop(key, None)
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, str(script), *extra],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_ewmbench_fixture_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._run_runner(
                RUNNERS / "ewmbench" / "run_ewmbench_official_runner.py",
                "--run-fixture",
                "--output-dir",
                tmp,
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])

    def test_evalcrafter_fixture_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._run_runner(
                RUNNERS / "evalcrafter" / "run_evalcrafter_official_runner.py",
                "--run-fixture",
                "--output-dir",
                tmp,
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])

    def test_ewmbench_mock_official_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "generated"
            generated.mkdir()
            out = Path(tmp) / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env["WORLDFOUNDRY_EWMBENCH_SCORER_BACKEND"] = "mock"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNERS / "ewmbench" / "run_ewmbench_official_runner.py"),
                    "--run-official",
                    "--generated-artifact-dir",
                    str(generated),
                    "--output-dir",
                    str(out),
                    "--json",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertEqual(scorecard["run"]["scorer_summary"]["backend"], "mock")

    def test_evalcrafter_mock_official_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "generated"
            generated.mkdir()
            out = Path(tmp) / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env["WORLDFOUNDRY_EVALCRAFTER_SCORER_BACKEND"] = "mock"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNERS / "evalcrafter" / "run_evalcrafter_official_runner.py"),
                    "--run-official",
                    "--generated-artifact-dir",
                    str(generated),
                    "--output-dir",
                    str(out),
                    "--json",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr or completed.stdout)
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            self.assertFalse(scorecard["normalization_ok"])
            self.assertFalse(scorecard["official_benchmark_verified"])
            self.assertFalse(scorecard["leaderboard_valid"])
            self.assertIn("only supports the raw-video backend", scorecard["run"]["error"])

    def test_bundled_assets_resolve_without_env(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.ewmbench.ewmbench_paths import resolve_task_manifest_path
        from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import resolve_prompt700_path

        task_manifest = resolve_task_manifest_path()
        prompt700 = resolve_prompt700_path()
        self.assertTrue(task_manifest.is_file())
        self.assertTrue(prompt700.is_file())

    def test_evalcrafter_official_metric_prompt_is_distinct_from_generation_prompt(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.evalcrafter.evalcrafter_prompts import (
            load_prompt_records,
            resolve_official_metric_prompt_path,
        )

        generation_prompt = load_prompt_records()[2]["prompt"]
        metric_prompt = resolve_official_metric_prompt_path("0002").read_text(encoding="utf-8").strip()
        self.assertEqual(generation_prompt, "goldfish in glass")
        self.assertEqual(metric_prompt, "in goldfish glass")


if __name__ == "__main__":
    unittest.main()
