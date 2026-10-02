"""Smoke tests for VMBench and World-in-World in-tree integrations."""

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


class VMBenchWorldInWorldSmokeTests(unittest.TestCase):
    def _run_runner(self, script: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("WORLDFOUNDRY_") and "BACKEND" not in key:
                if key.endswith("_ROOT") or key.endswith("_RESULTS_PATH") or key.endswith("_PROMPT_MANIFEST"):
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

    def test_vmbench_bundled_prompts_materialize(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.vmbench.vmbench_prompts import (
            materialize_vmbench_generation_requests,
        )

        requests = materialize_vmbench_generation_requests(limit=3)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].sample_id, "0001")
        self.assertEqual(requests[0].inputs["official_video_name"], "0001.mp4")

    def test_world_in_world_bundled_prompts_materialize(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.world_in_world.world_in_world_prompts import (
            materialize_world_in_world_generation_requests,
        )

        requests = materialize_world_in_world_generation_requests(limit=2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].task_name, "world-in-world")

    def test_vmbench_fixture_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._run_runner(
                RUNNERS / "vmbench" / "run_vmbench_official_runner.py",
                "--run-fixture",
                "--output-dir",
                tmp,
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertTrue(scorecard["official_results_imported"])

    def test_world_in_world_fixture_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = self._run_runner(
                RUNNERS / "world_in_world" / "run_world_in_world_official_runner.py",
                "--run-fixture",
                "--output-dir",
                tmp,
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertEqual(scorecard["prompt_count"], 184)


if __name__ == "__main__":
    unittest.main()
