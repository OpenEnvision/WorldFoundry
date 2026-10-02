"""Smoke tests for in-tree GenAI-Bench and t2v_metrics integrations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners"


class GenAIBenchT2VMetricsSmokeTests(unittest.TestCase):
    def _run_runner(self, script: Path, *extra: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("WORLDFOUNDRY_") and "BACKEND" not in key:
                if key.endswith("_ROOT") or key.endswith("_RESULTS_PATH") or key.endswith("_CONFIG_PATH"):
                    env.pop(key, None)
        if env_overrides:
            env.update(env_overrides)
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, str(script), *extra],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_genai_bench_fixture_normalizer(self) -> None:
        with self._temporary_output_dir() as tmp:
            completed = self._run_runner(
                RUNNERS / "genai_bench" / "run_genai_bench_official_runner.py",
                "--run-fixture",
                "--output-dir",
                tmp,
                "--json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads(Path(tmp, "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])

    def test_genai_bench_mock_official_runtime(self) -> None:
        with self._temporary_output_dir() as tmp:
            generated = Path(tmp) / "generated"
            generated.mkdir()
            out = Path(tmp) / "out"
            completed = self._run_runner(
                RUNNERS / "genai_bench" / "run_genai_bench_official_runner.py",
                "--run-official",
                "--generated-artifact-dir",
                str(generated),
                "--output-dir",
                str(out),
                "--json",
                env_overrides={"WORLDFOUNDRY_GENAI_BENCH_SCORER_BACKEND": "mock"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            self.assertTrue(scorecard["normalization_ok"])
            self.assertEqual(scorecard["run"]["scorer_summary"]["backend"], "mock")

    def test_bundled_genai_assets_resolve_without_env(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_prompts import (
            load_metadata,
            resolve_preference_pairs_path,
        )

        metadata = load_metadata()
        self.assertEqual(metadata["benchmark_id"], "genai-bench")
        pairs_path = resolve_preference_pairs_path()
        self.assertTrue(pairs_path.is_file())
        self.assertIn("genai-bench", str(pairs_path))

    def test_official_genai_output_fields_are_normalized(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners.genai_bench.genai_bench_metrics import (
            evaluate_genai_preference_rows,
        )

        result = evaluate_genai_preference_rows(
            [
                {"human_vote": "leftvote", "model_vote": "[[A>>B]]", "correct": True},
                {"human_vote": "bothbad_vote", "model_vote": "[[A=B]]", "correct": True},
                {"human_vote": "rightvote", "model_vote": "[[A>B]]", "correct": False},
            ],
            default_task="video_generation",
        )

        self.assertEqual(result["num_total"], 3)
        self.assertEqual(result["num_correct"], 2)
        self.assertAlmostEqual(result["pairwise_accuracy"], 2 / 3)
        self.assertEqual(result["per_task"]["video_generation"]["num_total"], 3)

    def test_vqa_score_import_without_ffmpeg(self) -> None:
        from worldfoundry.evaluation.tasks.execution.runners._scorers.vqa_score import (
            list_all_vqascore_models,
            package_root,
        )

        self.assertIn("clip-flant5-xxl", list_all_vqascore_models())
        self.assertTrue((package_root() / "score.py").is_file())

    def _temporary_output_dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
