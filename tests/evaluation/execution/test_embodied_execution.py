"""CPU regressions for embodied rollout persistence and shard completion."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from worldfoundry.core.io.serialization import iter_jsonl, write_jsonl
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.embodied.merge_results import merge_embodied_results
from worldfoundry.evaluation.tasks.embodied.orchestrator import EmbodiedEvalOrchestrator


class EmbodiedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = [GenerationRequest(sample_id=str(i), task_name="embodied") for i in range(2)]
        self.results = [
            GenerationResult(sample_id=str(i), status="succeeded", metadata={"task_success": i})
            for i in range(2)
        ]

    def write_shard(self, index, requests, results):
        root = self.root / f"shard{index}of2"
        root.mkdir()
        write_jsonl(root / "requests.jsonl", (row.to_dict() for row in requests))
        if results is not None:
            write_jsonl(root / "results.jsonl", (row.to_dict() for row in results))

    def test_merge_rejects_missing_shard(self):
        self.write_shard(0, self.requests[:1], self.results[:1])
        with self.assertRaisesRegex(ValueError, "incomplete.*shard1of2"):
            merge_embodied_results(self.root)
        self.assertFalse((self.root / "scorecard.json").exists())

    def test_merge_rejects_shard_without_results(self):
        self.write_shard(0, self.requests[:1], self.results[:1])
        self.write_shard(1, self.requests[1:], None)
        with self.assertRaisesRegex(ValueError, "incomplete.*shard1of2"):
            merge_embodied_results(self.root)

    def test_merge_rejects_partial_rollout_shard(self):
        self.write_shard(0, self.requests, self.results[:1])
        self.write_shard(1, [], [])
        with self.assertRaisesRegex(ValueError, "incomplete.*shard0of2"):
            merge_embodied_results(self.root)

    def test_merge_accepts_completed_empty_shard(self):
        self.write_shard(0, self.requests, self.results)
        self.write_shard(1, [], [])
        merged = merge_embodied_results(self.root, metric_ids=["task_success"])
        card = json.loads(merged.scorecard_path.read_text())
        self.assertEqual(merged.sample_count, 2)
        self.assertEqual(card["metrics"]["leaderboard"]["task_success"], 0.5)

    def test_interrupted_rollout_keeps_completed_results_and_planned_requests(self):
        for failure, status in ((asyncio.CancelledError, "interrupted"), (RuntimeError, "failed")):
            with self.subTest(failure=failure.__name__):
                root = self.root / failure.__name__
                root.mkdir()
                (root / "scorecard.json").write_text("old scorecard")
                runner = Mock()

                def generate(requests):
                    if requests[0].sample_id == "1":
                        # A completed rollout must be visible before the next one starts.
                        self.assertEqual(list(iter_jsonl(root / "results.jsonl")), [self.results[0].to_dict()])
                        raise failure()
                    return [self.results[0]]

                runner.generate.side_effect = generate
                orchestrator = EmbodiedEvalOrchestrator(
                    {"output_dir": str(root), "benchmarks": [{"benchmark_id": "libero"}]},
                )
                with (
                    patch("worldfoundry.evaluation.tasks.embodied.orchestrator.materialize_embodied_rollout_requests",
                          return_value=self.requests),
                    patch("worldfoundry.evaluation.tasks.embodied.orchestrator.build_embodied_closed_loop_runner",
                          return_value=runner),
                    self.assertRaises(failure),
                ):
                    asyncio.run(orchestrator.run())
                self.assertEqual(list(iter_jsonl(root / "requests.jsonl")), [row.to_dict() for row in self.requests])
                self.assertEqual(list(iter_jsonl(root / "results.jsonl")), [self.results[0].to_dict()])
                manifest = json.loads((root / "run_manifest.json").read_text())
                self.assertEqual(manifest["status"], status)
                self.assertEqual(manifest["stage"], "generate")
                self.assertEqual(manifest["sample_count"], 2)
                self.assertFalse((root / "scorecard.json").exists())
                runner.cleanup.assert_called_once()

    def test_completed_rollouts_keep_scoring_and_no_save_behavior(self):
        for no_save in (False, True):
            with self.subTest(no_save=no_save):
                root = self.root / str(no_save)
                runner = Mock()
                runner.generate.side_effect = lambda requests: [self.results[int(requests[0].sample_id)]]
                orchestrator = EmbodiedEvalOrchestrator(
                    {"output_dir": str(root), "benchmarks": [{"benchmark_id": "libero"}]}, no_save=no_save,
                )
                with (
                    patch("worldfoundry.evaluation.tasks.embodied.orchestrator.materialize_embodied_rollout_requests",
                          return_value=self.requests),
                    patch("worldfoundry.evaluation.tasks.embodied.orchestrator.build_embodied_closed_loop_runner",
                          return_value=runner),
                ):
                    result = asyncio.run(orchestrator.run())
                self.assertEqual(result.raw_results, tuple(self.results))
                self.assertEqual(result.evaluate_result.status, "succeeded")
                self.assertFalse(orchestrator.progress_path.exists())
                runner.cleanup.assert_called_once()
                if no_save:
                    self.assertFalse((root / "results.jsonl").exists())
                    self.assertFalse((root / "run_manifest.json").exists())
                else:
                    self.assertEqual(list(iter_jsonl(root / "results.jsonl")), [row.to_dict() for row in self.results])
                    card = json.loads((root / "scorecard.json").read_text())
                    self.assertEqual(card["metrics"]["leaderboard"]["task_success"], 0.5)


if __name__ == "__main__":
    unittest.main()
