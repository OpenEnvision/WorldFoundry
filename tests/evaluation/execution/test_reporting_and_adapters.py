"""Regressions for metric selection and run comparison."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldfoundry.evals.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evals.reporting.run_comparison import build_run_comparison
from worldfoundry.evals.tasks.embodied.metrics import ResultFieldMetric
from worldfoundry.evals.tasks.execution.orchestration.evaluate import run_evaluate
from worldfoundry.evals.tasks.metrics.registry import BuiltinExistingResultsMetric


class ReportingAndAdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.requests = [GenerationRequest(sample_id=str(i), task_name="review") for i in range(2)]
        video = self.root / "sample.mp4"
        video.touch()
        self.results = [
            GenerationResult(
                sample_id=q.sample_id,
                artifacts={"generated_video": ArtifactRef(str(video), "video")},
                metadata={"scores": {"quality": value, "scene_consistency": value, "motion_correctness": 0.6}},
            )
            for q, value in zip(self.requests, (0.9, 0.1))
        ]

    def run_scores(self, name, metrics, **kwargs):
        run_evaluate(
            output_dir=self.root / name, requests=self.requests, results=self.results, metrics=metrics, **kwargs
        )
        return json.loads((self.root / name / "scorecard.json").read_text())

    def test_failed_and_partial_scores_remain_visible_without_winning(self):
        metric = ResultFieldMetric("quality", higher_is_better=True)
        self.run_scores("complete", [metric])
        self.results[1] = GenerationResult(sample_id="1", status="failed", error="generation failed")
        self.run_scores("failed", [metric])
        self.results[1] = GenerationResult(sample_id="1")
        self.run_scores("partial", [metric])
        comparison = build_run_comparison(
            [self.root / name for name in ("complete", "failed", "partial")], labels=["complete", "failed", "partial"]
        )
        self.assertEqual(comparison["best_by_metric"]["quality"]["label"], "complete")
        self.assertEqual(comparison["metrics"]["quality"]["values"]["failed"], 0.9)
        self.assertEqual(comparison["metrics"]["quality"]["values"]["partial"], 0.9)

    def test_in_tree_rates_are_lower_is_better(self):
        for name, rate in (("low", 0.1), ("high", 0.9)):
            self.results = [
                GenerationResult(
                    sample_id=q.sample_id, artifacts=r.artifacts, metadata={"scores": {"violence_nsfw_rate": rate}}
                )
                for q, r in zip(self.requests, self.results)
            ]
            self.run_scores(name, ["violence_nsfw_rate"], benchmark_id="t2v-safety-bench")
        comparison = build_run_comparison([self.root / "low", self.root / "high"], labels=["low", "high"])
        self.assertEqual(comparison["best_by_metric"]["violence_nsfw_rate"]["label"], "low")

    def test_adding_an_in_tree_metric_preserves_comparability(self):
        self.run_scores("one", ["scene_consistency"], benchmark_id="ewmbench")
        self.run_scores("two", ["scene_consistency", "motion_correctness"], benchmark_id="ewmbench")
        comparison = build_run_comparison([self.root / "one", self.root / "two"], metric_ids=["scene_consistency"])
        self.assertEqual(comparison["metric_ids"], ["scene_consistency"])
        self.run_scores("primary_one", ["scene_consistency", "ewmbench_average"], benchmark_id="ewmbench")
        self.run_scores(
            "primary_two", ["scene_consistency", "motion_correctness", "ewmbench_average"], benchmark_id="ewmbench"
        )
        with self.assertRaisesRegex(ValueError, "ewmbench_average"):
            build_run_comparison(
                [self.root / "primary_one", self.root / "primary_two"], metric_ids=["ewmbench_average"]
            )

    def test_builtin_and_in_tree_metrics_can_be_selected_together(self):
        card = self.run_scores("mixed", ["scene_consistency", "artifact_count"], benchmark_id="ewmbench")
        self.assertEqual(card["metrics"]["leaderboard"]["scene_consistency"], 0.5)
        self.assertEqual(card["metrics"]["leaderboard"]["artifact_count"], 1)

    def test_missing_requested_numeric_metric_is_explicitly_invalid(self):
        card = self.run_scores("missing", ["numeric:absent_score"])
        self.assertFalse(card["eligibility"]["score_valid"])
        metric = card["metrics"]["per_metric"]["absent_score"]
        self.assertFalse(metric["valid"])
        self.assertEqual(metric["n_skipped"], 2)
        self.assertNotIn("absent_score", card["metrics"]["leaderboard"])

    def test_empty_numeric_collection_cannot_be_replaced_by_generation_success(self):
        self.results = [GenerationResult(sample_id=q.sample_id) for q in self.requests]
        card = self.run_scores("empty", ["numeric"])
        self.assertFalse(card["eligibility"]["score_valid"])

    def test_numeric_wildcard_records_missing_fields_and_keeps_partial_scores(self):
        for name, scores in (
            ("full", [{"fid": 0.2, "other": 0.5}, {"fid": 0.2, "other": 0.5}]),
            ("partial", [{"fid": 0.1, "other": 0.5}, {"other": 0.5}]),
        ):
            self.results = [
                GenerationResult(sample_id=q.sample_id, metadata={"scores": values})
                for q, values in zip(self.requests, scores)
            ]
            card = self.run_scores(name, ["numeric"], metric_batch_size=1)
        metric = card["metrics"]["per_metric"]["fid"]
        self.assertEqual((metric["n_total"], metric["n_valid"], metric["n_skipped"]), (2, 1, 1))
        self.assertIsNotNone(metric["definition"])
        rows = [json.loads(line) for line in (self.root / "partial/metrics/per_sample.jsonl").read_text().splitlines()]
        missing = next(row for row in rows[1]["metric_results"] if row["metric_id"] == "fid")
        self.assertFalse(missing["valid"])
        comparison = build_run_comparison(
            [self.root / "full", self.root / "partial"], labels=["full", "partial"], metric_ids=["fid"]
        )
        self.assertEqual(comparison["best_by_metric"]["fid"]["label"], "full")
        self.assertEqual(comparison["metrics"]["fid"]["values"]["partial"], 0.1)

    def test_numeric_wildcard_matches_explicit_fields_and_resolves_each_run(self):
        metric = BuiltinExistingResultsMetric(["numeric"])
        for name, field in (("first", "quality"), ("second", "another_quality")):
            self.results = [
                GenerationResult(sample_id=q.sample_id, metadata={"scores": {field: 0.5}}) for q in self.requests
            ]
            self.run_scores(name, [metric])
            card = self.run_scores(name, [metric], resume=True)
            self.assertIsNotNone(card["metrics"]["per_metric"][field]["definition"])
            self.assertEqual(set(card["metrics"]["leaderboard"]), {field, "generation_success"})
            self.run_scores(f"{name}-explicit", [f"numeric:{field}"])
            build_run_comparison([self.root / name, self.root / f"{name}-explicit"], metric_ids=[field])
        self.assertEqual(metric.metrics, ("numeric",))

    def test_builtin_metric_definitions_support_resume_and_subset_comparison(self):
        metric = BuiltinExistingResultsMetric(metrics=["numeric:quality"])
        with patch.object(metric, "compute_sample", wraps=metric.compute_sample) as compute:
            for resume in (False, True):
                self.run_scores("builtin", [metric], resume=resume)
            self.assertEqual(compute.call_count, 2)
        self.run_scores("extended", ["numeric:quality", "artifact_count"])
        comparison = build_run_comparison([self.root / "builtin", self.root / "extended"], metric_ids=["quality"])
        self.assertEqual(comparison["metric_ids"], ["quality"])
        self.assertNotIn("quality", comparison["best_by_metric"])

    def test_builtin_numeric_fields_preserve_registered_direction(self):
        for name, value in (("low", 1), ("high", 9)):
            self.results = [
                GenerationResult(sample_id=q.sample_id, metadata={"scores": {"fid": value}}) for q in self.requests
            ]
            self.run_scores(name, ["numeric:fid"])
        comparison = build_run_comparison([self.root / "low", self.root / "high"], labels=["low", "high"])
        self.assertEqual(comparison["best_by_metric"]["fid"]["label"], "low")


if __name__ == "__main__":
    unittest.main()
