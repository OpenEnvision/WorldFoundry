"""Native NumPy scores must aggregate identically before and after resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.api.metrics import aggregate_mean
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import run_evaluate


class NumpyMetric:
    name = "quality"
    version = "1"
    higher_is_better = True
    required_artifacts = ()

    def __init__(self):
        self.calls = 0

    def compute_sample(self, request, result):
        self.calls += 1
        return MetricResult(sample_id=request.sample_id, metric_id=self.name, raw_value=np.float32(0.75))

    def aggregate(self, results):
        return aggregate_mean(self.name, results)


class ScalarAggregationTests(unittest.TestCase):
    def test_numpy_scores_use_the_same_mean_and_validity_rules(self):
        for scalar in (float, np.float32, np.float64, np.int64):
            for normalized in (False, True):
                with self.subTest(scalar=scalar.__name__, normalized=normalized):
                    rows = [
                        MetricResult(
                            sample_id=str(i),
                            metric_id="quality",
                            raw_value=scalar(value),
                            normalized_value=scalar(value) if normalized else None,
                        )
                        for i, value in enumerate((1, 3))
                    ]
                    rows.append(MetricResult(sample_id="skip", metric_id="quality", valid=False, skip_reason="missing"))
                    result = aggregate_mean("quality", rows)
                    self.assertEqual((result.n_total, result.n_valid, result.n_skipped), (3, 2, 1))
                    self.assertTrue(result.valid)
                    self.assertEqual(result.raw_stats, {"mean": 2.0})
                    self.assertEqual(result.normalized_stats, {"mean": 2.0} if normalized else {})
                    self.assertEqual(result.skip_breakdown, {"missing": 1})

    def test_first_run_and_resume_keep_numpy_scores_and_counts(self):
        metric = NumpyMetric()
        requests = [GenerationRequest(sample_id="one", task_name="scalar-test")]
        results = [GenerationResult(sample_id="one", status="succeeded")]
        summaries = []
        with tempfile.TemporaryDirectory(prefix="scalar-tests-") as tmp:
            root = Path(tmp)
            for resume in (False, True):
                run_evaluate(output_dir=root, requests=requests, results=results, metrics=[metric], resume=resume)
                summaries.append(json.loads((root / "metrics" / "summary.json").read_text()))
        self.assertEqual(metric.calls, 1)
        for summary in summaries:
            self.assertEqual(summary["leaderboard"], {"quality": 0.75})
            quality = summary["per_metric"]["quality"]
            self.assertEqual((quality["n_valid"], quality["n_skipped"]), (1, 0))
        self.assertEqual(summaries[0]["per_metric"], summaries[1]["per_metric"])


if __name__ == "__main__":
    unittest.main()
