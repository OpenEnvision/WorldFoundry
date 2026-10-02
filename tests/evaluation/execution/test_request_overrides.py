"""Request overrides retain the live runner, metric, and configuration objects."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from worldfoundry.evaluation import framework
from worldfoundry.evaluation.api import GenerationRequest, GenerationResult, MetricResult, WorldModelConfig
from worldfoundry.evaluation.api.metrics import aggregate_mean
from worldfoundry.evaluation.tasks.embodied import evaluate as embodied
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark_suite
from worldfoundry.evaluation.tasks.execution.orchestration.contract import ContractRunRequest, run_contract
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import EvaluateRunRequest, run_evaluate
from worldfoundry.evaluation.tasks.execution.orchestration.existing_results import (
    ExistingResultsRunRequest,
    run_existing_results,
)


class RecordingRunner:
    model_id = "request-overrides"

    def __init__(self):
        self.calls = []
        self.cleaned = False

    def generate(self, requests):
        self.calls.extend(row.sample_id for row in requests)
        return [GenerationResult(sample_id=row.sample_id) for row in requests]

    def cleanup(self):
        self.cleaned = True


class RecordingMetric:
    name = "quality"
    version = "1"
    required_artifacts = ()
    higher_is_better = True

    def __init__(self):
        self.calls = []

    def compute_sample(self, request, result):
        self.calls.append(request.sample_id)
        return MetricResult(sample_id=request.sample_id, metric_id=self.name, raw_value=0.5)

    def aggregate(self, rows):
        return aggregate_mean(self.name, rows)


class RequestOverrideTests(unittest.TestCase):
    def test_generation_overrides_use_and_clean_the_supplied_runner(self):
        for entrypoint, request_type, options in (
            (run_evaluate, EvaluateRunRequest, {"mode": "model"}),
            (run_contract, ContractRunRequest, {}),
        ):
            with self.subTest(entrypoint=entrypoint.__name__), tempfile.TemporaryDirectory() as temp:
                runner, metric = RecordingRunner(), RecordingMetric()
                request = request_type(
                    output_dir=temp, runner=runner, metrics=[metric],
                    requests=[GenerationRequest(sample_id="one", task_name="review")], **options,
                )
                with patch("worldfoundry.core.worldfoundry_inference_context", side_effect=nullcontext):
                    result = entrypoint(request, resume=True)
                self.assertEqual(result.status, "succeeded")
                self.assertEqual(runner.calls, ["one"])
                self.assertTrue(runner.cleaned)
                self.assertEqual(metric.calls, ["one"])
                self.assertFalse(request.resume)

    def test_offline_overrides_use_the_supplied_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            metric = RecordingMetric()
            request = ExistingResultsRunRequest(
                output_dir=temp, metrics=[metric],
                requests=[GenerationRequest(sample_id="one", task_name="review")],
                results=[GenerationResult(sample_id="one")],
            )
            run_existing_results(request, metric_batch_size=1)
            self.assertEqual(metric.calls, ["one"])
            self.assertEqual(request.metric_batch_size, 32)

    def test_facades_preserve_nested_configuration_types(self):
        config = WorldModelConfig("review", "worldfoundry.pipeline")
        for module, request in (
            (framework, framework.WorldFoundryRunRequest(output_dir="original", model_config=config)),
            (embodied, embodied.VlaVaWamRunRequest(output_dir="original", spec={}, model_config=config)),
            (model_benchmark_suite, model_benchmark_suite.ModelBenchmarkSuiteRequest(
                output_dir="original", model_config=config,
            )),
        ):
            with self.subTest(module=module.__name__):
                updated = module._coerce_request(request, {"output_dir": Path("updated")})
                self.assertIs(updated.model_config, config)
                self.assertEqual(updated.output_dir, Path("updated"))
                self.assertEqual(request.output_dir, "original")


if __name__ == "__main__":
    unittest.main()
