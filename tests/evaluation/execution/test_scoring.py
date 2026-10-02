"""CPU regressions for shared scoring and resumable evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from worldfoundry.evaluation.api import AggregateResult, ArtifactRef, GenerationRequest, GenerationResult, MetricResult
from worldfoundry.evaluation.api.metrics import aggregate_mean
from worldfoundry.evaluation.models.runners.registry import ModelRunnerRegistry
from worldfoundry.evaluation.models.runners.resolver import ResolvedWorldModel
from worldfoundry.evaluation.reporting.run_comparison import build_run_comparison
from worldfoundry.evaluation.reporting.run_index import build_run_index
from worldfoundry.evaluation.reporting.scorecard import build_scorecard
from worldfoundry.evaluation.reporting.validation import validate_contract_file
from worldfoundry.evaluation.tasks.embodied.merge_results import merge_embodied_results
from worldfoundry.evaluation.tasks.embodied.metrics import ResultFieldMetric, metric_suite
from worldfoundry.evaluation.tasks.execution.orchestration.contract import run_contract
from worldfoundry.evaluation.tasks.execution.orchestration.evaluate import run_evaluate
from worldfoundry.evaluation.tasks.execution.orchestration.existing_results import run_existing_results
from worldfoundry.evaluation.tasks.metrics.registry import BuiltinExistingResultsMetric


class WeightedError:
    name = "weighted_error"
    version = "1"
    required_artifacts = ()
    higher_is_better = False

    def __init__(self, *, scale=1):
        self.parameters = {"scale": scale}
        self.calls = []

    def compute_sample(self, request, result):
        self.calls.append(request.sample_id)
        return MetricResult(
            sample_id=request.sample_id,
            metric_id=self.name,
            raw_value=result.metadata["value"] * self.parameters["scale"],
            components={"weight": result.metadata["weight"]},
        )

    def aggregate(self, results):
        values = [row for row in results if row.valid]
        mean = sum(row.raw_value * row.components["weight"] for row in values) / sum(
            row.components["weight"] for row in values
        )
        return AggregateResult(
            metric_id=self.name,
            n_total=len(results),
            n_valid=len(values),
            raw_stats={"mean": mean},
            confidence_interval={"low": 7, "high": 9},
            stderr=0.5,
        )


class Runner:
    model_id = "scoring-test"
    version = "1"

    def __init__(self, results):
        self.results = {row.sample_id: row for row in results}
        self.calls = []
        self.cleaned = False

    def generate(self, requests):
        self.calls.extend(row.sample_id for row in requests)
        return [self.results[row.sample_id] for row in requests]

    def cleanup(self):
        self.cleaned = True


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = [GenerationRequest(sample_id=str(i), task_name="scoring") for i in range(2)]
        self.results = [
            GenerationResult(sample_id=str(i), status="succeeded", metadata={"value": value, "weight": weight})
            for i, (value, weight) in enumerate(((1, 1), (9, 9)))
        ]

    def summary(self, name):
        return json.loads((self.root / name / "metrics" / "summary.json").read_text())

    def offline(self, name, metrics, **kwargs):
        return run_evaluate(
            output_dir=self.root / name,
            requests=self.requests,
            results=self.results,
            metrics=metrics,
            **kwargs,
        )

    def test_online_offline_aggregation_and_statistics_match(self):
        runner = Runner(self.results)
        run_contract(output_dir=self.root / "online", requests=self.requests, runner=runner, metrics=[WeightedError()])
        self.offline("offline", [WeightedError()])
        online = self.summary("online")["per_metric"]
        self.assertEqual(online, self.summary("offline")["per_metric"])
        self.assertEqual(online["weighted_error"]["mean"], 8.2)
        self.assertFalse(online["weighted_error"]["higher_is_better"])
        self.assertEqual(online["weighted_error"]["confidence_interval"], {"low": 7, "high": 9})
        self.assertEqual(online["weighted_error"]["stderr"], 0.5)
        self.assertTrue(runner.cleaned)
        for name in ("online", "offline"):
            for filename in ("scorecard.json", "run_manifest.json", "summary.json"):
                report = validate_contract_file(self.root / name / filename, check_artifacts=True)
                self.assertTrue(report["ok"], report["errors"])

    def test_online_offline_dictionary_results_keep_scores_and_artifacts(self):
        artifact = self.root / "output.txt"
        artifact.write_text("generated output")
        raw = [
            {"sample_id": q.sample_id, "model_id": "scoring-test", "status": "succeeded", "scores": {"quality": 0.8, "reward": 0.5},
             "metadata": {"extra": {"episode_id": q.sample_id, "scores": {"quality": 0.7, "aux": 0.2}}},
             "artifacts": {"text": str(artifact), "preview": {"uri": str(artifact)}}}
            for q in self.requests
        ]
        runner = Runner(self.results)
        with patch.object(runner, "generate", return_value=raw):
            run_contract(output_dir=self.root / "online", requests=self.requests, runner=runner,
                         metrics=[BuiltinExistingResultsMetric(["numeric:quality", "numeric:reward"])])
            run_evaluate(mode="model", output_dir=self.root / "facade", requests=self.requests, runner=runner,
                         metrics=["numeric:quality", "numeric:reward"])
        run_existing_results(output_dir=self.root / "offline", requests=self.requests, results=raw,
                             metrics=[BuiltinExistingResultsMetric(["numeric:quality", "numeric:reward"])])
        for name in ("online", "facade"):
            self.assertEqual(self.summary(name)["leaderboard"]["quality"], 0.7)
            self.assertEqual(self.summary(name)["leaderboard"].get("reward"), 0.5)
            self.assertEqual(self.summary(name)["per_metric"], self.summary("offline")["per_metric"])
            self.assertEqual((self.root / name / "results.jsonl").read_text(),
                             (self.root / "offline/results.jsonl").read_text())
            rows = [json.loads(line) for line in (self.root / name / "results.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["metadata"]["extra"],
                             {"episode_id": "0", "scores": {"quality": 0.7, "reward": 0.5, "aux": 0.2}})

    def test_partial_metric_bundle_tracks_missing_outputs_and_resumes_them(self):
        class PairMetric:
            name = "pair"
            version = "1"
            metric_ids = ("quality", "consistency")
            higher_is_better = True
            parameters = {}

            def __init__(self):
                self.calls = []

            def compute_sample(self, request, result):
                self.calls.append(request.sample_id)
                return {"quality": 0.9, **({"consistency": 1.0} if request.sample_id == "0" else {})}

            def aggregate(self, results):
                return aggregate_mean(results[0].metric_id, results)

        metric = PairMetric()
        self.offline("partial_bundle", [metric])
        summary = self.summary("partial_bundle")
        partial = summary["per_metric"]["consistency"]
        self.assertEqual((partial["n_total"], partial["n_valid"], partial["n_skipped"]), (2, 1, 1))
        self.assertEqual(summary["metrics"]["skipped"], 1)
        self.assertEqual(summary["leaderboard"], {"quality": 0.9, "consistency": 1.0})
        comparison = build_run_comparison([self.root / "partial_bundle"])
        self.assertNotIn("consistency", comparison["best_by_metric"])
        self.assertIn("quality", comparison["best_by_metric"])
        # Old checkpoints could contain incomplete bundles marked as successful.
        checkpoint = self.root / "partial_bundle/metrics/checkpoint.jsonl"
        cached = json.loads(checkpoint.read_text().splitlines()[0])
        cached.update(sample_id="1", request=self.requests[1].to_dict(), result=self.results[1].to_dict())
        cached["outputs"] = [{**cached["outputs"][0], "sample_id": "1"}]
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(cached) + "\n")
        metric.calls.clear()
        self.offline("partial_bundle", [metric], resume=True)
        self.assertEqual(metric.calls, ["1"])
        self.assertEqual(self.summary("partial_bundle")["per_metric"], summary["per_metric"])

    def test_overlapping_metric_ids_are_rejected_before_generation(self):
        runner = Runner(self.results)
        metrics = [ResultFieldMetric("quality"), BuiltinExistingResultsMetric(["numeric:quality"])]
        with self.assertRaisesRegex(ValueError, "duplicate metric id.*quality"):
            run_contract(output_dir=self.root / "contract", requests=self.requests, runner=runner, metrics=metrics)
        with self.assertRaisesRegex(ValueError, "duplicate metric id.*quality"):
            run_evaluate(mode="model", output_dir=self.root / "model", requests=self.requests, runner=runner,
                         metrics=[metrics[0], "numeric:quality"])
        self.assertEqual(runner.calls, [])

    def test_offline_metric_overlap_is_rejected_after_numeric_expansion(self):
        self.results = [GenerationResult(sample_id=q.sample_id, metadata={"scores": {"quality": 0.1}})
                        for q in self.requests]
        for selection in ("numeric:quality", "numeric"):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "duplicate metric id.*quality"):
                self.offline("overlap", [ResultFieldMetric("quality"), selection])
        self.assertFalse((self.root / "overlap/scorecard.json").exists())

    def test_legacy_dynamic_metric_overlap_fails_without_a_scorecard(self):
        def first(request, result):
            return {"quality": 0.1}

        def second(request, result):
            return {"quality": 0.9}

        with self.assertRaisesRegex(ValueError, "duplicate metric id.*quality"):
            run_existing_results(output_dir=self.root / "dynamic", requests=self.requests,
                                 results=self.results, metrics=[first, second])
        manifest = json.loads((self.root / "dynamic/run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((self.root / "dynamic/scorecard.json").exists())

    def test_embodied_metric_directions_use_known_definitions(self):
        for name, fid, success in (("low", 1, 1), ("high", 9, 0)):
            self.results = [GenerationResult(sample_id=q.sample_id,
                                            metadata={"fid": fid, "task_success": success, "custom": fid})
                            for q in self.requests]
            self.offline(name, metric_suite(["fid", "task_success", "custom"]))
        comparison = build_run_comparison([self.root / "low", self.root / "high"], labels=["low", "high"])
        self.assertEqual(comparison["best_by_metric"]["fid"]["label"], "low")
        self.assertEqual(comparison["best_by_metric"]["task_success"]["label"], "low")
        self.assertNotIn("custom", comparison["best_by_metric"])
        self.assertIsNone(ResultFieldMetric("custom").higher_is_better)
        self.assertFalse(ResultFieldMetric("custom", higher_is_better=False).higher_is_better)

    def test_failed_metric_preserves_other_scores(self):
        class Broken(WeightedError):
            name = "broken"

            def compute_sample(self, request, result):
                raise RuntimeError("judge unavailable")

        self.offline("partial", [WeightedError(), Broken()])
        summary = self.summary("partial")
        self.assertEqual(summary["leaderboard"]["weighted_error"], 8.2)
        self.assertFalse(summary["per_metric"]["broken"]["valid"])
        self.assertEqual(summary["failed_samples"], 2)

    def test_empty_or_failed_aggregation_is_not_a_valid_score(self):
        class BrokenAggregate(WeightedError):
            def aggregate(self, results):
                raise RuntimeError("aggregation failed")

        self.offline("aggregate", [BrokenAggregate()])
        card = json.loads((self.root / "aggregate" / "scorecard.json").read_text())
        self.assertFalse(card["eligibility"]["score_valid"])
        empty = build_scorecard(
            run={},
            benchmark={},
            model={},
            dataset={},
            generation={},
            metrics_summary={"failed_samples": 0, "leaderboard": {}, "per_metric": {}},
            artifacts={},
        )
        self.assertFalse(empty["eligibility"]["score_valid"])

    def test_resume_reuses_generation_and_only_recomputes_changed_metrics(self):
        runner = Runner(self.results)
        metric = WeightedError()
        run_contract(output_dir=self.root / "resume", requests=self.requests, runner=runner, metrics=[metric])
        metric.calls.clear()
        runner.calls.clear()
        run_contract(
            output_dir=self.root / "resume",
            requests=self.requests,
            runner=runner,
            metrics=[metric],
            resume=True,
        )
        self.assertEqual(runner.calls, [])
        self.assertEqual(metric.calls, [])
        metric.parameters["scale"] = 2
        run_contract(
            output_dir=self.root / "resume",
            requests=self.requests,
            runner=runner,
            metrics=[metric],
            resume=True,
        )
        self.assertEqual(runner.calls, [])
        self.assertEqual(metric.calls, ["0", "1"])
        self.assertEqual(self.summary("resume")["leaderboard"][metric.name], 16.4)

    def test_interrupted_scoring_preserves_finished_work(self):
        class Interruptible(WeightedError):
            interrupt = True

            def compute_sample(self, request, result):
                if request.sample_id == "1" and self.interrupt:
                    raise KeyboardInterrupt()
                return super().compute_sample(request, result)

        metric = Interruptible()
        with self.assertRaises(KeyboardInterrupt):
            self.offline("interrupted", [metric], metric_batch_size=1)
        manifest = json.loads((self.root / "interrupted" / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        metric.interrupt = False
        metric.calls.clear()
        self.offline("interrupted", [metric], metric_batch_size=1, resume=True)
        self.assertEqual(metric.calls, ["1"])
        self.assertEqual(self.summary("interrupted")["leaderboard"][metric.name], 8.2)

    def test_batch_outputs_align_by_sample_id(self):
        class Batched(WeightedError):
            def compute_batch(self, requests, results):
                return {q.sample_id: self.compute_sample(q, r) for q, r in reversed(list(zip(requests, results)))}

        metric = Batched()
        self.offline("batch", [metric])
        self.assertEqual(metric.calls, ["1", "0"])
        self.assertEqual(self.summary("batch")["leaderboard"][metric.name], 8.2)

    def test_batch_failure_falls_back_without_repeating_other_metrics(self):
        class BatchFailure(WeightedError):
            name = "batch_failure"

            def compute_batch(self, requests, results):
                raise RuntimeError("batch unavailable")

        first, second = WeightedError(), BatchFailure()
        self.offline("fallback", [first, second])
        self.assertEqual(first.calls, ["0", "1"])
        self.assertEqual(second.calls, ["0", "1"])
        self.assertEqual(self.summary("fallback")["leaderboard"][second.name], 8.2)

    def test_metric_ids_and_objects_share_the_scoring_path(self):
        self.offline("mixed", ["artifact_count", WeightedError()])
        summary = self.summary("mixed")
        self.assertEqual(summary["leaderboard"]["weighted_error"], 8.2)
        self.assertEqual(summary["leaderboard"]["artifact_count"], 0)

    def test_legacy_callable_keeps_raw_and_normalized_statistics(self):
        def metric(request, result):
            return MetricResult(sample_id=request.sample_id, metric_id="custom", raw_value=80, normalized_value=0.8)

        run_existing_results(
            output_dir=self.root / "legacy",
            requests=self.requests,
            results=self.results,
            metric=metric,
        )
        summary = self.summary("legacy")["per_metric"]["custom"]
        self.assertEqual(summary["raw_stats"]["mean"], 80)
        self.assertEqual(summary["normalized_stats"]["mean"], 0.8)
        self.assertIsNone(summary["higher_is_better"])

    def test_model_facade_resumes_generation_and_scoring(self):
        runner, metric = Runner(self.results), WeightedError()
        for resume in (False, True):
            run_evaluate(
                mode="model",
                output_dir=self.root / "model",
                requests=self.requests,
                runner=runner,
                metrics=[metric],
                resume=resume,
            )
        self.assertEqual(runner.calls, ["0", "1"])
        self.assertEqual(metric.calls, ["0", "1"])

    def test_adding_or_updating_a_metric_only_runs_that_metric(self):
        class Second(WeightedError):
            name = "second"

        first, second = WeightedError(), Second()
        self.offline("extend", [first])
        first.calls.clear()
        self.offline("extend", [first, second], resume=True)
        self.assertEqual(first.calls, [])
        self.assertEqual(second.calls, ["0", "1"])
        second.calls.clear()
        first.version = "2"
        self.offline("extend", [first, second], resume=True)
        self.assertEqual(first.calls, ["0", "1"])
        self.assertEqual(second.calls, [])

    def test_failed_generation_is_counted_in_metric_coverage(self):
        self.results[1] = GenerationResult(sample_id="1", status="failed", error="generation failed")
        self.offline("generation_failure", [WeightedError()])
        summary = self.summary("generation_failure")
        self.assertEqual(summary["failed_samples"], 1)
        self.assertEqual(summary["per_metric"]["weighted_error"]["n_total"], 2)
        self.assertEqual(summary["per_metric"]["weighted_error"]["n_valid"], 1)

    def test_generation_success_includes_failures_without_scoring_their_quality(self):
        for failed_count in (1, 2):
            results = [
                GenerationResult(sample_id=str(i), status="failed", error="rollout failed")
                if i < failed_count else GenerationResult(
                    sample_id=result.sample_id, status="succeeded",
                    metadata={**result.metadata, "scores": {"value": result.metadata["value"]}},
                )
                for i, result in enumerate(self.results)
            ]
            for name, metrics, quality in (
                ("embodied", [ResultFieldMetric("generation_success"), WeightedError()], "weighted_error"),
                ("builtin", [BuiltinExistingResultsMetric(["numeric:value"])], "value"),
            ):
                with self.subTest(metric=name, failed_count=failed_count):
                    output = f"{name}-{failed_count}"
                    run_existing_results(
                        output_dir=self.root / output, requests=self.requests, results=results, metrics=metrics,
                    )
                    summary = self.summary(output)
                    success = summary["per_metric"]["generation_success"]
                    self.assertEqual(success["mean"], (2 - failed_count) / 2)
                    self.assertEqual(success["n_valid"], 2)
                    self.assertEqual(success["n_skipped"], 0)
                    self.assertEqual(summary["failed_samples"], failed_count)
                    self.assertEqual(summary["per_metric"][quality]["n_valid"], 2 - failed_count)

    def test_unknown_metric_direction_does_not_select_a_winner(self):
        run_existing_results(
            output_dir=self.root / "unknown",
            requests=self.requests,
            results=self.results,
            metric=lambda request, result: {"custom": result.metadata["value"]},
        )
        comparison = build_run_comparison([self.root / "unknown" / "summary.json"])
        self.assertNotIn("custom", comparison["best_by_metric"])

    def test_embodied_shards_use_metric_aggregation(self):
        for index, request in enumerate(self.requests):
            root = self.root / "embodied" / f"shard{index}of2"
            root.mkdir(parents=True)
            result = GenerationResult(sample_id=request.sample_id, status="succeeded", metadata={"task_success": index})
            (root / "requests.jsonl").write_text(json.dumps(request.to_dict()) + "\n")
            (root / "results.jsonl").write_text(json.dumps(result.to_dict()) + "\n")
        merge_embodied_results(self.root / "embodied", metric_ids=["task_success"])
        summary = self.summary("embodied")["per_metric"]["task_success"]
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary["n_total"], 2)

    def test_invalid_metric_is_rejected_before_generation(self):
        runner = Runner(self.results)
        with self.assertRaises(ValueError):
            run_evaluate(
                mode="model",
                output_dir=self.root / "invalid",
                requests=self.requests,
                runner=runner,
                metrics=["cmmd"],
            )
        self.assertEqual(runner.calls, [])

    def test_result_field_changes_invalidate_previous_scores(self):
        self.results = [
            GenerationResult(sample_id=str(i), status="succeeded", metadata={"first": 1, "second": 2}) for i in range(2)
        ]
        self.offline("fields", [ResultFieldMetric(name="custom", field_names=("first",))])
        self.offline("fields", [ResultFieldMetric(name="custom", field_names=("second",))], resume=True)
        self.assertEqual(self.summary("fields")["leaderboard"]["custom"], 2)

    def test_runner_configuration_changes_regenerate_in_both_entrypoints(self):
        class ConfiguredRunner(Runner):
            def __init__(self, value, field):
                super().__init__([])
                setattr(self, field, {"value": value})
                self.results = {
                    q.sample_id: GenerationResult(sample_id=q.sample_id, metadata={"value": value, "weight": 1})
                    for q in self_requests
                }

        self_requests = self.requests
        for entrypoint in (run_contract, run_evaluate):
            for field in ("parameters", "config"):
                with self.subTest(entrypoint=entrypoint.__name__, field=field):
                    options = {"mode": "model"} if entrypoint is run_evaluate else {}
                    root = self.root / f"{entrypoint.__name__}-{field}"
                    for value in (1, 7):
                        runner = ConfiguredRunner(value, field)
                        entrypoint(output_dir=root, requests=self.requests, runner=runner,
                                   metrics=[WeightedError()], resume=value == 7, **options)
                        self.assertEqual(runner.calls, ["0", "1"])
                    summary = json.loads((root / "metrics" / "summary.json").read_text())
                    self.assertEqual(summary["leaderboard"]["weighted_error"], 7)

    def test_comparison_rejects_changed_metric_definitions(self):
        self.offline("original", [WeightedError()])
        for field, value in (("version", "2"), ("parameters", {"scale": 2})):
            metric = WeightedError()
            setattr(metric, field, value)
            self.offline(field, [metric])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "weighted_error"):
                build_run_comparison([self.root / "original", self.root / field])
        self.offline("same", [WeightedError(), ResultFieldMetric("value")])
        comparison = build_run_comparison([self.root / "original", self.root / "same"])
        self.assertEqual(comparison["metric_ids"], ["weighted_error"])

    def test_missing_artifact_regenerates_and_rescores(self):
        path = self.root / "value.txt"

        class FileRunner(Runner):
            value = 1

            def generate(self, requests):
                path.write_text(str(self.value))
                return super().generate(requests)

        class FileMetric(WeightedError):
            def compute_sample(self, request, result):
                self.calls.append(request.sample_id)
                return MetricResult(sample_id=request.sample_id, metric_id=self.name,
                                    raw_value=float(Path(result.artifacts["text"].uri).read_text()),
                                    components={"weight": 1})

        runner = FileRunner([GenerationResult(sample_id="0", artifacts={"text": ArtifactRef(str(path), "text")})])
        metric = FileMetric()
        options = dict(output_dir=self.root / "file", requests=self.requests[:1], runner=runner, metrics=[metric])
        run_contract(**options)
        path.unlink()
        runner.value = 3
        run_contract(**options, resume=True)
        self.assertTrue(path.exists())
        self.assertEqual(runner.calls, ["0", "0"])
        self.assertEqual(metric.calls, ["0", "0"])
        self.assertEqual(self.summary("file")["leaderboard"][metric.name], 3)
        path.unlink()
        run_existing_results(output_dir=self.root / "file", requests=self.requests[:1],
                             results=list(runner.results.values()), metrics=[metric], resume=True)
        self.assertEqual(self.summary("file")["failed_samples"], 1)

    def test_interrupted_rerun_replaces_old_report_in_index(self):
        self.offline("rerun", [WeightedError()])

        class Interrupted(WeightedError):
            def compute_sample(self, request, result):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.offline("rerun", [Interrupted()], resume=True)
        root = self.root / "rerun"
        manifest = json.loads((root / "run_manifest.json").read_text())
        row = build_run_index(root)["runs"][0]
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["run_id"], manifest["run_id"])
        self.assertEqual(row["metrics"], {})
        self.assertFalse((root / "scorecard.json").exists())

    def test_generation_interruption_records_current_run(self):
        class Interrupted(Runner):
            def generate(self, requests):
                raise KeyboardInterrupt()

        for entrypoint in (run_contract, run_evaluate):
            root = self.root / entrypoint.__name__
            options = {"mode": "model"} if entrypoint is run_evaluate else {}
            with self.subTest(entrypoint=entrypoint.__name__), self.assertRaises(KeyboardInterrupt):
                entrypoint(output_dir=root, requests=self.requests, runner=Interrupted(self.results),
                           metrics=[WeightedError()], **options)
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["stage"], "generate")
            self.assertTrue(validate_contract_file(root / "run_manifest.json")["ok"])
            self.assertEqual(build_run_index(root)["runs"][0]["status"], "interrupted")

    def test_reporting_failure_does_not_leave_a_successful_report(self):
        with patch("worldfoundry.evaluation.tasks.execution.orchestration.existing_results.write_run_report_artifacts",
                   side_effect=OSError("report unavailable")):
            with self.assertRaises(OSError):
                self.offline("report_failure", [WeightedError()])
        root = self.root / "report_failure"
        manifest = json.loads((root / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stage"], "report")
        self.assertEqual(build_run_index(root)["runs"][0]["metrics"], {})

    def test_missing_metric_values_are_skipped_without_becoming_errors(self):
        self.results[1] = GenerationResult(sample_id="1")
        self.offline("skips", [ResultFieldMetric("value")])
        summary = self.summary("skips")
        self.assertEqual(summary["metrics"]["executed"], 2)
        self.assertEqual(summary["metrics"]["successful"], 1)
        self.assertEqual(summary["metrics"]["skipped"], 1)
        self.assertEqual(summary["metrics"]["failed"], 0)
        self.assertEqual(summary["per_metric"]["value"]["n_skipped"], 1)
        ledger = [json.loads(row) for row in (self.root / "skips" / "sample_ledger.jsonl").read_text().splitlines()]
        self.assertEqual(ledger[1]["status"], "skipped")
        self.assertNotIn("errors", ledger[1])
        self.results[0] = GenerationResult(sample_id="0")
        self.offline("all_skipped", [ResultFieldMetric("value")])
        report = json.loads((self.root / "all_skipped" / "summary.json").read_text())
        self.assertEqual(report["counts"]["successful_samples"], 0)
        self.assertEqual(report["counts"]["skipped_samples"], 2)

    def test_configured_runner_is_not_created_on_resume(self):
        runner = Runner(self.results)
        active = []

        @contextmanager
        def inference_context():
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def create_runner(*args, **kwargs):
            self.assertTrue(active)
            return ResolvedWorldModel(runner.model_id, runner, "runner_target")

        with patch("worldfoundry.evaluation.models.resolve_world_model_runner",
                   side_effect=create_runner) as resolve, patch(
                       "worldfoundry.core.worldfoundry_inference_context", side_effect=inference_context,
                   ) as runtime:
            for name, resume in (("lazy", False), ("lazy", True), ("cache_hit", False)):
                run_evaluate(mode="model", output_dir=self.root / name, requests=self.requests,
                             model_config={"model_id": runner.model_id, "runner": "fixture"},
                             metrics=[WeightedError()], resume=resume,
                             generation_cache_dir=self.root / "cache", generation_cache_mode="read-write")
            self.assertEqual(resolve.call_count, 1)
            self.assertEqual(runtime.call_count, 1)
        self.assertEqual(runner.calls, ["0", "1"])

    def test_registry_creation_does_not_install_or_wrap_runtime(self):
        class PlainRunner:
            def __init__(self, config):
                self.model_id = config.model_id

            def generate(self, requests):
                return []

        registry = ModelRunnerRegistry(include_builtins=False)
        registry.register_runner("plain", runner_class=PlainRunner)
        with patch("worldfoundry.core.install_worldfoundry_inference_infra") as install:
            runner = registry.create({"model_id": "plain", "runner": "plain"})
            install.assert_not_called()
        self.assertNotIn("generate", vars(runner))


if __name__ == "__main__":
    unittest.main()
