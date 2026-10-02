from __future__ import annotations

from pathlib import Path

from worldfoundry.evaluation.tasks.execution.orchestration.interfaces import (
    BenchmarkRunner,
    BenchmarkSample,
    DatasetMaterializationPlan,
    OfficialRunResult,
    OfficialRunStage,
    OfficialBenchmarkRunner,
)


class TinyRunner:
    benchmark_id = "tiny"

    def load_manifest(self):
        return {"benchmark_id": self.benchmark_id}

    def materialization_plan(self):
        return DatasetMaterializationPlan(
            benchmark_id=self.benchmark_id,
            dataset_ids=("org/tiny",),
            commands=(("hf", "download", "org/tiny"),),
        )

    def iter_samples(self):
        return iter((BenchmarkSample(sample_id="sample-1", inputs={"prompt": "hello"}),))

    def evaluate(self, *, output_dir: str | Path, **kwargs):
        root = Path(output_dir)
        return OfficialRunResult(
            benchmark_id=self.benchmark_id,
            output_dir=root,
            scorecard_path=root / "scorecard.json",
            official_benchmark_verified=True,
            integration_evidence=True,
        )


class TinyOfficialRunner(TinyRunner):
    def prepare(self, *, output_dir: str | Path, **kwargs):
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="prepare",
            output_dir=Path(output_dir),
            status="ready",
            data={"kwargs": dict(kwargs)},
        )

    def run(self, prepared):
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="run",
            output_dir=prepared.output_dir,
            status="finished",
            data=prepared.data,
        )

    def collect(self, run_result):
        return OfficialRunStage(
            benchmark_id=self.benchmark_id,
            stage="collect",
            output_dir=run_result.output_dir,
            status="collected",
            data={"collected": True, **dict(run_result.data)},
        )

    def normalize(self, collected):
        return self.evaluate(output_dir=collected.output_dir)

    def report_metadata(self):
        return {"benchmark_id": self.benchmark_id, "runner": "tiny"}


def test_benchmark_runner_protocol_and_value_objects() -> None:
    runner = TinyRunner()
    sample = next(runner.iter_samples())
    plan = runner.materialization_plan()
    result = runner.evaluate(output_dir="tmp/tiny")

    assert isinstance(runner, BenchmarkRunner)
    assert not isinstance(runner, OfficialBenchmarkRunner)
    assert sample.to_dict()["inputs"]["prompt"] == "hello"
    assert plan.to_dict()["commands"] == [["hf", "download", "org/tiny"]]
    assert result.ok is True
    assert result.to_dict()["scorecard_path"] == "tmp/tiny/scorecard.json"


def test_official_benchmark_runner_protocol_and_stage_result() -> None:
    runner = TinyOfficialRunner()
    prepared = runner.prepare(output_dir="tmp/tiny-official", mode="contract")
    run_result = runner.run(prepared)
    collected = runner.collect(run_result)
    result = runner.normalize(collected)

    assert isinstance(runner, BenchmarkRunner)
    assert isinstance(runner, OfficialBenchmarkRunner)
    assert prepared.to_dict()["output_dir"] == "tmp/tiny-official"
    assert prepared.data["kwargs"]["mode"] == "contract"
    assert run_result.stage == "run"
    assert collected.data["collected"] is True
    assert runner.report_metadata()["runner"] == "tiny"
    assert result.ok is True
