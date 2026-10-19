from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark
from worldfoundry.evaluation.tasks.execution.orchestration.model_benchmark import ModelBenchmarkRunRequest
from worldfoundry.evaluation.tasks.execution.orchestration.service import ModelBenchmarkIntent, prepare_evaluation

pytestmark = pytest.mark.unit
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "worldfoundry/data/benchmarks/catalog"


@pytest.mark.parametrize("task_materialization", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_generation_preserves_resume_and_metric_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_materialization: bool, resume: bool
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    samples = data / "samples.jsonl"
    samples.write_text('{"sample_id":"a","prompt_id":"a","prompt":"a wooden room"}\n', encoding="utf-8")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "task.yaml").write_text(
        "name: resume-controls\nbenchmark_name: vbench\n"
        "protocol: external_official_benchmark\nevaluation_protocol: external_official_runner\n"
        "input_keys: [prompt, prompt_id]\noutput_keys: [generated_video]\n"
        "data:\n  metadata_path: samples.jsonl\n",
        encoding="utf-8",
    )
    request = ModelBenchmarkRunRequest(
        output_dir=tmp_path / "output",
        benchmark_id="vbench",
        benchmark_manifest_path=MANIFEST_PATH,
        model_id=CONTRACT_FIXTURE_MODEL_ID,
        task_name="resume-controls" if task_materialization else None,
        task_roots=(tasks,),
        dataset_root=data,
        requests_path=None if task_materialization else samples,
        resume=resume,
        metric_batch_size=7,
    )
    execute = Mock()
    monkeypatch.setattr(model_benchmark, "execute_evaluate_run", execute)

    assert model_benchmark._run_generation(request, tmp_path / "output") is execute.return_value
    forwarded = execute.call_args.args[0]
    assert forwarded.resume is resume
    assert forwarded.metric_batch_size == 7


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_is_rejected_before_generation(tmp_path: Path, batch_size: int) -> None:
    request = ModelBenchmarkRunRequest(
        output_dir=tmp_path / "output",
        benchmark_id="vbench",
        benchmark_manifest_path=MANIFEST_PATH,
        metric_batch_size=batch_size,
    )
    with pytest.raises(ValueError, match="metric_batch_size must be positive"):
        model_benchmark.run_model_benchmark(request)

    prepared = prepare_evaluation(
        ModelBenchmarkIntent(
            output_dir=tmp_path / "output",
            model_id="vchitect-2-t2v",
            benchmark_id="vbench",
            benchmark_mode="contract",
            metric_batch_size=batch_size,
        )
    )
    assert "metric_batch_size_invalid" in {issue.code for issue in prepared.issues}
