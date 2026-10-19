"""Offline cross-stage contracts for model generation, official scoring and restoration."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.eval_core.contract_fixture import CONTRACT_FIXTURE_MODEL_ID, CONTRACT_FIXTURE_RUNNER_TARGET
from worldfoundry.core.io.formats.serialization import write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark as mb
from worldfoundry.evaluation.tasks.execution.orchestration import model_benchmark_suite as suite
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import get_benchmark_generation_adapter
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_runner import run_benchmark_execution
from worldfoundry.evaluation.tasks.execution.orchestration.interfaces import OfficialRunResult
from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR

pytestmark = pytest.mark.unit


def test_generic_staging_resolves_relative_successes_only(tmp_path: Path) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    rows = []
    for name, status in (("good", "succeeded"), ("bad", "failed"), ("unknown", "unknown")):
        path = generation / f"{name}.mp4"
        path.write_bytes(name.encode())
        rows.append(
            GenerationResult(
                sample_id=name,
                status=status,
                artifacts={
                    "generated_video": ArtifactRef(uri=path.name if name == "good" else str(path), kind="video"),
                },
            ).to_dict()
        )
    write_jsonl(generation / "results.jsonl", rows)
    output = tmp_path / "inputs"
    assert mb._materialize_generated_artifacts(
        generation_output_dir=generation,
        generated_artifact_dir=output,
        artifact_manifest_path=tmp_path / "artifacts.jsonl",
        output_artifact="generated_video",
        allow_placeholders=False,
    ) == (1, 0)
    assert len(list(output.iterdir())) == 1
    assert next(output.iterdir()).read_bytes() == b"good"


@pytest.mark.parametrize(
    "mode,verified,child_valid,expected",
    [
        ("contract", False, False, False),
        ("official-run", False, False, False),
        ("official-run", True, False, False),
        ("official-run", True, True, True),
    ],
)
def test_outer_summary_preserves_metrics_and_narrows_eligibility(
    tmp_path: Path,
    mode: str,
    verified: bool,
    child_valid: bool,
    expected: bool,
) -> None:
    metrics = {
        "leaderboard": {"distance": 0.75},
        "per_metric": {
            "distance": {"mean": 0.75, "higher_is_better": False, "definition": {"name": "distance", "version": "1"}},
        },
        "summary": {"sample_count": 2, "successful_samples": 2},
    }
    card = tmp_path / "scorecard.json"
    card.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "normalization_ok": True,
                "eligibility": {"score_valid": True, "leaderboard_valid": child_valid},
            }
        )
    )
    child = OfficialRunResult(
        benchmark_id="vbench",
        output_dir=tmp_path,
        scorecard_path=card,
        official_benchmark_verified=verified,
        integration_evidence=True,
        metadata={"mode": mode, "contract_only": mode == "contract"},
    )
    summary = mb._model_benchmark_run_summary(
        request=mb.ModelBenchmarkRunRequest(
            output_dir=tmp_path, benchmark_id="vbench", benchmark_manifest_path=BENCHMARK_ZOO_DIR, model_id="m"
        ),
        status="succeeded",
        mode=mode,
        root=tmp_path,
        materialized_count=2,
        placeholder_count=0,
        generation_result=None,
        benchmark_payload=child.to_dict(),
        artifacts={},
    )
    assert summary["leaderboard"] == metrics["leaderboard"]
    assert summary["metrics"] == metrics
    assert summary["eligibility"]["leaderboard_valid"] is expected
    assert summary["eligibility"]["score_valid"] is (mode != "contract")
    assert (
        summary["comparison_identity"]["metric_definitions"]["distance"]
        == metrics["per_metric"]["distance"]["definition"]
    )


def test_failed_rerun_invalidates_root_success_without_deleting_inputs(tmp_path: Path) -> None:
    inputs = tmp_path / "user-inputs"
    inputs.mkdir()
    video = inputs / "sample.mp4"
    video.write_bytes(b"video")
    request = mb.ModelBenchmarkRunRequest(
        output_dir=tmp_path / "run",
        benchmark_id="vbench",
        benchmark_manifest_path=BENCHMARK_ZOO_DIR,
        benchmark_mode="contract",
        model_id="user-model",
        generated_artifact_dir=inputs,
    )
    first = mb.run_model_benchmark(request)
    assert video.read_bytes() == b"video"
    assert json.loads((first.output_dir / "summary.json").read_text())["eligibility"]["leaderboard_valid"] is False
    video.unlink()
    with pytest.raises(RuntimeError, match="no generated artifacts"):
        mb.run_model_benchmark(request)
    assert inputs.is_dir()
    assert json.loads((first.output_dir / "run_manifest.json").read_text())["status"] == "failed"
    assert not (first.output_dir / "summary.json").exists()
    assert not first.run_manifest_path.exists()


def test_official_subprocess_must_write_this_attempts_scorecard(tmp_path: Path) -> None:
    marker = tmp_path / "emit"
    marker.touch()
    command = (
        "import json,os; from pathlib import Path; "
        f"marker=Path({str(marker)!r}); "
        "card={'schema_version':'worldfoundry-scorecard','official_benchmark_verified':False,"
        "'integration_evidence':True,'run':{'status':'succeeded','run_id':'first'},"
        "'evaluation':{'available':True},'metrics':{'leaderboard':{'quality':0.8}}}; "
        "root=Path(os.environ['WORLDFOUNDRY_BENCHMARK_OUTPUT_DIR']); "
        "(root/'scorecard.json').write_text(json.dumps(card)) if marker.exists() else None"
    )
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "vbench",
                        "status": "confirmed_official_code",
                        "integration_status": "integrated",
                        "runner": {
                            "verification_status": "verified",
                            "validation_command": [sys.executable, "-c", command],
                            "expected_artifacts": ["scorecard.json"],
                        },
                    }
                ]
            }
        )
    )
    kwargs = dict(output_dir=tmp_path / "run", manifest_path=manifest, mode="official-validation", timeout_seconds=10)
    assert run_benchmark_execution("vbench", **kwargs).ok
    marker.unlink()
    second = run_benchmark_execution("vbench", **kwargs)
    assert second.metadata["returncode"] == 0
    assert not second.ok
    assert not second.integration_evidence
    assert json.loads(second.scorecard_path.read_text())["run"].get("run_id") != "first"


def test_suite_resume_checks_request_contents_and_required_scorecards(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    row = GenerationRequest(
        sample_id="one",
        task_name="video",
        inputs={"prompt": "old"},
        output_schema={"generated_video": {"kind": "video"}},
    )
    write_jsonl(path, [row.to_dict()])
    request = suite.ModelBenchmarkSuiteRequest(
        output_dir=tmp_path / "suite",
        benchmark_manifest_dir=BENCHMARK_ZOO_DIR,
        model_manifest_dir=None,
        model_ids=(CONTRACT_FIXTURE_MODEL_ID,),
        model_runner=CONTRACT_FIXTURE_RUNNER_TARGET,
        benchmark_ids=("vbench",),
        mode="contract",
        contract_fixture=True,
        requests_path=path,
    )
    first = suite.run_model_benchmark_suite(request)
    assert first.status == "succeeded"
    resumed = suite.run_model_benchmark_suite(replace(request, resume=True, metric_batch_size=7))
    assert resumed.cells[0]["resumed"] is True
    write_jsonl(path, [replace(row, inputs={"prompt": "new"}).to_dict()])
    second = suite.run_model_benchmark_suite(replace(request, resume=True))
    assert second.status == "succeeded"
    assert not second.cells[0]["resumed"]
    assert first.run_fingerprint != second.run_fingerprint
    cell_dir = Path(second.cells[0]["run_dir"])
    assert json.loads((cell_dir / "generation/requests.jsonl").read_text())["inputs"]["prompt"] == "new"
    generation_results = (cell_dir / "generation/results.jsonl").read_text()
    Path(second.cells[0]["benchmark_scorecard_path"]).unlink()
    third = suite.run_model_benchmark_suite(replace(request, resume=True))
    assert third.status == "succeeded"
    assert not third.cells[0]["resumed"]
    assert Path(third.cells[0]["benchmark_scorecard_path"]).is_file()
    assert (cell_dir / "generation/results.jsonl").read_text() == generation_results


def test_suite_comparison_retains_native_benchmark_scores(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "sample.mp4").write_bytes(b"video")
    source = tmp_path / "official.json"
    source.write_text(
        json.dumps(
            {
                "aesthetic_quality": [
                    0.75,
                    [
                        {"video_path": "sample.mp4", "video_results": 0.75},
                    ],
                ]
            }
        )
    )
    output = tmp_path / "suite"
    result = suite.run_model_benchmark_suite(
        suite.ModelBenchmarkSuiteRequest(
            output_dir=output,
            benchmark_manifest_dir=BENCHMARK_ZOO_DIR,
            model_manifest_dir=None,
            model_ids=("first", "second"),
            benchmark_ids=("vbench",),
            mode="official-validation",
            model_workers=1,
            generated_artifact_dir=videos,
            output_artifact="generated_video",
            benchmark_parameters={"official_results_path": str(source)},
        )
    )
    assert result.status == "succeeded"
    for cell in result.cells:
        root = Path(cell["run_dir"])
        child = json.loads((root / "benchmark/scorecard.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        assert summary["leaderboard"] == child["metrics"]["leaderboard"]
        assert summary["leaderboard"]["aesthetic_quality"] == 0.75
        assert summary["eligibility"]["score_valid"] is True
        assert summary["eligibility"]["leaderboard_valid"] is False
    comparison = json.loads((output / "comparison/comparison.json").read_text())
    assert "aesthetic_quality" in comparison["available_metric_ids"]
    assert "materialized_artifact_count" not in comparison["available_metric_ids"]


def test_adapter_consumes_explicit_root_and_split(tmp_path: Path) -> None:
    from worldfoundry.evaluation.tasks.execution.runners.aigcbench.aigcbench_prompts import PROMPT_MANIFEST_REL

    manifest = tmp_path / PROMPT_MANIFEST_REL
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([{"prompt_id": "custom", "prompt": "custom prompt", "prompt_type": "standard"}]))
    adapter = get_benchmark_generation_adapter("aigcbench")
    assert adapter is not None
    rows = adapter.materialize_requests(limit=None, dataset_root=tmp_path, split="validation")
    assert len(rows) == 1
    assert rows[0].sample_id == "custom"
    assert rows[0].split == "validation"
    assert rows[0].inputs["prompt"] == "custom prompt"


def test_adapter_rejects_unsupported_explicit_root(tmp_path: Path) -> None:
    adapter = get_benchmark_generation_adapter("videophy")
    assert adapter is not None
    with pytest.raises(ValueError, match="does not accept dataset_root"):
        adapter.materialize_requests(limit=None, dataset_root=tmp_path)
