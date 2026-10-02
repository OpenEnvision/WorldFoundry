from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from worldfoundry.cli.model_run import _field_kind
from worldfoundry.cli.main import _handle_direct_model_run
from worldfoundry.pipelines.ati.pipeline_ati import ATIPipeline


class _RecordingSynthesis:
    def __init__(self) -> None:
        self.call = None

    def predict(self, **kwargs):
        self.call = kwargs
        return {"status": "success", "video": "frames"}


def test_typed_cli_parses_operator_kwargs_as_json() -> None:
    assert _field_kind("operator_kwargs") == "json"


def test_ati_pipeline_forwards_typed_cli_operator_kwargs() -> None:
    synthesis = _RecordingSynthesis()
    pipeline = ATIPipeline(synthesis)

    result = pipeline(
        prompt="A forward camera move.",
        images="input.jpg",
        output_path="output.mp4",
        return_dict=True,
        operator_kwargs={
            "track_path": "tracks.pt",
            "num_inference_steps": 1,
            "width": 832,
        },
        width=640,
    )

    assert result["status"] == "success"
    assert synthesis.call == {
        "prompt": "A forward camera move.",
        "image": "input.jpg",
        "output_path": "output.mp4",
        "return_dict": True,
        "track_path": "tracks.pt",
        "num_inference_steps": 1,
        "width": 640,
    }


def test_ati_pipeline_rejects_non_mapping_operator_kwargs() -> None:
    pipeline = ATIPipeline(_RecordingSynthesis())

    with pytest.raises(TypeError, match="operator_kwargs must be a mapping"):
        pipeline(operator_kwargs=[("track_path", "tracks.pt")])


def test_direct_model_run_honors_fail_on_generation_error(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class _Result:
        exit_code = 1

        @staticmethod
        def to_dict():
            return {"status": "completed_with_failures", "exit_code": 1}

    def fake_execute(request):
        captured["request"] = request
        return _Result()

    monkeypatch.setattr("worldfoundry.evaluation.runner.execute_evaluate_run", fake_execute)
    schema = SimpleNamespace(
        runnable=True,
        model_id="ati-wan21-14b",
        requested_model_id="ati-wan21-14b",
        blocked_reason="",
        fields=(),
        task_id="default",
        variant_id="default",
        catalog_variant_id=None,
    )
    args = SimpleNamespace(
        model_run_schema=schema,
        requests_path=tmp_path / "requests.jsonl",
        split="default",
        output_dir=tmp_path / "run",
        metric=None,
        required_artifact=None,
        model_runner=None,
        model_manifest_dir=None,
        model_variant=None,
        model_parameter=None,
        model_runtime=None,
        model_config=None,
        dataset_id=None,
        run_id=None,
        fail_on_sample_error=False,
        fail_on_generation_error=True,
        no_artifacts_index=False,
        generation_cache_dir=None,
        generation_cache_mode="off",
        generation_cache_namespace="worldfoundry_run",
        json=True,
    )

    assert _handle_direct_model_run(args) == 1
    assert captured["request"].fail_on_sample_error is True
