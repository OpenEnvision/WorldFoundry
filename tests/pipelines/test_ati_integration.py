from __future__ import annotations

import pytest

from worldfoundry.cli.model_run import _field_kind
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
