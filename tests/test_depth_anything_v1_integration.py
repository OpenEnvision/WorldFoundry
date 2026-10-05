from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from worldfoundry.base_models.three_dimensions.depth.depth_anything.runtime_v1 import (
    DepthAnything1Representation,
)
from worldfoundry.cli.model_run import load_model_run_schema
from worldfoundry.pipelines.depth_anything.pipeline_depth_anything_v1 import (
    DepthAnything1Pipeline,
)


class _SavedDepthResult:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self.output_dir: str | None = None

    def save(self, output_dir: str | None = None) -> list[str]:
        self.output_dir = output_dir
        return [str(self.artifact)]


def test_depth_anything_v1_catalog_is_runnable() -> None:
    schema = load_model_run_schema("depth-anything-v1")

    assert schema.integration_status == "integrated"
    assert schema.runnable is True


def test_depth_anything_v1_accepts_unified_runner_input(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "input_depth.png"
    depth_result = _SavedDepthResult(artifact)
    captured = {}
    pipeline = DepthAnything1Pipeline(representation=object(), data_type="image", device="cpu")

    def fake_run_image(source: str, grayscale: bool = False):
        captured.update(source=source, grayscale=grayscale)
        return depth_result

    monkeypatch.setattr(pipeline, "run_image", fake_run_image)
    result = pipeline(
        output_path=tmp_path / "sample-0000_depth-anything-v1.png",
        return_dict=True,
        operator_kwargs={"input": "/fixtures/input.jpg"},
        grayscale=True,
    )

    assert captured == {"source": "/fixtures/input.jpg", "grayscale": True}
    assert depth_result.output_dir == str(tmp_path)
    assert result == {
        "status": "succeeded",
        "artifact_path": str(artifact),
        "artifacts": [str(artifact)],
    }


def test_depth_anything_v1_keeps_legacy_result_api(monkeypatch) -> None:
    depth_result = _SavedDepthResult(Path("input_depth.png"))
    pipeline = DepthAnything1Pipeline(representation=object(), data_type="image", device="cpu")
    monkeypatch.setattr(pipeline, "run_image", lambda source, grayscale=False: depth_result)

    assert pipeline("/fixtures/input.jpg") is depth_result


def test_depth_anything_v1_normalizes_transformers_output() -> None:
    predicted_depth = torch.ones(1, 37, 41)

    assert (
        DepthAnything1Representation._prediction_tensor(
            SimpleNamespace(predicted_depth=predicted_depth)
        )
        is predicted_depth
    )
