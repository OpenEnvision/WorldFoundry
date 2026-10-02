from __future__ import annotations

import torch

import worldfoundry.studio.inference.execution as execution
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.inference.execution import (
    BaseRuntimeDriver,
    PipelineContext,
    PreparedInputs,
    StudioManager,
)


def test_materializer_preserves_pipeline_encoded_video(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    video_path = output_dir / "model.mp4"
    original = b"pipeline-encoded-video"
    video_path.write_bytes(original)
    entry = CatalogEntry(
        model_id="tensor-and-path",
        display_name="Tensor and Path",
        module_path="tests.fake",
        class_name="FakePipeline",
        family="video",
        category="Video Generation",
        summary="test",
    )
    context = PipelineContext(
        entry=entry,
        pipeline=object(),
        cache_key="test",
        backend="auto",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    request = PreparedInputs(
        prompt="",
        input_path="",
        image=None,
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=None,
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=24,
        num_frames=5,
        output_dir=str(output_dir),
        output_path=str(video_path),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="auto",
        endpoint="",
        api_key="",
        device="cpu",
    )

    def unexpected_reencode(*args, **kwargs):
        raise AssertionError("an existing encoded video must not be overwritten")

    monkeypatch.setattr(execution, "export_frames_to_video", unexpected_reencode)
    record = StudioManager(workspace_root=str(tmp_path / "workspace")).materialize_run(
        context,
        request,
        result={
            "artifact_path": str(video_path),
            "video": torch.zeros(1, 3, 5, 8, 8),
        },
        mode="run",
    )

    assert video_path.read_bytes() == original
    assert record.preview_video == str(video_path)


def test_driver_requests_mapping_from_var_kwargs_component_pipeline(tmp_path) -> None:
    received = {}

    class VarKwargsPipeline:
        def __call__(self, **kwargs):
            received.update(kwargs)
            return {"status": "failed", "error": "missing checkpoint"}

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    entry = CatalogEntry(
        model_id="component",
        display_name="Component",
        module_path="tests.fake",
        class_name="VarKwargsPipeline",
        family="video",
        category="Video Generation",
        summary="test",
    )
    context = PipelineContext(
        entry=entry,
        pipeline=VarKwargsPipeline(),
        cache_key="test",
        backend="auto",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    request = PreparedInputs(
        prompt="demo",
        input_path="",
        image=None,
        image_path=None,
        video_path=None,
        last_frame=None,
        last_frame_path=None,
        reference_images=[],
        reference_image_paths=[],
        interactions=None,
        camera_view=None,
        task_type="",
        intrinsics=None,
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=16,
        num_frames=1,
        output_dir=str(output_dir),
        output_path=str(output_dir / "video.mp4"),
        call_kwargs={},
        load_kwargs={},
        model_ref="",
        backend="auto",
        endpoint="",
        api_key="",
        device="cpu",
    )

    result = BaseRuntimeDriver()._invoke(context, request, mode="run")

    assert received["return_dict"] is True
    assert result["error"] == "missing checkpoint"
