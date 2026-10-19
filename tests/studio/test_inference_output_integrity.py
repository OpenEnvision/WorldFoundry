"""Conditioning attachments cannot satisfy generated-output requirements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from worldfoundry.studio.inference import execution
from worldfoundry.studio.inference.catalog import CatalogEntry


@pytest.fixture()
def prepared_run(tmp_path):
    manager = execution.StudioManager(workspace_root=str(tmp_path / "studio"))
    entry = CatalogEntry(
        model_id="output-integrity",
        display_name="Output Integrity",
        module_path="tests.fake",
        class_name="EmptyPipeline",
        family="video",
        category="Video Generation",
        summary="test",
    )
    video_path = execution.export_frames_to_video(
        [np.zeros((16, 16, 3), dtype=np.uint8)] * 2, str(tmp_path / "source.mp4"), fps=8
    )
    request = manager.prepare_inputs(
        entry=entry,
        prompt="test",
        input_path="",
        image=Image.new("RGB", (16, 16)),
        video=video_path,
        last_frame=None,
        reference_files=None,
        interactions_text="",
        camera_view_text="",
        task_type="",
        intrinsics_text="",
        meta_path="",
        panorama_path="",
        scene_name="",
        fps=8,
        num_frames=2,
        call_kwargs_text="{}",
        load_kwargs_text="{}",
        model_ref="",
        backend="auto",
        endpoint="",
        api_key="",
        device="cpu",
    )
    context = execution.PipelineContext(
        entry=entry,
        pipeline=object(),
        cache_key="test",
        backend="auto",
        model_ref="",
        endpoint="",
        load_kwargs={},
        device="cpu",
    )
    return manager, context, request


def _require(request, kind):
    request.infer_metadata = {"task": {"outputs": [{"artifact_id": "generated", "kind": kind, "required": True}]}}


@pytest.mark.parametrize("kind", ("video", "generated_video", "image", "generated_image"))
@pytest.mark.parametrize("explicit_input_result", (False, True))
def test_input_media_does_not_satisfy_required_outputs(prepared_run, monkeypatch, kind, explicit_input_result) -> None:
    manager, context, request = prepared_run
    _require(request, kind)
    monkeypatch.setenv(execution.VIDEO_PREVIEW_IMAGE_ENV, "1")
    before_image = Path(request.image_path).read_bytes()
    before_video = Path(request.video_path).read_bytes()
    result = {"preview_image": request.image_path, "video_path": request.video_path} if explicit_input_result else {}
    with pytest.raises(RuntimeError, match="required inference output"):
        manager.materialize_run(context, request, result=result, mode="run")
    assert Path(request.image_path).read_bytes() == before_image
    assert Path(request.video_path).read_bytes() == before_video
    assert not Path(request.output_path).exists()
    assert not (Path(request.output_dir) / "preview.png").exists()
    assert not (Path(request.output_dir) / "manifest.json").exists()


def test_generated_outputs_are_selected_over_inputs(prepared_run) -> None:
    manager, context, request = prepared_run
    _require(request, "generated_image")
    generated = Path(request.output_dir) / "generated.png"
    Image.new("RGB", (16, 16), "green").save(generated)
    record = manager.materialize_run(context, request, result={}, mode="run")
    assert record.status == "succeeded"
    assert record.preview_image == str(generated)
    assert record.preview_video is None


def test_input_video_does_not_suppress_generated_frame_export(prepared_run) -> None:
    manager, context, request = prepared_run
    _require(request, "video")
    original = Path(request.video_path).read_bytes()
    result = {"artifact_paths": [request.video_path], "frames": [np.full((16, 16, 3), 90, dtype=np.uint8)] * 2}
    record = manager.materialize_run(context, request, result=result, mode="run")
    assert record.status == "succeeded"
    assert record.preview_video == request.output_path
    assert Path(request.output_path).is_file()
    assert Path(request.video_path).read_bytes() == original


def test_explicit_caller_owned_model_input_is_not_a_generated_asset(prepared_run, tmp_path) -> None:
    manager, context, request = prepared_run
    source = tmp_path / "input.obj"
    source.write_text("v 0 0 0\n")
    request.input_path = str(source)
    _require(request, "generated_3d_asset")
    with pytest.raises(RuntimeError, match="required inference output"):
        manager.materialize_run(context, request, result={"model_path": str(source)}, mode="run")
    assert source.read_text() == "v 0 0 0\n"


def test_blocked_model_result_does_not_promote_input_previews(prepared_run) -> None:
    manager, context, request = prepared_run
    _require(request, "video")
    record = manager.materialize_run(context, request, result={"status": "blocked"}, mode="run")
    assert record.status == "blocked"
    assert record.preview_video is None
    assert record.preview_image is None
