from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest
from PIL import Image

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)
from worldfoundry.evaluation.tasks.execution.runners.fetv.fetv_prompts import (
    FETV_BOUNDED_PROMPT_NAME,
    FETV_GENERATION_MANIFEST_NAME,
    copy_fetv_generated_videos,
    materialize_fetv_generation_requests,
)


def _write_video(path, frame_count: int = 20) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))
    assert writer.isOpened()
    for frame_index in range(frame_count):
        writer.write(np.full((32, 32, 3), frame_index * 5, dtype=np.uint8))
    writer.release()


def _write_generation_manifests(generation_dir, requests, results) -> None:
    generation_dir.mkdir()
    write_jsonl(generation_dir / "requests.jsonl", [request.to_dict() for request in requests])
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict() for result in results])


def _success_result(request, video_path) -> GenerationResult:
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id="test-model",
        artifacts={"generated_video": ArtifactRef(uri=str(video_path), kind="video")},
        status="succeeded",
    )


def test_adapter_persists_official_sent_mapping_and_decodes_frames(tmp_path) -> None:
    adapter = get_benchmark_generation_adapter("fetv")
    assert adapter is not None
    requests = adapter.materialize_requests(limit=1)
    request = requests[0]
    assert request.inputs["sent_index"] == 0
    assert request.inputs["official_video_name"] == "0.mp4"
    assert request.inputs["official_frame_dir_name"] == "sent0_frames"

    video_path = tmp_path / "arbitrary-model-output-name.mp4"
    _write_video(video_path)
    generation_dir = tmp_path / "generation"
    _write_generation_manifests(generation_dir, requests, [_success_result(request, video_path)])
    generated = tmp_path / "generated"
    artifact_manifest = tmp_path / "generated_artifacts.jsonl"

    count, placeholders = adapter.materialize_artifacts(
        generation_output_dir=generation_dir,
        generated_artifact_dir=generated,
        artifact_manifest_path=artifact_manifest,
        output_artifact="generated_video",
    )

    assert (count, placeholders) == (1, 0)
    frames = sorted((generated / "sent0_frames").glob("frame*.jpg"))
    assert len(frames) == 16
    for frame in frames:
        with Image.open(frame) as image:
            image.verify()
    manifest = read_jsonl_objects(generated / FETV_GENERATION_MANIFEST_NAME)
    assert manifest[0]["sample_id"] == request.sample_id
    assert manifest[0]["sent_index"] == 0
    assert manifest[0]["frame_count"] == 16
    assert read_jsonl_objects(generated / FETV_BOUNDED_PROMPT_NAME)[0]["prompt"] == request.inputs["prompt"]
    assert read_jsonl_objects(artifact_manifest) == manifest


def test_materializer_rejects_duplicate_sent_indices(tmp_path) -> None:
    request = materialize_fetv_generation_requests(limit=1)[0]
    duplicate = copy.deepcopy(request.to_dict())
    duplicate["sample_id"] = "different-sample"
    from worldfoundry.evaluation.api import GenerationRequest

    generation_dir = tmp_path / "generation"
    _write_generation_manifests(generation_dir, [request, GenerationRequest.from_dict(duplicate)], [])

    with pytest.raises(ValueError, match="duplicate FETV sent_index"):
        copy_fetv_generated_videos(
            generation_output_dir=generation_dir,
            generated_artifact_dir=tmp_path / "generated",
            artifact_manifest_path=tmp_path / "artifacts.jsonl",
        )


def test_materializer_rejects_incomplete_result_coverage(tmp_path) -> None:
    requests = materialize_fetv_generation_requests(limit=2)
    video_path = tmp_path / "one.mp4"
    _write_video(video_path)
    generation_dir = tmp_path / "generation"
    _write_generation_manifests(generation_dir, requests, [_success_result(requests[0], video_path)])

    with pytest.raises(ValueError, match="coverage mismatch"):
        copy_fetv_generated_videos(
            generation_output_dir=generation_dir,
            generated_artifact_dir=tmp_path / "generated",
            artifact_manifest_path=tmp_path / "artifacts.jsonl",
        )


def test_materializer_rejects_an_undecodable_video(tmp_path) -> None:
    request = materialize_fetv_generation_requests(limit=1)[0]
    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"not a video")
    generation_dir = tmp_path / "generation"
    _write_generation_manifests(generation_dir, [request], [_success_result(request, broken_video)])

    with pytest.raises(ValueError, match="unable to decode FETV generated video"):
        copy_fetv_generated_videos(
            generation_output_dir=generation_dir,
            generated_artifact_dir=tmp_path / "generated",
            artifact_manifest_path=tmp_path / "artifacts.jsonl",
        )
