from __future__ import annotations

import json

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)

from .wrbench_prompts import copy_wrbench_generated_videos
from .wrbench_runtime import discover_video_manifest


def test_wrbench_adapter_materializes_discoverable_runtime_manifest(tmp_path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    generated_dir = tmp_path / "generated"
    artifact_manifest = tmp_path / "generated_artifacts.jsonl"
    source_video = generation_dir / "model-output.mp4"
    source_video.write_bytes(b"not-decoded-by-this-contract-test")

    request = GenerationRequest(
        sample_id="family__T1__none__yaw_LR",
        task_name="wrbench",
        split="natural25",
        request_id="request-1",
        inputs={
            "prompt": "A controlled camera-motion prompt.",
            "world_state_prompt": "The object remains on the table.",
            "expected_state": "The object remains on the table.",
            "family_id": "family",
            "variant_id": "family__T1__none",
            "output_id": "family__T1__none__yaw_LR",
            "reasoning_tier": "T1",
            "event_delta": "spatial",
            "official_video_name": "family__T1__none__yaw_LR.mp4",
        },
        controls={
            "camera_intent": "yaw_LR",
            "camera_preset": "preset:yaw_LR",
            "camera_script": "yaw:left:60@40,yaw:right:60@41",
            "requires_go_return": True,
            "camera_sidecar": "requested-but-not-generated.camera.json",
            "target_camera_poses": [[1.0]],
        },
    )
    result = GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id="in-tree-model",
        artifacts={"generated_video": ArtifactRef(uri=source_video.name, kind="video")},
    )
    write_jsonl(generation_dir / "requests.jsonl", [request.to_dict()])
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict()])

    adapter = get_benchmark_generation_adapter("wrbench")
    assert adapter is not None
    assert adapter.artifact_materializer is not None
    counts = adapter.materialize_artifacts(
        generation_output_dir=generation_dir,
        generated_artifact_dir=generated_dir,
        artifact_manifest_path=artifact_manifest,
        output_artifact="generated_video",
    )

    assert counts == (1, 0)
    official_video = generated_dir / "family__T1__none__yaw_LR.mp4"
    assert official_video.read_bytes() == source_video.read_bytes()
    runtime_manifest = discover_video_manifest(generated_dir)
    assert runtime_manifest == (generated_dir / "videos_manifest.json").resolve()
    rows = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["video_id"] == request.sample_id
    assert row["path"] == str(official_video.resolve())
    assert row["video_path"] == str(official_video.resolve())
    assert row["model"] == "in-tree-model"
    assert row["camera"] == "yaw_LR"
    assert row["camera_type"] == "yaw_LR"
    assert row["world_state_prompt"] == request.inputs["world_state_prompt"]
    assert row["family_id"] == "family"
    assert row["variant_id"] == "family__T1__none"
    assert row["generation_prompt"] == request.inputs["prompt"]
    assert row["ti2v_prompt"] == request.inputs["prompt"]
    assert "camera_sidecar" not in row
    assert "target_camera_poses" not in row
    assert not list(generated_dir.glob("*.camera.json"))
    assert read_jsonl_objects(artifact_manifest)[0]["status"] == "copied"


def test_wrbench_materializer_does_not_manifest_failed_generation(tmp_path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    generated_dir = tmp_path / "generated"
    artifact_manifest = tmp_path / "generated_artifacts.jsonl"
    request = GenerationRequest(
        sample_id="failed-sample",
        task_name="wrbench",
        inputs={"prompt": "prompt", "official_video_name": "failed-sample.mp4"},
    )
    result = GenerationResult(
        sample_id=request.sample_id,
        status="failed",
        error="model error",
    )
    write_jsonl(generation_dir / "requests.jsonl", [request.to_dict()])
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict()])

    counts = copy_wrbench_generated_videos(
        generation_output_dir=generation_dir,
        generated_artifact_dir=generated_dir,
        artifact_manifest_path=artifact_manifest,
    )

    assert counts == (0, 0)
    assert json.loads((generated_dir / "videos_manifest.json").read_text(encoding="utf-8")) == []
    assert read_jsonl_objects(artifact_manifest)[0]["status"] == "generation_failed"
    assert not (generated_dir / "failed-sample.mp4").exists()
