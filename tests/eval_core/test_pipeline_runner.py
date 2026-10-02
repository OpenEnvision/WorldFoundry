from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from worldfoundry.evaluation.api import GenerationRequest
from worldfoundry.evaluation.models.pipelines.loading import call_pipeline_from_pretrained
from worldfoundry.evaluation.models.runners import pipeline as pipeline_runner_module
from worldfoundry.evaluation.models.runners.pipeline import WorldFoundryPipelineRunner
from worldfoundry.pipelines.pusa_vidgen.pipeline_pusa_vidgen import PusaVidGenPipeline
from worldfoundry.pipelines.echo_memory.pipeline_echo_memory import EchoMemoryContextK1Pipeline


@dataclass(frozen=True)
class _Profile:
    artifact_filename: str = "artifact.json"
    artifact_kind: str = "generated_world"
    task_family: str = "world_model"


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "success",
            "runtime": "recording",
            "artifact_kind": "generated_world",
            "artifact_path": kwargs["output_path"],
            "plan_path": str(Path(kwargs["output_path"]).with_suffix(".plan.json")),
        }


class _BlockedPipeline:
    def __call__(self, **kwargs):
        """
        Return a blocked plan payload for runner status propagation.

        Args:
            **kwargs: Normalized pipeline invocation values.
        """
        return {
            "status": "blocked",
            "runtime": "blocked-recording",
            "artifact_kind": "blocked_plan",
            "artifact_path": kwargs["output_path"],
            "backend_quality": "blocked_plan",
            "blocked_reason": "runtime is vendor-blocked",
        }


class _NativeInvocationPipeline:
    def __init__(self) -> None:
        self.invocations = []
        self.fallback_called = False

    def __call__(self, **kwargs):
        self.fallback_called = True
        raise AssertionError("fallback callable should not be used")

    def run_pipeline_invocation(self, invocation):
        self.invocations.append(invocation)
        return {
            "status": "success",
            "runtime": "native-invocation",
            "artifact_kind": "generated_world",
            "artifact_path": invocation.output_path,
        }


def test_pipeline_loader_preserves_unified_mapping_for_component_aware_loader() -> None:
    captured = {}

    class ComponentAwarePipeline:
        @classmethod
        def from_pretrained(cls, model_path=None, required_components=None, device="cuda"):
            captured.update(
                model_path=model_path,
                required_components=required_components,
                device=device,
            )
            return cls()

    model_path = {"model_path": "/models/action", "video_model_path": "/models/video"}
    pipeline = call_pipeline_from_pretrained(
        ComponentAwarePipeline,
        model_path=model_path,
        fallback_model_path="/models/action",
        required_components={"tokenizer": "/models/tokenizer"},
        device="cpu",
        model_id="component-aware",
    )

    assert isinstance(pipeline, ComponentAwarePipeline)
    assert captured == {
        "model_path": model_path,
        "required_components": {"tokenizer": "/models/tokenizer"},
        "device": "cpu",
    }


def test_echo_memory_pipeline_strips_unified_loader_metadata() -> None:
    pipeline = EchoMemoryContextK1Pipeline.from_pretrained(
        model_path={
            "model_id": "echo-memory-context-k1",
            "checkpoint_path": "/models/echo.safetensors",
            "wan_base_dir": "/models/wan",
            "pipeline_binding": "echo-memory-context-k1",
            "profile_id": "echo-memory-context-k1",
            "runtime_profile": "echo-memory-context-k1",
        },
        lazy=True,
    )

    runtime_kwargs = pipeline.synthesis_model.runtime_kwargs
    assert runtime_kwargs["checkpoint_path"] == "/models/echo.safetensors"
    assert runtime_kwargs["wan_base_dir"] == "/models/wan"
    assert "pipeline_binding" not in runtime_kwargs
    assert "profile_id" not in runtime_kwargs
    assert "runtime_profile" not in runtime_kwargs


def test_pipeline_runner_builds_operator_payload_and_consumes_runner_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_runner_module,
        "load_runtime_profile",
        lambda model_id: _Profile(),
    )
    pipeline = _RecordingPipeline()
    runner = WorldFoundryPipelineRunner(
        "recording-model",
        pipeline,
        pipeline_target="tests:RecordingPipeline",
    )

    result = runner.generate(
        [
            GenerationRequest(
                sample_id="sample-001",
                request_id="request-001",
                task_name="world-task",
                inputs={
                    "instruction": "move through the scene",
                    "image": "memory://image.png",
                    "proprio": [0.0, 1.0],
                    "camera_names": ["front"],
                },
                controls={
                    "sample_controls": {
                        "actions": [{"delta": [0.1]}],
                        "camera_pose": {"yaw": 12},
                    }
                },
                generation_kwargs={
                    "output_dir": str(tmp_path / "runner-output"),
                    "temperature": 0,
                    "operator_kwargs": {"camera_names": ["wrist"]},
                },
            )
        ]
    )[0]

    call = pipeline.calls[0]
    operator_kwargs = call["operator_kwargs"]

    assert result.status == "succeeded"
    assert result.artifacts["generated_world"].uri.endswith("sample-001_artifact.json")
    assert result.metadata["profile_task_family"] == "world_model"
    assert call["prompt"] == "move through the scene"
    assert call["images"] == "memory://image.png"
    assert call["interactions"] == [{"delta": [0.1]}]
    assert call["temperature"] == 0
    assert "output_dir" not in call
    assert Path(call["output_path"]).parent == tmp_path / "runner-output"
    assert operator_kwargs["sample_id"] == "sample-001"
    assert operator_kwargs["task_name"] == "world-task"
    assert operator_kwargs["proprio"] == [0.0, 1.0]
    assert operator_kwargs["camera_pose"] == {"yaw": 12}
    assert operator_kwargs["camera_names"] == ["wrist"]


def test_pipeline_runner_prefers_native_invocation_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_runner_module,
        "load_runtime_profile",
        lambda model_id: _Profile(),
    )
    pipeline = _NativeInvocationPipeline()
    runner = WorldFoundryPipelineRunner(
        "native-model",
        pipeline,
        pipeline_target="tests:NativeInvocationPipeline",
        output_dir=tmp_path,
    )

    result = runner.generate(
        [
            GenerationRequest(
                sample_id="native-001",
                request_id="request-001",
                task_name="world-task",
                inputs={"prompt": "navigate through the scene", "image": "memory://image.png"},
                generation_kwargs={"temperature": 0},
            )
        ]
    )[0]

    invocation = pipeline.invocations[0]
    assert result.status == "succeeded"
    assert pipeline.fallback_called is False
    assert invocation.prompt == "navigate through the scene"
    assert invocation.image == "memory://image.png"
    assert invocation.output_path.parent == tmp_path
    assert invocation.pipeline_kwargs == {"temperature": 0}


def test_pipeline_runner_propagates_blocked_plan_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_runner_module,
        "load_runtime_profile",
        lambda model_id: _Profile(artifact_filename="blocked.json", artifact_kind="blocked_plan"),
    )
    runner = WorldFoundryPipelineRunner(
        "blocked-model",
        _BlockedPipeline(),
        pipeline_target="tests:BlockedPipeline",
    )

    result = runner.generate(
        [
            GenerationRequest(
                sample_id="blocked-001",
                task_name="world-task",
                inputs={"instruction": "blocked request"},
                generation_kwargs={"output_dir": str(tmp_path)},
            )
        ]
    )[0]

    assert result.status == "blocked"
    assert result.error == "runtime is vendor-blocked"
    assert result.artifacts == {}
    assert result.metadata["backend_quality"] == "blocked_plan"


def test_pipeline_runner_fails_pusa_without_execute(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_runner_module,
        "load_runtime_profile",
        lambda model_id: _Profile(artifact_filename="pusa_request.json", artifact_kind="generated_video"),
    )
    runner = WorldFoundryPipelineRunner(
        "pusa-vidgen",
        PusaVidGenPipeline.from_pretrained(
            model_path={"checkpoint_root": tmp_path / "pusa", "base_model_root": tmp_path / "wan"},
            device="cpu",
        ),
        pipeline_target="worldfoundry.pipelines.pusa_vidgen.pipeline_pusa_vidgen:PusaVidGenPipeline",
    )

    result = runner.generate(
        [
            GenerationRequest(
                sample_id="pusa-001",
                task_name="sample_t2v",
                inputs={"prompt": "a small robot walks forward"},
                generation_kwargs={"output_dir": str(tmp_path / "out"), "seed": 7},
            )
        ]
    )[0]

    plan_path = tmp_path / "out" / "pusa-001_pusa_request.json"
    assert not plan_path.exists()
    assert result.status == "failed"
    assert result.artifacts == {}
    assert (
        result.error
        == "RuntimeError: Pusa VidGen requires execute=True; request-plan artifacts are no longer emitted."
    )
