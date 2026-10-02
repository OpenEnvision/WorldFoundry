from __future__ import annotations

import ast
import hashlib
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from worldfoundry.evaluation.api import (
    AggregateResult,
    ArtifactRef,
    BenchmarkSpec,
    GenerationRequest,
    GenerationResult,
    MetricResult,
    MetricSpec,
    WorldModelRunner,
    WorldModelConfig,
    WorldModelManifest,
    WorldTaskConfig,
    enrich_artifact_ref,
    local_path_for_uri,
)
from worldfoundry.evaluation.api.json_contract import canonical_json_dumps, json_sha256


def _round_trip(contract):
    restored = contract.__class__.from_json(contract.to_json())
    assert restored == contract
    assert restored.to_dict() == contract.to_dict()
    assert restored.stable_hash() == contract.stable_hash()
    assert hash(restored) == hash(contract)
    return restored


def test_artifact_ref_hash_and_json_round_trip(tmp_path: Path) -> None:
    payload = b"worldfoundry artifact payload\n"
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(payload)

    artifact = ArtifactRef.from_path(
        media_path,
        kind="video",
        mime_type="video/mp4",
        media_metadata={"fps": 24, "frame_count": 12, "resolution": [640, 480]},
    )

    assert is_dataclass(artifact)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)
    assert artifact.kind == "video"
    _round_trip(artifact)


def test_artifact_ref_requires_kind() -> None:
    with pytest.raises(ValueError, match="requires kind"):
        ArtifactRef.from_dict({"uri": "out/sample.mp4"})


def test_artifact_ref_from_uri_enriches_local_files(tmp_path: Path) -> None:
    media_path = tmp_path / "sample.mp4"
    media_path.write_bytes(b"video bytes")

    artifact = ArtifactRef.from_uri("sample.mp4", kind="video", base_dir=tmp_path)
    assert artifact.uri == "sample.mp4"
    assert artifact.kind == "video"
    assert artifact.sha256 == hashlib.sha256(b"video bytes").hexdigest()
    assert artifact.size_bytes == len(b"video bytes")
    assert artifact.mime_type == "video/mp4"

    remote = ArtifactRef.from_uri("https://example.com/sample.mp4", kind="video")
    assert local_path_for_uri(remote.uri) is None
    assert enrich_artifact_ref(remote, base_dir=tmp_path) == remote


def test_generation_request_and_result_restore_nested_artifact_refs() -> None:
    source = ArtifactRef.from_bytes(b"image", uri="memory://image.png", kind="image")
    request = GenerationRequest(
        sample_id="sample-001",
        task_name="sample_static_i2v",
        split="smoke",
        request_id="req-001",
        inputs={"prompt": "orbit camera", "image": source, "frames": [source]},
        controls={"camera": {"yaw": 10}},
        generation_kwargs={"seed": 7},
        output_schema={"generated_video": {"kind": "video"}},
        cache_policy={"mode": "reuse"},
    )

    restored_request = _round_trip(request)
    assert isinstance(restored_request.inputs["image"], ArtifactRef)
    assert isinstance(restored_request.inputs["frames"][0], ArtifactRef)
    assert restored_request.task_id == "sample_static_i2v"

    result = GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id="sample-world-model",
        artifacts={"generated_video": ArtifactRef(uri="out/sample.mp4", kind="video", sha256="0" * 64)},
        timings={"generate_seconds": 0.5},
    )

    restored_result = _round_trip(result)
    assert isinstance(restored_result.artifacts["generated_video"], ArtifactRef)


def test_json_contract_canonical_json_is_strict_and_stable() -> None:
    left = {"b": ("x", 2), "a": {"prompt": "向左转", "seed": 7}}
    right = {"a": {"seed": 7, "prompt": "向左转"}, "b": ["x", 2]}

    assert canonical_json_dumps(left) == '{"a":{"prompt":"向左转","seed":7},"b":["x",2]}'
    assert json_sha256(left) == json_sha256(right)


def test_model_metric_task_and_benchmark_contracts_round_trip() -> None:
    manifest = WorldModelManifest(
        model_id="sample-world-model",
        name="Sample World Model",
        aliases=("sample",),
        version="0.1",
        provider="local",
        capabilities=("i2v", "camera_control"),
        supported_tasks=("sample_static_i2v",),
        output_artifacts=("generated_video",),
        tags=("smoke",),
    )
    config = WorldModelConfig(
        model_id=manifest.model_id,
        runner="tests.local:SampleRunner",
        parameters={"quality": "fast"},
        runtime={"device": "cpu"},
        seed=123,
        manifest=manifest,
    )
    metric = MetricSpec(
        id="camera_error",
        aliases=("cam_err",),
        display_name="Camera Error",
        description="Reference camera-path error.",
        version="1",
        family="reference",
        capability="camera_control",
        requires_reference=True,
        required_artifacts=("generated_video", "camera_path"),
        output_unit="degrees",
        higher_is_better=False,
        normalizer="inverse_error_v1",
        aggregator="mean",
        statistics=("mean", "stderr"),
        primary=True,
        weight=2.0,
        implementation="tests.metrics:CameraErrorMetric",
        tags=("geometry",),
    )
    task = WorldTaskConfig(
        task_id="sample_static_i2v",
        protocol="open_loop",
        evaluation_protocol="reference_metrics",
        capability_track="core_video",
        input_keys=("prompt", "image"),
        output_keys=("generated_video",),
        metric_ids=(metric.id,),
        metric_groups=("camera",),
        tags=("image", "static"),
        generation_defaults={"num_frames": 16},
    )
    benchmark = BenchmarkSpec(
        benchmark_id="sample_image_static_smoke",
        version="v1",
        tasks=(task,),
        metrics=(metric,),
        splits=("smoke",),
        tags=("worldfoundry",),
        dataset_root="data/benchmarks/WorldFoundry",
    )

    for contract in (manifest, config, metric, task, benchmark):
        assert is_dataclass(contract)
        _round_trip(contract)

    restored_benchmark = BenchmarkSpec.from_json(benchmark.to_json())
    assert restored_benchmark.tasks[0].task_id == task.name
    assert restored_benchmark.metrics[0].metric_id == metric.id


def test_metric_results_round_trip_and_hash() -> None:
    artifact = ArtifactRef(uri="reports/judge.json", kind="judge_report", sha256="1" * 64)
    result = MetricResult(
        sample_id="sample-001",
        metric_id="clip_score",
        raw_value=0.31,
        normalized_value=0.73,
        components={"text": 0.8, "image": 0.66},
        valid=True,
        coverage=1.0,
        artifact_refs={"judge": artifact},
        judge_trace={"provider": "stub", "model": "stub-vlm"},
    )
    aggregate = AggregateResult(
        metric_id=result.metric_id,
        n_total=2,
        n_valid=1,
        n_skipped=1,
        raw_stats={"mean": 0.31},
        normalized_stats={"mean": 0.73},
        confidence_interval={"lower": 0.7, "upper": 0.76},
        stderr=0.03,
        skip_breakdown={"missing_artifact": 1},
    )

    assert isinstance(_round_trip(result).artifact_refs["judge"], ArtifactRef)
    _round_trip(aggregate)


def test_world_model_runner_protocol_accepts_minimal_runner() -> None:
    class LocalRunner:
        model_id = "sample-world-model"
        capabilities = {"i2v"}

        @classmethod
        def from_config(cls, config: WorldModelConfig) -> "LocalRunner":
            assert config.model_id == cls.model_id
            return cls()

        def generate(self, requests):
            return [GenerationResult(sample_id=request.sample_id, model_id=self.model_id) for request in requests]

        def cleanup(self) -> None:
            self.cleaned = True

    runner = LocalRunner.from_config(WorldModelConfig(model_id="sample-world-model", runner="local"))
    request = GenerationRequest(sample_id="sample-001", task_name="sample_static_i2v")

    assert isinstance(runner, WorldModelRunner)
    assert runner.generate([request])[0].sample_id == request.sample_id


def test_api_contract_imports_are_stdlib_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_root = repo_root / "src" / "worldfoundry" / "evaluation" / "api"
    local_modules = {path.stem for path in api_root.glob("*.py")}
    allowed_external = set(sys.stdlib_module_names) | {"__future__"}

    def assert_modules(path: Path, modules: set[str]) -> None:
        for mod in modules:
            if mod == "worldfoundry":
                continue
            assert mod in allowed_external, f"{path} imports non-stdlib module {mod!r}"

    for path in api_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".", 1)[0]
                    if name == "worldfoundry":
                        assert alias.name.startswith(
                            "worldfoundry.evaluation.api"
                        ), f"{path} imports disallowed first-party {alias.name!r}"
                    else:
                        assert name in allowed_external, f"{path} imports non-stdlib module {name!r}"
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    modules = {node.module.split(".", 1)[0]} if node.module else set()
                    modules -= local_modules
                    assert_modules(path, modules)
                else:
                    if not node.module:
                        continue
                    if node.module.startswith("worldfoundry."):
                        assert node.module.startswith(
                            "worldfoundry.evaluation.api."
                        ), f"{path} imports disallowed first-party {node.module!r}"
                        continue
                    modules = {node.module.split(".", 1)[0]}
                    assert modules <= allowed_external, f"{path} imports non-stdlib modules: {modules - allowed_external}"
