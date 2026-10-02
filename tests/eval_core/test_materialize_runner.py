from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.evaluation.runner import (
    MATERIALIZED_REQUESTS_SCHEMA_VERSION,
    materialize_generation_requests,
    materialize_requests_from_dataset_manifest,
    materialize_requests_from_benchmark,
)
from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest


def test_materialize_generation_requests_preserves_inputs_controls_and_outputs() -> None:
    requests = materialize_generation_requests(
        [
            {
                "sample_id": "sample-a",
                "initial_context": {"prompt": "orbit around object"},
                "ref_image": "image/a.png",
                "control_sequence": [{"camera": "left"}],
                "expected_outputs": {"generated_video": {"kind": "video", "fps": 8}},
            }
        ],
        task_name="synthetic_i2v",
        split="smoke",
        input_keys=("prompt", "ref_image"),
        output_keys=("generated_video",),
        generation_defaults={"num_frames": 16},
    )

    request = requests[0]
    assert request.sample_id == "sample-a"
    assert request.request_id == "synthetic_i2v:sample-a"
    assert request.inputs["prompt"] == "orbit around object"
    assert request.inputs["ref_image"] == "image/a.png"
    assert request.controls["control_sequence"] == [{"camera": "left"}]
    assert request.generation_kwargs["num_frames"] == 16
    assert request.output_schema["generated_video"]["fps"] == 8


def test_materialize_requests_from_benchmark_rejects_benchmark_zoo(tmp_path: Path) -> None:
    class BenchmarkZooDefinition:
        source_kind = "benchmark_zoo"

    with pytest.raises(ValueError, match="Benchmark Zoo entries do not materialize samples"):
        materialize_requests_from_benchmark(
            BenchmarkZooDefinition(),
            tmp_path,
            limit=1,
            split="smoke",
        )


def test_materialize_requests_from_dataset_manifest_uses_sample_manifest(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        '{"sample_id": "sample-a", "prompt": "move forward", "control_sequence": [{"action": "forward"}]}\n'
        '{"sample_id": "sample-b", "prompt": "turn right"}\n',
        encoding="utf-8",
    )
    manifest = build_dataset_manifest(
        samples_path=samples_path,
        root=tmp_path,
        dataset_id="manifest-materialize",
        split="smoke",
    )
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)

    materialized = materialize_requests_from_dataset_manifest(
        manifest_path,
        task_name="manifest_task",
        input_keys=("prompt",),
        output_keys=("generated_video",),
        limit=1,
    )

    assert materialized.schema_version == MATERIALIZED_REQUESTS_SCHEMA_VERSION
    assert materialized.benchmark_name == "manifest-materialize"
    assert materialized.sample_count == 1
    request = materialized.requests[0]
    assert request.sample_id == "sample-a"
    assert request.task_name == "manifest_task"
    assert request.split == "smoke"
    assert request.inputs["prompt"] == "move forward"
    assert request.controls["control_sequence"] == [{"action": "forward"}]
