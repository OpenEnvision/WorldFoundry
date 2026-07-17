from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from worldfoundry.core.io.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationResult
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)
from worldfoundry.evaluation.tasks.execution.runners.t2v_compbench import (
    run_t2v_compbench_official_runner as runner_module,
)
from worldfoundry.evaluation.tasks.execution.runners.t2v_compbench.run_t2v_compbench_official_runner import (
    build_metric_rows,
    normalize_t2v_compbench_results,
)
from worldfoundry.evaluation.tasks.execution.runners.t2v_compbench.t2v_compbench_prompts import (
    CANONICAL_PROMPT_COUNT,
    CATEGORY_ORDER,
    GENERATION_MANIFEST_NAME,
    copy_t2v_compbench_generated_videos,
    materialize_t2v_compbench_generation_requests,
    materialize_t2v_compbench_official_layout,
    validate_t2v_compbench_generation_manifest,
)


def _write_video(path, *, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 32))
    assert writer.isOpened()
    for frame_index in range(frame_count):
        writer.write(np.full((32, 32, 3), frame_index * 20, dtype=np.uint8))
    writer.release()


def _success_result(request, video_path) -> GenerationResult:
    return GenerationResult(
        sample_id=request.sample_id,
        request_id=request.request_id,
        model_id="model-under-test",
        artifacts={"generated_video": ArtifactRef(uri=str(video_path), kind="video")},
        status="succeeded",
    )


def _write_generation_manifests(generation_dir, requests, results) -> None:
    generation_dir.mkdir()
    write_jsonl(generation_dir / "requests.jsonl", [request.to_dict() for request in requests])
    write_jsonl(generation_dir / "results.jsonl", [result.to_dict() for result in results])


def test_generation_adapter_materializes_the_exact_official_prompt_suite() -> None:
    adapter = get_benchmark_generation_adapter("t2v-compbench")
    assert adapter is not None
    requests = adapter.materialize_requests(limit=None)

    assert len(requests) == CANONICAL_PROMPT_COUNT
    assert requests[0].sample_id == "t2v-compbench-consistent_attribute_binding-0001"
    assert requests[0].inputs["category_id"] == CATEGORY_ORDER[0]
    assert requests[0].inputs["prompt_id"] == "consistent_attribute_binding-0001"
    assert requests[0].inputs["official_video_name"] == "0001.mp4"
    assert requests[199].inputs["official_video_name"] == "0200.mp4"
    assert requests[200].inputs["category_id"] == CATEGORY_ORDER[1]
    assert requests[-1].sample_id == "t2v-compbench-generative_numeracy-0200"


def test_generated_results_are_strictly_joined_and_staged_by_manifest(tmp_path) -> None:
    requests = materialize_t2v_compbench_generation_requests(limit=2)
    first_video = tmp_path / "model-output-a.mp4"
    second_video = tmp_path / "model-output-b.mp4"
    _write_video(first_video)
    _write_video(second_video)
    generation_dir = tmp_path / "generation"
    _write_generation_manifests(
        generation_dir,
        requests,
        [_success_result(requests[0], first_video), _success_result(requests[1], second_video)],
    )
    generated = tmp_path / "generated"
    artifact_manifest = tmp_path / "generated_artifacts.jsonl"

    count, placeholders = copy_t2v_compbench_generated_videos(
        generation_output_dir=generation_dir,
        generated_artifact_dir=generated,
        artifact_manifest_path=artifact_manifest,
    )

    assert (count, placeholders) == (2, 0)
    assert (generated / "consistent_attr" / "0001.mp4").is_file()
    assert (generated / "consistent_attr" / "0002.mp4").is_file()
    manifest = read_jsonl_objects(generated / GENERATION_MANIFEST_NAME)
    assert manifest == read_jsonl_objects(artifact_manifest)
    assert [row["sample_id"] for row in manifest] == [request.sample_id for request in requests]

    # Prove the runner bridge uses IDs rather than assuming the generated file
    # is already named or organized like the official repository.
    arbitrary = generated / "arbitrary-model-layout" / "clip.mp4"
    arbitrary.parent.mkdir()
    shutil.move(generated / "consistent_attr" / "0001.mp4", arbitrary)
    (generated / "consistent_attr" / "0002.mp4").unlink()
    (generated / "consistent_attr").rmdir()
    manifest = manifest[:1]
    manifest[0]["relative_path"] = "arbitrary-model-layout/clip.mp4"
    write_jsonl(generated / GENERATION_MANIFEST_NAME, manifest)

    report = materialize_t2v_compbench_official_layout(
        generated_video_dir=generated,
        official_layout_dir=tmp_path / "official-layout",
        selected_categories=["consistent_attribute_binding"],
    )
    assert report["sample_count"] == 1
    assert report["bounded"] is True
    assert report["full_suite"] is False
    assert (tmp_path / "official-layout" / "consistent_attr" / "0001.mp4").is_file()
    rerun = materialize_t2v_compbench_official_layout(
        generated_video_dir=generated,
        official_layout_dir=tmp_path / "official-layout",
        selected_categories=["consistent_attribute_binding"],
    )
    assert rerun["sample_count"] == 1


def test_bridge_fails_closed_for_missing_results_bad_video_and_unmapped_video(tmp_path) -> None:
    requests = materialize_t2v_compbench_generation_requests(limit=2)
    valid_video = tmp_path / "valid.mp4"
    _write_video(valid_video)
    incomplete = tmp_path / "incomplete"
    _write_generation_manifests(incomplete, requests, [_success_result(requests[0], valid_video)])
    with pytest.raises(ValueError, match="coverage mismatch"):
        copy_t2v_compbench_generated_videos(
            generation_output_dir=incomplete,
            generated_artifact_dir=tmp_path / "unused-a",
            artifact_manifest_path=tmp_path / "unused-a.jsonl",
        )

    broken_video = tmp_path / "broken.mp4"
    broken_video.write_bytes(b"not a video")
    broken = tmp_path / "broken-generation"
    _write_generation_manifests(
        broken,
        requests[:1],
        [_success_result(requests[0], broken_video)],
    )
    with pytest.raises(ValueError, match="not decodable"):
        copy_t2v_compbench_generated_videos(
            generation_output_dir=broken,
            generated_artifact_dir=tmp_path / "unused-b",
            artifact_manifest_path=tmp_path / "unused-b.jsonl",
        )

    complete = tmp_path / "complete"
    _write_generation_manifests(
        complete,
        requests[:1],
        [_success_result(requests[0], valid_video)],
    )
    generated = tmp_path / "generated"
    copy_t2v_compbench_generated_videos(
        generation_output_dir=complete,
        generated_artifact_dir=generated,
        artifact_manifest_path=tmp_path / "generated.jsonl",
    )
    _write_video(generated / "unmapped.mp4")
    with pytest.raises(ValueError, match="no manifest mapping"):
        validate_t2v_compbench_generation_manifest(generated_video_dir=generated)


def test_bounded_component_is_evidence_but_not_a_full_benchmark_claim(tmp_path) -> None:
    video_root = tmp_path / "official"
    _write_video(video_root / "consistent_attr" / "0001.mp4")
    results = [
        {
            "metric_id": "consistent_attribute_binding",
            "raw_score": 0.75,
            "sample_count": 1,
            "source": "official_csv_final_score",
            "source_csv": None,
            "per_sample_rows": [{"sample_id": "0001", "raw_score": 0.75}],
        }
    ]

    metric_rows, _, leaderboard = build_metric_rows(results)
    assert "t2v_compbench_average" not in leaderboard
    assert next(row for row in metric_rows if row["metric_id"] == "t2v_compbench_average")[
        "available"
    ] is False

    scorecard = normalize_t2v_compbench_results(
        results,
        benchmark_id="t2v-compbench",
        output_dir=tmp_path / "scorecard",
        source_dir=None,
        dataset_root=None,
        video_root=video_root,
        upstream_results_path=None,
        command=[["official-evaluator"]],
        duration_seconds=0.1,
        returncode=0,
        stdout_path=None,
        stderr_path=None,
        requested_categories=["consistent_attribute_binding"],
    )
    assert scorecard["official_component_verified"] is True
    assert scorecard["integration_evidence"] is True
    assert scorecard["official_benchmark_verified"] is False
    assert scorecard["eligibility"]["leaderboard_valid"] is False
    assert scorecard["evaluation"]["scope"] == "bounded"


def test_official_execution_path_passes_only_runner_arguments(monkeypatch, tmp_path) -> None:
    video_root = tmp_path / "official"
    _write_video(video_root / "consistent_attr" / "0001.mp4")
    args = runner_module.build_parser().parse_args(
        [
            "--video-root",
            str(video_root),
            "--category",
            "consistent_attribute_binding",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    runner_module.normalize_path_args(args)
    monkeypatch.setattr(runner_module, "build_official_commands", lambda _args, _root: [])

    def fake_run_command_sequence(commands, *, root, timeout, stdout_path, stderr_path):
        assert commands == []
        assert root == args.t2v_compbench_root
        assert timeout == args.timeout
        assert stdout_path.name == "upstream_stdout.log"
        assert stderr_path.name == "upstream_stderr.log"
        return [], 0.0, 1

    monkeypatch.setattr(runner_module, "run_command_sequence", fake_run_command_sequence)
    scorecard = runner_module.run_t2v_compbench(args)
    assert scorecard["run"]["returncode"] == 1
    assert scorecard["integration_evidence"] is False
