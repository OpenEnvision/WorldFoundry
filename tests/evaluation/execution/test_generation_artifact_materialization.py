# SPDX-FileCopyrightText: 2026 OpenEnvision contributors
# SPDX-License-Identifier: Apache-2.0
"""Registered benchmark layouts consume the standard generation result ledger."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from worldfoundry.core.io.formats.serialization import read_jsonl_objects, write_jsonl
from worldfoundry.evaluation.api import ArtifactRef, GenerationResult
from worldfoundry.evaluation.tasks.execution.framework.runner_common import iter_generation_video_artifacts
from worldfoundry.evaluation.tasks.execution.orchestration.benchmark_generation import (
    get_benchmark_generation_adapter,
)

pytestmark = pytest.mark.unit

_CASES = [
    ("aigcbench", "aigcbench.aigcbench_prompts", "0001.mp4"),
    ("phyeduvideo", "phyeduvideo.phyeduvideo_prompts", "0001.mp4"),
    ("physvidbench", "physvidbench.physvidbench_prompts", "0001.mp4"),
    ("videophy", "videophy.videophy_prompts", "0001.mp4"),
    ("videophy2", "videophy2.videophy2_prompts", "0001.mp4"),
    ("phygenbench", "phygenbench.phygenbench_prompts", "output_video_0001.mp4"),
    ("phyground", "phyground.phyground_prompts", "0001.mp4"),
    ("mirabench", "mirabench.mirabench_prompts", "0001.mp4"),
    ("ewmbench", "ewmbench.ewmbench_prompts", "0001.mp4"),
    ("evalcrafter", "evalcrafter.evalcrafter_prompts", "0001.mp4"),
    ("physics-iq", "physics_iq.physics_iq_prompts", "0001.mp4"),
    ("physics-iq-verified", "physics_iq.physics_iq_prompts", "0001.mp4"),
    ("phyfps-bench-gen", "phyfps_bench_gen.phyfps_prompts", "0001.mp4"),
    ("videoverse", "videoverse.videoverse_prompts", "0001.mp4"),
]


@pytest.mark.parametrize("benchmark_id,module_name,filename", _CASES)
@pytest.mark.parametrize("uri_kind", ["relative", "file"])
def test_registered_layout_uses_successful_artifact_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_id: str,
    module_name: str,
    filename: str,
    uri_kind: str,
) -> None:
    module = importlib.import_module("worldfoundry.evaluation.tasks.execution.runners." + module_name)
    if hasattr(module, "load_prompt_records"):
        monkeypatch.setattr(module, "load_prompt_records", lambda **kwargs: [{"prompt_id": "0001"}])
    generation = tmp_path / "generation"
    generation.mkdir()
    source = generation / "backend-output.mp4"
    source.write_bytes(b"local generated artifact")
    uri = source.as_uri() if uri_kind == "file" else source.name
    rows = [
        GenerationResult(
            sample_id=sample_id, status=status, artifacts={"selected_video": ArtifactRef(uri=uri, kind="video")}
        ).to_dict()
        for sample_id, status in [("0001", "succeeded"), ("0002", "failed"), ("0003", "unknown")]
    ]
    write_jsonl(generation / "results.jsonl", rows)
    adapter = get_benchmark_generation_adapter(benchmark_id)
    assert adapter is not None
    artifacts = tmp_path / "artifacts"
    manifest = tmp_path / "artifacts.jsonl"
    assert adapter.materialize_artifacts(
        generation_output_dir=generation,
        generated_artifact_dir=artifacts,
        artifact_manifest_path=manifest,
        output_artifact="selected_video",
    ) == (1, 0)
    assert sorted(path.name for path in artifacts.iterdir()) == [filename]
    assert (artifacts / filename).read_bytes() == source.read_bytes()
    assert [row["sample_id"] for row in read_jsonl_objects(manifest)] == ["0001"]

    rows[0]["status"] = "failed"
    write_jsonl(generation / "results.jsonl", rows)
    assert adapter.materialize_artifacts(
        generation_output_dir=generation,
        generated_artifact_dir=artifacts,
        artifact_manifest_path=manifest,
        output_artifact="selected_video",
    ) == (0, 0)
    assert list(artifacts.iterdir()) == []
    assert source.read_bytes() == b"local generated artifact"
    assert list(read_jsonl_objects(manifest)) == []


@pytest.mark.parametrize("benchmark_id", ["aigcbench", "phyeduvideo", "physvidbench"])
def test_failed_only_ledger_does_not_fall_back_to_unidentified_video_files(
    tmp_path: Path,
    benchmark_id: str,
) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    source = generation / "orphan.mp4"
    source.write_bytes(b"failed sample artifact")
    write_jsonl(
        generation / "results.jsonl",
        [
            GenerationResult(
                sample_id="0001",
                status="failed",
                artifacts={"generated_video": ArtifactRef(uri=str(source), kind="video")},
            ).to_dict()
        ],
    )
    adapter = get_benchmark_generation_adapter(benchmark_id)
    assert adapter is not None
    assert adapter.materialize_artifacts(
        generation_output_dir=generation,
        generated_artifact_dir=tmp_path / "artifacts",
        artifact_manifest_path=tmp_path / "artifacts.jsonl",
        output_artifact="generated_video",
    ) == (0, 0)
    assert not list((tmp_path / "artifacts").iterdir())


def test_video_artifact_reader_retains_missing_artifacts_for_manifest_diagnostics(tmp_path: Path) -> None:
    rows = [
        GenerationResult(
            sample_id="missing",
            status="succeeded",
            artifacts={"generated_video": ArtifactRef(uri="absent.mp4", kind="video")},
        ).to_dict(),
        GenerationResult(
            sample_id="remote",
            status="succeeded",
            artifacts={"generated_video": ArtifactRef(uri="s3://bucket/video.mp4", kind="video")},
        ).to_dict(),
        GenerationResult(
            sample_id="failed",
            status="failed",
            artifacts={"generated_video": ArtifactRef(uri="absent.mp4", kind="video")},
        ).to_dict(),
    ]
    write_jsonl(tmp_path / "results.jsonl", rows)
    results = list(iter_generation_video_artifacts(tmp_path))
    assert [(result.sample_id, path) for result, _, path in results] == [
        ("missing", tmp_path / "absent.mp4"),
        ("remote", None),
    ]
    assert results[0][2] is not None and not results[0][2].is_file()
