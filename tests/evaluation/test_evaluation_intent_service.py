from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.tasks.datasets import build_dataset_manifest, write_dataset_manifest
from worldfoundry.evaluation.tasks.execution.orchestration.service import (
    GenerateAndScoreIntent,
    prepare_evaluation,
)


def _dataset_manifest(tmp_path: Path) -> tuple[Path, Path]:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"sample_id": "sample-1", "prompt": "A red cube slides across a table."}) + "\n",
        encoding="utf-8",
    )
    manifest = build_dataset_manifest(samples_path=samples, dataset_id="custom-video-prompts")
    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(manifest, manifest_path)
    return manifest_path, samples


def test_generate_and_score_compiles_valid_dataset_to_existing_evaluate_request(tmp_path: Path) -> None:
    manifest_path, _ = _dataset_manifest(tmp_path)

    prepared = prepare_evaluation(
        GenerateAndScoreIntent(
            output_dir=tmp_path / "output",
            model_id="hailuo-2p3",
            dataset_manifest=manifest_path,
            metrics=("artifact_count",),
        )
    )

    assert prepared.ready is True
    assert prepared.intent_kind == "generate_and_score"
    assert prepared.classification == "custom_dataset_metric_evaluation"
    assert prepared.request is not None
    assert prepared.request.mode == "model"
    assert prepared.request.dataset_id == "custom-video-prompts"
    assert prepared.request.requests[0].inputs["prompt"] == "A red cube slides across a table."
    assert prepared.config_sources["dataset_validation"]["ok"] is True


def test_generate_and_score_rejects_dataset_changed_after_manifest_creation(tmp_path: Path) -> None:
    manifest_path, samples = _dataset_manifest(tmp_path)
    samples.write_text(
        samples.read_text(encoding="utf-8")
        + json.dumps({"sample_id": "sample-2", "prompt": "A blue ball falls."})
        + "\n",
        encoding="utf-8",
    )

    prepared = prepare_evaluation(
        GenerateAndScoreIntent(
            output_dir=tmp_path / "output",
            model_id="hailuo-2p3",
            dataset_manifest=manifest_path,
            metrics=("artifact_count",),
        )
    )

    assert prepared.ready is False
    assert prepared.request is None
    assert {issue.code for issue in prepared.issues} == {"dataset_manifest_invalid"}
    assert prepared.config_sources["dataset_validation"]["ok"] is False
