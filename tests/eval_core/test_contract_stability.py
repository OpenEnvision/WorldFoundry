from __future__ import annotations

import json
from pathlib import Path

from worldfoundry.evaluation.api import ArtifactRef, GenerationRequest, GenerationResult
from worldfoundry.cli import main


def test_generation_request_and_result_snapshot_fields_are_stable() -> None:
    request = GenerationRequest(
        sample_id="sample-0001",
        task_name="contract-task",
        split="smoke",
        inputs={"prompt": "move forward"},
        generation_kwargs={"seed": 7},
        output_schema={"artifacts": ["generated_video"]},
    )
    result = GenerationResult(
        sample_id="sample-0001",
        model_id="test-contract-model",
        artifacts={
            "generated_video": ArtifactRef(
                uri="memory://sample-0001/generated_video.json",
                kind="json",
                metadata={"contract": "stable"},
            )
        },
        status="succeeded",
    )

    assert request.to_dict() == {
        "sample_id": "sample-0001",
        "task_name": "contract-task",
        "split": "smoke",
        "request_id": None,
        "inputs": {"prompt": "move forward"},
        "controls": {},
        "generation_kwargs": {"seed": 7},
        "output_schema": {"artifacts": ["generated_video"]},
        "cache_policy": {},
        "schema_version": "worldfoundry-generation-request",
    }
    assert result.to_dict()["schema_version"] == "worldfoundry-generation-result"
    assert result.to_dict()["artifacts"]["generated_video"]["schema_version"] == "worldfoundry-artifact-ref"
    assert result.to_dict()["artifacts"]["generated_video"]["uri"] == "memory://sample-0001/generated_video.json"


def test_contract_run_writes_stable_public_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "contract_run"

    assert main(["contract", "run", "--output-dir", str(output_dir), "--json"]) == 0

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    scorecard = json.loads((output_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "worldfoundry-run-manifest"
    assert manifest["runner"] == "contract_runner"
    assert manifest["status"] == "succeeded"
    assert manifest["model"]["model_type"] == "contract"
    assert manifest["benchmark"]["evaluation_protocol"] == "contract_runner"
    assert scorecard["schema_version"] == "worldfoundry-scorecard"
    assert scorecard["benchmark"]["benchmark_name"] == "contract_cli"
    assert scorecard["model"]["model_id"] == "test-contract-model"
    assert scorecard["eligibility"]["leaderboard_valid"] is False
