from __future__ import annotations

from worldfoundry.evals.api import ArtifactRef, GenerationRequest, GenerationResult


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
