from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.pipelines.cosmos.pipeline_cosmos_predict2 import CosmosPredict2Pipeline


def test_cosmos_predict2_accepts_workspace_dispatch_fields() -> None:
    requests = []
    pipeline = object.__new__(CosmosPredict2Pipeline)
    pipeline.native_pipeline = lambda request: (
        requests.append(request)
        or SimpleNamespace(
            sample=torch.zeros(1, 3, 1, 2, 2),
            latents=None,
            metadata={"backend": "test"},
        )
    )

    result = pipeline(
        prompt="A test scene",
        image=torch.zeros(3, 2, 2),
        output_dir="unused-workspace-directory",
        task_type="image-to-world",
        num_frames=1,
        height=2,
        width=2,
        return_dict=True,
    )

    assert len(requests) == 1
    assert result["generated_video"].shape == (1, 3, 1, 2, 2)
    assert result["metadata"] == {"backend": "test"}


def test_cosmos_predict2_still_rejects_unknown_inference_options() -> None:
    pipeline = object.__new__(CosmosPredict2Pipeline)

    with pytest.raises(TypeError, match="unsupported Cosmos Predict2 inference options"):
        pipeline(
            prompt="A test scene",
            image=torch.zeros(3, 2, 2),
            unsupported_workspace_option=True,
        )
