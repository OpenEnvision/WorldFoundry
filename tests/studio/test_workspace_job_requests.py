"""Shared inference requests used by the CLI and Workspace."""

import pytest

from worldfoundry.studio.inference.requests import (
    InferenceRequestError,
    JobCreateRequest,
    build_inference_run_kwargs,
)
from worldfoundry.studio.workspace_job import _build_infer_payload, build_parser


def test_cli_parameters_build_an_inference_request() -> None:
    args = build_parser().parse_args(
        [
            "infer",
            "--model-id",
            "zeroscope",
            "--guidance-scale",
            "1.5",
            "--frames",
            "8",
            "--size",
            "640x360",
            "--device",
            "cpu",
        ]
    )

    payload = _build_infer_payload(args)

    assert payload.device == "cpu"
    assert payload.params["num_frames"] == 8
    assert payload.params["width"] == 640
    assert payload.params["height"] == 360
    assert payload.call_kwargs["guidance_scale"] == 1.5


def test_shared_request_accepts_workspace_settings() -> None:
    _, kwargs = build_inference_run_kwargs(
        JobCreateRequest(model_id="zeroscope"),
        settings={"device": "cpu", "fps": 8, "num_frames": 8},
    )

    assert kwargs["device"] == "cpu"


def test_shared_request_rejects_unsupported_backend() -> None:
    with pytest.raises(InferenceRequestError, match="does not support backend=api_init"):
        build_inference_run_kwargs(JobCreateRequest(model_id="zeroscope", backend="api_init"))
