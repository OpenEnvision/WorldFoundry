# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Studio attention selections respect the declared load and inference boundaries."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.inference.requests import (
    InferenceRequestError,
    JobCreateRequest,
    _merge_common_params,
    build_inference_run_kwargs,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("attention_key", ["attention_backend", "attention-backend"])
def test_native_wan_attention_selection_reaches_generation(attention_key: str) -> None:
    import torch

    from worldfoundry.base_models.diffusion_model.contracts import DiffusionOutput, DiffusionRequest
    from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline

    requests: list[DiffusionRequest] = []

    class WeightFreeNativePipeline:
        model_id = "wan2.1-t2v-1.3b"

        def __call__(self, request: DiffusionRequest) -> DiffusionOutput:
            requests.append(request)
            sample = torch.zeros(1, 3, 1, 8, 8)
            return DiffusionOutput(sample=sample, latents=sample)

    _, run_kwargs = build_inference_run_kwargs(
        JobCreateRequest(
            model_id="wan2.1-t2v-1.3b",
            prompt="A boat on a lake",
            params={attention_key: "sdpa"},
            model_ref="/tmp/unused-wan-checkpoint",
        )
    )
    load_kwargs = json.loads(run_kwargs["load_kwargs_text"])
    call_kwargs = json.loads(run_kwargs["call_kwargs_text"])
    pipeline = Wan2p1T2VPipeline(native_pipeline=WeightFreeNativePipeline(), device="cpu")
    result = pipeline(prompt=run_kwargs["prompt"], return_dict=True, **call_kwargs)

    assert len(requests) == 1
    assert result["sample"].shape == (1, 3, 1, 8, 8)
    assert load_kwargs["attention_backend"] == "sdpa"
    assert "attention_backend" not in call_kwargs


@pytest.mark.parametrize("attention_key", ["attention_backend", "attention-backend"])
@pytest.mark.parametrize(
    ("load", "call", "stream"),
    [(True, False, False), (False, True, False), (False, False, True), (True, True, False)],
    ids=["load-only", "call-only", "stream-only", "both-boundaries"],
)
def test_attention_selection_routes_only_to_declared_boundaries(
    attention_key: str, load: bool, call: bool, stream: bool
) -> None:
    entry = replace(
        find_entry("wan2.1-t2v-1.3b"),
        load_params=("attention_backend",) if load else (),
        call_params=("attention_backend",) if call else (),
        stream_params=("attention_backend",) if stream else (),
        supports_stream=stream,
    )
    call_kwargs, load_kwargs = _merge_common_params(
        entry, JobCreateRequest(model_id=entry.model_id, params={attention_key: "sdpa"})
    )

    assert load_kwargs == ({"attention_backend": "sdpa"} if load else {})
    assert call_kwargs == ({"attention_backend": "sdpa"} if call or stream else {})


@pytest.mark.parametrize("attention_key", ["attention_backend", "attention-backend"])
def test_explicit_attention_json_overrides_common_selection_independently(attention_key: str) -> None:
    entry = replace(
        find_entry("wan2.1-t2v-1.3b"),
        load_params=("attention_backend",),
        call_params=("attention_backend",),
        stream_params=(),
    )
    call_kwargs, load_kwargs = _merge_common_params(
        entry,
        JobCreateRequest(
            model_id=entry.model_id,
            params={attention_key: "sdpa"},
            load_kwargs={attention_key: "torch"},
            call_kwargs={attention_key: "flash_attn_2"},
        ),
    )

    assert load_kwargs == {"attention_backend": "torch"}
    assert call_kwargs == {"attention_backend": "flash_attn_2"}


def test_native_wan_explicit_load_attention_json_overrides_common_selection() -> None:
    _, run_kwargs = build_inference_run_kwargs(
        JobCreateRequest(
            model_id="wan2.1-t2v-1.3b",
            prompt="A boat on a lake",
            params={"attention_backend": "sdpa"},
            load_kwargs={"attention-backend": "torch"},
            model_ref="/tmp/unused-wan-checkpoint",
        )
    )

    assert json.loads(run_kwargs["load_kwargs_text"])["attention_backend"] == "torch"
    assert "attention_backend" not in json.loads(run_kwargs["call_kwargs_text"])


def test_native_wan_explicit_unsupported_call_attention_json_is_rejected() -> None:
    with pytest.raises(InferenceRequestError, match="call_kwargs=attention_backend"):
        build_inference_run_kwargs(
            JobCreateRequest(
                model_id="wan2.1-t2v-1.3b",
                prompt="A boat on a lake",
                params={"attention_backend": "sdpa"},
                call_kwargs={"attention_backend": "sdpa"},
                model_ref="/tmp/unused-wan-checkpoint",
            )
        )


@pytest.mark.parametrize("selection", [None, "", "auto"])
def test_native_wan_automatic_attention_respects_load_boundary(selection: str | None) -> None:
    call_kwargs, load_kwargs = _merge_common_params(
        find_entry("wan2.1-t2v-1.3b"),
        JobCreateRequest(model_id="wan2.1-t2v-1.3b", params={"attention_backend": selection}),
    )

    assert load_kwargs == ({"attention_backend": "auto"} if selection == "auto" else {})
    assert "attention_backend" not in call_kwargs
