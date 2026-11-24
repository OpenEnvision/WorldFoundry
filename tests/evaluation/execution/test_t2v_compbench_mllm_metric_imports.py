# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""The T2V-CompBench MLLM metric scripts import with the canonical LLaVA helpers."""

from __future__ import annotations

import importlib

import pytest

from worldfoundry.base_models.llm_mllm_core.mllm.llava_next.llava import constants, conversation, mm_utils

pytestmark = pytest.mark.unit

_PACKAGE = "worldfoundry.evals.tasks.execution.runners.t2v_compbench.runtime.t2v_compbench.mllm_metrics"


@pytest.mark.parametrize(
    "script",
    [
        "compbench_eval_action_binding",
        "compbench_eval_consistent_attr",
        "compbench_eval_dynamic_attr",
        "compbench_eval_interaction",
    ],
)
def test_metric_script_binds_the_canonical_llava_helpers(script: str) -> None:
    metric = importlib.import_module(f"{_PACKAGE}.{script}")

    assert metric.IMAGE_TOKEN_INDEX == constants.IMAGE_TOKEN_INDEX
    assert metric.DEFAULT_IMAGE_TOKEN == constants.DEFAULT_IMAGE_TOKEN
    assert metric.conv_templates is conversation.conv_templates
    assert metric.process_images is mm_utils.process_images
    assert metric.tokenizer_image_token is mm_utils.tokenizer_image_token
