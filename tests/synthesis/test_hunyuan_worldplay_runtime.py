# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _inference_call(namespace: dict[str, Any]) -> Callable[..., Any]:
    # This contract stops before encoding; CUDA-only model imports are covered by runtime smoke.
    source = (
        Path(__file__).resolve().parents[2]
        / "worldfoundry/synthesis/visual_generation/hunyuan_world/hunyuan_worldplay/runtime.py"
    )
    pipeline = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_HunyuanWorldPlayInternalPipeline"
    )
    call = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), call],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace["__call__"]


@pytest.mark.parametrize("rewrite", [None, False, True])
@pytest.mark.parametrize("image_conditioned", [False, True])
def test_optional_prompt_rewrite_keeps_the_raw_prompt(rewrite: bool | None, image_conditioned: bool) -> None:
    warnings: list[str] = []
    prompts: list[str] = []

    class Encoded(Exception):
        pass

    def encode(prompt: str, *args: Any, **kwargs: Any) -> None:
        prompts.append(prompt)
        raise Encoded

    inference_call = _inference_call(
        {
            "torch": torch,
            "np": np,
            "Image": Image,
            "loguru": SimpleNamespace(logger=SimpleNamespace(warning=warnings.append)),
            "dist": SimpleNamespace(is_initialized=lambda: False),
            "get_parallel_state": lambda: SimpleNamespace(sp_enabled=False),
            "get_rank": lambda: 1,
            "auto_offload_model": lambda *args, **kwargs: nullcontext(),
        }
    )
    pipeline = SimpleNamespace(
        ideal_resolution="480p",
        ideal_task=None,
        config=SimpleNamespace(guidance_scale=6.0, embedded_guidance_scale=None, flow_shift=7.0),
        transformer=SimpleNamespace(config=SimpleNamespace(guidance_embed=False)),
        _create_scheduler=lambda shift: object(),
        get_closest_resolution_given_reference_image=lambda *args: (16, 16),
        get_closest_resolution_given_original_size=lambda *args: (16, 16),
        get_latent_size=lambda *args: (1, 1, 1),
        vae_temporal_compression_ratio=4,
        get_task_mask=lambda *args: None,
        execution_device="cpu",
        text_encoder=None,
        enable_offloading=False,
        do_classifier_free_guidance=False,
        clip_skip=None,
        encode_prompt=encode,
    )
    options = {} if rewrite is None else {"prompt_rewrite": rewrite}
    with pytest.raises(Encoded):
        inference_call(
            cast(Any, pipeline),
            prompt="The supplied prompt.",
            aspect_ratio="1:1",
            video_length=1,
            reference_image=Image.new("RGB", (16, 16)) if image_conditioned else None,
            **options,
        )
    assert prompts == ["The supplied prompt."]
    assert warnings == (
        ["HY-WorldPlay prompt rewriting is unavailable in this build; using the raw prompt"] if rewrite else []
    )
    assert inspect.signature(inference_call).parameters["prompt_rewrite"].default is False
