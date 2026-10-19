# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Catalog and profile agreement for implemented model support surfaces."""

import importlib

import pytest

from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "model_id",
    [
        "dreamdojo",
        "dreamdojo-student",
        "last-r1",
        "magi2-preview",
        "minimax-h3",
        "worldlabs-marble-1.1",
        "lingbot-world-v2",
        "lingbot-world-v2-1.3b",
        "kandinsky6-lite",
        "kandinsky6-lite-distill",
        "kandinsky6-lite-pretrain",
        "kandinsky6-pro",
        "kandinsky6-pro-pretrain",
    ],
)
def test_catalog_and_runtime_profile_mark_implemented_integrations(model_id: str) -> None:
    entry = load_model_zoo_registry().get(model_id)
    assert entry.integration_status == "integrated"
    profile = load_runtime_profile(model_id)
    assert profile.integration_status == "integrated"


def test_studio_controls_match_native_model_routes() -> None:
    from worldfoundry.studio.inference.catalog import find_entry

    dreamdojo = find_entry("dreamdojo")
    assert {"vae_path", "text_encoder_path", "text_embeddings_path"} <= set(dreamdojo.load_params)
    assert "dataset_path" not in dreamdojo.load_params
    assert "action_file" in dreamdojo.call_params
    assert dreamdojo.default_call_kwargs["num_inference_steps"] == 35
    magi = find_entry("magi2-preview")
    assert {"use_refiner", "refiner_height", "refiner_width", "refiner_num_inference_steps"} <= set(magi.call_params)


@pytest.mark.parametrize(
    "model_id, export",
    [
        ("irasim", "IRASimPipeline"),
        ("pandora", "PandoraPipeline"),
        ("biwm-wan21", "BiWMPipeline"),
        ("biwm-wan22", "BiWMPipeline"),
    ],
)
def test_retired_models_are_not_public_integrations(model_id: str, export: str) -> None:
    assert model_id not in load_model_zoo_registry().keys()
    assert not hasattr(importlib.import_module("worldfoundry.pipelines.component_pipelines"), export)
