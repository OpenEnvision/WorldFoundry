from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from worldfoundry.studio.inference.catalog import (
    COGVIDEOX_DEFAULT_VARIANT_ID,
    COGVIDEOX_STUDIO_PARENT_ID,
    STUDIO_HIDDEN_CATALOG_MODEL_IDS,
    _cogvideox_folded_studio_model_ids,
    cogvideox_runtime_model_id,
    find_entry,
    find_runtime_entry,
)
from worldfoundry.studio.serving.workspace import _model_payload


def test_folded_cogvideox_pipelines_are_hidden_from_studio_rows() -> None:
    folded = _cogvideox_folded_studio_model_ids()

    assert {"cogvideox_2b_t2v", "cogvideox_5b_i2v", "cogvideox-2b-t2v", "cogvideox-5b-i2v"} <= folded
    assert folded <= STUDIO_HIDDEN_CATALOG_MODEL_IDS
    assert COGVIDEOX_STUDIO_PARENT_ID not in folded


def test_cogvideox_card_exposes_t2v_and_i2v_as_variants() -> None:
    entry = find_entry("cogvideox")
    payload = _model_payload(entry)
    variant_ids = [variant["variant_id"] for variant in payload["variants"]]
    by_id = {variant["variant_id"]: variant for variant in payload["variants"]}

    assert payload["id"] == COGVIDEOX_STUDIO_PARENT_ID
    assert payload["name"] == "CogVideoX"
    assert payload["default_variant_id"] == COGVIDEOX_DEFAULT_VARIANT_ID
    assert set(variant_ids) == {"cogvideox_2b_t2v", "cogvideox_5b_t2v", "cogvideox_5b_i2v"}
    assert by_id["cogvideox_2b_t2v"]["workload_type"] == "t2v"
    assert by_id["cogvideox_5b_i2v"]["workload_type"] == "i2v"
    assert by_id["cogvideox_5b_i2v"]["default_input_path"]


@pytest.mark.parametrize(
    "model_id",
    (
        "cogvideox",
        "cogvideox_2b_t2v",
        "cogvideox-2b-t2v",
        "cogvideox_5b_i2v",
        "cogvideox-5b-i2v",
        "cogvideox_5b_t2v",
    ),
)
def test_folded_cogvideox_ids_resolve_to_the_shared_card(model_id: str) -> None:
    entry = find_entry(model_id)

    assert entry.model_id == COGVIDEOX_STUDIO_PARENT_ID
    assert entry.display_name == "CogVideoX"


def test_cogvideox_runtime_entry_keeps_variant_pipelines() -> None:
    two_b = find_runtime_entry("cogvideox_2b_t2v")
    i2v = find_runtime_entry("cogvideox-5b-i2v")

    assert two_b.model_id == "cogvideox_2b_t2v"
    assert "pipeline_cogvideox_2b_t2v" in two_b.module_path
    assert i2v.model_id == "cogvideox_5b_i2v"
    assert "pipeline_cogvideox_5b_i2v" in i2v.module_path
    assert cogvideox_runtime_model_id("cogvideox", "cogvideox_2b_t2v") == "cogvideox_2b_t2v"
    assert cogvideox_runtime_model_id("sana", "cogvideox_2b_t2v") is None
