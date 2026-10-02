from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from worldfoundry.base_models.diffusion_model.recipes.sana_variants import SANA_VARIANTS
from worldfoundry.studio.inference.catalog import (
    SANA_DEFAULT_IMAGE_VARIANT_ID,
    STUDIO_HIDDEN_CATALOG_MODEL_IDS,
    _sana_folded_studio_model_ids,
    find_entry,
)
from worldfoundry.studio.serving.workspace import _model_payload


def test_folded_sana_checkpoints_are_hidden_from_studio_rows() -> None:
    folded = _sana_folded_studio_model_ids()

    assert folded
    assert folded <= STUDIO_HIDDEN_CATALOG_MODEL_IDS
    assert all(SANA_VARIANTS[model_id].runner in {"image", "sprint", "controlnet"} for model_id in folded)
    assert "sana-video-2b-480p" not in folded
    assert "sana-streaming-2b-720p" not in folded


def test_sana_card_exposes_image_checkpoints_as_variants() -> None:
    entry = find_entry("sana")
    payload = _model_payload(entry)
    variant_ids = [variant["variant_id"] for variant in payload["variants"]]
    folded = _sana_folded_studio_model_ids()

    assert payload["id"] == "sana"
    assert payload["workload_type"] == "image"
    assert payload["default_variant_id"] == SANA_DEFAULT_IMAGE_VARIANT_ID
    assert set(variant_ids) == folded
    assert "default" not in variant_ids
    default = next(variant for variant in payload["variants"] if variant["variant_id"] == SANA_DEFAULT_IMAGE_VARIANT_ID)
    assert default["load_kwargs"]["model_id"] == SANA_DEFAULT_IMAGE_VARIANT_ID
    assert default["call_kwargs"]["height"] == 1024


@pytest.mark.parametrize("model_id", sorted(_sana_folded_studio_model_ids()))
def test_folded_sana_checkpoint_ids_resolve_to_the_shared_card(model_id: str) -> None:
    entry = find_entry(model_id)

    assert entry.model_id == "sana"
    assert model_id in entry.aliases
