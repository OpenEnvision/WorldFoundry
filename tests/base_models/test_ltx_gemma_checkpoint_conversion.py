from __future__ import annotations

from worldfoundry.base_models.diffusion_model.models.encoders.ltx.component import (
    convert_ltx_gemma_state_dict,
)


def test_ltx_gemma_converter_removes_qat_vision_model_wrapper() -> None:
    patch_weight = object()
    language_weight = object()
    projector_weight = object()

    converted = convert_ltx_gemma_state_dict(
        {
            "vision_tower.vision_model.embeddings.patch_embedding.weight": patch_weight,
            "language_model.model.embed_tokens.weight": language_weight,
            "multi_modal_projector.mm_input_projection_weight": projector_weight,
        }
    )

    assert (
        converted[
            "model.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        ]
        is patch_weight
    )
    assert "model.model.vision_tower.embeddings.patch_embedding.weight" not in converted
    assert converted["model.model.language_model.embed_tokens.weight"] is language_weight
    assert converted["model.lm_head.weight"] is language_weight
    assert converted["model.model.multi_modal_projector.mm_input_projection_weight"] is projector_weight


def test_ltx_gemma_converter_preserves_unwrapped_vision_keys() -> None:
    weight = object()

    converted = convert_ltx_gemma_state_dict(
        {"vision_tower.embeddings.position_embedding.weight": weight}
    )

    assert (
        converted[
            "model.model.vision_tower.vision_model.embeddings.position_embedding.weight"
        ]
        is weight
    )
