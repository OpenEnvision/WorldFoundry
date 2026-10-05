from __future__ import annotations

import pytest

from worldfoundry.synthesis.visual_generation.stable_video_infinity.worldfoundry_runtime import (
    normalize_svi_lora_state_dict,
)


def test_svi_lora_training_prefix_is_removed() -> None:
    marker_a = object()
    marker_b = object()

    normalized = normalize_svi_lora_state_dict(
        {
            "pipe.dit.blocks.0.self_attn.q.lora_A.default.weight": marker_a,
            "pipe.dit.blocks.0.self_attn.q.lora_B.default.weight": marker_b,
        }
    )

    assert normalized == {
        "blocks.0.self_attn.q.lora_A.default.weight": marker_a,
        "blocks.0.self_attn.q.lora_B.default.weight": marker_b,
    }


def test_svi_lora_prefix_normalization_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="key collision"):
        normalize_svi_lora_state_dict(
            {
                "pipe.dit.blocks.0.self_attn.q.lora_A.weight": object(),
                "blocks.0.self_attn.q.lora_A.weight": object(),
            }
        )
