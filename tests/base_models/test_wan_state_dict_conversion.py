from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers.wan import (
    convert_diffusers_wan_transformer_state_dict,
)


def test_diffusers_converter_accepts_official_native_wan_shard_keys() -> None:
    state_dict = {
        "patch_embedding.weight": torch.ones(1),
        "text_embedding.0.bias": torch.ones(1),
        "time_embedding.2.weight": torch.ones(1),
        "time_projection.1.weight": torch.ones(1),
        "img_emb.proj.0.bias": torch.ones(1),
        "head.head.weight": torch.ones(1),
        "blocks.0.self_attn.q.weight": torch.ones(1),
    }

    converted = convert_diffusers_wan_transformer_state_dict(state_dict)

    assert set(converted) == set(state_dict)


def test_diffusers_converter_still_maps_diffusers_names() -> None:
    state_dict = {
        "condition_embedder.text_embedder.linear_1.bias": torch.ones(1),
        "blocks.0.attn1.to_q.weight": torch.ones(1),
        "blocks.0.attn1.to_gate_compress.weight": torch.ones(2, 2),
        "blocks.0.to_gate_compress.bias": torch.ones(2),
    }

    converted = convert_diffusers_wan_transformer_state_dict(state_dict)

    assert set(converted) == {
        "text_embedding.0.bias",
        "blocks.0.self_attn.q.weight",
        "blocks.0.self_attn.gate_compress.weight",
        "blocks.0.self_attn.gate_compress.bias",
    }
