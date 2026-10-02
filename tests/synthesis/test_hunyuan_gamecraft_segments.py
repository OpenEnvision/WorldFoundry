from __future__ import annotations

import torch

from worldfoundry.synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.runtime import (
    HunyuanGameCraftRuntime,
)


def test_gamecraft_deduplicates_autoregressive_action_boundaries() -> None:
    runtime = HunyuanGameCraftRuntime.__new__(HunyuanGameCraftRuntime)
    segment_index = 0

    def predict_per_action(**_kwargs):
        nonlocal segment_index
        sample = torch.full((1, 3, 33, 2, 2), float(segment_index) / 4.0)
        segment_index += 1
        return {
            "samples": [sample],
            "last_latents": None,
            "ref_latents": None,
        }

    runtime.predict_per_action = predict_per_action
    frames = runtime.predict(
        ref_images=None,
        last_latents=None,
        ref_latents=None,
        action_list=["w", "s", "d", "a"],
        action_speed_list=[0.2] * 4,
        prompt="village",
        negative_prompt="",
        size=(2, 2),
        video_length=129,
        guidance_scale=2.0,
        infer_steps=50,
        flow_shift=5.0,
    )

    assert len(frames) == 129
    assert [int(frames[index][0, 0, 0]) for index in (0, 32, 33, 64, 65, 96, 97, 128)] == [
        0,
        0,
        63,
        63,
        127,
        127,
        191,
        191,
    ]
