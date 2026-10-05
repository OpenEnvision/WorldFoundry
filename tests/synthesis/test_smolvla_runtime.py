from __future__ import annotations

from worldfoundry.synthesis.action_generation.smolvla.runtime import (
    SmolVLARuntime,
    SmolVLARuntimeConfig,
)


def _runtime() -> SmolVLARuntime:
    return SmolVLARuntime(
        SmolVLARuntimeConfig(
            checkpoint_location="unused",
            backbone_location="unused",
            device="cpu",
            torch_dtype="float32",
            seed=0,
            compile_model=False,
            denoising_steps=10,
            state_keys=("observation.state", "state"),
            camera_aliases=(
                ("observation.images.image", "image"),
                ("observation.images.image2", "image2"),
            ),
            prompt_suffix="\n",
        )
    )


def test_camera_aliases_cover_flat_and_nested_observation_layouts() -> None:
    runtime = _runtime()
    layouts = (
        {"observation.images.image": "a", "observation.images.image2": "b"},
        {"images": {"image": "c", "image2": "d"}},
        {"observation": {"observation.images.image": "e", "observation.images.image2": "f"}},
        {"observation": {"images": {"image": "g", "image2": "h"}}},
    )

    assert [runtime._images(layout, None) for layout in layouts] == [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
        ["g", "h"],
    ]


def test_state_aliases_cover_flat_and_nested_observation_layouts() -> None:
    runtime = _runtime()

    assert runtime._state({"observation.state": [1] * 8}) == [1] * 8
    assert runtime._state({"observation": {"state": [2] * 8}}) == [2] * 8
