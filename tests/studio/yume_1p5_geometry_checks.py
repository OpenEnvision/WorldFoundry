"""Regression coverage for Yume-1.5 T2V portrait/landscape geometry."""

import pytest
import torch

from worldfoundry.synthesis.visual_generation.yume.yume_runtime.yume_1p5.worldfoundry_runtime import (
    Yume1p5Runtime,
)


@pytest.mark.parametrize("size", [(704, 1280), (1280, 704)])
def test_t2v_passes_requested_orientation_to_sampler(size: tuple[int, int]) -> None:
    """Request geometry is height-first; the Wan sampler needs width-first."""

    class GeometryCaptured(Exception):
        pass

    class RecordingSampler:
        def generate(self, _caption: str, **kwargs: object) -> None:
            assert kwargs["size"] == (size[1], size[0])
            raise GeometryCaptured

    runtime = Yume1p5Runtime(RecordingSampler(), "cpu", torch.bfloat16)
    with pytest.raises(GeometryCaptured):
        runtime.predict_per_interaction(
            prompt="room",
            image=None,
            video=None,
            interaction_idx=0,
            interaction="forward",
            interaction_caption="Move forward.",
            interaction_speed=100.0,
            interaction_distance=4.0,
            task_type="t2v",
            size=size,
            seed=42,
            max_area=size[0] * size[1],
            current_latent_num=8,
            current_frame_num=32,
            num_euler_timesteps=1,
        )
