from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import numpy as np
import torch

from worldfoundry.base_models.perception_core.tracking.cotracker.visualizer import (
    Visualizer,
    _cool,
    _normalize,
    _rainbow,
)


def test_cotracker_color_maps_do_not_require_matplotlib():
    assert _normalize(5, 0, 10) == 0.5
    assert len(_rainbow(0.5)) == 4
    assert _cool(0.25) == (0.25, 0.75, 1.0, 1.0)


def test_official_cotracker_visualizer_renders_without_saving():
    video = torch.zeros((1, 3, 3, 32, 48), dtype=torch.float32)
    tracks = torch.tensor([[[[5, 8]], [[18, 12]], [[30, 20]]]], dtype=torch.float32)
    visibility = torch.ones((1, 3, 1), dtype=torch.bool)

    rendered = Visualizer(tracks_leave_trace=2).visualize(
        video, tracks, visibility, save_video=False
    )

    assert rendered.shape == (1, 12, 3, 32, 48)
    assert np.asarray(rendered).std() > 0
