from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "transformers" dependency at import time; skip when it is unavailable.
pytest.importorskip("transformers")

import torch

from worldfoundry.base_models.perception_core.detection.grounding_dino.models.GroundingDINO.bertwarper import (
    _prepare_head_mask,
)


def test_transformers_five_head_mask_compatibility():
    assert _prepare_head_mask(None, 3) == [None, None, None]
    one_dimensional = _prepare_head_mask(torch.ones(4), 3)
    two_dimensional = _prepare_head_mask(torch.ones(3, 4), 3)
    assert one_dimensional.shape == (3, 1, 4, 1, 1)
    assert two_dimensional.shape == (3, 1, 4, 1, 1)
