"""Behavioral checks for the second core ownership/correctness review."""

import numpy as np
import torch


def test_sam_mask_data_keeps_fields_aligned_after_move():
    from worldfoundry.base_models.perception_core.segment.mask_data import MaskData

    data = MaskData(
        scores=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float16),
        boxes=np.array([[0], [1], [2]]),
        labels=["a", "b", "c"],
    )
    data.filter(torch.tensor([True, False, True]))
    added = MaskData(scores=torch.tensor([1.0], dtype=torch.float16), boxes=np.array([[3]]), labels=["d"])
    data.cat(added)
    added["labels"][0] = "changed"
    data.filter(torch.tensor([2, 0]))
    data.to_numpy()
    np.testing.assert_array_equal(data["scores"], [1.0, 0.25])
    assert data["scores"].dtype == np.float32
    np.testing.assert_array_equal(data["boxes"], [[3], [0]])
    assert data["labels"] == ["d", "a"]
