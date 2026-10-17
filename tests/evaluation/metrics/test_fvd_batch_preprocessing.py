"""FVD bounds host preprocessing without changing whole-dataset features."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from worldfoundry.evaluation.tasks.metrics.fvd import fvd_core

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_preprocessing_is_bounded_and_preserves_features(batch_size: int) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        videos = np.random.default_rng(7).integers(0, 256, size=(5, 2, 8, 10, 3), dtype=np.uint8)

        def encoder(batch: torch.Tensor) -> torch.Tensor:
            return batch.mean((2, 3, 4))

        expected = encoder(fvd_core.preprocess_videos_uint8(videos))
        preprocess = fvd_core.preprocess_videos_uint8
        with patch.object(fvd_core, "preprocess_videos_uint8", wraps=preprocess) as spy:
            actual = fvd_core.extract_i3d_features(videos, encoder, torch.device("cpu"), batch_size)
        assert [len(call.args[0]) for call in spy.call_args_list] == [
            min(batch_size, 5 - start) for start in range(0, 5, batch_size)
        ]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
