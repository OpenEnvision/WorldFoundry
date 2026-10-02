"""Optional, short real-weight checks for state leaking between inference requests."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from worldfoundry.pipelines.pi3.pipeline_pi3 import Pi3Pipeline


def _numerical_result(result):
    arrays = {name: value.copy() for name, value in result.numpy_data.items() if isinstance(value, np.ndarray)}
    assert {"points", "masks", "camera_poses", "depth_map"} <= arrays.keys()
    assert arrays["masks"].dtype == np.bool_
    assert all(value.size and np.isfinite(value).all() for value in arrays.values())
    return arrays


def _assert_same_result(result, expected):
    arrays = _numerical_result(result)
    assert arrays.keys() == expected.keys()
    for key in arrays:
        assert arrays[key].dtype == expected[key].dtype
        np.testing.assert_array_equal(arrays[key], expected[key], err_msg=key)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pi3x_reused_pipeline_matches_fresh_model_after_different_and_failed_requests(monkeypatch):
    checkpoint_root = os.environ.get("WORLDFOUNDRY_REGRESSION_CHECKPOINT_ROOT")
    input_root = os.environ.get("WORLDFOUNDRY_REGRESSION_INPUT_ROOT")
    if not checkpoint_root or not input_root:
        pytest.skip("set regression checkpoint and multiview input roots for real Pi3X inference")
    weights = Path(checkpoint_root) / "yyfz233--Pi3X"
    images = [str(Path(input_root) / f"frame-{index}.png") for index in [0, 16, 32]]
    assert weights.is_dir(), f"Configured Pi3X checkpoint is missing: {weights}"
    assert all(Path(path).is_file() for path in images), "Configured multiview frames are missing"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pipeline = fresh = None
    try:
        pipeline = Pi3Pipeline.from_pretrained(model_path=str(weights), mode="pi3x", device="cuda:0")
        torch.manual_seed(42)
        first = pipeline(images=images)
        first_values = _numerical_result(first)
        changed_images = [images[2], images[0]]
        torch.manual_seed(42)
        changed = pipeline(images=changed_images)
        changed_values = _numerical_result(changed)
        assert changed_values["depth_map"].shape[1] == 2
        with pytest.raises(ValueError, match="images is required"):
            pipeline(images=[])
        torch.manual_seed(42)
        repeated = pipeline(images=images)
        _assert_same_result(repeated, first_values)
        _assert_same_result(first, first_values)
        _assert_same_result(changed, changed_values)
        fresh = Pi3Pipeline.from_pretrained(model_path=str(weights), mode="pi3x", device="cuda:0")
        torch.manual_seed(42)
        isolated = fresh(images=changed_images)
        _assert_same_result(isolated, changed_values)
    finally:
        del pipeline, fresh
        gc.collect()
        torch.cuda.empty_cache()
