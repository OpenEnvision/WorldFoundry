from pathlib import Path

import numpy as np
import pytest

from worldfoundry.synthesis.visual_generation.geometry_priors import _load_prompt_depth


def test_prior_depth_requires_a_real_condition() -> None:
    with pytest.raises(ValueError, match="requires prompt_depth_path"):
        _load_prompt_depth(None, (4, 6), device="cpu")


def test_prior_depth_loads_the_provided_condition(tmp_path: Path) -> None:
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    path = tmp_path / "prior.npy"
    np.save(path, source)

    result = _load_prompt_depth(path, source.shape, device="cpu")

    np.testing.assert_array_equal(result.numpy(), source)
