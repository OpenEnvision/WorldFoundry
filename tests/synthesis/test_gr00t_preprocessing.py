# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared state/action transforms preserve family metadata and inversion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import ComposedModalityTransform
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.concat import ConcatTransform
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.schema import StateActionMetadata
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.state_action import (
    Normalizer,
    StateActionToTensor,
    StateActionTransform,
)

pytestmark = pytest.mark.unit


def test_being_data_config_builds_shared_inference_transforms() -> None:
    from worldfoundry.synthesis.action_generation.being_h05.preprocessing import data_config

    assert data_config.StateActionToTensor is StateActionToTensor
    assert data_config.StateActionTransform is StateActionTransform
    assert data_config.ComposedModalityTransform is ComposedModalityTransform


def test_dreamzero_tensor_transform_remains_invertible_in_composed_chain() -> None:
    from worldfoundry.synthesis.action_generation.dreamzero import preprocessing

    assert preprocessing.StateActionToTensor is StateActionToTensor
    transform = StateActionToTensor(
        apply_to=["action.x", "action.missing"],
        output_dtypes={"action.x": "torch.float32"},
        input_dtypes={"action.x": "np.float64"},
    )
    chain = preprocessing.ComposedModalityTransform(transforms=[transform])
    assert chain.eval() is chain
    assert transform.training is False
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = chain.apply({"action.x": data.copy()})
    assert result["action.x"].dtype == torch.float32
    restored = chain.unapply(result)
    np.testing.assert_array_equal(restored["action.x"], data)
    assert restored["action.x"].dtype == np.float64
    assert transform.model_dump(mode="json") == {"apply_to": ["action.x", "action.missing"]}


@pytest.mark.parametrize("mode", ["mean_std", "min_max", "q99"])
def test_shared_normalizer_preserves_exact_round_trip(mode: str) -> None:
    stats = {"mean": [0.0], "std": [2.0], "min": [-2.0], "max": [2.0], "q01": [-2.0], "q99": [2.0]}
    normalizer = Normalizer(mode, stats)
    values = torch.tensor([[-2.0], [0.0], [2.0]])
    normalized = normalizer.forward(values)
    torch.testing.assert_close(normalized, values / 2, atol=0, rtol=0)
    torch.testing.assert_close(normalizer.inverse(normalized), values, atol=0, rtol=0)


def test_concat_preserves_action_order_and_metadata_class_identity() -> None:
    transform = ConcatTransform(video_concat_order=[], action_concat_order=["action.b", "action.a"])
    descriptor = StateActionMetadata(absolute=True, shape=(2,), continuous=True)
    metadata = SimpleNamespace(modalities=SimpleNamespace(action={"a": descriptor, "b": descriptor}))
    transform.set_metadata(metadata)
    data = {"action.a": torch.tensor([[1.0, 2.0]]), "action.b": torch.tensor([[3.0, 4.0]])}
    result = transform.apply(dict(data))
    torch.testing.assert_close(result["action"], torch.tensor([[3.0, 4.0, 1.0, 2.0]]), atol=0, rtol=0)
    restored = transform.unapply(result)
    for key, value in data.items():
        torch.testing.assert_close(restored[key], value, atol=0, rtol=0)


def test_dreamzero_composition_propagates_pipeline_and_per_horizon_statistics() -> None:
    from worldfoundry.synthesis.action_generation.dreamzero.preprocessing import (
        ComposedModalityTransform as DreamZeroComposed,
    )
    from worldfoundry.synthesis.action_generation.dreamzero.preprocessing import (
        ConcatTransform as DreamZeroConcat,
    )
    from worldfoundry.synthesis.action_generation.dreamzero.preprocessing import (
        DatasetMetadata,
        EmbodimentTag,
        PerHorizonActionTransform,
    )

    concat = DreamZeroConcat(video_concat_order=[])
    per_horizon = PerHorizonActionTransform(apply_to=["action.x"], normalization_modes={"action.x": "mean_std"})
    chain = DreamZeroComposed(transforms=[per_horizon, concat])
    metadata = DatasetMetadata.model_validate(
        {
            "statistics": {"state": {}, "action": {}},
            "modalities": {"state": {}, "action": {}, "video": {}},
            "embodiment_tag": next(iter(EmbodimentTag)),
        }
    )
    chain.set_metadata(metadata)
    assert concat._transform_pipeline == chain.transforms
    stats = {"x": {"mean": [[0.0], [1.0]], "std": [[1.0], [2.0]], "q01": [[-1.0], [-1.0]]}}
    chain.set_per_horizon_statistics(stats)
    result = per_horizon.apply({"action.x": torch.tensor([[1.0], [3.0]])})
    torch.testing.assert_close(result["action.x"], torch.ones(2, 1), atol=0, rtol=0)
    assert per_horizon.dataset_metadata is metadata
