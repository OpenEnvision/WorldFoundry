from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "hydra" dependency at import time; skip when it is unavailable.
pytest.importorskip("hydra")

from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import (
    LTXVideoOnlyModelConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType


class _CapturedModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _CapturedConfigurator(LTXVideoOnlyModelConfigurator):
    MODEL_CLS = _CapturedModel


def _config(**updates):
    transformer = {
        "dropout": 0.0,
        "attention_bias": True,
        "num_vector_embeds": None,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "use_linear_projection": False,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "double_self_attention": False,
        "upcast_attention": False,
        "standardization_norm": "rms_norm",
        "norm_elementwise_affine": False,
        "qk_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "caption_proj_before_connector": True,
    }
    transformer.update(updates)
    return {"transformer": transformer}


def test_ltx_video_configurator_preserves_explicit_ltx2_rope_layout() -> None:
    model = _CapturedConfigurator.from_config(
        _config(use_middle_indices_grid=True, rope_type="split")
    )

    assert model.kwargs["use_middle_indices_grid"] is True
    assert model.kwargs["rope_type"] is LTXRopeType.SPLIT


def test_ltx_video_configurator_keeps_legacy_defaults_when_fields_are_absent() -> None:
    model = _CapturedConfigurator.from_config(_config())

    assert model.kwargs["use_middle_indices_grid"] is False
    assert model.kwargs["rope_type"] is LTXRopeType.INTERLEAVED
