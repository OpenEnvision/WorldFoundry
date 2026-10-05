"""Regression coverage for DreamX-World's causal model binding."""

from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.models.denoisers import wan_dreamx
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamx_world import causal


def test_dreamx_wrapper_uses_in_tree_causal_camera_model(monkeypatch) -> None:
    fake_model = torch.nn.Identity()
    monkeypatch.setattr(wan_dreamx, "load_wan_config", lambda _: {})
    monkeypatch.setattr(
        causal.CausalWanModel,
        "from_config",
        classmethod(lambda cls, config, **kwargs: fake_model),
    )

    wrapper = wan_dreamx.WanDiffusionCameraWrapper("unused.json")

    assert wrapper.model is fake_model
