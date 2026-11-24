# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Lyra-1 shares stateless timm helpers without changing patch or residual tensors."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_lyra1_uses_timm_drop_path_and_tuple_helpers() -> None:
    from timm import layers

    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils import attention

    assert attention.DropPath is layers.DropPath
    assert attention.to_2tuple is layers.to_2tuple
    assert attention.to_3tuple is layers.to_3tuple
    layer = attention.DropPath(0.5).eval()
    x = torch.randn(2, 3, 8)
    assert not layer.state_dict()
    assert layer(x) is x


def test_lyra1_patch_embed_preserves_list_configuration() -> None:
    from worldfoundry.synthesis.visual_generation.lyra_1.lyra1_runtime.src.models.utils.attention import PatchEmbed3D

    reference = PatchEmbed3D(patch_size=(1, 2, 3), in_chans=3, embed_dim=8).eval()
    candidate = PatchEmbed3D(patch_size=[1, 2, 3], in_chans=3, embed_dim=8).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 4, 3, 4, 6)
    with torch.no_grad():
        expected, actual = reference(x), candidate(x)
    assert actual.shape == (2, 16, 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
