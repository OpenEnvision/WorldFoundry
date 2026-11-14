# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest
import torch
from timm.models.vision_transformer import LayerScale

from worldfoundry.base_models.llm_mllm_core.mllm.prismatic import PrismaticProjector, ls_apply_patch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fused", [False, True])
def test_prismatic_projector_keeps_native_fc_keys_and_construction_order(fused: bool) -> None:
    layer = PrismaticProjector(fused, 8, 12)
    expected = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    if fused:
        expected += ["fc3.weight", "fc3.bias"]
    assert list(layer.state_dict()) == expected
    assert layer.fc1.weight.shape == ((32 if fused else 12), 8)
    assert layer(torch.randn(2, 7, 8)).shape == (2, 7, 12)


@pytest.mark.parametrize("inplace", [False, True])
def test_ls_apply_patch_clones_gamma_and_preserves_mutation_policy(inplace: bool) -> None:
    layer = LayerScale(8, init_values=0.3, inplace=inplace)
    old = layer.gamma
    ls_apply_patch(layer)
    assert list(layer.state_dict()) == ["scale_factor"]
    assert layer.scale_factor is not old
    torch.testing.assert_close(layer.scale_factor, old, rtol=0, atol=0)
    x = torch.randn(2, 7, 8)
    expected = x * old
    output = layer(x)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert (output is x) is inplace
    with pytest.raises(AttributeError, match="gamma"):
        ls_apply_patch(layer)


def test_prismatic_owner_avoids_action_runtime_and_tracking_initializers() -> None:
    code = """
import sys
import torch
from timm.models.vision_transformer import LayerScale
# Test initialization added by the owner, beyond its required timm dependency.
before = set(sys.modules)
import worldfoundry.base_models.llm_mllm_core.mllm.prismatic
assert not any(name.startswith('worldfoundry.synthesis') for name in sys.modules)
assert not any(name.startswith(('hydra', 'omegaconf', 'wandb', 'triton')) for name in set(sys.modules) - before)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
