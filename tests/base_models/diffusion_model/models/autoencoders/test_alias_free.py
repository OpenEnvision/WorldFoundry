# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.alias_free import (
    DownSample1d,
    LowPassFilter1d,
    kaiser_sinc_filter1d,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kernel_size", [11, 12])
def test_low_pass_filter_preserves_registered_kernel_and_depthwise_channels(kernel_size: int) -> None:
    layer = LowPassFilter1d(cutoff=0.25, half_width=0.3, stride=2, kernel_size=kernel_size)
    assert list(layer.state_dict()) == ["filter"]
    assert layer.filter.shape == (1, 1, kernel_size)
    torch.testing.assert_close(layer.filter, kaiser_sinc_filter1d(0.25, 0.3, kernel_size), rtol=0, atol=0)
    x = torch.randn(2, 3, 63)
    expected = torch.cat([layer(channel) for channel in x.split(1, dim=1)], dim=1)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


def test_down_sample_uses_shared_filter_without_changing_checkpoint_slots() -> None:
    layer = DownSample1d(ratio=3, kernel_size=11)
    assert type(layer.lowpass) is LowPassFilter1d
    assert list(layer.state_dict()) == ["lowpass.filter"]
    assert layer(torch.randn(2, 3, 63)).shape == (2, 3, 21)


@pytest.mark.parametrize("cutoff", [-0.1, 0.6])
def test_low_pass_filter_keeps_cutoff_errors(cutoff: float) -> None:
    with pytest.raises(ValueError):
        LowPassFilter1d(cutoff=cutoff)


def test_kaiser_sinc_zero_cutoff_keeps_native_error() -> None:
    with pytest.raises(UnboundLocalError):
        kaiser_sinc_filter1d(0.0, 0.3, 12)


def test_alias_free_owner_avoids_codec_and_tracking_initializers() -> None:
    code = """
import sys
import worldfoundry.base_models.diffusion_model.models.autoencoders.alias_free
prefixes = ('worldfoundry.base_models.diffusion_model.models.autoencoders.kandinsky6',
            'worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio',
            'worldfoundry.synthesis', 'hydra', 'omegaconf', 'wandb', 'triton')
assert not any(name.startswith(prefixes) for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
