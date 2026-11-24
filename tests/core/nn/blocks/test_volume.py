# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.nn.blocks.volume import center_down_sample_3d, pad_for_3d_conv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kernel", [(2, 4, 4), (4, 8, 8)])
def test_pad_for_3d_conv_preserves_trailing_replication_and_pool_stride(
    dtype: torch.dtype, kernel: tuple[int, int, int]
) -> None:
    x = torch.arange(3 * 5 * 7, dtype=dtype).reshape(1, 1, 3, 5, 7)
    padding = tuple((step - size % step) % step for size, step in zip(x.shape[-3:], kernel, strict=True))
    expected = F.pad(x, (0, padding[2], 0, padding[1], 0, padding[0]), mode="replicate")
    padded = pad_for_3d_conv(x, kernel)
    torch.testing.assert_close(padded, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        center_down_sample_3d(padded, kernel), F.avg_pool3d(expected, kernel, stride=kernel), rtol=0, atol=0
    )
    assert padded.dtype == dtype
