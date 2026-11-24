# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.core.attention.rotary.embedding import rotate_half_chunked
from worldfoundry.core.attention.rotary.rope import rotate_half

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [4, 5, 16])
def test_rotate_half_chunked_keeps_torch_split_order(width: int) -> None:
    x = torch.arange(2 * width).reshape(2, width)
    left, right = x.chunk(2, dim=-1)
    torch.testing.assert_close(rotate_half_chunked(x), torch.cat([-right, left], dim=-1), rtol=0, atol=0)


def test_existing_rotate_half_still_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="even"):
        rotate_half(torch.ones(2, 5))
