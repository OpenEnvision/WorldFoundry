# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.models import (
    visual_transformer as mirror,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.utils import (
    inference_utils as mirror_images,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.models.models import (
    visual_transformer as hyworld,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hyworldmirror_2p0.utils import (
    inference_utils as hyworld_images,
)

pytestmark = pytest.mark.unit


def test_hyworld_uses_mirror_special_token_and_image_padding_helpers() -> None:
    assert hyworld.expand_and_flatten_special_tokens is mirror.expand_and_flatten_special_tokens
    assert hyworld_images._apply_padding is mirror_images._apply_padding
    token = torch.arange(16).reshape(1, 2, 2, 4)
    result = hyworld.expand_and_flatten_special_tokens(token, 2, 3).reshape(2, 3, 2, 4)
    torch.testing.assert_close(result[:, 0], token[:, 0].expand(2, -1, -1), rtol=0, atol=0)
    torch.testing.assert_close(result[:, 1:], token[:, 1:].expand(2, 2, -1, -1), rtol=0, atol=0)


def test_mirror_image_padding_keeps_white_border_and_noop_identity() -> None:
    image = torch.zeros(3, 2, 4)
    padded = hyworld_images._apply_padding(image, 4)
    assert padded.shape == (3, 4, 4)
    assert torch.equal(padded[:, [0, -1]], torch.ones(3, 2, 4))
    assert hyworld_images._apply_padding(image, 1) is image
