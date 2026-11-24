# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""VGGSfM frame ranking binds the canonical DINOv2 loader and attention policy."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("spatial_similarity", [False, True])
def test_vggsfm_frame_ranking_loads_in_tree_dinov2(spatial_similarity: bool) -> None:
    script = textwrap.dedent(
        """
        import sys
        from types import ModuleType
        import torch

        # Feature extraction and COLMAP are unrelated to frame ranking.
        lightglue = ModuleType("lightglue")
        lightglue.ALIKED = lightglue.SIFT = lightglue.SuperPoint = object
        sys.modules["lightglue"] = lightglue
        sys.modules["pycolmap"] = ModuleType("pycolmap")

        from worldfoundry.base_models.perception_core.general_perception.dinov2.hub import backbones
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.dependency.vggsfm_utils import (
            generate_rank_by_dino,
        )

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Module()])
                self.blocks[0].attn = torch.nn.Module()
                self.blocks[0].attn.fused_attn = True

            def forward(self, images, *, is_training):
                assert is_training
                assert not self.training
                assert not self.blocks[0].attn.fused_attn
                return {
                    "x_norm_clstoken": images.mean(dim=(2, 3)),
                    "x_norm_patchtokens": images.flatten(2).transpose(1, 2),
                }

        calls = []
        def build(*, pretrained):
            calls.append(pretrained)
            return Encoder()

        backbones.dinov2_vitb14_reg = build
        torch.manual_seed(0)
        indices = generate_rank_by_dino(
            torch.rand(5, 3, 4, 4), query_frame_num=3, image_size=4,
            device="cpu", spatial_similarity=sys.argv[1] == "True",
        )
        assert calls == [True]
        assert len(indices) == 3
        assert all(0 <= index < 5 for index in indices)
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(spatial_similarity)],
        check=True,
        capture_output=True,
        text=True,
    )
