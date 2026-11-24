# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Adapted from Robbyant/lingbot-world.
"""LingBot camera embeddings with an optional direction-only control branch."""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.camera_geometry import (
    get_plucker_embeddings as _get_plucker_embeddings,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.camera_geometry import world_ray_directions


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
) -> torch.Tensor:
    if only_rays_d:
        rays_d = world_ray_directions(c2ws_mat, Ks, height, width)
        return rays_d.view([c2ws_mat.shape[0], height, width, 3])
    return _get_plucker_embeddings(c2ws_mat, Ks, height, width)
