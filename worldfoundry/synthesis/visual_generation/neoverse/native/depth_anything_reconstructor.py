"""Depth Anything 3 Gaussian reconstructor used by NeoVerse inference."""

import torch.nn as nn
import torch.nn.functional as F
from worldfoundry.core.model_loading import hash_state_dict_keys

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
    DepthAnything3,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
)
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.sh_utils import RGB2SH

from .geometry import depth_to_world_coords_points
from .rasterization import Gaussians, Rasterizer


class DA3GaussianRenderer:
    """Matching pipe.reconstructor.gs_renderer.rasterizer."""

    def __init__(self):
        """Init."""
        self.rasterizer = Rasterizer()


class DepthAnything3Reconstructor(nn.Module):
    """
    Adapter that wraps DepthAnything3 to match the NeoVerse reconstructor interface.

    Predicts depth + camera parameters via DA3, then constructs pseudo Gaussian
    Splatting representations compatible with the existing pipeline.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    PATCH_SIZE = 14

    def __init__(self, model_name="da3-giant", gaussian_scale=0.0001, **kwargs):
        """Init.

        Args:
            model_name: The model name.
            gaussian_scale: The gaussian scale.
        """
        super().__init__()
        self.da3 = DepthAnything3(model_name=model_name)
        self.gs_renderer = DA3GaussianRenderer()
        self.gaussian_scale = gaussian_scale

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict.

        Args:
            state_dict: The state dict.
            strict: The strict.
            assign: The assign.
        """
        return self.da3.load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, views, **kwargs):
        """Forward.

        Args:
            views: The views.
        """
        imgs = views["img"]  # [B, S, 3, H, W]
        timestamps = views["timestamp"]  # [B, S]
        B, S, C, H, W = imgs.shape

        # Normalize for DA3 (ImageNet normalization)
        mean = imgs.new_tensor(self.IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = imgs.new_tensor(self.IMAGENET_STD).view(1, 1, 3, 1, 1)
        imgs_normalized = (imgs - mean) / std

        # DA3 forward: expects [B, N, 3, H, W]
        da3_output = self.da3.forward(imgs_normalized)

        # Extract depth and crop back to original size
        depth = da3_output.depth  # [B, S, H, W]
        depth = depth.reshape(B * S, depth.shape[-2], depth.shape[-1])

        # Extract camera parameters
        # DA3 outputs w2c extrinsics; invert to get c2w
        w2c = as_homogeneous(da3_output.extrinsics)  # [B, S, 4, 4]
        # Convert camera pose to first camera coordinate system
        c2w = w2c[:, :1] @ affine_inverse(w2c)  # [B, S, 4, 4]
        c2w = c2w.float()
        intrinsics = da3_output.intrinsics  # [B, S, 3, 3]

        # Unproject depth to 3D world coordinates
        c2w_flat = c2w.reshape(B * S, 4, 4)
        K_flat = intrinsics.reshape(B * S, 3, 3)
        world_coords, _, valid_mask = depth_to_world_coords_points(depth, c2w_flat, K_flat)
        # world_coords: [B*S, H, W, 3], valid_mask: [B*S, H, W]

        # Get pixel colors (original, un-normalized)
        pixel_rgb = imgs.permute(0, 1, 3, 4, 2).reshape(B * S, H, W, 3)  # [B*S, H, W, 3]

        # Build pseudo-Gaussians per frame
        splats = []
        for b in range(B):
            static_flag = views["is_static"][b, 0]
            batch_gaussians = []
            for s in range(S):
                idx = b * S + s
                mask = valid_mask[idx]  # [H, W]
                pts = world_coords[idx][mask]  # [N_valid, 3]
                rgb = pixel_rgb[idx][mask]  # [N_valid, 3]

                N_valid = pts.shape[0]
                if N_valid == 0:
                    continue

                harmonics = RGB2SH(rgb).unsqueeze(1)  # [N_valid, 1, 3]
                scales = pts.new_full((N_valid, 3), self.gaussian_scale)
                rotations = pts.new_zeros(N_valid, 4)
                rotations[:, 0] = 1.0  # identity quaternion [1, 0, 0, 0]
                opacities = pts.new_ones(N_valid)

                gs = Gaussians(
                    means=pts,
                    harmonics=harmonics,
                    opacities=opacities,
                    scales=scales,
                    rotations=rotations,
                    timestamp=-1 if static_flag else timestamps[b, s].item(),
                )
                batch_gaussians.append(gs)
            splats.append(batch_gaussians)

        predictions = {
            "splats": splats,
            "rendered_extrinsics": c2w,  # [B, S, 4, 4]
            "rendered_intrinsics": intrinsics,  # [B, S, 3, 3]
            "rendered_timestamps": timestamps,  # [B, S]
        }
        return predictions

    @staticmethod
    def state_dict_converter():
        """State dict converter."""
        return ModelDictConverter()

class ModelDictConverter:
    """Model dict converter implementation."""
    def __init__(self):
        """Init."""
        pass

    def from_civitai(self, state_dict):
        """From civitai.

        Args:
            state_dict: The state dict.
        """
        if hash_state_dict_keys(state_dict) == '252f1c3923a62665aee9b32f1b18afb5':
            config = {
                "model_name": "da3-giant",
                "gaussian_scale": 0.001,
                "strict_load": False,
                "upcast_to_float32": True,
            }
        else:
            config = {}
        return state_dict, config
