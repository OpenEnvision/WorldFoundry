"""Released NeoVerse reconstruction assembled from the shared Mirror modules."""

import torch
from torch import nn

from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.heads.camera_head import CameraHead
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.heads.dense_head import DPTHead
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.models.visual_transformer import VisualGeometryTransformer
from worldfoundry.base_models.three_dimensions.point_clouds.hunyuan_mirror.models.utils.camera_utils import vector_to_camera_matrices
from worldfoundry.core.model_loading import hash_state_dict_keys

from .rasterization import GaussianSplatRenderer


class NeoVerseWorldMirror(nn.Module):
    """Match the released inference path, which disables the motion heads.

    The checkpoint predates Mirror 2.0's normalized RoPE and changed prediction
    heads. Video inputs require timestamped Gaussians, even with motion disabled.
    """

    def __init__(self, checkpoint_ignored_keys=()):
        super().__init__()
        self.checkpoint_ignored_keys = tuple(checkpoint_ignored_keys)
        self.visual_geometry_transformer = VisualGeometryTransformer(enable_cond=True)
        self.cam_head = CameraHead(dim_in=2048)
        self.pts_head = DPTHead(dim_in=2048, output_dim=4, patch_size=14, activation="inv_log+expp1")
        self.depth_head = DPTHead(dim_in=2048, output_dim=2, patch_size=14, activation="exp+expp1")
        self.gs_head = DPTHead(
            dim_in=2048, output_dim=2, patch_size=14, features=256,
            is_gsdpt=True, activation="exp+expp1",
        )
        self.gs_renderer = GaussianSplatRenderer(
            sh_degree=0, enable_prune=True, voxel_size=0.002, is_4dgs=True,
        )

    def forward(self, views, *, is_inference=True, use_motion=False):
        if not is_inference or use_motion:
            raise ValueError("NeoVerse reconstruction supports inference with use_motion=False.")
        images = views["img"]
        tokens, patch_start = self.visual_geometry_transformer(images)
        camera = self.cam_head(tokens)[-1]
        extrinsics, intrinsics = vector_to_camera_matrices(camera, image_hw=images.shape[-2:])
        bottom = torch.tensor([0, 0, 0, 1], device=extrinsics.device).view(1, 1, 1, 4)
        world_to_camera = torch.cat((extrinsics, bottom.repeat(*extrinsics.shape[:2], 1, 1)), dim=2)
        predictions = {
            "camera_params": camera,
            "camera_poses": torch.linalg.inv(world_to_camera),
            "camera_intrs": intrinsics,
        }
        for name, head in (("depth", self.depth_head), ("pts3d", self.pts_head)):
            predictions[name], predictions[name + "_conf"] = head(
                tokens, images=images, patch_start_idx=patch_start,
            )
        features, predictions["gs_depth"], predictions["gs_depth_conf"] = self.gs_head(
            tokens, images=images, patch_start_idx=patch_start,
        )
        predictions = self.gs_renderer.render(
            gs_feats=features, images=images, predictions=predictions, views=views,
            context_predictions={}, is_inference=True,
        )
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                predictions[key] = value.float()
            elif isinstance(value, list):
                for batch in value:
                    for gaussians in batch:
                        gaussians.to(torch.float32)
        return predictions

    @staticmethod
    def state_dict_converter():
        return NeoVerseWorldMirrorStateDictConverter()


class NeoVerseWorldMirrorStateDictConverter:
    """Exclude only the unused motion branches; strictly load every active role."""

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) != "1a1d001a35f78f3a7796a1e719ead340":
            raise ValueError("Unsupported NeoVerse WorldMirror checkpoint layout.")
        unused_prefixes = (
            "visual_geometry_transformer.motion_",
            "velocity_fwd_head.", "velocity_bwd_head.",
            "gs_fwd_attr_head.", "gs_bwd_attr_head.",
        )
        ignored = tuple(key for key in state_dict if key.startswith(unused_prefixes))
        active = {key: value for key, value in state_dict.items() if not key.startswith(unused_prefixes)}
        return active, {"strict_load": True, "checkpoint_ignored_keys": ignored}
