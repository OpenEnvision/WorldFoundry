# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# isort: split
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.cfg import create_object

# isort: split
from addict import Dict

from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.model.utils.transform import (
    pose_encoding_to_extri_intri,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    map_pdf_to_opacity,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.pose_align import (
    batch_align_poses_umeyama,
)
from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.utils.ray_utils import (
    get_extrinsic_from_camray,
)


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            self.cam_enc = cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = gs_adapter if isinstance(gs_adapter, nn.Module) else create_object(_wrap_cfg(gs_adapter))
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert gs_head.out_dim == gs_out_dim, f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert gs_head["output_dim"] == gs_out_dim, (
                    f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                )
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],  # noqa: B006 - Keeps the upstream default object and signature.
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        input_extrinsics: torch.Tensor | None = None,
        input_intrinsics: torch.Tensor | None = None,
        use_aligned_pred_cam: bool = False,
        gs_down_ratio: int = 1,
        gs_scale_extra_multiplier: float = 1.0,
        gs_ds_feature_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view
            input_extrinsics: Unnormalized input extrinsics (B, N, 4, 4) for the GS branch
            input_intrinsics: Input intrinsics (B, N, 3, 3) for the GS branch
            use_aligned_pred_cam: Unproject Gaussians with predicted cameras aligned to the
                input-extrinsics frame by the inverse Umeyama Sim(3)
            gs_down_ratio: Spatial downsampling factor of the GS head output
            gs_scale_extra_multiplier: Multiplier applied to the maximum Gaussian scale
            gs_ds_feature_mode: Average-pool full-resolution GS image features when downsampling

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, aux_feats = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
        )
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        # Process features through depth head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = self._process_ray_pose_estimation(output, H, W)
            else:
                output = self._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = self._process_gs_head(
                    feats,
                    H,
                    W,
                    output,
                    x,
                    extrinsics,
                    intrinsics,
                    input_extrinsics=input_extrinsics,
                    input_intrinsics=input_intrinsics,
                    use_aligned_pred_cam=use_aligned_pred_cam,
                    gs_down_ratio=gs_down_ratio,
                    gs_scale_extra_multiplier=gs_scale_extra_multiplier,
                    gs_ds_feature_mode=gs_ds_feature_mode,
                )

        output = self._process_mono_sky_estimation(output)

        # Extract auxiliary features if requested
        output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

        return output

    def _process_mono_sky_estimation(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output

        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(output.depth, None, non_sky_mask, max_depth=non_sky_max)
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic)  # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = (
                torch.eye(3, 3)[None, None]
                .repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1)
                .clone()
                .to(pred_extrinsic.device)
            )
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(self, feats: list[torch.Tensor], H: int, W: int) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        *,
        input_extrinsics: torch.Tensor | None = None,
        input_intrinsics: torch.Tensor | None = None,
        use_aligned_pred_cam: bool = False,
        gs_down_ratio: int = 1,
        gs_scale_extra_multiplier: float = 1.0,
        gs_ds_feature_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available.

        By default the Gaussians are unprojected with the predicted cameras and
        ``extrinsics`` only aligns their scale. With ``use_aligned_pred_cam`` and
        ``input_extrinsics``, the predicted cameras are mapped into the input
        extrinsics frame by the inverse Umeyama Sim(3), GS depths are rescaled
        accordingly, and ``input_intrinsics`` replace the predicted intrinsics.
        """
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert ctx_extr is not None and ctx_intr is not None, "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        original_down_ratio = self.gs_head.down_ratio
        original_ds_feature_mode = self.gs_head.gs_ds_feature_mode
        if gs_down_ratio != 1:
            self.gs_head.down_ratio = gs_down_ratio
        self.gs_head.gs_ds_feature_mode = gs_ds_feature_mode
        try:
            gs_outs = self.gs_head(
                feats=feats,
                H=H,
                W=W,
                patch_start_idx=0,
                images=in_images,
            )
        finally:
            self.gs_head.down_ratio = original_down_ratio
            self.gs_head.gs_ds_feature_mode = original_ds_feature_mode
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf
        H_gs, W_gs = raw_gaussians.shape[2], raw_gaussians.shape[3]

        gs_extr = ctx_extr
        gs_intr = ctx_intr
        gs_depths = output.depth
        if use_aligned_pred_cam and input_extrinsics is not None:
            input_extr = as_homogeneous(input_extrinsics)
            rots, trans, pose_scales = batch_align_poses_umeyama(
                ctx_extr.detach().float(),
                input_extr.detach().float(),
                retry_degenerate=True,
            )
            pose_scales = torch.clamp(pose_scales.to(dtype=gs_depths.dtype), min=1e-6)
            scale_view = pose_scales.view(-1, 1, 1, 1)
            gs_depths = gs_depths / scale_view
            if getattr(self.gs_adapter, "pred_offset_depth", False):
                raw_gaussians = raw_gaussians.clone()
                raw_gaussians[..., -1] = raw_gaussians[..., -1] / scale_view

            # Umeyama gives s * R @ input_pose + t ~= predicted_pose, so the
            # predicted cameras map into the input frame as R^T @ (pose - t) / s.
            dtype = ctx_extr.dtype
            rots_inv = rots.transpose(-1, -2).to(dtype=dtype)
            pred_c2w = affine_inverse(ctx_extr)
            aligned_c2w = torch.zeros_like(pred_c2w)
            aligned_c2w[..., :3, :3] = rots_inv[:, None] @ pred_c2w[..., :3, :3]
            aligned_c2w[..., :3, 3] = (
                rots_inv[:, None] @ (pred_c2w[..., :3, 3:] - trans.to(dtype=dtype)[:, None, :, None])
            ).squeeze(-1) / pose_scales.to(dtype=dtype)[:, None, None]
            aligned_c2w[..., 3, 3] = 1.0
            gs_extr = affine_inverse(aligned_c2w)
            if input_intrinsics is not None:
                gs_intr = input_intrinsics
            gt_extr = None

        if (H_gs, W_gs) != (H, W):
            gs_depths = gs_depths[:, :, :: H // H_gs, :: W // W_gs]
            gs_intr = gs_intr.clone()
            gs_intr[..., 0, :] *= W_gs / W
            gs_intr[..., 1, :] *= H_gs / H

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=gs_extr,
            intrinsics=gs_intr,
            depths=gs_depths,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H_gs, W_gs),
            gt_extrinsics=gt_extr,
            gs_scale_extra_multiplier=gs_scale_extra_multiplier,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers, strict=False):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3 = create_object(anyview)
        self.da3_metric = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],  # noqa: B006 - Keeps the upstream default object and signature.
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        input_extrinsics: torch.Tensor | None = None,
        input_intrinsics: torch.Tensor | None = None,
        use_aligned_pred_cam: bool = False,
        gs_down_ratio: int = 1,
        gs_scale_extra_multiplier: float = 1.0,
        gs_ds_feature_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view
            input_extrinsics: Unnormalized input extrinsics (B, N, 4, 4) for the GS branch
            input_intrinsics: Input intrinsics (B, N, 3, 3) for the GS branch
            use_aligned_pred_cam: Unproject Gaussians with predicted cameras aligned to the
                input-extrinsics frame by the inverse Umeyama Sim(3)
            gs_down_ratio: Spatial downsampling factor of the GS head output
            gs_scale_extra_multiplier: Multiplier applied to the maximum Gaussian scale
            gs_ds_feature_mode: Average-pool full-resolution GS image features when downsampling

        Returns:
            Dictionary containing aligned depth predictions and camera parameters
        """
        # Get predictions from both branches
        output = self.da3(
            x,
            extrinsics,
            intrinsics,
            export_feat_layers=export_feat_layers,
            infer_gs=infer_gs,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            input_extrinsics=input_extrinsics,
            input_intrinsics=input_intrinsics,
            use_aligned_pred_cam=use_aligned_pred_cam,
            gs_down_ratio=gs_down_ratio,
            gs_scale_extra_multiplier=gs_scale_extra_multiplier,
            gs_ds_feature_mode=gs_ds_feature_mode,
        )
        metric_output = self.da3_metric(x)

        # Apply metric scaling and alignment
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
