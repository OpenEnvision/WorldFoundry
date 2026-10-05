# SPDX-License-Identifier: Apache-2.0
"""InSpatio depth-weighted RGBD splatting, used for camera control."""


import torch


class DepthWarper:
    """Batched depth-image forward splatting used by the fast render backend."""

    def __init__(self):
        self._grid_key = None
        self._grid = None

    def bilinear_splatting(self, frame1, mask1, depth1, flow12, flow12_mask, is_image=False):
        b, c, h, w = frame1.shape
        if flow12_mask is None:
            flow12_mask = torch.ones((b, 1, h, w), device=flow12.device, dtype=flow12.dtype)

        grid = self.create_grid(b, h, w, frame1.device, frame1.dtype)
        trans_pos = flow12 + grid
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()

        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
        ], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth = torch.clamp(depth1, min=0, max=1000)
        log_depth = torch.log(1 + sat_depth)
        depth_weights = torch.exp(log_depth / log_depth.max().clamp_min(1e-6) * 50)
        if depth1.dim() == 3:
            valid_mask = (depth1 >= 0).to(depth1).unsqueeze(1)
            depth_weights = depth_weights.unsqueeze(1)
        else:
            valid_mask = (depth1 >= 0).to(depth1)

        def make_weight(prox_weight):
            return torch.moveaxis(
                prox_weight * mask1 * flow12_mask * valid_mask / depth_weights,
                [0, 1, 2, 3],
                [0, 3, 1, 2],
            )

        weight_nw = make_weight(prox_weight_nw)
        weight_sw = make_weight(prox_weight_sw)
        weight_ne = make_weight(prox_weight_ne)
        weight_se = make_weight(prox_weight_se)

        warped_frame = torch.zeros((b, h + 2, w + 2, c), dtype=torch.float32, device=frame1.device)
        warped_weights = torch.zeros((b, h + 2, w + 2, 1), dtype=torch.float32, device=frame1.device)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b, device=frame1.device)[:, None, None]
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        known_mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        warped_frame2 = torch.where(
            known_mask,
            cropped_frame / cropped_weights,
            torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device),
        )
        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, known_mask.to(frame1)

    def create_grid(self, b, h, w, device=None, dtype=torch.float32):
        key = (h, w, str(device), dtype)
        if self._grid_key != key:
            y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                  torch.arange(w, device=device, dtype=dtype), indexing="ij")
            self._grid = torch.stack([x, y])[None]
            self._grid_key = key
        return self._grid.expand(b, -1, -1, -1)
