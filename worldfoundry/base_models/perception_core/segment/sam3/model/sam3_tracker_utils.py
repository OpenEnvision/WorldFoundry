# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
import torch.nn.functional as F

from worldfoundry.base_models.perception_core.segment.sam3.model.edt import edt_triton
from worldfoundry.base_models.perception_core.segment.sam_components.mask_ops import (
    sample_random_points_from_errors as sample_random_points_from_errors,
)


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, H, W = gt_masks.shape

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = (~gt_masks & pred_masks).squeeze(1)
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = (gt_masks & ~pred_masks).squeeze(1)

    if padding:
        padded_fp_masks = torch.zeros(B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device)
        padded_fp_masks[:, 1 : H + 1, 1 : W + 1] = fp_masks
        padded_fn_masks = torch.zeros(B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device)
        padded_fn_masks[:, 1 : H + 1, 1 : W + 1] = fn_masks
    else:
        padded_fp_masks = fp_masks
        padded_fn_masks = fn_masks

    fn_mask_dt = edt_triton(padded_fn_masks)
    fp_mask_dt = edt_triton(padded_fp_masks)
    if padding:
        fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]

    fn_max, fn_argmax = fn_mask_dt.reshape(B, -1).max(dim=-1)
    fp_max, fp_argmax = fp_mask_dt.reshape(B, -1).max(dim=-1)
    is_positive = fn_max > fp_max
    chosen = torch.where(is_positive, fn_argmax, fp_argmax)
    points_x = chosen % W
    points_y = chosen // W

    labels = is_positive.long()
    points = torch.stack([points_x, points_y], -1)
    return points.unsqueeze(1), labels.unsqueeze(1)


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min((t for t in cond_frame_outputs if t < frame_idx), default=None)
            if idx_first is None:
                # Maybe we are tracking in reverse
                idx_first = max((t for t in cond_frame_outputs if t > frame_idx), default=None)
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_best_gt_match_from_multimasks(pred_multimasks, gt_masks, pred_scores=None):
    """
    Get the mask with the best match to GT masks (based on IoU) from pred_multimasks.
    Optionally, use `pred_scores` to break ties in case all IoUs are zeros.
    """
    assert pred_multimasks.ndim == 4 and gt_masks.ndim == 4
    if pred_multimasks.size(1) == 1:
        return pred_multimasks  # only a single mask channel, nothing to select

    pred_multimasks_binary = pred_multimasks > 0
    area_i = torch.sum(pred_multimasks_binary & gt_masks, dim=(2, 3)).float()
    area_u = torch.sum(pred_multimasks_binary | gt_masks, dim=(2, 3)).float()
    ious = area_i / torch.clamp(area_u, min=1.0)

    # In case all IoUs are zeros (e.g. because the GT mask is empty), use pred_scores
    # to break ties and select the best mask
    if pred_scores is not None:
        has_nonzero_ious = torch.any(ious > 0).expand_as(ious)
        scores = torch.where(has_nonzero_ious, ious, pred_scores)
    else:
        scores = ious

    # Finally, take the best mask prediction (with the highest score)
    best_scores_inds = torch.argmax(scores, dim=-1)
    batch_inds = torch.arange(scores.size(0), device=scores.device)
    best_pred_mask = pred_multimasks[batch_inds, best_scores_inds].unsqueeze(1)
    return best_pred_mask


def fill_holes_in_mask_scores(
    mask,
    max_area=None,
    fill_holes=True,
    remove_sprinkles=True,
    fill_hole_area=None,
    sprinkle_removal_area=None,
):
    # Support onevision-style keyword args
    if fill_hole_area is not None and max_area is None:
        max_area = fill_hole_area
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    Holes are those small connected components in either background or foreground.

    Note that it uses the optional "cc_torch" package for fast connected components when a
    compatible prebuilt wheel is installed. Otherwise, it falls back to a slower triton
    implementation, or skimage if the tensor is on CPU.
    """

    if max_area <= 0:
        return mask  # nothing to fill in this case

    if fill_holes:
        # We remove small connected components in background by changing them to foreground
        # with a small positive mask score (0.1).
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        # We remove small connected components in foreground by changing them to background
        # with a small negative mask score (-0.1). Here we only remove connected components
        # whose areas are under both `max_area` and half of the entire mask's area. This
        # removes sprinkles while avoids filtering out tiny objects that we want to track.
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask


def _get_connected_components_with_padding(mask):
    """Get connected components from masks (possibly padding them to an even size)."""
    from worldfoundry.base_models.perception_core.segment.sam3.perflib.connected_components import (
        connected_components,
    )

    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape
    # make sure both height and width are even (to be compatible with cc_torch)
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        labels, counts = connected_components(mask)
    else:
        # pad the mask to make its height and width even
        # padding format is (padding_left,padding_right,padding_top,padding_bottom)
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
        labels, counts = connected_components(mask_pad)
        labels = labels[:, :, :H, :W]
        counts = counts[:, :, :H, :W]

    return labels, counts
