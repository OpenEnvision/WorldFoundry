import torch
import torch.nn.functional as F

from .calculate_psnr import calculate_psnr
from .calculate_ssim import calculate_ssim

# isort: split
from .calculate_lpips import calculate_lpips


def evaluate_video_metrics(
    ground_truth_frames,
    generated_frames,
    device="cpu",
    only_final=False,
    fvd_method="styleganv",
    resize_hw=None,
):
    """
    Computes FVD, SSIM, PSNR, and LPIPS metrics between ground truth and generated videos.
    Args:
        ground_truth_frames (List[Tensor]): List of Tensors, each Tensor with shape [B, T, C, H, W],
                                            or any shape you consistently gather from the dataloader.
        generated_frames    (List[Tensor]): List of Tensors, same shape as ground_truth_frames.
        device (torch.device): Device on which to perform computations (e.g., "cuda" or "cpu").
        only_final (bool): If True, compute metrics only on the last frame. Otherwise, on all frames.
        fvd_method (str): One of ["styleganv", "videogpt"], per your usage in `calculate_fvd`.
        resize_hw (tuple): A (height, width) tuple if you want to resize frames to a fixed resolution
                           before evaluation (e.g., (64, 64)).
    Returns:
        Dict: A dictionary containing "fvd", "ssim", "psnr", and "lpips" metrics.
    """
    # 1. Concatenate all videos
    # Example: if ground_truth_frames is a list of shape [B, T, C, H, W]
    # for each batch, we can concatenate along dim=0 to combine them.
    if ground_truth_frames[0].dim() == 3:  # shape: [C, H, W]
        ground_truth_frames = [frame.reshape(1, 1, *frame.shape) for frame in ground_truth_frames]
        generated_frames = [frame.reshape(1, 1, *frame.shape) for frame in generated_frames]
    if ground_truth_frames[0].dim() == 4:  # shape: [N, C, H, W]
        ground_truth_frames = [frame.unsqueeze(0) for frame in ground_truth_frames]
        generated_frames = [frame.unsqueeze(0) for frame in generated_frames]
    elif ground_truth_frames[0].dim() == 5:
        pass
    gt_videos = torch.cat(ground_truth_frames, dim=0).to(device)  # shape: [N, T, C, H, W]
    gen_videos = torch.cat(generated_frames, dim=0).to(device)  # shape: [N, T, C, H, W]

    # 2. Optional: Ensure data is in [0,1] & resize
    # If your data is already [0,1], you can skip normalization.
    assert gt_videos.dim() == 5 and gt_videos.shape == gen_videos.shape, (
        f"Shape mismatch: {gt_videos.shape} vs {gen_videos.shape}"
    )
    assert gt_videos.min() >= 0 and gt_videos.max() <= 1
    assert gen_videos.min() >= 0 and gen_videos.max() <= 1

    # Resize to the same shape if desired:
    if resize_hw is not None:
        new_h, new_w = resize_hw
        # Interpolate expects shape [N, C, T, H, W] or [N, C, H, W] depending on mode,
        # so we may need to rearrange dimensions.
        # We'll do [N, T, C, H, W] -> [N*T, C, H, W], then reshape back.

        N, T, C, H, W = gt_videos.shape

        # Flatten the video dimension for resizing
        gt_videos_2d = gt_videos.view(N * T, C, H, W)
        gen_videos_2d = gen_videos.view(N * T, C, H, W)

        gt_videos_2d = F.interpolate(gt_videos_2d, size=(new_h, new_w), mode="bilinear", align_corners=False)
        gen_videos_2d = F.interpolate(gen_videos_2d, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Restore the time dimension
        gt_videos = gt_videos_2d.view(N, T, C, new_h, new_w)
        gen_videos = gen_videos_2d.view(N, T, C, new_h, new_w)

    # 3. Compute the metrics
    result = {}
    # result["fvd"]  = calculate_fvd(gt_videos, gen_videos, device, method=fvd_method, only_final=only_final)
    result["ssim"] = calculate_ssim(gt_videos, gen_videos, only_final=only_final)
    result["psnr"] = calculate_psnr(gt_videos, gen_videos, only_final=only_final)
    result["lpips"] = calculate_lpips(gt_videos, gen_videos, device=device, only_final=only_final)

    return result
