from utils.dino_utils import extract_dinov3_features


def dino_mse_metric(pred_frames, gt_frames, dino_model=None, dino_processor=None, device="cuda:0", batch_size=8):
    """Compute the per-frame MSE between the DINOv3 patch features of predicted and ground-truth frames.

    Args:
        pred_frames: Predicted frames, shape [f, c, h, w], range [0, 1].
        gt_frames: Ground-truth frames, shape [f, c, h, w], range [0, 1].
        dino_model: DINOv3 model from ``load_dinov3_model``.
        dino_processor: DINOv3 image processor from ``load_dinov3_model``.
        device: Device the features are computed on.
        batch_size: Frames per forward pass.

    Returns:
        Dict with the per-frame ``dino_mse`` list and its mean ``avg_dino_mse``.
    """
    result_dict = {}

    gt_features = extract_dinov3_features(
        gt_frames, model=dino_model, processor=dino_processor, device=device, batch_size=batch_size
    )  # [f, 196, 768]

    pred_features = extract_dinov3_features(
        pred_frames, model=dino_model, processor=dino_processor, device=device, batch_size=batch_size
    )  # [f, 196, 768]

    mse_per_frame = ((pred_features - gt_features) ** 2).reshape(pred_features.shape[0], -1).mean(dim=1)

    result_dict["dino_mse"] = mse_per_frame.cpu().tolist()
    result_dict["avg_dino_mse"] = float(mse_per_frame.mean().cpu())

    return result_dict


def merge_dino_results(result_dict, new_result):
    """Append the per-frame DINO scores of ``new_result`` to ``result_dict`` and update the mean; either may be None."""
    if result_dict is None:
        return new_result
    if new_result is None:
        return result_dict

    result_dict["dino_mse"].extend(new_result["dino_mse"])
    result_dict["avg_dino_mse"] = sum(result_dict["dino_mse"]) / len(result_dict["dino_mse"])
    return result_dict
