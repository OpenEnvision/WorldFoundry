from utils.dino_utils import load_dinov3_model, extract_dinov3_features

def dino_mse_metric(pred_frames, gt_frames, dino_model=None, dino_processor=None, device='cuda:0', batch_size=8):
    """
    计算DINOv3特征的MSE指标

    Args:
        pred_frames: 预测视频帧，形状 [f, c, h, w]，范围 [0, 1]
        gt_frames: ground truth视频帧，形状 [f, c, h, w]，范围 [0, 1]
        dino_model: DINOv3模型（如果为None，会自动加载）
        dino_processor: DINOv3 processor（如果为None，会自动加载）
        device: 计算设备
        batch_size: 批处理大小

    Returns:
        result_dict: 包含dino_mse相关指标的字典
    """
    result_dict = {}

    # 提取GT帧的DINOv3特征
    gt_features = extract_dinov3_features(
        gt_frames,
        model=dino_model,
        processor=dino_processor,
        device=device,
        batch_size=batch_size
    )  # [f, 196, 768]

    # 提取预测帧的DINOv3特征
    pred_features = extract_dinov3_features(
        pred_frames,
        model=dino_model,
        processor=dino_processor,
        device=device,
        batch_size=batch_size
    )  # [f, 196, 768]

    mse_per_frame = ((pred_features - gt_features) ** 2).reshape(pred_features.shape[0], -1).mean(dim=1)

    result_dict['dino_mse'] = mse_per_frame.cpu().tolist()
    result_dict['avg_dino_mse'] = float(mse_per_frame.mean().cpu())

    return result_dict

def merge_dino_results(result_dict, new_result):
    """合并两个dino结果字典"""
    if result_dict is None:
        return new_result
    if new_result is None:
        return result_dict

    result_dict['dino_mse'].extend(new_result['dino_mse'])
    result_dict['avg_dino_mse'] = sum(result_dict['dino_mse']) / len(result_dict['dino_mse'])
    return result_dict
