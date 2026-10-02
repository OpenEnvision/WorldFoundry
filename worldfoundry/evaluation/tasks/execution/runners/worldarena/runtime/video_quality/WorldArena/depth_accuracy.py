import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from .utils import load_video, load_dimension_info
from worldfoundry.core.distributed.collectives.evaluation import (
    get_rank,
    distribute_list_to_rank,
    gather_list_of_dict,
    get_world_size
)
from worldfoundry.core.execution.device import get_current_torch_device

class DepthEstimator:
    def __init__(self, model_path, device):
        self.device = device
        # 强制加载 fp16
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_path, 
            local_files_only=True,
            torch_dtype=torch.float16 
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, images):
        T, C, H, W = images.shape
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        inputs = self.processor(images=list(images_np), return_tensors="pt").to(self.device)
        
        target_dtype = next(self.model.parameters()).dtype
        for k in inputs:
            if torch.is_floating_point(inputs[k]):
                inputs[k] = inputs[k].to(target_dtype)

        # 修正 FutureWarning: 使用新的 amp.autocast 写法
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 插值回当前视频的原始分辨率
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        return depth_map

def depth_accuracy(depth_model, video_list, gt_root, device):
    total_abs_rel = 0.0
    cnt = 0
    video_results = []
    TARGET_FRAMES = 40

    def uniform_sample_tensor(img_tensor, n):
        T = img_tensor.shape[0]
        if T <= n:
            return img_tensor
        # 均匀生成索引：0, ..., T-1
        indices = torch.linspace(0, T - 1, n).long()
        return img_tensor[indices]

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        # 1. 路径匹配: 生成是 .../task/episode/trial/video, GT是 .../task/episode/video
        path_parts = video_path.rstrip('/').split('/')
        task_id, episode_id, file_name = path_parts[-4], path_parts[-3], path_parts[-1]
        
        rel_path = os.path.join(task_id, episode_id, file_name)
        gt_video_path = os.path.join(gt_root, rel_path)

        if not os.path.exists(gt_video_path):
            continue

        # 2. 加载与采样
        try:
            gen_images = load_video(video_path) # [T, C, H, W]
            gt_images = load_video(gt_video_path)
            
            # 均匀采样 40 帧
            gen_images = uniform_sample_tensor(gen_images, TARGET_FRAMES).to(device)
            gt_images = uniform_sample_tensor(gt_images, TARGET_FRAMES).to(device)
            
            # 强制对齐长度 (防止原视频本身就少于40帧的情况)
            min_len = min(len(gen_images), len(gt_images))
            gen_images, gt_images = gen_images[:min_len], gt_images[:min_len]
        except Exception as e:
            if get_rank() == 0: print(f"Load Error: {e}")
            continue

        # 3. 深度估计
        gen_depth = depth_model.infer(gen_images)
        gt_depth = depth_model.infer(gt_images)
        if gen_depth.shape != gt_depth.shape:
            gen_depth = F.interpolate(
                gen_depth, 
                size=(gt_depth.shape[2], gt_depth.shape[3]), 
                mode="bicubic", 
                align_corners=False
            )
        # 4. 深度对齐与误差计算
        # 使用全图的中值进行 scale 对齐
        gt_median = torch.median(gt_depth)
        gen_median = torch.median(gen_depth)
        if gen_median > 0:
            scale = gt_median / gen_median
            gen_depth = gen_depth * scale

        # AbsRel 计算
        error_map = torch.abs(gen_depth - gt_depth) / (gt_depth + 1e-6)
        valid_mask = gt_depth > 1e-3 # 过滤掉过近的噪点
        video_error = error_map[valid_mask].mean() if valid_mask.any() else error_map.mean()

        video_results.append({
            'video_path': video_path,
            'video_results': video_error.item()
        })
        total_abs_rel += video_error.item()
        cnt += 1

    all_results = total_abs_rel / cnt if cnt > 0 else 0.0
    return all_results, video_results

def compute_depth_accuracy(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    gt_path = kwargs.get('gt_path', None)
    if not gt_path:
        raise ValueError("depth_accuracy 需要 gt_path 参数")

    # 注意：dimension 必须与 __init__.py 里的名称完全一致
    video_list = load_dimension_info(json_dir, dimension='depth_accuracy')
    
    if isinstance(submodules_list, dict):
        checkpoint_path = submodules_list.get('model', submodules_list)
    else:
        checkpoint_path = submodules_list

    depth_model = DepthEstimator(checkpoint_path, device)
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = depth_accuracy(depth_model, video_list, gt_path, device)

    # 分布式聚合
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        if len(video_results) > 0:
            all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
        else:
            all_results = 0.0
    
    return all_results, video_results
