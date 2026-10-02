from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from EWMBench.utils import dino_transform, load_dimension_info, load_video
from worldfoundry.base_models.perception_core.general_perception.dinov2.models import build_model_from_cfg
from worldfoundry.core.distributed.collectives.evaluation import (
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)
from worldfoundry.core.execution.device import get_current_torch_device
from worldfoundry.core.execution.inference_runtime import adaptive_batched_inference, resolve_inference_batch_size


def scene_consistency(model, video_list, device):
    model.eval()
    batch_size = resolve_inference_batch_size(32, device=device, scope="ewmbench_dino")
    sim = 0.0
    cnt = 0
    video_results = []

    image_transform = dino_transform(518)

    def encode(batch):
        return model.forward_features(batch)["x_norm_patchtokens"]

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0

        images = load_video(video_path)
        _, _, height, width = images.shape
        max_side = max(height, width)
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top
        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom))
        images = image_transform(padded_images)
        if len(images) < 2:
            video_results.append({
                "video_path": video_path,
                "video_results": 0.0,
                "video_sim": 0.0,
                "cnt_per_video": 0,
                "error": "insufficient_frames",
            })
            continue
        if torch.device(device).type == "cuda":
            images = images.pin_memory()

        image_features = adaptive_batched_inference(
            images,
            encode,
            batch_size=batch_size,
            device=device,
            pad_to_batch_size=True,
            scope="ewmbench_dino",
            persistent_forward=True,
        )
        image_features = F.normalize(image_features, dim=-1, p=2)
        adjacent = F.cosine_similarity(image_features[:-1], image_features[1:], dim=1).mean(dim=-1).clamp_min_(0)
        first = F.cosine_similarity(image_features[0:1], image_features[1:], dim=1).mean(dim=-1).clamp_min_(0)
        sim_per_images = float(((adjacent + first) * 0.5).mean().item())
        transition_count = len(image_features) - 1
        video_sim = sim_per_images * transition_count
        cnt += transition_count

        sim += video_sim
        video_results.append({
            "video_path": video_path,
            "video_results": sim_per_images,
            "video_sim": video_sim,
            "cnt_per_video": transition_count,
        })
    sim_per_frame = sim / cnt if cnt else 0.0
    return sim_per_frame, video_results


def compute_scene_consistency(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()

    config_path = submodules_list.get("config") or Path(__file__).resolve().parents[1] / "dino_config.yaml"
    checkpoint_path = submodules_list["model"]

    if checkpoint_path is None:
        raise ValueError("Checkpoint path must be provided in submodules_list")

    cfg = OmegaConf.load(str(config_path))
    dino_model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dino_model = dino_model.to(device)

    print(f"Loading model weights from: {checkpoint_path}")
    ori_state_dict = dino_model.state_dict()
    state_dict = torch.load(checkpoint_path)
    state_dict_toload = {}
    for k, v in state_dict.items():
        if k.startswith("teacher"):
            k_toload = k.replace("teacher.", "")
            k_toload = k_toload.replace("backbone.", "")
            if k_toload in ori_state_dict.keys():
                state_dict_toload.update({k_toload: v})
    print(dino_model.load_state_dict(state_dict_toload, strict=False))
    print("Initialize DINO success")

    video_list = load_dimension_info(json_dir, dimension="scene_consistency")
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = scene_consistency(dino_model, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum(d.get("video_sim", 0.0) for d in video_results)
        cnt = sum(d.get("cnt_per_video", 0) for d in video_results)
        all_results = sim / cnt if cnt else 0.0
    return all_results, video_results
