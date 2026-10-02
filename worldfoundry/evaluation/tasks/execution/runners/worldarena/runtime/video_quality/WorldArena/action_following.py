"""CLIP feature diversity used by WorldArena action-following evaluation."""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.core.execution.device import get_current_torch_device
from worldfoundry.core.utils import batched_image_features, mean_pairwise_cosine_distance

from worldfoundry.core.distributed.collectives.evaluation import get_rank
from .utils import clip_transform_nocrop, load_dimension_info, load_video


def compute_cost_matrix(results):
    cost_matrices = {}
    for task_id, episodes in results.items():
        for episode_id, gids in episodes.items():
            gid_list = sorted(gids, key=int)
            cost_matrices.setdefault(task_id, {})[episode_id] = mean_pairwise_cosine_distance(
                [gids[gid] for gid in gid_list]
            )
    return cost_matrices


def action_following(clip_model, preprocess, video_list, device):
    del preprocess
    results = {}
    image_transform = clip_transform_nocrop(224)

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        parts = Path(video_path).parts
        task_id, episode_id, gid = parts[-4], parts[-3], parts[-2]
        results.setdefault(task_id, {}).setdefault(episode_id, {})

        images = load_video(video_path)
        padded_images = []
        for image in images:
            height, width = image.shape[1], image.shape[2]
            max_dim = max(height, width)
            pad_width_left = (max_dim - width) // 2
            pad_height_top = (max_dim - height) // 2
            padded_images.append(
                F.pad(
                    image,
                    (
                        pad_width_left,
                        max_dim - width - pad_width_left,
                        pad_height_top,
                        max_dim - height - pad_height_top,
                    ),
                    "constant",
                    0,
                )
            )

        images = image_transform(torch.stack(padded_images))
        image_features = batched_image_features(
            images,
            clip_model.encode_image,
            device=device,
            scope="clip_video",
            normalize=True,
            output_device=device,
        )
        results[task_id][episode_id][gid] = image_features.mean(dim=0).cpu()

    return compute_cost_matrix(results)


def compute_action_following(json_dir, submodules_list, **kwargs):
    del kwargs
    device = get_current_torch_device()
    clip_model, preprocess = clip.load(submodules_list["model"], device=device)
    print("Loaded CLIP model for action following evaluation")
    video_list = load_dimension_info(json_dir, dimension="action_following")
    return action_following(clip_model, preprocess, video_list, device)


__all__ = ["action_following", "compute_action_following", "compute_cost_matrix"]
