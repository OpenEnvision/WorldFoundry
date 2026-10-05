from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from EWMBench.utils import clip_transform_nocrop, load_dimension_info, load_video
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.core.execution.device import get_current_torch_device
from worldfoundry.core.utils import batched_image_features, mean_pairwise_cosine_distance

from worldfoundry.core.distributed.collectives.evaluation import get_rank


def compute_cost_matrix(results):
    cost_matrices = {}

    for task_id, episodes in results.items():
        for episode_id, gids in episodes.items():
            gid_list = sorted(gids.keys(), key=int)
            avg_cost = mean_pairwise_cosine_distance([gids[gid] for gid in gid_list])

            if task_id not in cost_matrices:
                cost_matrices[task_id] = {}
            cost_matrices[task_id][episode_id] = avg_cost

    return cost_matrices


def diversity(clip_model, preprocess, video_list, device):
    del preprocess
    results = {}
    image_transform = clip_transform_nocrop(224)

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        parts = Path(video_path).parts
        task_id = parts[-4]
        episode_id = parts[-3]
        gid = parts[-2]

        if task_id not in results:
            results[task_id] = {}
        if episode_id not in results[task_id]:
            results[task_id][episode_id] = {}

        images = load_video(video_path)

        padded_images = []
        for image in images:
            max_h, max_w = image.shape[1], image.shape[2]
            max_dim = max(max_h, max_w)
            pad_w_left = (max_dim - max_w) // 2
            pad_w_right = max_dim - max_w - pad_w_left
            pad_h_top = (max_dim - max_h) // 2
            pad_h_bottom = max_dim - max_h - pad_h_top
            padded = F.pad(image, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), "constant", 0)
            padded_images.append(padded)

        images = image_transform(torch.stack(padded_images))
        image_features = batched_image_features(
            images,
            clip_model.encode_image,
            device=device,
            scope="clip_video",
            normalize=True,
            output_device=device,
        )
        video_feature = image_features.mean(dim=0).cpu()

        results[task_id][episode_id][gid] = video_feature

        del images, image_features, video_feature

    return compute_cost_matrix(results)


def compute_diversity(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()

    checkpoint_path = submodules_list["model"]

    clip_model, preprocess = clip.load(checkpoint_path, device=device)

    print("Loaded CLIP model for diversity evaluation")

    video_list = load_dimension_info(json_dir, dimension="diversity")

    return diversity(clip_model, preprocess, video_list, device)
