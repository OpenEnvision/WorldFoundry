import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from vbench.utils import dino_transform, dino_transform_Image, load_dimension_info, load_video

from worldfoundry.core.distributed.collectives.evaluation import (
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)
from worldfoundry.core.execution.inference_runtime import adaptive_batched_inference, resolve_inference_batch_size
from worldfoundry.core.utils.tensors.torch import temporal_feature_consistency

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def subject_consistency(model, video_list, device, read_frame):
    model.eval()
    batch_size = resolve_inference_batch_size(32, device=device, scope="vbench_dino")
    sim = 0.0
    cnt = 0
    video_results = []
    if read_frame:
        image_transform = dino_transform_Image(224)
    else:
        image_transform = dino_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
        if len(images) < 2:
            raise ValueError("subject consistency requires at least two frames")
        if torch.device(device).type == "cuda":
            images = images.pin_memory()
        image_features = adaptive_batched_inference(
            images,
            model,
            batch_size=batch_size,
            device=device,
            pad_to_batch_size=True,
            scope="vbench_dino",
        )
        image_features = F.normalize(image_features, dim=-1, p=2)
        sim_per_images = float(temporal_feature_consistency(image_features).item())
        transition_count = len(image_features) - 1
        video_sim = sim_per_images * transition_count
        cnt += transition_count
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_subject_consistency(json_dir, device, submodules_list, **kwargs):
    dino_model = torch.hub.load(**submodules_list).to(device)
    read_frame = submodules_list['read_frame']
    logger.info("Initialize DINO success")
    video_list, _ = load_dimension_info(json_dir, dimension='subject_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = subject_consistency(dino_model, video_list, device, read_frame)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
