import os

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.core.distributed.collectives.evaluation import (
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)
from worldfoundry.core.execution.device import get_current_torch_device
from worldfoundry.core.execution.inference_runtime import adaptive_batched_inference, resolve_inference_batch_size
from worldfoundry.core.utils.tensors.torch import temporal_feature_consistency

from .dynamic_degree import DynamicDegree
from .utils import clip_transform, load_dimension_info, load_video


def background_consistency(clip_model, preprocess, video_list, device, read_frame, raft_model_path):
    clip_model.eval()
    batch_size = resolve_inference_batch_size(32, device=device, scope="worldarena_clip")
    sim = 0.0
    cnt = 0
    args_new = edict({
        "model": raft_model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    dynamic = DynamicDegree(args_new, device)
    video_results = []
    image_transform = clip_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        cnt_per_video = 0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(preprocess(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
        if len(images) < 2:
            video_results.append({
                'video_path': video_path,
                'video_results': 0.0,
                'video_sim': 0.0,
                'cnt_per_video': 0,
                'error': 'insufficient_frames',
            })
            continue
        if torch.device(device).type == "cuda":
            images = images.pin_memory()
        image_features = adaptive_batched_inference(
            images,
            clip_model.encode_image,
            batch_size=batch_size,
            device=device,
            pad_to_batch_size=True,
            scope="worldarena_clip",
        )
        image_features = F.normalize(image_features, dim=-1, p=2)
        sim_per_image = float(temporal_feature_consistency(image_features).item())
        cnt_per_video = len(image_features) - 1
        dynamic_score = dynamic.infer(video_path)
        if dynamic_score <= 0.1213:
            sim_per_image = sim_per_image * dynamic_score

        video_sim = sim_per_image * cnt_per_video
        sim += video_sim
        cnt += cnt_per_video
        video_results.append({
            'video_path': video_path, 
            'video_results': sim_per_image,
            'video_sim': video_sim,
            'cnt_per_video': cnt_per_video})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt if cnt else 0.0
    return sim_per_frame, video_results


def compute_background_consistency(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    vit_path = submodules_list.get('clip_model')
    raft_model_path = submodules_list.get('raft_model')
    read_frame = submodules_list.get('read_frame', False)

    if vit_path is None or raft_model_path is None:
        raise ValueError("background_consistency requires clip_model and raft_model checkpoints from config")

    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(json_dir, dimension='background_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = background_consistency(
        clip_model,
        preprocess,
        video_list,
        device,
        read_frame,
        raft_model_path,
    )
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum([d['video_sim'] for d in video_results])
        cnt = sum([d['cnt_per_video'] for d in video_results])
        all_results = sim / cnt if cnt else 0.0
    return all_results, video_results
