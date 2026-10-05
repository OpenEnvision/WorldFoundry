import logging
import os

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

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
from .utils import dino_transform, dino_transform_Image, load_dimension_info, load_video

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def subject_consistency(model, video_list, device, read_frame, raft_model_path):
    model.eval()
    batch_size = resolve_inference_batch_size(32, device=device, scope="worldarena_dino")
    sim = 0.0
    cnt = 0
    video_results = []
    args_new = edict({
        "model": raft_model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    dynamic = DynamicDegree(args_new, device)
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
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)
        if torch.device(device).type == "cuda":
            images = images.pin_memory()
        image_features = adaptive_batched_inference(
            images,
            model,
            batch_size=batch_size,
            device=device,
            pad_to_batch_size=True,
            scope="worldarena_dino",
        )
        image_features = F.normalize(image_features, dim=-1, p=2)
        sim_per_images = float(temporal_feature_consistency(image_features).item())
        transition_count = len(image_features) - 1
        dynamic_score = dynamic.infer(video_path)

        # ===== 唯一耦合点：阈值判断 =====
        if dynamic_score <= 0.1213:
            sim_per_images = sim_per_images * dynamic_score

        video_sim = sim_per_images * transition_count
        sim += video_sim
        cnt += transition_count
        video_results.append({
            'video_path': video_path,
            'video_results': sim_per_images,
            'video_sim': video_sim,
            'cnt_per_video': transition_count,
        })
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt if cnt else 0.0
    return sim_per_frame, video_results


def compute_subject_consistency(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    submodules_kwargs = dict(submodules_list)
    read_frame = submodules_kwargs.pop('read_frame', False)
    raft_model_path = submodules_kwargs.pop('raft_model', None)
    dino_weight_path = submodules_kwargs.pop('path', None)
    if raft_model_path is None:
        raise ValueError("subject_consistency requires raft_model checkpoint from config")

    # Always construct model from local repo without triggering hub URL download.
    # Then load checkpoint weights from config ckpt.subject_consistency.weight.
    if dino_weight_path is None:
        raise ValueError("subject_consistency requires local dino weight path from config")
    if not os.path.isfile(dino_weight_path):
        raise FileNotFoundError(f"subject_consistency dino weight not found: {dino_weight_path}")

    dino_model = torch.hub.load(pretrained=False, **submodules_kwargs).to(device)

    ckpt = torch.load(dino_weight_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = ckpt['state_dict']
        elif 'teacher' in ckpt and isinstance(ckpt['teacher'], dict):
            state_dict = ckpt['teacher']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # remove possible wrappers from different training/export styles
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('backbone.'):
            nk = nk[len('backbone.'):]
        cleaned_state_dict[nk] = v

    missing, unexpected = dino_model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"DINO missing keys when loading local ckpt: {len(missing)}")
    if unexpected:
        logger.warning(f"DINO unexpected keys when loading local ckpt: {len(unexpected)}")

    dino_model.eval()
    logger.info("Initialize DINO success")
    video_list, _ = load_dimension_info(json_dir, dimension='subject_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = subject_consistency(
        dino_model,
        video_list,
        device,
        read_frame,
        raft_model_path,
    )
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum(d.get('video_sim', 0.0) for d in video_results)
        cnt = sum(d.get('cnt_per_video', 0) for d in video_results)
        all_results = sim / cnt if cnt else 0.0
    return all_results, video_results
