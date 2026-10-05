import torch
import torch.nn.functional as F
from .utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm
from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.core.execution.device import get_current_torch_device

from worldfoundry.core.distributed.collectives.evaluation import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_linear_head,
)

batch_size = 32


def get_aesthetic_model(cache_folder_or_file):
    """Load the aesthetic linear head (delegates to the shared in-tree loader)."""
    if cache_folder_or_file is None:
        raise ValueError("aesthetic_quality_head_ckpt is required")
    return load_laion_aesthetic_linear_head(cache_folder_or_file)


def laion_aesthetic(aesthetic_model, clip_model, video_list, device):
    aesthetic_model.eval()
    clip_model.eval()
    aesthetic_avg = 0.0
    num = 0
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        image_transform = clip_transform(224)

        aesthetic_scores_list = []
        for i in range(0, len(images), batch_size):
            image_batch = images[i:i + batch_size]
            image_batch = image_transform(image_batch)
            image_batch = image_batch.to(device)

            with torch.no_grad():
                image_feats = clip_model.encode_image(image_batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

            aesthetic_scores_list.append(aesthetic_scores)

        aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
        normalized_aesthetic_scores = aesthetic_scores / 10
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
        aesthetic_avg += cur_avg.item()
        num += 1
        video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    aesthetic_avg /= num
    return aesthetic_avg, video_results


def compute_aesthetic_quality(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    vit_path = submodules_list.get('clip_model')
    aes_path = submodules_list.get('aesthetic_head')
    if vit_path is None or aes_path is None:
        raise ValueError("aesthetic_quality requires clip_model and aesthetic_head checkpoints from config")
    if get_rank() == 0:
        aesthetic_model = get_aesthetic_model(aes_path).to(device)
        barrier()
    else:
        barrier()
        aesthetic_model = get_aesthetic_model(aes_path).to(device)
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(json_dir, dimension='aesthetic_quality', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = laion_aesthetic(aesthetic_model, clip_model, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
