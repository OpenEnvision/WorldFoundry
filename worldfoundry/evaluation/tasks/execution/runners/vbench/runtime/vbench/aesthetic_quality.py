import torch
import torch.nn.functional as F
from tqdm import tqdm
from vbench.utils import clip_transform, load_dimension_info, load_video

from worldfoundry.base_models.perception_core.general_perception import openai_clip as clip
from worldfoundry.core.distributed.collectives.evaluation import (
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)
from worldfoundry.core.execution.inference_runtime import adaptive_batched_inference, resolve_inference_batch_size
from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_linear_head as get_aesthetic_model,
)

batch_size = 32


def laion_aesthetic(aesthetic_model, clip_model, video_list, device):
    aesthetic_model.eval()
    clip_model.eval()
    resolved_batch_size = resolve_inference_batch_size(batch_size, device=device, scope="vbench_aesthetic")
    image_transform = clip_transform(224)
    aesthetic_avg = 0.0
    num = 0
    video_results = []

    def score_batch(image_batch):
        image_feats = clip_model.encode_image(image_batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        return aesthetic_model(image_feats).squeeze(dim=-1)

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        images = image_transform(images)
        if torch.device(device).type == "cuda":
            images = images.pin_memory()

        aesthetic_scores = adaptive_batched_inference(
            images,
            score_batch,
            batch_size=resolved_batch_size,
            device=device,
            pad_to_batch_size=True,
            scope="vbench_aesthetic",
            persistent_forward=True,
        )
        normalized_aesthetic_scores = aesthetic_scores / 10
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
        aesthetic_avg += cur_avg.item()
        num += 1
        video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    aesthetic_avg /= num
    return aesthetic_avg, video_results


def compute_aesthetic_quality(json_dir, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    aes_path = submodules_list[1]
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
    
