import torch
from utils.utils import clip_transform_Image
import torch.nn.functional as F

def visual_quality_metric(images, imaging_model, aesthetic_model, clip_model, batch_size=8, device='cuda:0'):
    result_dict = {}
    imaging_scores = []
    aesthetic_scores = []
    image_transform = clip_transform_Image(224)

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)

        imaging_batch = imaging_model(batch).cpu().tolist()
        imaging_batch = [item[0] for item in imaging_batch]
        imaging_scores.extend(imaging_batch)

        clip_batch = image_transform(batch)
        image_feats = clip_model.encode_image(clip_batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aesthetic_batch = aesthetic_model(image_feats).squeeze(dim=-1)
        aesthetic_scores.extend(aesthetic_batch.cpu().tolist())

        del batch, clip_batch, image_feats, aesthetic_batch
        torch.cuda.empty_cache()

    imaging_scores = [i/100 for i in imaging_scores]
    result_dict["imaging"] = imaging_scores
    result_dict["avg_imaging"] = sum(imaging_scores) / len(imaging_scores)

    aesthetic_scores = [i/10 for i in aesthetic_scores]
    result_dict["aesthetic"] = aesthetic_scores
    result_dict["avg_aesthetic"] = sum(aesthetic_scores) / len(aesthetic_scores)
    return result_dict

def merge_visual_results(result_dict, new_result):
    """合并两个visual quality结果字典"""
    if result_dict is None:
        return new_result
    if new_result is None:
        return result_dict

    result_dict['imaging'].extend(new_result['imaging'])
    result_dict['aesthetic'].extend(new_result['aesthetic'])
    result_dict['avg_imaging'] = sum(result_dict['imaging']) / len(result_dict['imaging'])
    result_dict['avg_aesthetic'] = sum(result_dict['aesthetic']) / len(result_dict['aesthetic'])
    return result_dict