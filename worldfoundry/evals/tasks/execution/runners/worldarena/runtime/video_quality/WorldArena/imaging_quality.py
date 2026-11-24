from pyiqa.archs.musiq_arch import MUSIQ
from torchvision import transforms
from tqdm import tqdm

# isort: split
from .utils import load_dimension_info, load_video

# isort: split
from worldfoundry.core.distributed.collectives.evaluation import (
    distribute_list_to_rank,
    gather_list_of_dict,
    get_rank,
    get_world_size,
)
from worldfoundry.core.platform.device import get_current_torch_device


def transform(images, preprocess_mode="shorter"):
    if preprocess_mode.startswith("shorter"):
        _, _, h, w = images.size()
        if min(h, w) > 512:
            scale = 512.0 / min(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(images)
            if preprocess_mode == "shorter_centercrop":
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == "longer":
        _, _, h, w = images.size()
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(images)

    elif preprocess_mode == "None":
        return images / 255.0

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.0


def technical_quality(model, video_list, device, **kwargs):
    preprocess_mode = kwargs.get("imaging_quality_preprocessing_mode", "longer")
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        images = transform(images, preprocess_mode)
        acc_score_video = 0.0
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
        video_results.append({"video_path": video_path, "video_results": acc_score_video / len(images)})
    average_score = sum([o["video_results"] for o in video_results]) / len(video_results)
    average_score = average_score / 100.0
    return average_score, video_results


def compute_imaging_quality(json_dir, submodules_list, **kwargs):
    device = get_current_torch_device()
    model_path = submodules_list["model_path"]

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False

    video_list, _ = load_dimension_info(json_dir, dimension="image_quality", lang="en")
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = technical_quality(model, video_list, device, **kwargs)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)

    for video_result in video_results:
        video_result["video_results"] = video_result["video_results"] / 100

    return all_results, video_results
