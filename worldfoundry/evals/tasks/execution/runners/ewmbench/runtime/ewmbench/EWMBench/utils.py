import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

CACHE_DIR = os.environ.get("VBENCH_CACHE_DIR")
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vbench")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def dino_transform(n_px):
    return Compose(
        [
            Resize(size=n_px, antialias=False),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def clip_transform_nocrop(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC, antialias=False),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def load_video(
    video_path, square_resize=None, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None
):
    file_list = sorted(os.listdir(video_path))
    img_files = [f for f in file_list if f.endswith((".jpg", ".jpeg", ".png"))]

    if len(img_files) >= 1:
        frames = []
        img_files = sorted(img_files)
        for fname in img_files:
            img = Image.open(os.path.join(video_path, fname))
            img = img.convert("RGB")
            img = np.array(img).astype(np.uint8)
            frames.append(img)
        buffer = np.array(frames).astype(np.uint8)
    else:
        raise NotImplementedError
    frames = buffer
    if num_frames and not video_path.endswith(".mp4"):
        frame_indices = get_frame_indices(num_frames, len(frames), sample="middle")
        frames = frames[frame_indices]

    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)

    return frames


def load_dimension_info(json_dir, dimension):
    video_list = []
    full_prompt_list = load_json(json_dir)

    file_path = Path(json_dir)
    file_stem = file_path.stem
    parts = file_stem.split("_")
    for prompt_dict in full_prompt_list:
        if parts[0] == "gt":
            if "video_list" in prompt_dict:
                cur_video_list = (
                    prompt_dict["video_list"]
                    if isinstance(prompt_dict["video_list"], list)
                    else [prompt_dict["video_list"]]
                )
                video_list += cur_video_list
        elif dimension in prompt_dict["dimension"] and "video_list" in prompt_dict:
            cur_video_list = (
                prompt_dict["video_list"]
                if isinstance(prompt_dict["video_list"], list)
                else [prompt_dict["video_list"]]
            )
            video_list += cur_video_list
    return video_list


def init_submodules(dimension_list, local=False, **kwargs):
    submodules_dict = {}

    if local:
        logger.info(
            "\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded."
        )

    os.makedirs(CACHE_DIR, exist_ok=True)

    for dimension in dimension_list:
        if dimension == "semantics":
            submodules_dict[dimension] = {
                "caption_model": kwargs.get(f"{dimension}_caption_model_ckpt", None),
                "clip_model": kwargs.get(f"{dimension}_clip_model_ckpt", None),
            }
        else:
            model_ckpt = kwargs.get(f"{dimension}_model_ckpt", None)

            submodules_dict[dimension] = {
                "model": model_ckpt,
            }

    return submodules_dict


def save_json(data, path, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
