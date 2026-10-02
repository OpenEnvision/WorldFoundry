import os
import json
import numpy as np
import logging
import torch
import re
import random
import cv2
from pathlib import Path
from PIL import Image, ImageSequence
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torchvision.transforms.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get('WORLDARENA_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'worldarena')


from worldfoundry.core.distributed.collectives.evaluation import (
    get_rank,
    barrier,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dino_transform(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_transform_Image(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_transform_nocrop(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def read_video_frames_cv2(video_path):
    """Read video frames (mp4 or image folder) into a RGB numpy list."""
    frames = []

    if os.path.isdir(video_path):
        exts = {
            "jpg", "jpeg", "png", "bmp", "tif", "tiff",
            "JPG", "JPEG", "PNG", "BMP", "TIF", "TIFF",
        }
        for fname in sorted(os.listdir(video_path)):
            if fname.split(".")[-1] not in exts:
                continue
            img = cv2.imread(os.path.join(video_path, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames

    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    raise NotImplementedError


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices



def load_video(video_path, square_resize=None, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    if os.path.isdir(video_path):
        file_list = sorted(os.listdir(video_path))
        img_files = [f for f in file_list if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) >= 1:
            frames = []
            img_files = sorted(img_files)
            for fname in img_files:
                img = Image.open(os.path.join(video_path, fname))
                img = img.convert('RGB')
                img = np.array(img).astype(np.uint8)
                frames.append(img)
            buffer = np.array(frames).astype(np.uint8)
        else:
            raise NotImplementedError
    elif video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        buffer = np.array([frame]).astype(np.uint8)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
                num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError

    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
            num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]

    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)

    return frames


    
def load_dimension_info(json_dir, dimension, lang=None):
    """
    Load video list and prompt information based on a specified dimension and language from a JSON file.
    
    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation to filter the video prompts.
    - lang (str): The language key used to retrieve the appropriate prompt text.
    
    Returns:
    - video_list (list): A list of video file paths that match the specified dimension.
    - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding video list.
    
    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension and compiles a list of video paths and associated prompts in the specified language.
    
    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
    - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
    """

    video_list = []
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)

    file_path = Path(json_dir)
    file_stem = file_path.stem
    parts = file_stem.split('_')

    if lang is None:
        for prompt_dict in full_prompt_list:
            if parts[0] == 'gt':
                if 'video_list' in prompt_dict:
                    cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
                    video_list += cur_video_list
            elif dimension in prompt_dict.get('dimension', []) and 'video_list' in prompt_dict:
                cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
                video_list += cur_video_list
        return video_list

    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict.get('dimension', []) and 'video_list' in prompt_dict:
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            prompt = prompt_dict.get(f'prompt_{lang}', prompt_dict.get('prompt', ''))
            if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
            else:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]

    return video_list, prompt_dict_ls

def init_submodules(dimension_list, local=False, **kwargs):
    submodules_dict = {}
    read_frame = kwargs.get("read_frame", False)

    if local:
        logger.info("\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded.")


    for dimension in dimension_list:
        if dimension == 'background_consistency':
            submodules_dict[dimension] = {
                'clip_model': kwargs.get(f"{dimension}_clip_ckpt"),
                'raft_model': kwargs.get(f"{dimension}_raft_ckpt"),
                'read_frame': read_frame,
            }

        elif dimension == 'dynamic_degree':
            submodules_dict[dimension] = {
                'model': kwargs.get(f"{dimension}_raft_ckpt"),
            }

        elif dimension == 'flow_score':
            submodules_dict[dimension] = {
                'model': kwargs.get(f"{dimension}_raft_ckpt"),
            }

        elif dimension == 'photometric_smoothness':
            submodules_dict[dimension] = {
                'config': kwargs.get(f"{dimension}_cfg_ckpt"),
                'model': kwargs.get(f"{dimension}_model_ckpt"),
            }

        elif dimension == 'motion_smoothness':
            submodules_dict[dimension] = {
                'model': kwargs.get(f"{dimension}_model_ckpt"),
            }

        elif dimension == 'subject_consistency':
            submodules_dict[dimension] = {
                'repo_or_dir': kwargs.get(f"{dimension}_repo_ckpt"),
                'path': kwargs.get(f"{dimension}_weight_ckpt"),
                'model': kwargs.get(f"{dimension}_model_name", 'dino_vitb16'),
                'source': 'local',
                'read_frame': read_frame,
                'raft_model': kwargs.get(f"{dimension}_raft_ckpt"),
            }

        elif dimension == 'aesthetic_quality':
            submodules_dict[dimension] = {
                'clip_model': kwargs.get(f"{dimension}_clip_ckpt"),
                'aesthetic_head': kwargs.get(f"{dimension}_head_ckpt"),
            }

        elif dimension in ['imaging_quality', 'image_quality']:
            # Handle legacy and new naming for the MUSIQ-based image quality metric
            submodules_dict[dimension] = {
                'model_path': kwargs.get(f"{dimension}_musiq_ckpt"),
            }

        elif dimension == 'semantic_alignment':
            submodules_dict[dimension] = {
                'caption_model': kwargs.get(f"{dimension}_caption_model_ckpt", None),
                'clip_model': kwargs.get(f"{dimension}_clip_model_ckpt", None)
            }

        else:
            model_ckpt = kwargs.get(f"{dimension}_model_ckpt", None)
            submodules_dict[dimension] = {
                'model': model_ckpt,
            }

    return submodules_dict



def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_json(path):
    """
    Load a JSON file from the given file path.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
