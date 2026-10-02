import io
import os
import re
import yaml
import cv2
import json
import random
import numpy as np
import logging

from PIL import Image, ImageSequence
from tqdm import tqdm
from pathlib import Path
from bisect import bisect_left

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from worldfoundry.base_models.perception_core.detection.yolo_world import config_path as yolo_world_config_path
from worldfoundry.base_models.perception_core.human_anatomy.vit_detector import config_path as vit_detector_config_path
from worldfoundry.base_models.perception_core.tracking.cotracker import checkpoint_path as cotracker_checkpoint_path
from worldfoundry.base_models.perception_core.optical_flow.raft import checkpoint_path as raft_checkpoint_path
from worldfoundry.base_models.capabilities import vbench_asset_path
from collections import defaultdict
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

CACHE_DIR = os.environ.get('VBENCH2_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench2')
from worldfoundry.core.distributed.collectives.evaluation import (
    get_rank,
    barrier,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return scene_list


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

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    if video_path.endswith('.gif'):
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
        frame_ls = [frame]
        buffer = np.array(frame_ls)
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
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
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
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def read_frames_decord_by_fps(
        video_path, sample_fps=2, sample='rand', fix_start=None, 
        max_num_frames=-1,  trimmed30=False, num_frames=8
    ):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_frames(video_path):
    frame_list = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) # get fps
    interval = 1
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    while video.isOpened():
        success, frame = video.read()
        if success:
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)  # convert to rgb
            image_pil = Image.fromarray(image_np)
            frame = preprocess(image_pil)
            frame = frame[None]
            frame_list.append(frame)
        else:
            break
    video.release()
    assert frame_list != []
    frame_list = extract_frame(frame_list, interval)
    return frame_list 
    
    
def extract_frame(frame_list, interval=1):
    extract = []
    for i in range(0, len(frame_list), interval):
        extract.append(frame_list[i])
    return extract

    
def load_dimension_info(json_dir, dimension, lang):
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
    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict['dimension'][0].lower() and 'video_list' in prompt_dict:
            prompt = prompt_dict[f'prompt_{lang}']
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            if 'auxiliary_info' in prompt_dict:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info']}]
            else:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
    return video_list, prompt_dict_ls

def init_submodules(dimension_list, local=False, read_frame=False):
    submodules_dict = {}
    if local:
        logger.info("\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded.")
    for dimension in dimension_list:
        os.makedirs(CACHE_DIR, exist_ok=True)

        if dimension == 'Multi-View_Consistency':
            submodules_dict[dimension] = {
                'raft': str(raft_checkpoint_path()),
                "checkpoint": str(cotracker_checkpoint_path()),
            }
                
        elif dimension == 'Camera_Motion':
            submodules_dict[dimension] = {
                "checkpoint": str(cotracker_checkpoint_path()),
            }
            
        elif dimension == 'Human_Identity':
            submodules_dict[dimension] = {
                "model": str(vbench_asset_path("vbench2_arcface_checkpoint")),
                "retina": str(vbench_asset_path("vbench_retinaface_checkpoint")),
            }
        
        elif dimension == 'Instance_Preservation':
            submodules_dict[dimension] = {}
        
        elif dimension == 'Human_Anatomy':
            default_config = str(vit_detector_config_path())
            submodules_dict[dimension] = {
                "detector_config": str(yolo_world_config_path()),
                "detector_weights": str(vbench_asset_path("vbench2_yolo_world_checkpoint")),
                "analyzer_configs": {
                    "human": {"cfg_path": default_config, "weight_path": str(vbench_asset_path("vbench2_human_anomaly_checkpoint")), "threshold": 0.4545454545454546},
                    "face": {"cfg_path": default_config, "weight_path": str(vbench_asset_path("vbench2_face_anomaly_checkpoint")), "threshold": 0.30303030303030304},
                    "hand": {"cfg_path": default_config, "weight_path": str(vbench_asset_path("vbench2_hand_anomaly_checkpoint")), "threshold": 0.3232}
                },
                "batch_size" : 128
            }
                    
        elif dimension in ["Human_Clothes", "Composition", "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Motion_Rationality", "Mechanics", "Thermotics", "Material"]:
            submodules_dict[dimension] = {"llava": str(vbench_asset_path("vbench2_llava_video_qwen2_model"))}
                
        elif dimension in ["Complex_Landscape", "Complex_Plot", "Human_Interaction", "Motion_Order_Understanding"]:
            submodules_dict[dimension] = {
                "llava": str(vbench_asset_path("vbench2_llava_video_qwen2_model")),
                "qwen": str(vbench_asset_path("vbench2_qwen2_5_7b_model"))
            }
        else:
            submodules_dict[dimension]={}

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

def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    """
    prompt = Path(path).stem
    number_ending = r'-\d+$' # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, '', prompt)
    return prompt
