import os
import torch
from PIL import Image, ImageSequence
# from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import functional as F
import json
from tqdm import tqdm
import numpy as np
import cv2
import subprocess
from pathlib import Path
import hashlib
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

CACHE_DIR = os.environ.get('MIND_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'mind')

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
        #ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_musiq_spaq_path():
    musiq_spaq_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
    if not os.path.isfile(musiq_spaq_path):
        wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(musiq_spaq_path)]
        subprocess.run(wget_command, check=True)
    return musiq_spaq_path

def get_aes_path():
    aes_path = f'{CACHE_DIR}/vitl_model/sa_0_4_vit_l_14_linear.pth'
    if not os.path.isfile(aes_path):
        os.makedirs(os.path.dirname(aes_path), exist_ok=True)
        wget_command = ['wget' ,'https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true', '-O', aes_path]
        subprocess.run(wget_command, check=True)
    return aes_path

def get_vitl_path():
    vit_l_path = f'{CACHE_DIR}/clip_model/ViT-L-14.pt'
    if not os.path.isfile(vit_l_path):
        wget_command = ['wget' ,'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', '-P', os.path.dirname(vit_l_path)]
        subprocess.run(wget_command, check=True)
    return vit_l_path

def ensure_all_models_downloaded():
    """在并行前预先下载所有模型文件，避免多进程冲突"""
    from tqdm import tqdm
    tqdm.write("Pre-downloading model files...")

    download_tasks = [
        ("MUSIQ", get_musiq_spaq_path),
        ("Aesthetic", get_aes_path),
        ("ViT-L/CLIP", get_vitl_path),
    ]

    for name, download_fn in tqdm(download_tasks, desc="Downloading models"):
        try:
            path = download_fn()
            tqdm.write(f"  {name}: {os.path.isfile(path)}")
        except Exception as e:
            tqdm.write(f"  {name} download failed: {e}")

    tqdm.write("All models ready.\n")

def transform_image(images):
    # 输入: images (Tensor) - 形状为 [B, C, H, W], 范围[0,255]
    # 输出: 归一化后, 范围[0,1]的 [B, C, 1280, 720] Tensor
    b, c, h, w = images.size()
    if h * 1280 == w * 720:
        images = transforms.Resize(size=(720, 1280), antialias=False)(images)
        return images / 255.0
    else:
        scale = max(1280.0/w, 720.0/h)
        images = transforms.Resize(size=(round(h*scale), round(w*scale)), antialias=False)(images)
        images = F.center_crop(images, (720, 1280))
        return images / 255.0

def min_max_normalization(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(value - min_value) / (max_value - min_value) for value in data]
    return normalized_data

def save_video(tensor, output_path):
    frames = tensor.permute(0, 2, 3, 1).numpy()
    frames = (frames * 255).astype(np.uint8)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()

def expand_to_batch_dim(tensor, batch_size):
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

# legacy code, preserve
from torchvision.io import read_video
import av

def load_sample_video(video_path, mark_time, total_time, max_time = None) -> torch.Tensor:
    video_length = total_time - mark_time
    frames,_,_ = read_video(video_path, pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2)[:video_length]
    if max_time is not None:
        frames = frames[:max_time]
    return transform_image(frames)

def load_gt_video(video_path, mark_time, total_time, max_time = None) -> torch.Tensor:
    start_time = mark_time // 24 - 1
    frames,_,_ = read_video(video_path, pts_unit='sec', start_pts=start_time)

    frames = frames.permute(0, 3, 1, 2)
    video_length = total_time - mark_time
    frames = frames[len(frames) - video_length:]
    if max_time is not None:
        frames = frames[:max_time]
    return transform_image(frames)

def get_video_length(video_path) -> int:
    """获取视频帧数"""
    with av.open(video_path) as container:
        return container.streams.video[0].frames

class VideoStreamReader:
    """使用av进行流式视频读取，内存安全"""
    def __init__(self, video_path, start_frame=0, total_frames=None):
        self.video_path = video_path
        self.container = av.open(video_path)
        self.video_stream = self.container.streams.video[0]

        stream_total_frames = self.video_stream.frames
        self.total_frames = stream_total_frames if total_frames is None else total_frames
        self.current_pos = start_frame
        self.yielder = self.container.decode(video=0)
        tqdm.write(f"[VideoStreamReader] {video_path}: stream_total={stream_total_frames}, start_frame={start_frame}, total_frames={self.total_frames}")

        if start_frame > 0:
            tqdm.write(f"[VideoStreamReader] Skipping {start_frame} frames...")
            for _ in range(start_frame):
                frame = next(self.yielder, None)
                if frame is None:
                    break

    def read_batch(self, batch_size):
        """读取一批帧，返回(is_ended, frames_tensor)"""
        if self.current_pos >= self.total_frames:
            return True, None

        frames_to_read = min(batch_size, self.total_frames - self.current_pos)
        frames_list = []

        for i in range(frames_to_read):
            try:
                frame = next(self.yielder, None)
            except Exception as e:
                tqdm.write(f"[VideoStreamReader] ERROR: Exception at read_batch frame {i}/{frames_to_read}, current_pos={self.current_pos}: {e}")
            if frame is None:
                break
            self.current_pos += 1
            frames_list.append(torch.from_numpy(frame.to_rgb().to_ndarray()).float())

        if not frames_list:
            return True, None

        frames_tensor = torch.stack(frames_list).permute(0, 3, 1, 2)
        frames_tensor = transform_image(frames_tensor)

        is_ended = self.current_pos >= self.total_frames
        frames_tensor = torch.clamp(frames_tensor, 0.0, 1.0)
        return is_ended, frames_tensor

    def __del__(self):
        if hasattr(self, 'container'):
            self.container.close()

def load_time_from_json(json_path):
    with open(json_path, 'r') as f:
        item = json.load(f)
    return item['mark_time'], item['total_time']

def extract_actions_from_json(json_path, mark_time=None, video_max_time=97):
    with open(json_path, 'r') as f:
        action_data = json.load(f)

    if mark_time is None:
        mark_time = action_data['mark_time']

    data = action_data['data']
    end_time = min(mark_time + video_max_time, len(data))

    actions = []
    for i in range(mark_time, end_time):
        frame = data[i]
        ws = frame['ws']
        ad = frame['ad']
        ud = frame['ud']
        lr = frame['lr']

        action_parts = []

        if ws == 1:
            action_parts.append('forward')
        elif ws == 2:
            action_parts.append('backward')

        if ad == 1:
            action_parts.append('left')
        elif ad == 2:
            action_parts.append('right')

        if ud == 1:
            action_parts.append('look_up')
        elif ud == 2:
            action_parts.append('look_down')

        if lr == 1:
            action_parts.append('look_left')
        elif lr == 2:
            action_parts.append('look_right')

        if len(action_parts) == 0:
            actions.append('no_op')
        else:
            actions.append('+'.join(action_parts))

    return actions

def print_gpu_memory():
    tqdm.write(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    tqdm.write(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def crop_video_frames(video_path, max_frames, cache_dir, start_frame=0):
    """
    Crop video to first max_frames frames using ffmpeg.
    Returns path to cropped video (from cache if exists).
    
    Args:
        video_path: Path to original video
        max_frames: Number of frames to keep
        cache_dir: Directory for caching cropped videos
        start_frame: Starting frame position for cropping (default: 0)
    
    Returns:
        Path to cropped video file
    """
    video_path = Path(video_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    
    # Generate cache filename based on original path and max_frames
    video_hash = hashlib.sha1(str(video_path).encode()).hexdigest()[:12]
    cache_filename = f"{video_path.stem}_{video_hash}_start{start_frame}_frames{max_frames}.mp4"
    cache_path = cache_dir / "cropped_videos" / cache_filename
    
    # Check if cached version exists
    if cache_path.exists():
        tqdm.write(f"Using cached cropped video: {cache_path}")
        return cache_path

    # Create cache directory
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Cropping video {video_path.name} to {max_frames} frames starting from frame {start_frame}...")
    
    if start_frame == 0:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vframes", str(max_frames),
            "-c:v", "libx264", "-crf", "18",
            "-y", str(cache_path)
        ]
    else:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select=gte(n\,{start_frame})",  # 从指定帧开始
            "-vframes", str(max_frames),
            "-c:v", "libx264",
            "-crf", "18",
            "-y", str(cache_path)
        ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        tqdm.write(f"Cropped video saved to: {cache_path}")
    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error cropping video: {e.stderr}")
        raise
    
    return cache_path