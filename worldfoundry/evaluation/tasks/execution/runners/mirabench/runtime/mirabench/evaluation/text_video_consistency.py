import os
import json
import random
from weakref import WeakKeyDictionary

import numpy as np

import torch
from worldfoundry.base_models.perception_core.video_text.viclip import SimpleTokenizer, ViCLIP
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
from decord import VideoReader, cpu

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

_TEXT_FEATURE_CACHES = WeakKeyDictionary()


def get_text_features(model, input_text, tokenizer):
    del tokenizer
    text_feature_dict = _TEXT_FEATURE_CACHES.setdefault(model, {})
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)      
        text_feature_dict[input_text] = text_features
    return text_features

def get_vid_features(model, input_frames):
    with torch.inference_mode():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

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


def read_frames_decord_by_fps(
        video_path, sample_fps=2, sample='rand', fix_start=None, 
        max_num_frames=-1,  trimmed30=False, num_frames=8
    ):
    import decord
    decord.bridge.set_bridge("torch")
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
    

def get_video_feature(clip_model, video_path, device, sample="middle"):
    image_transform = clip_transform(224)
    with torch.inference_mode():
        images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
        images = image_transform(images)
        images = images.to(device, non_blocking=True)
        return get_vid_features(clip_model, images.unsqueeze(0))


def EvaluateTextVideoConsistency(
    clip_model,
    video_path,
    tokenizer,
    device,
    text,
    sample="middle",
    video_feature=None,
):
    query = text
    with torch.inference_mode():
        clip_feat = (
            video_feature
            if video_feature is not None
            else get_video_feature(clip_model, video_path, device, sample=sample)
        )
        text_feat = get_text_features(clip_model, query, tokenizer)
        logit_per_text =  clip_feat @ text_feat.T
        score_per_video = float(logit_per_text[0][0].item())
    return score_per_video
