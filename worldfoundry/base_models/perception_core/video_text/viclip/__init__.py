"""ViCLIP inference helpers."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip import ViCLIP

SimpleTokenizer = _Tokenizer

_V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def get_viclip(size: str = "l", pretrain: str | os.PathLike | None = None) -> dict[str, object]:
    """Build a ViCLIP inference model and tokenizer."""
    if size.lower() != "l":
        raise ValueError("Only ViCLIP-L is integrated in WorldFoundry base_models.")
    tokenizer = _Tokenizer()
    model_path = str(pretrain) if pretrain is not None else str(checkpoint_path())
    return {"viclip": ViCLIP(tokenizer=tokenizer, pretrain=model_path), "tokenizer": tokenizer}


def checkpoint_path() -> Path:
    """Resolve the ViCLIP checkpoint from the shared VBench metric assets."""
    capability = BASE_MODEL_CAPABILITIES["vbench_metric_checkpoint_assets"]
    for asset in capability.assets:
        if asset.id == "vbench_viclip_checkpoint":
            status = asset.check()
            return Path(status["matched_path"] or status["local_path"])
    raise RuntimeError("vbench_viclip_checkpoint is not registered in BASE_MODEL_CAPABILITIES.")


def get_text_feat_dict(texts, clip, tokenizer, text_feat_d=None):
    if text_feat_d is None:
        text_feat_d = {}
    for text in texts:
        text_feat_d[text] = clip.get_text_features(text, tokenizer, text_feat_d)
    return text_feat_d


def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def _frame_from_video(video):
    return frame_from_video(video)


def normalize_frames(data):
    return (data / 255.0 - _V_MEAN) / _V_STD


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device("cuda")):  # noqa: B008 - Default evaluated once at definition, as upstream.
    if len(vid_list) < fnum:
        raise ValueError(f"ViCLIP expects at least {fnum} frames, got {len(vid_list)}.")
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(frame[:, :, ::-1], target_size) for frame in vid_list]
    vid_tube = [np.expand_dims(normalize_frames(frame), axis=(0, 1)) for frame in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    return torch.from_numpy(vid_tube).to(device, non_blocking=True).float()


def retrieve_text(frames, texts, models=None, topk=5, device=torch.device("cuda")):  # noqa: B008 - Default evaluated once at definition, as upstream.
    if not isinstance(models, dict) or models.get("viclip") is None or models.get("tokenizer") is None:
        raise ValueError("models must contain loaded 'viclip' and 'tokenizer' entries.")
    clip, tokenizer = models["viclip"], models["tokenizer"]
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, {})
    text_feats_tensor = torch.cat([text_feat_d[text] for text in texts], 0)
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)
    return [texts[i] for i in idxs.numpy()[0].tolist()], probs.numpy()[0]


__all__ = [
    "SimpleTokenizer",
    "ViCLIP",
    "_Tokenizer",
    "_frame_from_video",
    "checkpoint_path",
    "frame_from_video",
    "frames2tensor",
    "get_text_feat_dict",
    "get_viclip",
    "get_vid_feat",
    "normalize_frames",
    "retrieve_text",
]
