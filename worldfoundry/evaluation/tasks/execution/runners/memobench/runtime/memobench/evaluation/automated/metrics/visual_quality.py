import numpy as np
import torch
import requests
import os
import cv2
from PIL import Image

from worldfoundry.evaluation.tasks.metrics._shared.aesthetic import (
    load_laion_aesthetic_mlp_head,
)
from worldfoundry.evaluation.tasks.metrics._shared.clip_embed import open_clip_bundle

_AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
_AESTHETIC_CACHE = os.path.expanduser("~/.cache/aesthetic_predictor.pth")

_aes_model = None
_aes_device = None


def _load_aesthetic(device: str = None):
    """Load the MemoBench aesthetic scorer.

    Provenance: the CLIP ViT-L/14 backbone and improved-aesthetic-predictor MLP
    head construction are delegated to the shared in-tree helpers
    (``metrics._shared.clip_embed`` / ``metrics._shared.aesthetic``). Only the
    MemoBench download-to-cache staging protocol stays here.
    """
    global _aes_model, _aes_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess, _ = open_clip_bundle("openai:ViT-L-14", device)

    if _aes_model is not None:
        return _aes_model, clip_model, preprocess, _aes_device
    _aes_device = device

    # Download aesthetic head weights if not cached
    if not os.path.exists(_AESTHETIC_CACHE):
        os.makedirs(os.path.dirname(_AESTHETIC_CACHE), exist_ok=True)
        print("[AestheticScore] Downloading LAION predictor weights (~55 MB)…")
        r = requests.get(_AESTHETIC_URL, timeout=60)
        r.raise_for_status()
        with open(_AESTHETIC_CACHE, "wb") as f:
            f.write(r.content)

    _aes_model = load_laion_aesthetic_mlp_head(_AESTHETIC_CACHE).to(device)

    return _aes_model, clip_model, preprocess, _aes_device


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def _aesthetic_score_single(aes_model, clip_model, preprocess, device, img: np.ndarray) -> float:
    t = preprocess(_bgr_to_pil(img)).unsqueeze(0).to(device)
    feat = clip_model.encode_image(t).float()
    feat = feat / feat.norm(dim=-1, keepdim=True)
    score = aes_model(feat).item()
    return float(score)


def aesthetic_score(frames, sample_step: int = 5, device: str = None) -> float:
    """
    Mean LAION aesthetic score over sampled frames. Higher is better (~0–10).
    """
    aes_model, clip_model, preprocess, device = _load_aesthetic(device)
    scores = []
    for idx in frames.iter_indices(0, frames.num_frames - 1, sample_step):
        img = frames.get(idx)
        scores.append(_aesthetic_score_single(aes_model, clip_model, preprocess, device, img))
    return round(float(np.mean(scores)), 4) if scores else 0.0


# CLIP-IQA+ via pyiqa; falls back to MUSIQ then Laplacian.
_iqa_metric = None
_iqa_device = None
_iqa_backend = None   # "clipiqa+" | "musiq" | "laplacian"


def _load_iqa(device: str = None):
    global _iqa_metric, _iqa_device, _iqa_backend
    if _iqa_metric is not None:
        return _iqa_metric, _iqa_device, _iqa_backend

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _iqa_device = device

    try:
        import pyiqa
        # CLIP-IQA+: CLIP-based perceptual IQA, no-reference, [0, 1]
        _iqa_metric = pyiqa.create_metric("clipiqa+", device=device)
        _iqa_backend = "clipiqa+"
        return _iqa_metric, _iqa_device, _iqa_backend
    except Exception:
        pass

    try:
        import pyiqa
        # MUSIQ fallback: multi-scale IQA trained on diverse distortions
        _iqa_metric = pyiqa.create_metric("musiq", device=device)
        _iqa_backend = "musiq"
        return _iqa_metric, _iqa_device, _iqa_backend
    except Exception:
        pass

    _iqa_backend = "laplacian"
    return None, device, "laplacian"


def _iqa_single(metric, backend: str, device: str, img: np.ndarray) -> float:
    """Score one frame → [0, 1], higher = better quality."""
    if backend == "laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        return float(1.0 - np.exp(-lap_var / 500.0))

    import torch
    from torchvision import transforms
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = float(metric(t).item())

    if backend == "clipiqa+":
        # CLIP-IQA+ output is already in [0, 1]
        return float(np.clip(raw, 0.0, 1.0))
    else:  # musiq: ~0–100
        return float(np.clip(raw / 100.0, 0.0, 1.0))


def image_quality_score(frames, sample_step: int = 5, device: str = None) -> float:
    """
    Perceptual image quality score in [0, 1]. Higher = better.

    Uses CLIP-IQA+ (Wang et al., AAAI 2023) — a CLIP-based no-reference IQA
    model assessing perceptual quality via prompt-pair contrastive learning.
    Sensitive to blocking/mosaic, blur, noise, and compression artifacts.
    Also adopted by WorldScore (Duan et al., ICCV 2025).
    Falls back to MUSIQ, then Laplacian sharpness.
    """
    metric, device, backend = _load_iqa(device)
    scores = []
    for idx in frames.iter_indices(0, frames.num_frames - 1, sample_step):
        scores.append(_iqa_single(metric, backend, device, frames.get(idx)))
    return round(float(np.mean(scores)), 4) if scores else 0.0


_ir_model = None
_ir_device = None
_ir_load_error = None   # stores error message on first load failure


def _load_image_reward(device: str = None):
    global _ir_model, _ir_device, _ir_load_error
    if _ir_load_error is not None:
        raise RuntimeError(_ir_load_error)
    if _ir_model is not None:
        return _ir_model, _ir_device

    try:
        import ImageReward as IR
    except ImportError:
        _ir_load_error = "ImageReward not installed. Run: pip install image-reward"
        raise RuntimeError(_ir_load_error)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _ir_model = IR.load("ImageReward-v1.0", device=device)
    except Exception as e:
        _ir_load_error = str(e)
        raise
    _ir_device = device
    return _ir_model, _ir_device


def image_reward_score(frames, prompt: str, n_sample: int = 5,
                       device: str = None) -> float:
    """
    ImageReward (2023) human-preference score averaged over sampled frames.

    Penalises low-quality, distorted, or hallucinated frames that don't match
    the text description — including mosaic artifacts, identity drift, and
    content mismatch. Only meaningful when a descriptive prompt is provided.

    Raw ImageReward scores (~-2 to +2) are mapped to [0, 1] via sigmoid:
        sigmoid(raw) = 1 / (1 + exp(-raw))
    so that score=0 maps to 0.5, and typical good outputs (raw≈1) map to ~0.73.

    Returns mean sigmoid-normalised score over n_sample uniformly spaced frames.
    Returns None if prompt is empty or ImageReward is not installed.
    """
    if not prompt or not prompt.strip():
        return None

    try:
        model, device = _load_image_reward(device)
    except Exception as e:
        import warnings
        warnings.warn(f"[ImageReward] {e}")
        return None

    N = frames.num_frames
    indices = [int(i * (N - 1) / (n_sample - 1)) for i in range(n_sample)]

    pil_frames = [
        Image.fromarray(cv2.cvtColor(frames.get(idx), cv2.COLOR_BGR2RGB))
        for idx in indices
    ]

    scores = []
    for pil in pil_frames:
        try:
            raw = model.score(prompt, [pil])
            # model.score returns a list; take first element
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            # sigmoid normalisation: maps (-∞,+∞) → (0,1)
            sig = float(1.0 / (1.0 + np.exp(-float(raw))))
            scores.append(sig)
        except Exception:
            continue

    return round(float(np.mean(scores)), 4) if scores else None
