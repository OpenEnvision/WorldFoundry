import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_dino = None
_dino_device = None

_DINO_TRANSFORM = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_dino(device: str | None = None) -> tuple:
    global _dino, _dino_device
    if _dino is not None:
        return _dino, _dino_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _dino_device = device

    _dino = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
        verbose=False,
    )
    _dino = _dino.to(device).eval()
    return _dino, _dino_device


@torch.no_grad()
def _dino_embed(img: np.ndarray, model, device: str) -> torch.Tensor:
    """Extract L2-normalised DINOv2 CLS token from a BGR frame."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = _DINO_TRANSFORM(pil).unsqueeze(0).to(device)
    feat = model(t)  # [1, 768]
    return F.normalize(feat, dim=-1)  # unit sphere


@torch.no_grad()
def _dino_patch_embed(img: np.ndarray, model, device: str) -> torch.Tensor:
    """Extract L2-normalised DINOv2 patch tokens from a BGR frame.

    Returns (N_patches, D) tensor where N_patches = (224/14)^2 = 256.
    Uses forward_features() to obtain per-patch representations, which
    capture local spatial information rather than just global CLS.
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = _DINO_TRANSFORM(pil).unsqueeze(0).to(device)
    feats = model.forward_features(t)  # dict with 'x_norm_patchtokens'
    patches = feats["x_norm_patchtokens"]  # [1, N, D]
    patches = F.normalize(patches[0], dim=-1)  # [N, D] unit norm per patch
    return patches


def _dino_similarity(f0: torch.Tensor, fk: torch.Tensor) -> float:
    """Cosine similarity between two L2-normalised DINOv2 embeddings → [0, 1]."""
    sim = float(F.cosine_similarity(f0, fk).item())
    return float(np.clip(sim, 0.0, 1.0))


def identity_consistency(
    frames, n_sample: int = 9, device: str | None = None, start: int = 0, end: int | None = None
) -> dict:
    """
    Visual identity consistency via DINOv2 CLS token cosine similarity.

    Samples n_sample frames uniformly from [start, end]. For each sampled
    frame, computes the DINOv2 cosine similarity against frame_0 (the anchor).

    For phase-aware V-D-R evaluation pass start=r_start, end=N-1 to measure
    whether visual identity survived the hidden interval (frame_0 vs R phase).

    Returns:
        identity_consistency     : mean cosine similarity (frame_0 vs sampled frames)
        min_identity_consistency : minimum per-frame similarity (worst-case drift)
    """
    model, device = _load_dino(device)

    N = frames.num_frames
    if end is None:
        end = N - 1
    start = max(0, min(start, N - 1))
    end = max(start, min(end, N - 1))
    n_range = end - start + 1

    if n_sample <= 1 or n_range <= 1:
        indices = [start]
    else:
        indices = [start + int(i * (n_range - 1) / (n_sample - 1)) for i in range(n_sample)]

    # Anchor: always frame_0
    f0 = _dino_embed(frames.get(0), model, device)

    sims = []
    for idx in indices:
        fk = _dino_embed(frames.get(idx), model, device)
        sims.append(_dino_similarity(f0, fk))

    if not sims:
        return {"identity_consistency": 1.0, "min_identity_consistency": 1.0}

    return {
        "identity_consistency": round(float(np.mean(sims)), 4),
        "min_identity_consistency": round(float(np.min(sims)), 4),
    }


def object_centric_identity_consistency(
    frames,
    n_sample: int = 9,
    top_k_frac: float = 0.4,
    device: str | None = None,
    start: int = 0,
    end: int | None = None,
) -> dict:
    """
    Object-centric visual identity consistency using DINOv2 patch tokens.

    Unlike identity_consistency() which uses the global CLS token (dominated
    by background), this metric focuses on the high-similarity patch regions,
    which tend to correspond to the persistent foreground target object.

    Method:
      1. Extract per-patch DINOv2 features for frame_0 → anchor patch tokens.
      2. For each sampled frame k, compute per-patch cosine similarity with
         the corresponding patch in frame_0: sim_patches [N_patches].
      3. Take the top top_k_frac fraction of patches by similarity
         (the "stable" regions — likely the target object rather than
         dynamic background).
      4. Report mean and min of those top-K patch similarities.

    Returns:
      obj_identity_consistency     : mean of top-K patch similarities
      obj_min_identity_consistency : min of top-K patch similarities
    """
    model, device = _load_dino(device)

    N = frames.num_frames
    if end is None:
        end = N - 1
    start = max(0, min(start, N - 1))
    end = max(start, min(end, N - 1))
    n_range = end - start + 1

    if n_sample <= 1 or n_range <= 1:
        indices = [start]
    else:
        indices = [start + int(i * (n_range - 1) / (n_sample - 1)) for i in range(n_sample)]

    # Anchor: always frame_0 patch tokens
    p0 = _dino_patch_embed(frames.get(0), model, device)  # [N_patches, D]
    N_patches = p0.shape[0]
    top_k = max(1, int(N_patches * top_k_frac))

    mean_sims, min_sims = [], []
    for idx in indices:
        pk = _dino_patch_embed(frames.get(idx), model, device)  # [N_patches, D]
        # Per-patch cosine similarity (both are already L2-normalised)
        patch_sims = (p0 * pk).sum(dim=-1).clamp(0.0, 1.0)  # [N_patches]
        # Focus on top-K most stable patches
        top_vals, _ = patch_sims.topk(top_k)
        mean_sims.append(float(top_vals.mean().item()))
        min_sims.append(float(top_vals.min().item()))

    if not mean_sims:
        return {
            "obj_identity_consistency": 1.0,
            "obj_min_identity_consistency": 1.0,
        }

    return {
        "obj_identity_consistency": round(float(np.mean(mean_sims)), 4),
        "obj_min_identity_consistency": round(float(np.min(min_sims)), 4),
    }


_depth_model = None
_depth_processor = None
_depth_device = None


def _load_depth_anything(device: str | None = None):
    global _depth_model, _depth_processor, _depth_device
    if _depth_model is not None:
        return _depth_model, _depth_processor, _depth_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _depth_device = device

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    _depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    _depth_model = (
        AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    )

    return _depth_model, _depth_processor, _depth_device


@torch.no_grad()
def _depth_embed(model, processor, device: str, img_bgr: np.ndarray) -> torch.Tensor:
    """Estimate depth with Depth Anything V2, min-max normalise, L2-normalise flat vector."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    depth = model(**inputs).predicted_depth.squeeze()  # [H, W]
    d_min, d_max = depth.min(), depth.max()
    depth = (depth - d_min) / (d_max - d_min + 1e-8)  # [0, 1]
    flat = depth.flatten().unsqueeze(0)  # [1, H*W]
    return F.normalize(flat, dim=-1)


def _depth_similarity(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Cosine similarity between two L2-normalised depth vectors → [0, 1]."""
    sim = float(F.cosine_similarity(d1, d2).item())
    return float(np.clip(sim, 0.0, 1.0))


def geometry_3d_consistency(
    frames, device: str | None = None, n_sample: int = 9, start: int = 0, end: int | None = None
) -> dict:
    """
    Geometric 3D consistency via Depth Anything V2 cosine similarity.

    For each consecutive pair of n_sample uniformly spaced frames within
    [start, end], estimates a dense depth map using Depth Anything V2 ViT-S
    (Yang et al., NeurIPS 2024), min-max normalises it to [0, 1], and computes
    cosine similarity between the flattened depth vectors.

    High similarity → 3D scene geometry is temporally stable.
    Low similarity  → scene depth structure is inconsistent or incoherent.

    For phase-aware V-D-R evaluation call once for V phase and once for R phase
    (excluding D phase where the camera deliberately moves away).

    Returns:
        geo_consistency     : mean cosine similarity across consecutive sampled pairs
        min_geo_consistency : minimum cosine similarity (worst-case geometry break)
    """
    model, processor, device = _load_depth_anything(device)
    N = frames.num_frames
    if end is None:
        end = N - 1
    start = max(0, min(start, N - 1))
    end = max(start, min(end, N - 1))
    n_range = end - start + 1

    if n_sample <= 1 or n_range <= 1:
        return {"geo_consistency": 0.0, "min_geo_consistency": 0.0}

    indices = [start + int(i * (n_range - 1) / (n_sample - 1)) for i in range(n_sample)]

    sims = []
    for i in range(len(indices) - 1):
        d1 = _depth_embed(model, processor, device, frames.get(indices[i]))
        d2 = _depth_embed(model, processor, device, frames.get(indices[i + 1]))
        sims.append(_depth_similarity(d1, d2))

    if not sims:
        return {"geo_consistency": 0.0, "min_geo_consistency": 0.0}

    return {
        "geo_consistency": round(float(np.mean(sims)), 4),
        "min_geo_consistency": round(float(np.min(sims)), 4),
    }
