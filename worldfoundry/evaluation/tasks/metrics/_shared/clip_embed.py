"""Lazy CLIP embedding helpers for paper-reimplemented metrics."""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F

from .images import ImageInput, load_rgb_image

DEFAULT_CLIP_MODEL = "openai:ViT-B-32"


@lru_cache(maxsize=4)
def _open_clip_bundle(model: str, device: str):
    import open_clip

    pretrained, arch = model.split(":", 1)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        arch,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(arch)
    clip_model.eval()
    return clip_model, preprocess, tokenizer


@torch.no_grad()
def encode_clip_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
) -> np.ndarray:
    """Return L2-normalized CLIP text embeddings."""
    device_t = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _, tokenizer = _open_clip_bundle(model, device_t)
    tokens = tokenizer(texts).to(device_t)
    feats = clip_model.encode_text(tokens)
    feats = F.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy()


@torch.no_grad()
def encode_clip_images(
    images: list[ImageInput],
    *,
    model: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
) -> np.ndarray:
    """Return L2-normalized CLIP image embeddings."""
    device_t = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, _ = _open_clip_bundle(model, device_t)
    batch = torch.stack([preprocess(load_rgb_image(image)) for image in images], dim=0).to(device_t)
    feats = clip_model.encode_image(batch)
    feats = F.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy()


def cosine_similarity_vectors(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity between two 1-D embedding vectors."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def clip_image_text_cosine(
    image: ImageInput,
    text: str,
    *,
    model: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
) -> float:
    """CLIP cosine similarity between one image and one text prompt."""
    image_feat = encode_clip_images([image], model=model, device=device)[0]
    text_feat = encode_clip_texts([text], model=model, device=device)[0]
    return cosine_similarity_vectors(image_feat, text_feat)


__all__ = [
    "DEFAULT_CLIP_MODEL",
    "clip_image_text_cosine",
    "cosine_similarity_vectors",
    "encode_clip_images",
    "encode_clip_texts",
]
