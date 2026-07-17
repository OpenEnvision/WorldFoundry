"""WorldFoundry facade for Fréchet Denoised Distance (FDD)."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import sqrtm

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor"


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _dae_classes() -> tuple[Any, Any]:
    prepend_import_path(VENDOR_ROOT)
    from DAE.model import AutoEncoder, AutoEncoderConfig

    return AutoEncoder, AutoEncoderConfig


def _resolve_checkpoint(checkpoint: str | Path | None) -> str | Path | io.BytesIO:
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.is_file():
            return path
        raise FileNotFoundError(f"FDD DAE checkpoint not found: {checkpoint}")
    env_path = os.environ.get("WORLDFOUNDRY_FDD_DAE_CKPT")
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "FDD requires the pretrained DAE checkpoint. Install gdown or set "
            "WORLDFOUNDRY_FDD_DAE_CKPT to a local .pt file."
        ) from exc
    file_id = "1j7MVFWYfRNZLQ3uChe7TGQG8L1Oaf9Gt"
    file_bytes = io.BytesIO()
    gdown.download(id=file_id, output=file_bytes, quiet=True)
    file_bytes.seek(0)
    return file_bytes


def _load_images(images: str | Path | Sequence[str | Path | np.ndarray]) -> list[np.ndarray]:
    from PIL import Image

    if isinstance(images, (str, Path)):
        root = Path(images)
        if root.is_dir():
            suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)
            return [np.array(Image.open(path).convert("RGB")) for path in paths]
        raise ValueError(f"expected image directory for FDD, got {images}")
    loaded: list[np.ndarray] = []
    for item in images:
        if isinstance(item, np.ndarray):
            loaded.append(item)
        else:
            loaded.append(np.array(Image.open(item).convert("RGB")))
    return loaded


@lru_cache(maxsize=4)
def _load_dae_model(checkpoint_key: str) -> Any:
    AutoEncoder, AutoEncoderConfig = _dae_classes()
    ckpt = _resolve_checkpoint(None if checkpoint_key == "__default__" else checkpoint_key)
    config = AutoEncoderConfig()
    model = AutoEncoder(config, ckpt=ckpt)
    model.eval()
    return model


def _extract_activations(
    images: Sequence[np.ndarray],
    *,
    model: Any,
    batch_size: int,
    image_shape: int,
    device: str,
) -> np.ndarray:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class ImagenetDAEDataset(Dataset):
        def __init__(self, items: Sequence[np.ndarray], transform_fn: Any) -> None:
            self.items = list(items)
            self.transform_fn = transform_fn

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> Any:
            image = self.items[index]
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            pil_image = Image.fromarray(image.astype(np.uint8))
            return self.transform_fn(pil_image)

    transform = transforms.Compose(
        [
            transforms.Resize((image_shape, image_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(ImagenetDAEDataset(images, transform), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    all_latent_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            latent_vectors = model.encode(batch).cpu().numpy()
            all_latent_vectors.append(latent_vectors)
    return np.concatenate(all_latent_vectors, axis=0)


def compute_fdd(
    reference: str | Path | Sequence[str | Path | np.ndarray],
    generated: str | Path | Sequence[str | Path | np.ndarray],
    *,
    checkpoint: str | Path | None = None,
    batch_size: int = 64,
    image_shape: int = 299,
    device: str | None = None,
) -> float:
    """Compute Fréchet Denoised Distance between two image sets."""
    import torch

    ref_images = _load_images(reference)
    gen_images = _load_images(generated)
    if not ref_images or not gen_images:
        raise ValueError("FDD requires non-empty reference and generated image sets")
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_key = str(checkpoint) if checkpoint is not None else "__default__"
    model = _load_dae_model(ckpt_key)
    act_ref = _extract_activations(
        ref_images, model=model, batch_size=batch_size, image_shape=image_shape, device=device_name
    )
    act_gen = _extract_activations(
        gen_images, model=model, batch_size=batch_size, image_shape=image_shape, device=device_name
    )
    mu1, sigma1 = act_ref.mean(axis=0), np.cov(act_ref, rowvar=False)
    mu2, sigma2 = act_gen.mean(axis=0), np.cov(act_gen, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


__all__ = ["compute_fdd", "package_root"]
