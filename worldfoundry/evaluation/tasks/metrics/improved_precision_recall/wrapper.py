"""WorldFoundry facade for Improved Precision and Recall (α-precision / β-recall)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.imports import prepend_import_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root() -> Path:
    return PACKAGE_ROOT


@lru_cache(maxsize=1)
def _ipr_class() -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from improved_precision_recall import IPR

    return IPR


def compute_improved_precision_recall(
    reference: str | Path,
    generated: str | Path,
    *,
    batch_size: int = 50,
    k: int = 3,
    num_samples: int = 5000,
    device: str | None = None,
) -> dict[str, float]:
    """Compute improved precision and recall between image directories."""
    import torch

    device_t = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ipr = _ipr_class()(batch_size=batch_size, k=k, num_samples=num_samples, device=device_t)
    with torch.no_grad():
        ipr.compute_manifold_ref(str(reference))
        precision, recall = ipr.precision_and_recall(str(generated))
    return {"precision": float(precision), "recall": float(recall)}


def _custom_loader(*args: Any, **kwargs: Any) -> Any:
    prepend_import_path(PACKAGE_ROOT)
    from improved_precision_recall import get_custom_loader

    return get_custom_loader(*args, **kwargs)


def _load_image_tensor(image: np.ndarray, device: str) -> Any:
    from torchvision import transforms

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8 and arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = __import__("PIL").Image.fromarray(arr.astype(np.uint8))
    pil = pil.resize((224, 224), __import__("PIL").Image.BICUBIC)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil).unsqueeze(0).to(device)


def compute_realism_score(
    reference: str | Path,
    generated: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
    *,
    batch_size: int = 50,
    k: int = 3,
    num_samples: int = 5000,
    device: str | None = None,
) -> float | dict[str, float]:
    """Compute IPR realism score(s) relative to a reference image manifold."""
    import torch

    prepend_import_path(PACKAGE_ROOT)
    device_t = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ipr = _ipr_class()(batch_size=batch_size, k=k, num_samples=num_samples, device=device_t)
    with torch.no_grad():
        ipr.compute_manifold_ref(str(reference))
        if isinstance(generated, (str, Path)):
            loader = _custom_loader(str(generated), batch_size=batch_size, num_samples=num_samples)
            scores: list[float] = []
            for batch in loader:
                scores.append(float(ipr.realism(batch)))
            if not scores:
                raise ValueError(f"no images found in generated path: {generated}")
            return {"mean_realism": float(np.mean(scores)), "realism_scores": scores}
        if isinstance(generated, np.ndarray):
            return float(ipr.realism(_load_image_tensor(generated, device_t)))
        scores = []
        for item in generated:
            if isinstance(item, (str, Path)):
                loader = _custom_loader(str(item), batch_size=1, num_samples=1)
                batch = next(iter(loader))
                scores.append(float(ipr.realism(batch)))
            else:
                scores.append(float(ipr.realism(_load_image_tensor(np.asarray(item), device_t))))
        if not scores:
            raise ValueError("generated sequence is empty")
        if len(scores) == 1:
            return scores[0]
        return {"mean_realism": float(np.mean(scores)), "realism_scores": scores}


__all__ = [
    "compute_improved_precision_recall",
    "compute_realism_score",
    "package_root",
]
