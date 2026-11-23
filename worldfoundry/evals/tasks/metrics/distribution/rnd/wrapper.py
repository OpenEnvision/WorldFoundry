"""Random Network Distillation (RND) diversity score (arXiv:2010.06715)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _split_indices(n: int, train_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.permutation(n)
    k = min(max(1, train_size), n - 1)
    return indices[:k], indices[k:]


def compute_rnd(
    features: np.ndarray,
    *,
    train_size: int | None = None,
    runs: int = 3,
    epochs: int = 20,
    start_epoch: int = 10,
    learning_rate: float = 1e-3,
    hidden_dim: int = 256,
    seed: int = 0,
) -> float:
    """Estimate RND diversity score on feature vectors (higher is more diverse)."""
    import torch
    import torch.nn as nn

    data = np.asarray(features, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2:
        raise ValueError("RND requires at least two samples")
    n, dim = data.shape
    k = train_size if train_size is not None else max(1, n // 2)
    run_scores: list[float] = []
    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        train_idx, val_idx = _split_indices(n, k, rng)
        train_x = torch.from_numpy(data[train_idx])
        val_x = torch.from_numpy(data[val_idx])

        target = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for param in target.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        rnd_values: list[float] = []
        for epoch in range(epochs):
            predictor.train()
            optimizer.zero_grad()
            with torch.no_grad():
                target_train = target(train_x)
            pred_train = predictor(train_x)
            loss = loss_fn(pred_train, target_train)
            loss.backward()
            optimizer.step()
            if epoch + 1 >= start_epoch:
                predictor.eval()
                with torch.no_grad():
                    target_val = target(val_x)
                    pred_val = predictor(val_x)
                    mse_val = float(loss_fn(pred_val, target_val).item())
                    mse_tr = float(loss_fn(predictor(train_x), target(train_x)).item())
                denom = mse_val + mse_tr
                rnd_i = (mse_val - mse_tr) / denom if denom > 0 else 0.0
                rnd_values.append(rnd_i)
        if rnd_values:
            run_scores.append(float(np.mean(rnd_values)))
    if not run_scores:
        return 0.0
    return float(np.mean(run_scores))


def compute_rnd_from_images(
    images: np.ndarray,
    *,
    image_size: int = 64,
    **kwargs: Any,
) -> float:
    """Flatten and downsample RGB images, then compute RND on pixel features."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 4:
        h, w = arr.shape[1], arr.shape[2]
        if h != image_size or w != image_size:
            from skimage.transform import resize

            resized = np.stack([resize(img, (image_size, image_size), preserve_range=True) for img in arr])
            arr = resized
        flat = arr.reshape(arr.shape[0], -1) / 255.0
    else:
        flat = arr.reshape(arr.shape[0], -1)
    return compute_rnd(flat, **kwargs)


__all__ = ["compute_rnd", "compute_rnd_from_images"]
