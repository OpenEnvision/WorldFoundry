"""Classification Accuracy Score (CAS) from arXiv:1905.10887."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_cas_from_predictions(
    real_labels: np.ndarray | list[Any],
    predicted_labels: np.ndarray | list[Any],
    *,
    num_classes: int | None = None,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute Top-k CAS given classifier predictions on real validation labels."""
    labels = np.asarray(real_labels)
    if labels.ndim != 1:
        raise ValueError("real_labels must be a 1-D array")
    preds = np.asarray(predicted_labels)
    if preds.ndim == 1:
        top1 = preds
        available_topk = (1,)
    elif preds.ndim == 2:
        top1 = preds[:, 0]
        available_topk = tuple(k for k in topk if k <= preds.shape[1])
    else:
        raise ValueError("predicted_labels must be rank-1 labels or rank-2 class rankings")
    if labels.shape[0] != top1.shape[0]:
        raise ValueError("real_labels and predicted_labels must have the same length")
    results: dict[str, float] = {}
    results["cas_top1"] = float(np.mean(labels == top1))
    if preds.ndim == 2:
        for k in available_topk:
            if k == 1:
                continue
            matches = (preds[:, :k] == labels[:, None]).any(axis=1)
            results[f"cas_top{k}"] = float(np.mean(matches))
    if num_classes is not None and num_classes > 0:
        results["num_classes"] = float(num_classes)
    results["cas"] = results["cas_top1"]
    return results


def train_classifier_and_compute_cas(
    synthetic_images: np.ndarray,
    synthetic_labels: np.ndarray | list[Any],
    real_images: np.ndarray,
    real_labels: np.ndarray | list[Any],
    *,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str | None = None,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Train a classifier on synthetic images only, then evaluate Top-k accuracy on real images."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_x = torch.from_numpy(np.asarray(synthetic_images, dtype=np.float32))
    if train_x.ndim == 4 and train_x.shape[1] not in (1, 3):
        train_x = train_x.permute(0, 3, 1, 2)
    train_y = torch.from_numpy(np.asarray(synthetic_labels, dtype=np.int64))
    val_x = torch.from_numpy(np.asarray(real_images, dtype=np.float32))
    if val_x.ndim == 4 and val_x.shape[1] not in (1, 3):
        val_x = val_x.permute(0, 3, 1, 2)
    _val_y = torch.from_numpy(np.asarray(real_labels, dtype=np.int64))

    channels = int(train_x.shape[1])
    model = nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, num_classes),
    ).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device_t)
            batch_y = batch_y.to(device_t)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_x.to(device_t))
        probabilities = torch.softmax(logits, dim=-1)
        max_k = min(max(topk), num_classes)
        top_indices = probabilities.topk(max_k, dim=-1).indices.cpu().numpy()

    return compute_cas_from_predictions(
        np.asarray(real_labels),
        top_indices,
        num_classes=num_classes,
        topk=topk,
    )


__all__ = [
    "compute_cas_from_predictions",
    "train_classifier_and_compute_cas",
]
