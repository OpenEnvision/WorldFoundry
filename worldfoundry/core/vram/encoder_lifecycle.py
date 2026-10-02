# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle helpers for one-shot encoder and initialization stages.

Responsibility: construct a text/VAE encoder, run it once under
``no_grad``, then drop references and trim CUDA allocator blocks so
peak VRAM belongs to the DiT.

This module is not a model loader, tokenizer, or VAE implementation.
It does not decide *which* encoder to run — callers pass ``config`` /
``stage`` / attribute names. Torch is imported lazily so CPU-only
Studio tooling can call the same helpers without an accelerator
runtime.

Public surface:
- :func:`setup_one_shot_encoder` / :func:`ensure_one_shot_encoder`
- :func:`run_one_shot_encoder_stage` (always ``release`` in ``finally``)
- :func:`release_one_shot_encoder_references` / :func:`offload_module_to_cpu`
- :func:`move_tensors_to_cpu` / :func:`collect_and_release_cuda_memory`

Model integrations share the same reference-release, GC, and allocator
cleanup so peak VRAM is not left pinned after a one-shot encode.
"""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Construct / reload — config owns the encoder type; this only places it
# ──────────────────────────────────────────────────────────────────────────


def setup_one_shot_encoder(
    config: Any,
    *,
    device: Any | Callable[[], Any] | None = None,
    torch_module: Any | None = None,
) -> Any:
    """Instantiate an encoder config and move torch modules to ``device``.

    ``device`` may be a factory so the CUDA device is resolved after
    ``config.setup()``, not at import. Non-``nn.Module`` encoders are
    returned unchanged.
    """

    encoder = config.setup()
    if device is None:
        return encoder
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    module_cls = getattr(getattr(torch, "nn", None), "Module", None)
    if module_cls is not None and isinstance(encoder, module_cls):
        encoder = encoder.to(device=device() if callable(device) else device)
    return encoder


def ensure_one_shot_encoder(
    encoder: Any | None,
    config: Any | None,
    *,
    device: Any | Callable[[], Any] | None = None,
    name: str = "encoder",
    required: bool = True,
    torch_module: Any | None = None,
) -> Any | None:
    """Return an existing encoder or lazily reconstruct it from ``config``.

    A missing config is fatal only when ``required`` — optional stages
    (e.g. a second text tower) must be allowed to stay ``None``.
    """

    if encoder is not None:
        return encoder
    if config is None:
        if required:
            raise RuntimeError(f"{name} is unloaded and has no configuration for reloading.")
        return None
    return setup_one_shot_encoder(config, device=device, torch_module=torch_module)


# ──────────────────────────────────────────────────────────────────────────
# Release / offload — drop refs then trim so the DiT can claim the slack
# ──────────────────────────────────────────────────────────────────────────


def release_one_shot_encoder_references(
    owner: Any,
    *attrs: str,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> tuple[str, ...]:
    """Clear encoder attributes and release allocator blocks they retained.

    Every named attr is set to ``None`` even if already empty, so a
    later ``ensure_*`` cannot revive a half-cleared owner. The returned
    tuple lists attrs that were non-``None`` *before* the wipe.
    """

    released: list[str] = []
    for attr in attrs:
        if getattr(owner, attr, None) is not None:
            released.append(attr)
        setattr(owner, attr, None)
    collect_and_release_cuda_memory(
        device=device,
        synchronize_cuda=synchronize_cuda,
        empty_cuda_cache=empty_cuda_cache,
        torch_module=torch_module,
    )
    return tuple(released)


def offload_module_to_cpu(
    module: Any,
    *,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> Any:
    """Move a reusable module to CPU and release its no-longer-used CUDA blocks."""

    moved = module.cpu()
    collect_and_release_cuda_memory(
        device=device,
        synchronize_cuda=synchronize_cuda,
        empty_cuda_cache=empty_cuda_cache,
        torch_module=torch_module,
    )
    return moved


def collect_and_release_cuda_memory(
    *,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> None:
    """Collect unreachable objects and optionally trim the CUDA allocator.

    Sync is off by default: a full device sync here would serialize the
    DiT's next launch. ``empty_cache`` is the usual peak-VRAM lever.
    """

    gc.collect()
    if not empty_cuda_cache and not synchronize_cuda:
        return
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available) or not is_available():
        return
    if synchronize_cuda:
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize() if device is None else synchronize(device)
    if empty_cuda_cache:
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def move_tensors_to_cpu(value: Any, *, torch_module: Any | None = None) -> Any:
    """Recursively detach tensor containers from accelerator memory.

    Dict / list / tuple are rebuilt; other objects (strings, numpy) pass
    through so encoder metadata is not coerced.
    """

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    is_tensor = getattr(torch, "is_tensor", None)
    if callable(is_tensor) and is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item, torch_module=torch) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item, torch_module=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(item, torch_module=torch) for item in value)
    return value


# ──────────────────────────────────────────────────────────────────────────
# One-shot stage — release even when the encoder raises
# ──────────────────────────────────────────────────────────────────────────


def run_one_shot_encoder_stage(
    stage: Callable[[], Any],
    *,
    release: Callable[[], Any] | None = None,
    cpu_result: bool = True,
    torch_module: Any | None = None,
) -> Any:
    """Run an encoder-only stage without gradients and always release it."""

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    no_grad = getattr(torch, "no_grad", None)
    context = no_grad() if callable(no_grad) else nullcontext()
    try:
        with context:
            result = stage()
        return move_tensors_to_cpu(result, torch_module=torch) if cpu_result else result
    finally:
        if release is not None:
            released = release()
            del released


def _maybe_import_torch() -> Any | None:
    """Load torch on demand so CPU-only callers never import CUDA at module import."""
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


__all__ = [
    "collect_and_release_cuda_memory",
    "ensure_one_shot_encoder",
    "move_tensors_to_cpu",
    "offload_module_to_cpu",
    "release_one_shot_encoder_references",
    "run_one_shot_encoder_stage",
    "setup_one_shot_encoder",
]
