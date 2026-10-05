# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Inference-side enforcement for the checkpoint DMD timestep contract.

The denoising tables themselves live in :mod:`worldcrafter.fast.timestep_grid`.  This
module deliberately contains no schedule math: it only locates a checkpoint's
sidecar, validates the latent shape, and selects the normal or empty-history
table from the conditioning tensors that will actually enter the transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .timestep_grid import (
    DmdTimestepContract,
    DmdTimestepStep,
    resolve_dmd_contract_path,
)


@dataclass(frozen=True)
class DmdInferenceTrace:
    """The immutable contract trace selected for one denoised chunk."""

    fingerprint: str
    empty_history: bool
    stages: tuple[tuple[DmdTimestepStep, ...], ...]

    @property
    def num_steps(self) -> int:
        return sum(len(stage) for stage in self.stages)


def load_dmd_inference_contract(
    checkpoint_path: str | Path,
    *,
    expected_latent_shape: Sequence[int] | None = None,
    expected_fingerprint: str | None = None,
) -> DmdTimestepContract:
    """Load and authenticate the sidecar next to ``checkpoint_path``.

    A DMD checkpoint without its sidecar is intentionally unusable.  Rebuilding
    schedule from inference flags could silently change the checkpoint
    timesteps and generated output.
    """

    sidecar = resolve_dmd_contract_path(checkpoint_path)
    return DmdTimestepContract.load_json(
        sidecar,
        expected_latent_shape=expected_latent_shape,
        expected_fingerprint=expected_fingerprint,
    )


def _history_nonempty_mask(
    history_tensors: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    mask = None
    batch_size = None
    for history in history_tensors:
        if history is None:
            continue
        if history.ndim < 1:
            raise ValueError(
                f"DMD history tensor must have a batch dimension, got shape={tuple(history.shape)}"
            )
        if batch_size is None:
            batch_size = int(history.shape[0])
            mask = torch.zeros(batch_size, dtype=torch.bool, device=history.device)
        elif int(history.shape[0]) != batch_size:
            raise ValueError(
                "DMD history tensors disagree on batch size: "
                f"expected {batch_size}, got {int(history.shape[0])}"
            )
        current = history.detach().reshape(batch_size, -1).ne(0).any(dim=1)
        mask = mask | current.to(device=mask.device)

    if mask is None:
        # A missing conditioning bank is the same semantic condition as the
        # all-zero placeholders used by the pipelines for the first T2V chunk.
        return torch.zeros(1, dtype=torch.bool)
    return mask


def history_is_empty(history_tensors: Iterable[torch.Tensor | None]) -> bool:
    """Classify the real conditioning bank, rejecting a mixed batch.

    Empty/non-empty examples require different step counts, so a batch cannot
    share one denoising loop when only some rows have history.
    """

    nonempty = _history_nonempty_mask(history_tensors)
    has_nonempty = bool(nonempty.any().item())
    has_empty = bool((~nonempty).any().item())
    if has_nonempty and has_empty:
        raise ValueError(
            "DMD inference cannot mix empty-history and non-empty-history samples in one batch; "
            "split the batch so every sample follows one checkpoint schedule."
        )
    return not has_nonempty


def resolve_dmd_inference_trace(
    contract: DmdTimestepContract | None,
    *,
    latent_shape: Sequence[int],
    history_tensors: Iterable[torch.Tensor | None],
    num_stages: int,
) -> DmdInferenceTrace:
    """Validate and select the exact full rollout trace for a chunk."""

    if contract is None:
        raise RuntimeError(
            "DMD inference requires dmd_timestep_contract.json from the checkpoint; "
            "no contract was loaded."
        )

    actual_shape = tuple(int(value) for value in latent_shape)
    expected_shape = tuple(int(value) for value in contract.latent_shape)
    if actual_shape != expected_shape:
        raise ValueError(
            "DMD latent shape does not match the checkpoint contract: "
            f"checkpoint={expected_shape}, inference={actual_shape}"
        )

    if len(contract.normal_stages) != int(num_stages):
        raise ValueError(
            "DMD pyramid stage count does not match the checkpoint contract: "
            f"checkpoint={len(contract.normal_stages)}, inference={int(num_stages)}"
        )

    empty_history = history_is_empty(history_tensors)
    stages = tuple(
        tuple(contract.stage(stage_index, empty_history=empty_history))
        for stage_index in range(int(num_stages))
    )
    return DmdInferenceTrace(
        fingerprint=contract.fingerprint,
        empty_history=empty_history,
        stages=stages,
    )
