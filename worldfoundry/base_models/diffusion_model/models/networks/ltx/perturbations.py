"""Spatio-Temporal Guidance (STG) skip masks for LTX attention branches.

Perturbations zero selected self-attn or A↔V cross-attn paths per sample
and block so guidance can compare a full pass against a skipped one.
Masks are ``[0, 1]`` and broadcast over ``B T D`` token tensors.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch._prims_common import DeviceLikeType


class PerturbationType(IntEnum):
    """Attention branches skipped by STG; values index rows of the batched keep masks."""

    SKIP_VIDEO_SELF_ATTN = 0
    SKIP_AUDIO_SELF_ATTN = 1
    SKIP_A2V_CROSS_ATTN = 2
    SKIP_V2A_CROSS_ATTN = 3


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        """True when this rule skips ``perturbation_type`` at ``block`` (or all blocks)."""
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        """True if any listed perturbation matches this type and block."""
        if self.perturbations is None:
            return False

        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        """No-op config used when STG is disabled for a sample."""
        return PerturbationConfig([])


class BatchedPerturbationConfig:
    """Materialize per-block keep masks before compilation or CUDA graph capture.

    The compute tensor has shape ``(types, blocks, batch)``, indexed by
    :class:`PerturbationType`. Its CPU mirror supplies sync-free eager branch
    decisions; compiled paths consume masks as tensor inputs, without
    retaining the Python rule lists.
    """

    def __init__(
        self,
        perturbations: list[PerturbationConfig],
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        keep = [
            [[not config.is_perturbed(kind, block) for config in perturbations] for block in range(num_blocks)]
            for kind in PerturbationType
        ]
        self._block_masks_cpu = torch.tensor(keep, dtype=dtype, device="cpu")
        self._block_masks = self._block_masks_cpu if device is None else self._block_masks_cpu.to(device)

    @classmethod
    def from_masks(
        cls, block_masks: torch.Tensor, block_masks_cpu: torch.Tensor | None = None
    ) -> "BatchedPerturbationConfig":
        """Use prebuilt masks; eager skip decisions additionally require the CPU mirror."""
        obj = cls.__new__(cls)
        obj._block_masks = block_masks
        obj._block_masks_cpu = block_masks_cpu
        return obj

    @property
    def block_masks(self) -> torch.Tensor:
        """Compute-device keep masks, indexed by perturbation type, block and sample."""
        return self._block_masks

    @property
    def block_masks_cpu(self) -> torch.Tensor | None:
        """CPU keep masks used outside compiled regions to choose attention branches."""
        return self._block_masks_cpu

    def batch_slice(self, start: int, end: int) -> "BatchedPerturbationConfig":
        """Slice batch views without rebuilding the host masks during capture."""
        cpu_mask = self._block_masks_cpu[:, :, start:end] if self._block_masks_cpu is not None else None
        return self.from_masks(self._block_masks[:, :, start:end], cpu_mask)

    def mask(self, perturbation_type: PerturbationType, block: int) -> torch.Tensor:
        """Return an owned ``(B, 1, 1)`` keep mask so compiled inputs do not alias."""
        return self._block_masks[perturbation_type, block].reshape(-1, 1, 1).clone()

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        """True if at least one sample skips this branch."""
        assert self._block_masks_cpu is not None, "eager skip decisions require the CPU mask mirror"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).any())

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        """True if every sample skips this branch."""
        assert self._block_masks_cpu is not None, "eager skip decisions require the CPU mask mirror"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).all())

    @staticmethod
    def empty(
        batch_size: int,
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPerturbationConfig":
        """All-keep config for a batch that is not using STG."""
        return BatchedPerturbationConfig(
            [PerturbationConfig.empty() for _ in range(batch_size)], num_blocks, device, dtype
        )
