from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
    from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import BatchedPerturbationConfig

ModelType = TypeVar("ModelType", covariant=True, bound=torch.nn.Module)


class ModelConfigurator(Protocol[ModelType]):
    """Protocol for model loader classes that instantiates models from checkpoint metadata."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> ModelType: ...


class LTXModelProtocol(Protocol):
    """Velocity-model forward interface shared by ``LTXModel`` and its multi-GPU wrappers.
    ``forward`` pins the real signature (enforced structurally); ``__call__`` mirrors it
    so protocol-typed values stay callable via ``model(...)``.
    """

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks, delegated through any wrappers to the ``LTXModel``."""
        ...

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...
