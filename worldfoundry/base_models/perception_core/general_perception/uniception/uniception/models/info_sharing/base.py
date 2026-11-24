"""
Base Information Sharing Class for UniCeption
"""

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@dataclass
class InfoSharingInput:
    pass


@dataclass
class InfoSharingOutput:
    pass


class UniCeptionInfoSharingBase(nn.Module):
    "Information Sharing Base Class for UniCeption"

    def __init__(
        self,
        name: str,
        size: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Base class for all models in UniCeption.
        """
        super().__init__(*args, **kwargs)

        self.name: str = name
        self.size: str | None = size

    def forward(
        self,
        model_input: InfoSharingInput,
    ) -> InfoSharingOutput:
        """
        Forward interface for the UniCeption information sharing models.

        Args:
            model_input (InfoSharingInput): Input to the model.
                This is also includes the other fields that are required by the specific implementation of the model.

        Returns:
            InfoSharingOutput: Output of the model.
        """

        raise NotImplementedError

    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        """
        Wrapper for Gradient Checkpointing
        """

        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module


@dataclass
class MultiViewTransformerInput(InfoSharingInput):
    """
    Input class for Multi-View Transformer.
    """

    features: list[Tensor]
    additional_input_tokens: Tensor | None = None
    additional_input_tokens_per_view: list[Tensor] | None = None


@dataclass
class MultiViewTransformerOutput(InfoSharingOutput):
    """
    Output class for Multi-View Transformer.
    """

    features: list[Tensor]
    additional_token_features: Tensor | None = None
    additional_token_features_per_view: list[Tensor] | None = None


@dataclass
class MultiSetTransformerInput(InfoSharingInput):
    """
    Input class for Multi-Set Transformer.
    """

    features: list[Tensor]
    additional_input_tokens: Tensor | None = None


@dataclass
class MultiSetTransformerOutput(InfoSharingOutput):
    """
    Output class for Multi-Set Transformer.
    """

    features: list[Tensor]
    additional_token_features: Tensor | None = None
