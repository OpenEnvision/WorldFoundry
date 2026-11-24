"""Small model interfaces shared by the in-tree MolmoBot inference modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


class OLMoOutput(NamedTuple):
    """Backbone output needed by action inference.

    MolmoBot does not decode language, so ``logits`` is normally ``None``.  The
    action head consumes ``internal['layer_hidden_states']`` instead.
    """

    logits: torch.Tensor | None = None
    attn_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None
    hidden_states: Sequence[torch.Tensor] | None = None
    internal: dict[str, Sequence[torch.Tensor]] | None = None


class ModelBase(torch.nn.Module):
    """Base type retained for configuration/type compatibility."""


__all__ = ["ModelBase", "OLMoOutput"]
