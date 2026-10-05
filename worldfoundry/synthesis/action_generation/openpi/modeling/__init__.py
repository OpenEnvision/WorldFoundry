"""OpenPI JAX and PyTorch model implementations used by inference.

Importing a lightweight submodule such as ``action_tokenizer`` must not load the
JAX policy stack. GigaBrain uses the tokenizer without depending on Flax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ModelType
    from .pi0_config import Pi0Config

__all__ = ["ModelType", "Pi0Config"]


def __getattr__(name: str):
    if name == "ModelType":
        from .model import ModelType

        return ModelType
    if name == "Pi0Config":
        from .pi0_config import Pi0Config

        return Pi0Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
