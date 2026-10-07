"""Pick one installed attention kernel for a DiT attention slot."""

from __future__ import annotations

import logging

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels.flash_attention_2 import flash_attention_2
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels.flash_attention_3 import flash_attention_3
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels.sage import sage
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.kernels.sdpa import sdpa

logger = logging.getLogger("kandinsky")

# Historic name used by export code that forces the SDPA implementation.
_sdpa = sdpa


def resolve_attention_engine(requested: str) -> str:
    """Resolve a config or CLI name to an engine that can be constructed.

    ``flash_attention_3`` falls back to ``auto`` when FA3 is not installed.
    Other explicit engines stay as requested.
    """
    if requested == "flash_attention_3" and flash_attention_3 is None:
        logger.warning("flash_attention_3 requested but not installed; falling back to auto")
        return "auto"
    return requested


class SelfAttentionEngine:
    """Selects one attention kernel at construction time."""

    _ENGINES = ("flash_attention_3", "flash_attention_2", "sage", "sdpa", "auto")

    def __init__(self, engine: str = "auto"):
        assert engine in self._ENGINES, f"Unknown attention engine: {engine!r}"
        engine = resolve_attention_engine(engine)
        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 requested but not installed.")
            self._fn = flash_attention_3
        elif engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 requested but not installed.")
            self._fn = flash_attention_2
        elif engine == "sage":
            if sage is None:
                raise RuntimeError("sageattention requested but not installed.")
            self._fn = sage
        elif engine == "sdpa":
            self._fn = sdpa
        else:
            self._fn = sdpa
            if sage is not None:
                self._fn = sage
            if flash_attention_2 is not None:
                self._fn = flash_attention_2
            if flash_attention_3 is not None:
                self._fn = flash_attention_3

    def get_attention(self):
        return self._fn
