"""``torch.compile`` mechanics for DiffusionVideoDecoder (no DiffVAEMode / resolve)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.optimizations.ltx.compile import CompilationConfig

if TYPE_CHECKING:
    from ..diffusion_video_decoder import DiffusionVideoDecoder


def configure_natten_backend(module_root: nn.Module, backend: str | None) -> None:
    """Pin NATTEN on every ``NeighborhoodAttention3D`` under ``module_root``.
    Installs a :class:`.attention.NattenAttention`
    as ``attention_function`` (combined / det / chunked all route through it) and mirrors
    the pin onto ``natten_backend`` for introspection. ``backend=None`` restores NATTEN
    auto-selection.
    """
    from .attention import NattenAttention, NeighborhoodAttention3D

    attn_fn = NattenAttention(backend)
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = attn_fn
            module.natten_backend = backend


def compile_diffusion_decoder(
    decoder: DiffusionVideoDecoder,
    *,
    config: CompilationConfig | None = None,
    compile_det_stages: bool = True,
) -> DiffusionVideoDecoder:
    """``torch.compile`` pre-diffusion and/or diffusion-step block methods.
    Diff-step (matches ``feature/diffvae-attn-w-chunking-backup``):
      - Combined: ``forward_combined``
      - Chunked: ``forward_attn_mlp`` only (inject stays eager in-place outside
        the compiled region — keeps peak VRAM near ChunkedEager).
    Det stages optional via ``compile_det_stages``.
    """
    cfg = config or CompilationConfig()

    try:
        import natten

        natten.use_kv_parallelism_in_fused_na(False)
    except ImportError:
        pass

    compile_kwargs: dict[str, Any] = {
        "mode": cfg.mode,
        "backend": cfg.backend,
        "fullgraph": cfg.fullgraph,
        "dynamic": cfg.dynamic,
    }

    def _compile(fn: Callable[..., Any]) -> Callable[..., Any]:
        with (
            torch._inductor.config.patch(**cfg.inductor_config),
            torch._dynamo.config.patch(**cfg.dynamo_config),  # type: ignore[attr-defined]
        ):
            return torch.compile(fn, **compile_kwargs)

    from .chunked.block import ChunkedDiffusionNABlock
    from .combined.block import CombinedDiffusionNABlock
    from .dsl_kernels.block import DSLDiffusionNABlock

    if compile_det_stages:
        for stage_blocks in decoder.det_stages:
            for block in stage_blocks:
                block.forward = _compile(block.forward)  # type: ignore[method-assign]

    for block in decoder.diff_blocks:
        if isinstance(block, DSLDiffusionNABlock):
            # Fused stage-5 is a CuTe launch; do not torch.compile it.
            continue
        if isinstance(block, ChunkedDiffusionNABlock):
            block.forward_attn_mlp = _compile(block.forward_attn_mlp)  # type: ignore[method-assign]
        elif isinstance(block, CombinedDiffusionNABlock):
            block.forward_combined = _compile(block.forward_combined)  # type: ignore[method-assign]

    decoder.mark_dynamic_shapes = True
    return decoder
