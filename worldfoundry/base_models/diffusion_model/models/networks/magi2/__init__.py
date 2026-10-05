"""Checkpoint-compatible network math for the MAGI-2 two-stage audio-video model.

Preview vs refiner
------------------
``preview_dit.Transformer`` is the low-res 40-layer MoE + MHC generator.
``refiner_dit.Transformer`` is the dense 30-layer GQA 1080p upscaler.
``data_proxy`` packs latents into the preview token stream; ``sampler``
runs CFG flow-matching; ``loading`` maps Hub shards onto these modules.

Ported from SandAI's Apache-2.0 MAGI-2-preview inference code
(https://github.com/SandAI-org/MAGI-2-preview). This package holds only
``nn.Module`` architecture, tensor-shape contracts, and forward math as
single-GPU PyTorch. The upstream code assumes an 8×Hopper deployment with
context-parallel (cp), expert-parallel (ep), and data-parallel (dp) sharding;
every parallel op there is guarded by ``get_world_size(dim) > 1`` and collapses
to identity at size 1, so this port fixes cp=ep=dp=1 and drops the all-to-all /
ulysses machinery while preserving the numerics and the fp32/bf16 dtype split.

Two-stage model: ``magi2_preview`` (low-res 40-layer 256-expert MoE DiT) →
``magi2_refiner`` (dense 30-layer GQA DiT upscaling to 1080p). Novel pieces:
4-stream MHC hyper-connections, 12-head × 256-expert top-6 MoE with SwiGLU7,
FA3 attention with per-head sinks, and an element-wise Fourier coord embedding.
"""

from .sampler import (
    CFGConfig,
    Magi2PreviewSampler,
    Magi2SamplingConfig,
    ModelForward,
)

__all__: list[str] = [
    "CFGConfig",
    "Magi2PreviewSampler",
    "Magi2SamplingConfig",
    "ModelForward",
]
