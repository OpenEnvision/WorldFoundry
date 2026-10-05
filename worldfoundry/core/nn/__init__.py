"""Reusable neural-network operators with no model identity.

- blocks: shared layers, normalization, patching, module utilities and EMA.
- transformer: shape helpers, DiT modulation, execution operators and ViT blocks.
- diffusion: convolutional layers, schedules, noise helpers and timestep embeddings.
- latent: VAE output types, distributions and checkpoint-compatible codec blocks.
- checkpointing: model checkpoint hooks, activation modes and gradient helpers.

Public operators resolve lazily. Attention implementations live in core.attention;
source inventory and duplicate audits live in scripts/model_zoo.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Lazy export table — resolve symbols only on first attribute access
# ──────────────────────────────────────────────────────────────────────────


_EXPORT_MODULES = {
    "AdaLayerNorm": "worldfoundry.core.nn.transformer.dit",
    "CheckpointMode": "worldfoundry.core.nn.checkpointing.activation_checkpointing",
    "AttentionBackendInfo": "worldfoundry.core.attention.backends.native",
    "AdaZeroCallable": "worldfoundry.core.nn.transformer.ops",
    "DEFAULT_TRANSFORMER_OPS": "worldfoundry.core.nn.transformer.ops",
    "DropPath": "worldfoundry.core.nn.blocks.layers",
    "DomainAwareLinear": "worldfoundry.core.nn.blocks.layers",
    "FlowMatchScheduler": "worldfoundry.core.nn.diffusion.schedulers",
    "GatedAttentionCallable": "worldfoundry.core.nn.transformer.ops",
    "InferenceCheckpointModule": "worldfoundry.core.nn.checkpointing.checkpoint_compat",
    "LayerNorm2d": "worldfoundry.core.nn.blocks.layers",
    "LayerScale": "worldfoundry.core.nn.blocks.layers",
    "LitEma": "worldfoundry.core.nn.blocks.ema",
    "ModuleDeviceDtypeMixin": "worldfoundry.core.nn.blocks.module_properties",
    "SamHeadMLP": "worldfoundry.core.nn.blocks.layers",
    "SamMLPBlock": "worldfoundry.core.nn.blocks.layers",
    "SchedulerInterface": "worldfoundry.core.nn.diffusion.schedulers",
    "SACConfig": "worldfoundry.core.nn.checkpointing.activation_checkpointing",
    "Mlp": "worldfoundry.core.nn.blocks.layers",
    "NativeVAE2DDecoder": "worldfoundry.core.nn.latent.vae2d",
    "PositionEmbeddingRandom": "worldfoundry.core.nn.blocks.layers",
    "PostSACallable": "worldfoundry.core.nn.transformer.ops",
    "PreAttentionCallable": "worldfoundry.core.nn.transformer.ops",
    "ProjectedTimestepEmbedding": "worldfoundry.core.nn.diffusion.timestep",
    "PytorchAdaZeroFunction": "worldfoundry.core.nn.transformer.ops",
    "PytorchGatedAttention": "worldfoundry.core.nn.transformer.ops",
    "PytorchPostSAFunction": "worldfoundry.core.nn.transformer.ops",
    "PytorchPreAttention": "worldfoundry.core.nn.transformer.ops",
    "DiTModulation": "worldfoundry.core.nn.transformer.dit",
    "DiTFinalLayer": "worldfoundry.core.nn.transformer.dit",
    "ConcatenatedLinear": "worldfoundry.core.nn.transformer.dit",
    "ConditioningProjection": "worldfoundry.core.nn.transformer.dit",
    "MLPEmbedder": "worldfoundry.core.nn.transformer.dit",
    "TransformerMLP": "worldfoundry.core.nn.transformer.dit",
    "SinusoidalTimestepEmbedder": "worldfoundry.core.nn.transformer.dit",
    "PatchGridSpec": "worldfoundry.core.nn.blocks.patching",
    "PatchEmbed": "worldfoundry.core.nn.blocks.layers",
    "PatchEmbed_Mlp": "worldfoundry.core.nn.blocks.layers",
    "Permute": "worldfoundry.core.nn.blocks.layers",
    "PixelUnshuffle": "worldfoundry.core.nn.blocks.layers",
    "PreNormTransformerBlock": "worldfoundry.core.nn.transformer.vit",
    "QKVSelfAttention": "worldfoundry.core.nn.transformer.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.nn.transformer.vit_qkv",
    "RopePreNormTransformerBlock": "worldfoundry.core.nn.transformer.vit",
    "RMSNorm": "worldfoundry.core.nn.transformer.dit",
    "SwiGLU": "worldfoundry.core.nn.blocks.layers",
    "SwiGLUFFN": "worldfoundry.core.nn.blocks.layers",
    "SwiGLUFFNFused": "worldfoundry.core.nn.blocks.layers",
    "XFORMERS_AVAILABLE": "worldfoundry.core.nn.blocks.layers",
    "XFORMERS_ENABLED": "worldfoundry.core.nn.blocks.layers",
    "TransformerShapeSpec": "worldfoundry.core.nn.transformer.shape",
    "TransformerAttentionOps": "worldfoundry.core.nn.transformer.ops",
    "TransformerOpsConfig": "worldfoundry.core.nn.transformer.ops",
    "TimestepEmbedding": "worldfoundry.core.nn.diffusion.timestep",
    "DiagonalGaussianDistribution": "worldfoundry.core.nn.latent.distributions",
    "AutoencoderKLOutput": "worldfoundry.core.nn.latent.distributions",
    "DecoderOutput": "worldfoundry.core.nn.latent.distributions",
    "DiracDistribution": "worldfoundry.core.nn.latent.distributions",
    "Timesteps": "worldfoundry.core.nn.diffusion.timestep",
    "add_residual": "worldfoundry.core.nn.blocks.stochastic_depth",
    "activation_layer": "worldfoundry.core.nn.transformer.dit",
    "apply_gate": "worldfoundry.core.nn.transformer.dit",
    "apply_gate_with_prefix": "worldfoundry.core.nn.transformer.dit",
    "apply_prenorm_transformer_residuals": "worldfoundry.core.nn.transformer.vit",
    "apply_rotary_embedding": "worldfoundry.core.attention.rotary.rope",
    "attention_backend_info": "worldfoundry.core.attention.backends.native",
    "attention_head_dim": "worldfoundry.core.nn.transformer.shape",
    "causal_attention_mask": "worldfoundry.core.nn.transformer.shape",
    "drop_add_residual_stochastic_depth": "worldfoundry.core.nn.blocks.stochastic_depth",
    "drop_path": "worldfoundry.core.nn.blocks.layers",
    "get_branges_scales": "worldfoundry.core.nn.blocks.stochastic_depth",
    "get_same_padding": "worldfoundry.core.nn.blocks.layers",
    "get_timestep_embedding": "worldfoundry.core.nn.diffusion.timestep",
    "layer_scale": "worldfoundry.core.nn.blocks.normalization",
    "list_sum": "worldfoundry.core.nn.blocks.layers",
    "merge_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "make_2tuple": "worldfoundry.core.nn.blocks.layers",
    "mlp_hidden_size": "worldfoundry.core.nn.transformer.shape",
    "modulate_sequence": "worldfoundry.core.nn.transformer.dit",
    "modulate_sequence_with_prefix": "worldfoundry.core.nn.transformer.dit",
    "named_apply": "worldfoundry.core.nn.blocks.module_utils",
    "normalization_layer": "worldfoundry.core.nn.transformer.dit",
    "patchify_image": "worldfoundry.core.nn.blocks.patching",
    "rms_norm": "worldfoundry.core.nn.blocks.normalization",
    "rotary_frequencies": "worldfoundry.core.attention.rotary.rope",
    "scale_shift": "worldfoundry.core.nn.transformer.dit",
    "rotate_half": "worldfoundry.core.attention.rotary.rope",
    "scaled_dot_product_attention": "worldfoundry.core.attention.backends.native",
    "sinusoidal_embedding_1d": "worldfoundry.core.nn.transformer.shape",
    "split_attention_heads": "worldfoundry.core.nn.transformer.shape",
    "transformer_shape_spec": "worldfoundry.core.nn.transformer.shape",
    "to_2tuple": "worldfoundry.core.nn.blocks.layers",
    "to_3tuple": "worldfoundry.core.nn.blocks.layers",
    "unpatchify_image": "worldfoundry.core.nn.blocks.patching",
    "val2list": "worldfoundry.core.nn.blocks.layers",
    "val2tuple": "worldfoundry.core.nn.blocks.layers",
    "ceil_to_divisible": "worldfoundry.core.nn.blocks.volume",
    "chunked_interpolate": "worldfoundry.core.nn.blocks.volume",
    "pixel_shuffle_3d": "worldfoundry.core.nn.blocks.volume",
    "pixel_unshuffle_3d": "worldfoundry.core.nn.blocks.volume",
    "velocity_to_denoised": "worldfoundry.core.nn.transformer.dit",
    "zero_module": "worldfoundry.core.nn.blocks.layers",
}


# ──────────────────────────────────────────────────────────────────────────
# Attribute resolve — import_module once, then cache on this package
# ──────────────────────────────────────────────────────────────────────────


def __getattr__(name: str) -> Any:
    """Materialize one public symbol on first access.

    Raises:
        AttributeError: ``name`` is not in the lazy export table.
    """

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Union materialized globals with the lazy ``__all__`` list for completion."""

    return sorted({*globals(), *__all__})


__all__ = sorted(_EXPORT_MODULES)
