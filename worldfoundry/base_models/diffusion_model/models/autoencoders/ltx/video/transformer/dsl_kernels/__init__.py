"""Blackwell CuTe DSL pathway for DiffVAE (deferred stage-4 + fused stage-5)."""

from .apply_dsl import enable_blackwell_dsl
from .attn import NA_SOFTMAX_BOUND_BUFFER, DSLAttention, na_attention_dsl, na_dsl_available, require_softmax_bound
from .block import FUSED_CTX_BIAS_BUFFER, FUSED_CTX_WEIGHT_BUFFER, DSLDiffusionNABlock
from .chain import DSLChannelLinear, DSLDiffusionBlockChain, alloc_block_volume, linear_into_block_volume
from .joint_attn import DSLJointAttention, dsl_joint_supported, na_attention_joint_dsl

__all__ = [
    "FUSED_CTX_BIAS_BUFFER",
    "FUSED_CTX_WEIGHT_BUFFER",
    "NA_SOFTMAX_BOUND_BUFFER",
    "DSLAttention",
    "DSLChannelLinear",
    "DSLDiffusionBlockChain",
    "DSLDiffusionNABlock",
    "DSLJointAttention",
    "alloc_block_volume",
    "dsl_joint_supported",
    "enable_blackwell_dsl",
    "linear_into_block_volume",
    "na_attention_dsl",
    "na_attention_joint_dsl",
    "na_dsl_available",
    "require_softmax_bound",
]
