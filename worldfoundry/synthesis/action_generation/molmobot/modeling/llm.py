"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import dataclasses
import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import field

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from ..config import BaseConfig, StrEnum
from ..tokenizer import TokenizerConfig
from ..torch_utils import BufferCache, ensure_finite_

log = logging.getLogger(__name__)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class AttentionLayerNormType(StrEnum):
    olmo = "olmo"
    """
    Use the Attention LayerNorm (QK-norm) from OLMo.
    """

    qwen3 = "qwen3"
    """
    Use the Attention LayerNorm (QK-norm) from Qwen3.
    """


class ActivationType(StrEnum):
    quick_gelu = "quick_gelu"
    gelu = "gelu"
    gelu_pytorch_tanh = "gelu_pytorch_tanh"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class LlmActivationCheckpointMode(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    two_in_three = "two_in_three"
    """
    Checkpoint two out of every three transformer layers.
    """

    three_in_four = "three_in_four"
    """
    Checkpoint three out of four of every transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """


class AttentionType(StrEnum):
    """Attention to use"""

    sdpa = "sdpa"
    direct = "direct"
    flash = "flash"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """

    moe = "moe"


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


class RopeType(StrEnum):
    default = "default"
    llama3 = "llama3"
    yarn = "yarn"


def init_weights(
    config: LlmConfig,
    module: nn.Linear | nn.Embedding,
    d: int | None = None,
    layer_id: int | None = None,
    std_factor: float = 1.0,
    type_of_module: ModuleType | None = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            if hasattr(module, "weight"):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
            else:
                nn.init.trunc_normal_(module, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            if hasattr(module, "weight"):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                nn.init.normal_(module, mean=0.0, std=std)
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def init_normal(
    module: nn.Linear | nn.Embedding,
    std: float,
    init_cutoff_factor: float | None = None,
):
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        if hasattr(module, "weight"):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.trunc_normal_(module, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        if hasattr(module, "weight"):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        else:
            nn.init.normal_(module, mean=0.0, std=std)

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


@dataclasses.dataclass
class LlmConfig(BaseConfig):
    """Configuration for a multi-layer transformer"""

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: int | None = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for Llama2-style grouped query attention.
    """

    head_dim: int | None = None
    """
    The head dimensionality for the attention mechanism.
    """

    qkv_bias: bool = False  # qwen models use bias in kvq layers
    """
    Do QKV projection a bias
    """

    clip_qkv: float | None = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: int | None = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: float = 10000.0
    """
    RoPE theta parameter.
    """

    rope_type: RopeType = RopeType.default
    """
    RoPE type to use. Default is the original RoPE, llama3 is the new RoPE used in Llama3.
    """

    rope_factor: float | None = None
    """
    RoPE scaling factor. Used for Llama3/YaRN style RoPE.
    """

    rope_high_freq_factor: float | None = None
    """
    RoPE high frequency scaling factor. Used for Llama3 style RoPE.
    """

    rope_low_freq_factor: float | None = None
    """
    RoPE low frequency scaling factor. Used for Llama3 style RoPE.
    """

    rope_original_max_position_embeddings: int | None = None
    """
    The original max position embeddings used during pretraining.
    Used for Llama3/YaRN style RoPE.
    """

    rope_attention_factor: float | None = None
    """
    RoPE scaling factor to be applied to the computed cos/sin.
    If None, the value is inferred from `factor`, `mscale`, and `mscale_all_dim` as avaialble.
    Used for YaRN style RoPE.
    """

    rope_beta_fast: float | None = None
    """
    RoPE parameter to set the boundary for extrapolation (only) in the linear ramp function.
    Used for YaRN style RoPE.
    """

    rope_beta_slow: float | None = None
    """
    RoPE parameter to set the boundary for interpolation (only) in the linear ramp function.
    Used for YaRN style RoPE.
    """

    rope_mscale: float | None = None
    """
    Used for YaRN style RoPE.
    If `rope_attention_factor` is None and both `rope_mscale` and `rope_mscale_all_dim` are provided,
    `rope_mscale` acts scalar augmenting `log(factor)` when computing the numerator for the inferred
    value of `rope_attention_factor`. If not provided, `attention_factor` will be calculated based on
    `factor` only.
    """

    rope_mscale_all_dim: float | None = None
    """
    Used for YaRN style RoPE.
    If `rope_attention_factor` is None and both `rope_mscale` and `rope_mscale_all_dim` are provided,
    `rope_mscale_all_dim` acts scalar augmenting `log(factor)` when computing the denominator
    for the inferred value of `rope_attention_factor`. If not provided, `attention_factor` will be
    calculated based on `factor` only.
    """

    rope_truncate: bool | None = None
    """
    Whether to truncate the correction range. Used for YaRN style RoPE.
    """

    attention_type: AttentionType = AttentionType.sdpa
    """
    Attention implementation to use.
    """

    full_attention_layers: tuple[int] | None = None
    """
    If provided, apply full attention to the layers in the list while applying sliding attention to the rest.
    """

    sliding_attention_rope_scaling: bool = False
    """
    A legacy flag to use rope scaling for sliding attention.
    """

    float32_attention: bool = True
    """
    Compute attention in float32
    """

    attention_dropout: float = 0.0
    """
    The dropout probability within the attention modules.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    attention_layer_norm_type: AttentionLayerNormType = AttentionLayerNormType.olmo
    """
    The type of attention layer norm to use.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    response_residual_dropout: float = 0.0
    """
    Dropout applied only to loss/response tokens
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    layer_norm_eps: float | None = None
    """
    epsilon for layer norms
    """

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    max_position_embeddings: int | None = None
    """
    Max positional embeddings to use in RoPE cache
    """

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    """

    bias_for_layer_norm: bool | None = None
    """
    Whether or not to include bias parameters in layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    norm_after: bool = False

    moe_num_experts: int | None = 8
    """
    The number of experts to use in the MoE block.
    """

    moe_top_k: int | None = 2
    """
    The number of experts to select for each token.
    """

    moe_mlp_impl: str | None = "sparse"
    """
    Choose "grouped" for grouped GEMM installable via `pip install git+https://git@github.com/tgale96/grouped_gemm.git@66c7195e35e8c4f22fa6a014037ef511bfa397cb`.
    """

    moe_log_expert_assignment: bool | None = False
    """
    Whether to log the expert assignment.
    """

    moe_shared_expert: bool | None = False
    """
    Whether to have an always-used expert like in [DeepSeekMoE](https://arxiv.org/abs/2401.06066).
    """

    moe_lbl_in_fp32: bool | None = False
    """
    Whether to perform load balancing in FP32.
    """

    moe_interleave: bool | None = False
    """
    Interleave sequential with MoE blocks starting with sequential.
    """

    moe_loss_weight: float | None = 0.1
    """
    The weight to use for the MoE load balancing loss.
    """

    moe_zloss_weight: float | None = None
    """
    Weight for MoE router z-loss where None means no router z-loss. 0.001 is a common value.
    """

    moe_dropless: bool | None = True
    """
    Whether to use [dMoE](https://arxiv.org/abs/2211.15841).
    """

    moe_capacity_factor: float | None = 1.25
    """
    The capacity factor to use in the MoE block. Only applies if not using dMoE.
    """

    embedding_dropout: float = 0.1
    """The dropout probability for embeddings."""

    scale_logits: bool = False
    """If ``True``, scale the output logits by ``1 / sqrt(d_model)``."""

    vocab_size: int = 50257
    """Vocabulary size of the model."""

    additional_vocab_size: int | None = 128
    """Number of additional tokens to have the input embeddings for"""

    weight_tying: bool = True
    """Whether to tie output linear weights to the input embedding"""

    embedding_size: int | None = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    use_position_ids: bool = True
    """
    Whether to use position IDs in the model.
    The model operation regarding positional embeddings changes depending on this variable.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    init_path: str | None = None
    """Path to initial the LLM with"""

    init_incremental: int | None = None

    new_embedding_init_range: float = 0.02
    """How to initialize embedding for new tokens"""

    initializer_range: float = 0.02
    """
    standard deviation to for initializing the weight models
    """

    normalize_input_embeds: bool = False
    """
    Normalize input embeddings (both for text and images) before
    """

    activation_checkpoint: LlmActivationCheckpointMode | None = LlmActivationCheckpointMode.whole_layer
    """
    Where to use activation checkpoint if activation_checkpoint is enabled
    """

    compile: str | None = "blocks"
    """
    How to compile the transformer if compilation is requested
    """

    fix_pad_tokenizer: bool = False
    """
    Use embedding_size instead of vocab_size for padding the tokenizer
    """

    init_std: float | None = 0.02
    init_fn: InitFnType = InitFnType.normal
    init_cutoff_factor: float | None = None

    def __post_init__(self):
        if self.full_attention_layers is not None:
            self.full_attention_layers = tuple(self.full_attention_layers)  # type: ignore[assignment]

    def build(self, cache, device=None) -> Llm:
        return Llm(self, cache, device)

    def build_tokenizer(self):
        pad_tokenizer_to = self.embedding_size or self.vocab_size if self.fix_pad_tokenizer else self.vocab_size
        return self.tokenizer.build(pad_tokenizer_to=pad_tokenizer_to)

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            return self.n_heads
        else:
            return self.n_kv_heads

    def should_checkpoint_block(self, block_idx: int) -> bool:
        strategy = self.activation_checkpoint
        if strategy is None:
            return False
        return bool(
            (strategy == LlmActivationCheckpointMode.whole_layer)
            or (strategy == LlmActivationCheckpointMode.one_in_two and block_idx % 2 == 0)
            or (strategy == LlmActivationCheckpointMode.one_in_three and block_idx % 3 == 0)
            or (strategy == LlmActivationCheckpointMode.one_in_four and block_idx % 4 == 0)
            or (strategy == LlmActivationCheckpointMode.two_in_three and block_idx % 3 != 0)
            or (strategy == LlmActivationCheckpointMode.three_in_four and block_idx % 4 != 0)
        )

    @classmethod
    def update_legacy_settings(cls, config):
        """Remove deprecated keys from old checkpoints."""
        # Remove deprecated keys that are no longer used
        deprecated_keys = ["mm_rope", "debug"]
        for key in deprecated_keys:
            if key in config:
                log.warning(f"Removing deprecated '{key}' key from {cls.__name__}.")
                del config[key]
        return config


class Llm(nn.Module):
    def __init__(self, config: LlmConfig, cache, device):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([OLMoBlock.build(i, config, cache, device) for i in range(config.n_layers)])
        self.ln_f = LayerNorm.build(config)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        if config.additional_vocab_size is not None:
            self.wte = Embedding(
                config.embedding_size or config.vocab_size,
                config.additional_vocab_size,
                config.d_model,
                device=device,
                initializer_range=config.initializer_range,
                new_embed_initializer_range=config.new_embedding_init_range,
            )
        else:
            self.wte = nn.Embedding(config.embedding_size or config.vocab_size, config.d_model, device=device)
        if not config.weight_tying:
            self.ff_out = nn.Linear(
                config.d_model,
                config.embedding_size or config.vocab_size,
                bias=config.include_bias,
                device=device,
            )

    def reset_parameters(self) -> None:
        if self.config.additional_vocab_size:
            init_weights(
                self.config,
                self.wte.embedding,  # type: ignore
                std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
                type_of_module=ModuleType.emb,
            )
            if self.config.additional_vocab_size is not None:
                nn.init.normal_(self.wte.new_embedding, std=self.config.new_embedding_init_range)
        else:
            init_weights(
                self.config,
                self.wte.weight,  # type: ignore
                std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
                type_of_module=ModuleType.emb,
            )

        self.ln_f.reset_parameters()
        if hasattr(self, "ff_out"):
            init_weights(self.config, self.ff_out, type_of_module=ModuleType.final_out)
        for block in self.blocks:
            block.reset_parameters()

    def apply_compile(self, **kwargs):
        if self.config.compile == "blocks":
            for block in self.blocks:
                block.compile(**kwargs)
        elif self.config.compile is not None:
            raise NotImplementedError(self.config.compile)

    # No forward method since this is only used as part of a `Molmo` model


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: str | torch.device,
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def reset_parameters(self):
        nn.init.normal_(self.embedding, std=self.initializer_range)
        nn.init.normal_(self.new_embedding, std=self.new_embed_initializer_range)

    def forward(self, x: torch.Tensor, logits=False) -> torch.Tensor:
        if logits:
            # Used for computing the logits with weight tying being used
            # We re-use the forward method since using the raw weight can cause errors with FSDP
            return F.linear(x, self.embedding, None)
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class Dropout(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        mask_p: float = 0,
        broadcast_dims: Sequence[int] = (),
    ):
        super().__init__(p, inplace)
        self.mask_p = mask_p
        self.broadcast_dims = broadcast_dims

    def forward(self, input: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param input: A tensor of shape `(batch_size, seq_len, embed_dim)`
        :param drop_mask: A tensor of shape `(batch_size, seq_len)` with values of zero or one.
        """
        if self.p == 0.0 and (self.mask_p is None or self.mask_p == 0.0):
            return input
        else:
            if self.mask_p > 0.0 and self.training:
                assert drop_mask is not None
                drop_mask = drop_mask.to(input.dtype)
                keep_prob = 1.0 - self.p
                keep_prob2 = 1.0 - self.mask_p
                keep_prob = drop_mask * keep_prob2 + (1 - drop_mask) * keep_prob
                keep_prob = keep_prob.unsqueeze(-1)
                dropout_shape = list(input.shape)
                keep_prob = keep_prob.broadcast_to(dropout_shape)
                multiplier = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier.div_(keep_prob)
                return input * multiplier
            elif self.p > 0.0 and len(self.broadcast_dims) > 0 and self.training:
                keep_prob = 1.0 - self.p
                dropout_shape = list(input.shape)
                for dim in self.broadcast_dims:
                    dropout_shape[dim] = 1
                keep = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier = keep.broadcast_to(input.shape)
                multiplier.div_(keep_prob)
                input = input * multiplier
            else:
                return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: LlmConfig,
        *,
        size: int | None = None,
        elementwise_affine: bool | None = True,
        eps: float = 1e-05,
        weight_initializer: Callable | None = torch.ones,
        bias_initializer: Callable | None = torch.zeros,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.eps = self.config.layer_norm_eps or eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(weight_initializer(self.normalized_shape, device=device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(bias_initializer(self.normalized_shape, device=device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: LlmConfig, size: int | None = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: LlmConfig,
        size: int | None = None,
        low_precision: bool = False,
        elementwise_affine: bool | None = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: LlmConfig,
        size: int | None = None,
        elementwise_affine: bool | None = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: LlmConfig, cache: BufferCache, device=None, rope_type: RopeType | None = None):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rope_type = rope_type or config.rope_type

    def apply_llama3_scaling_factor(self, inv_freq: torch.Tensor) -> torch.Tensor:
        factor = self.config.rope_factor
        low_freq_factor = self.config.rope_low_freq_factor
        high_freq_factor = self.config.rope_high_freq_factor
        old_context_len = self.config.rope_original_max_position_embeddings
        if any(x is None for x in [factor, low_freq_factor, high_freq_factor, old_context_len]):
            return inv_freq

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq

        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def compute_yarn_inv_freq(self, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
        def get_mscale(scale, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        # Compute the inverse frequencies
        def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
            """Inverse dimension formula to find the dimension based on the number of rotations"""
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
            """Find dimension range bounds based on rotations"""
            low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
            high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        factor = self.config.rope_factor
        attention_factor = self.config.rope_attention_factor
        mscale = self.config.rope_mscale
        mscale_all_dim = self.config.rope_mscale_all_dim
        original_max_position_embeddings = (
            self.config.rope_original_max_position_embeddings or self.config.max_position_embeddings
        )

        if attention_factor is None:
            if mscale and mscale_all_dim:
                attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
            else:
                attention_factor = get_mscale(factor)

        beta_fast = self.config.rope_beta_fast or 32
        beta_slow = self.config.rope_beta_slow or 1

        # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
        # to expand the possible context length. In other words, interpolation = apply scaling factor.
        pos_freqs = self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        truncate = self.config.rope_truncate or True
        low, high = find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            self.config.rope_theta,
            original_max_position_embeddings,
            truncate,
        )

        # Get n-dimensional rotational scaling corrected for extrapolation
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

        return inv_freq, attention_factor

    def compute_rope_parameters(
        self,
        seq_len: int,
        dim: int,
        device: torch.device,
        rope_type: RopeType,
    ) -> torch.Tensor:
        if rope_type == RopeType.default:
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            attention_factor = 1.0
        elif rope_type == RopeType.llama3:
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            inv_freq = self.apply_llama3_scaling_factor(inv_freq)
            attention_factor = 1.0
        else:  # rope_type == RopeType.yarn
            inv_freq, attention_factor = self.compute_yarn_inv_freq(seq_len, dim, device)

        return inv_freq, attention_factor

    def get_rotary_embedding(
        self,
        seq_len: int,
        device: torch.device,
        rope_type: RopeType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_type = rope_type or self.rope_type
        prefix = f"{rope_type!s}_" if rope_type != RopeType.default else ""
        sin_key = f"{prefix}rope_pos_sin"
        cos_key = f"{prefix}rope_pos_cos"
        if (
            (pos_sin := self.__cache.get(sin_key)) is not None
            and (pos_cos := self.__cache.get(cos_key)) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                # This guard seems to prevent certain kinds of compiling errors and graphs breaks,
                # presumably due to the buffer cache modification confusing the compiler,
                # but it is hard to pin down why its sometimes needed and sometimes isn't
                if not torch.compiler.is_compiling():
                    self.__cache[sin_key] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                if not torch.compiler.is_compiling():
                    self.__cache[cos_key] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = (
                self.config.head_dim if self.config.head_dim is not None else self.config.d_model // self.config.n_heads
            )
            inv_freq, attention_factor = self.compute_rope_parameters(seq_len, dim, device, rope_type)
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
            pos_sin, pos_cos = pos_sin * attention_factor, pos_cos * attention_factor
        self.__cache[sin_key] = pos_sin
        self.__cache[cos_key] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, hs // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(B, nh, T, hs)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings on the ordinary inference path."""
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            batch_size = q_.shape[0]
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            cache_length = (
                self.config.max_position_embeddings or self.config.max_sequence_length
                if position_ids is not None
                else key_len
            )
            pos_sin, pos_cos = self.get_rotary_embedding(cache_length, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if position_ids is not None:
                if query_len != key_len:
                    raise ValueError("Query and key lengths must match when position_ids are provided.")
                pos_sin = pos_sin[0, 0][position_ids].view(batch_size, 1, key_len, pos_sin.shape[-1])
                pos_cos = pos_cos[0, 0][position_ids].view(batch_size, 1, key_len, pos_cos.shape[-1])
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len],
                pos_cos[:, :, key_len - query_len : key_len],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, activation_type, split_inputs=False) -> Activation:
        if split_inputs:
            if activation_type == ActivationType.swiglu:
                return LlamaSwiGLU()
            else:
                raise NotImplementedError(f"Unknown two-input activation: '{activation_type}'")
        else:
            if activation_type == ActivationType.quick_gelu:
                return QuickGELU()
            elif activation_type == ActivationType.gelu:
                return GELU(approximate="none")
            elif activation_type == ActivationType.gelu_pytorch_tanh:
                return GELU(approximate="tanh")
            elif activation_type == ActivationType.relu:
                return ReLU(inplace=False)
            elif activation_type == ActivationType.silu:
                return SiLU(inplace=False)
            elif activation_type == ActivationType.swiglu:
                return SwiGLU()
            else:
                raise NotImplementedError(f"Unknown activation: '{activation_type}'")


class QuickGELU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @property
    def output_multiplier(self) -> float:
        return 1.0


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class LlamaSwiGLU(Activation):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.silu(x1) * x2

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: LlmConfig, cache: BufferCache, device=None):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        if config.head_dim is None:
            assert config.d_model % config.n_heads == 0
        self.head_dim = config.head_dim if config.head_dim is not None else config.d_model // config.n_heads
        # Retained as a constant sentinel so the shared block implementations
        # take their direct inference path.
        self.fine_grained_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout, mask_p=config.response_residual_dropout)

        # Layer norms.
        self.k_norm: LayerNormBase | None = None
        self.q_norm: LayerNormBase | None = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            k_norm_size = (
                self.head_dim
                if config.attention_layer_norm_type == AttentionLayerNormType.qwen3
                else self.head_dim * config.effective_n_kv_heads
            )
            self.k_norm = LayerNormBase.build(
                config,
                size=k_norm_size,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            q_norm_size = self.head_dim if config.attention_layer_norm_type == AttentionLayerNormType.qwen3 else None
            self.q_norm = LayerNormBase.build(
                config, size=q_norm_size, elementwise_affine=config.attention_layer_norm_with_affine
            )

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=config.block_type == BlockType.llama)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        input_dim = config.head_dim * config.n_heads if config.head_dim is not None else config.d_model
        self.attn_out = nn.Linear(input_dim, config.d_model, bias=config.include_bias, device=device)

        if self.config.block_type != BlockType.moe:
            # Feed-forward output projection.
            self.ff_out = nn.Linear(
                int(self.act.output_multiplier * self.hidden_size),
                config.d_model,
                bias=config.include_bias,
                device=device,
            )
            self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            rope_type: RopeType | None = None
            if config.full_attention_layers is not None:
                if config.sliding_attention_rope_scaling:
                    rope_type = RopeType.default if layer_id in config.full_attention_layers else config.rope_type
                else:
                    rope_type = config.rope_type if layer_id in config.full_attention_layers else RopeType.default
            self.rotary_emb = RotaryEmbedding(config, self.__cache, device, rope_type=rope_type)

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        if k.size(1) != v.size(1) or q.size(1) % k.size(1) != 0:
            raise ValueError(f"Invalid grouped-query attention heads: q={q.size(1)}, k={k.size(1)}, v={v.size(1)}.")
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=q.size(1) != k.size(1),
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        value_scaling=None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if (
            self.q_norm is not None
            and self.k_norm is not None
            and self.config.attention_layer_norm_type != AttentionLayerNormType.qwen3
        ):
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, self.head_dim)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, self.head_dim)
        if (
            self.q_norm is not None
            and self.k_norm is not None
            and self.config.attention_layer_norm_type == AttentionLayerNormType.qwen3
        ):
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, self.head_dim).transpose(1, 2)

        if self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if value_scaling is not None:
            v = v * value_scaling[:, None, :, None]

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype)

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            drop_mask=drop_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: LlmConfig, cache: BufferCache, device=None) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            module = OLMoSequentialBlock(layer_id, config, cache, device)
        elif config.block_type == BlockType.llama:
            module = OLMoLlamaBlock(layer_id, config, cache, device)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")
        return module


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: LlmConfig, cache: BufferCache, device=None):
        super().__init__(layer_id, config, cache, device)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)

        self.fused_dims = (
            config.n_heads * self.head_dim,
            config.effective_n_kv_heads * self.head_dim,
            config.effective_n_kv_heads * self.head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias or config.qkv_bias, device=device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=device)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        value_scaling: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        if not self.config.norm_after:
            if self.fine_grained_checkpoint_fn is not None:
                atten_in = self.fine_grained_checkpoint_fn(self.attn_norm, x)
            else:
                atten_in = self.attn_norm(x)
        else:
            atten_in = x
        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores. Context-parallel training is intentionally not
        # part of the in-tree action inference runtime.
        if self.fine_grained_checkpoint_fn is not None:
            att, cache = self.fine_grained_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                value_scaling=value_scaling,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                value_scaling=value_scaling,
            )

        if self.config.norm_after:
            if self.fine_grained_checkpoint_fn is not None:
                att = self.fine_grained_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att, drop_mask=drop_mask)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self.fine_grained_checkpoint_fn is not None:
                x = self.fine_grained_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)
        if self.fine_grained_checkpoint_fn is not None:  # noqa: SIM108 - Branches carry per-case comments.
            x = self.fine_grained_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self.fine_grained_checkpoint_fn is not None:
                x = self.fine_grained_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: LlmConfig, cache: BufferCache, device=None):
        super().__init__(layer_id, config, cache, device)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        q_proj_out_dim = config.n_heads * self.head_dim
        k_proj_out_dim = config.effective_n_kv_heads * self.head_dim
        v_proj_out_dim = config.effective_n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.d_model, q_proj_out_dim, bias=config.qkv_bias, device=device)
        self.k_proj = nn.Linear(config.d_model, k_proj_out_dim, bias=config.qkv_bias, device=device)
        self.v_proj = nn.Linear(config.d_model, v_proj_out_dim, bias=config.qkv_bias, device=device)

        # Feed-forward input projection.
        self.ff_proj1 = nn.Linear(config.d_model, self.hidden_size // 2, bias=False, device=device)
        self.ff_proj2 = nn.Linear(config.d_model, self.hidden_size // 2, bias=False, device=device)
        if self.config.norm_after:
            raise NotImplementedError()

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj1, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj2, d=self.config.d_model, layer_id=None)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        response_dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # For GQA
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        og_dtype = q.dtype
        k = k.to(q.device)
        v = v.to(q.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        assert response_dropout_p == 0.0, "Response dropout is not supported in Llama."

        if self.config.float32_attention:
            q, k = q.to(torch.float), k.to(torch.float)

        if self.config.attention_type == AttentionType.direct:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

            if is_causal:
                assert attn_mask is None

                query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
                attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
            elif attn_mask is not None:
                attn_bias = attn_mask
            else:
                attn_bias = torch.zeros_like(attn_weights)

            attn_weights += attn_bias

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training).to(v.dtype)

            att = torch.matmul(attn_weights, v)
        elif self.config.attention_type == AttentionType.sdpa:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError(self.config.attention_type)
        att = att.to(og_dtype)
        return att

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        drop_mask: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        if self.fine_grained_checkpoint_fn is not None:
            att, cache = self.fine_grained_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att, drop_mask=drop_mask)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self.fine_grained_checkpoint_fn is not None:
            x = self.fine_grained_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x1 = self.ff_proj1(x)
        x2 = self.ff_proj2(x)
        if self.fine_grained_checkpoint_fn is not None:
            x = self.fine_grained_checkpoint_fn(self.act, x1, x2)  # type: ignore
        else:
            x = self.act(x1, x2)
        x = self.ff_out(x)
        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache
