"""DreamX UMT5 encoder backed by the canonical Wan T5 layers.

:class:`WanT5EncoderModel` is a naming / loading adapter
for DreamX-World so its checkpoints attach to
:mod:`~...model.WanTextEncoder`.

``context`` token contract is unchanged.  VAE counterpart:
:mod:`~...autoencoders.wan.variants.dreamx_world`.

Compatibility wrapper, not a new tokenizer.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import (
    T5LayerNorm,
    T5RelativeEmbedding,
    T5SelfAttention,
    init_weights,
)
from worldfoundry.core.model_loading import load_model
from worldfoundry.core.nn import ModuleDeviceDtypeMixin


class WanT5EncoderModel(ModuleDeviceDtypeMixin, nn.Module):
    """DreamX-World name for the canonical Wan UMT5 encoder."""
    def __init__(
        self,
        vocab: int | nn.Embedding,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) else nn.Embedding(vocab, dim)
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim,
                    dim_attn,
                    dim_ffn,
                    num_heads,
                    num_buckets,
                    shared_pos,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)
        self.apply(init_weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        value = self.dropout(self.token_embedding(input_ids))
        position = (
            self.pos_embedding(value.size(1), value.size(1))
            if self.pos_embedding is not None
            else None
        )
        for block in self.blocks:
            value = block(value, attention_mask, pos_bias=position)
        return (self.dropout(self.norm(value)),)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        additional_kwargs: dict | None = None,
        low_cpu_mem_usage: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "WanT5EncoderModel":
        del low_cpu_mem_usage
        parameters = set(inspect.signature(cls.__init__).parameters) - {"self", "cls"}
        config = {
            key: value
            for key, value in (additional_kwargs or {}).items()
            if key in parameters
        }
        return load_model(
            cls,
            pretrained_model_path,
            config=config,
            torch_dtype=torch_dtype,
            device="cpu",
        )
