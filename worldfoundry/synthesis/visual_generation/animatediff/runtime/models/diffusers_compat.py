from __future__ import annotations

import torch

try:
    from diffusers.models.modeling_utils import ModelMixin, load_state_dict
except ImportError:  # diffusers < 0.20
    from diffusers.modeling_utils import ModelMixin as ModelMixin
    from diffusers.modeling_utils import load_state_dict as load_state_dict

try:
    from diffusers.models.attention import AdaLayerNorm, CrossAttention, FeedForward
except ImportError:  # diffusers >= 0.38 renamed CrossAttention to Attention.
    from diffusers.models.attention import AdaLayerNorm as AdaLayerNorm
    from diffusers.models.attention import FeedForward as FeedForward
    from diffusers.models.attention_processor import Attention

    class CrossAttention(Attention):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._use_memory_efficient_attention_xformers = False
            self._slice_size = None

        def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.head_to_batch_dim(tensor)

        def _attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            return self.batch_to_head_dim(hidden_states)

        def _sliced_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            sequence_length: int,
            dim: int,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self._attention(query, key, value, attention_mask)

        def _memory_efficient_attention_xformers(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            import xformers.ops

            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            return self.batch_to_head_dim(hidden_states)
