# This file is originally from AnimateDiff/animatediff/models/motion_module.py at main · guoyww/AnimateDiff
# SPDX-License-Identifier: Apache-2.0 license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file was released under [ Apache-2.0 license], with the full license text available at [https://github.com/guoyww/AnimateDiff?tab=Apache-2.0-1-ov-file#readme].
from typing import Any

import torch.nn.functional as F
from einops import rearrange, repeat

from worldfoundry.base_models.three_dimensions.depth.temporal.attention import (
    apply_rotary_emb,
)
from worldfoundry.core.nn.blocks.layers import zero_module as zero_module

from .attention import CrossAttention

try:
    import xformers
    import xformers.ops  # noqa: F401 - Import checks optional dependency availability.

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


from worldfoundry.base_models.three_dimensions.depth.temporal.motion import (
    TemporalAttention as CanonicalTemporalAttention,
)
from worldfoundry.base_models.three_dimensions.depth.temporal.motion import TemporalModule as CanonicalTemporalModule
from worldfoundry.base_models.three_dimensions.depth.temporal.motion import (
    TemporalTransformer3DModel as CanonicalTemporalTransformer3DModel,
)
from worldfoundry.base_models.three_dimensions.depth.temporal.motion import (
    TemporalTransformerBlock as CanonicalTemporalTransformerBlock,
)


class TemporalModule(CanonicalTemporalModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, transformer_layer=TemporalTransformer3DModel, **kwargs)


class TemporalTransformer3DModel(CanonicalTemporalTransformer3DModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, block_layer=TemporalTransformerBlock, **kwargs)


class TemporalTransformerBlock(CanonicalTemporalTransformerBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, attention_layer=TemporalAttention, **kwargs)


class TemporalAttention(CanonicalTemporalAttention, CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)

        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)

        encoder_hidden_states = (
            repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
            if encoder_hidden_states is not None
            else encoder_hidden_states
        )

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if self.freqs_cis is not None:
            seq_len = query.shape[1]
            freqs_cis = self.freqs_cis[:seq_len].to(query.device)
            query, key = apply_rotary_emb(query, key, freqs_cis)

        if attention_mask is not None and attention_mask.shape[-1] != query.shape[1]:
            target_length = query.shape[1]
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        use_memory_efficient = XFORMERS_AVAILABLE and self._use_memory_efficient_attention_xformers
        if use_memory_efficient and (dim // self.heads) % 8 != 0:
            # print('Warning: the dim {} cannot be divided by 8. Fall into normal attention'.format(dim // self.heads))
            use_memory_efficient = False

        # attention, what we cannot get enough of
        if use_memory_efficient:
            query = self.reshape_heads_to_4d(query)
            key = self.reshape_heads_to_4d(key)
            value = self.reshape_heads_to_4d(value)

            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                raise NotImplementedError
                # hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
