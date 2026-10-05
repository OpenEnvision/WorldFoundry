# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""minWM HY15 cached attention; kernels and sequence parallelism are shared."""

import torch

from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)


@torch.compiler.disable
def sequence_parallel_attention_txt(
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    attn_mode=None,
    text_mask=None,
    attn_param=None,
    block_idx=None,
    kv_cache=None,
    cache_txt=False,
):
    encoder_query = q
    encoder_key = k
    encoder_value = v
    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
    if enable_sp:

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // sp_size
            return encoder_state.narrow(dim, sp_rank * local_heads, local_heads)

        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
    encoder_query = encoder_query.transpose(1, 2)
    encoder_key = encoder_key.transpose(1, 2)
    encoder_value = encoder_value.transpose(1, 2)
    t_kv = {}
    if cache_txt:
        t_kv["k_txt"] = encoder_key
        t_kv["v_txt"] = encoder_value
    encoder_hidden_states = scaled_dot_product_attention(
        encoder_query, encoder_key, encoder_value, dropout_p=0.0, is_causal=False
    )
    encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    if enable_sp:
        encoder_hidden_states = sequence_model_parallel_all_gather(encoder_hidden_states, dim=2).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(q.dtype)
    (b, s, a, d) = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(b, s, -1)
    return (encoder_hidden_states, t_kv)


@torch.compiler.disable
def sequence_parallel_attention_vision(q, k, v, block_idx=None, kv_cache=None, cache_vision=False):
    assert kv_cache is not None
    query = q
    key = k
    value = v
    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    if enable_sp:
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    cache_vision_key = kv_cache[block_idx]["k_vision"]
    cache_vision_value = kv_cache[block_idx]["v_vision"]
    vision_kv = {}
    if cache_vision:
        vision_kv["k_vision"] = key
        vision_kv["v_vision"] = value
    if cache_vision_key is not None:
        key = torch.cat([cache_vision_key, key], dim=2)
        value = torch.cat([cache_vision_value, value], dim=2)
    encoder_key = kv_cache[block_idx]["k_txt"]
    encoder_value = kv_cache[block_idx]["v_txt"]
    key = torch.cat([encoder_key, key], dim=2)
    value = torch.cat([encoder_value, value], dim=2)
    hidden_states = scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2)
    if enable_sp:
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        hidden_states = hidden_states.to(query.dtype)
    (b, s, a, d) = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)
    return (hidden_states, vision_kv)
