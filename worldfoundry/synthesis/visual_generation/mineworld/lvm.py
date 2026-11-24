"""
Wrap the Huggingface Transformers Llama to PyTorch Lightning Module.
"""

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP as LlamaMLP
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import (
    QwenRMSNorm as LlamaRMSNorm,
)
from worldfoundry.core.attention import (
    scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention,
)
from worldfoundry.core.attention.rotary.embedding import apply_rotary_pos_emb
from worldfoundry.synthesis.visual_generation.mineworld.diagonal_decoding import (
    decode_n_tokens,
    decode_one_token,
    decode_some_token,
    img_diagd_decode_n_tokens,
    prefill,
    video_diagd_decode_n_tokens,
)
from worldfoundry.synthesis.visual_generation.mineworld.utils import (
    get_obj_from_str,
    instantiate_from_config,
)

torch.backends.cuda.matmul.allow_tf32 = False

logger = logging.get_logger(__name__)


class LlamaLVM(torch.nn.Module):
    def __init__(
        self,
        transformer_config,
        model_class: str,
        tokenizer_config=None,
    ):
        super().__init__()
        self.config = instantiate_from_config(transformer_config)
        self.transformer = get_obj_from_str(model_class)(self.config)
        if tokenizer_config is not None:
            self.tokenizer = instantiate_from_config(tokenizer_config)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        device=None,
        config: LlamaConfig | None = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq, _ = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, device, dtype):
        """
        Set the cosine and sine cache for positional embeddings.

        Args:
            seq_len (int): The sequence length.
            device (str): The device on which the cache tensors will be stored.
            dtype: The data type of the cache tensors.
        """
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Forward pass of the LlamaRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape [bs, num_attention_heads, seq_len, head_size].
            seq_len (int): The sequence length. If greater than the cached length, the cache will be updated.

        Returns:
            tuple: A tuple containing two tensors, the cosine and sine embeddings, both of shape [1, 1, seq_len, dim].
        """
        if seq_len > self.max_position_embeddings:
            raise ValueError("seq length should less than max embedding")

        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        assert (self.head_dim * self.num_heads) == self.hidden_size, "hidden_size must be divisible by num_heads"
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.max_batch_size = getattr(config, "max_batch_size", 1)
        self.init_kv_cache()

    def init_kv_cache(self, dtype=torch.float16):
        cache_shape = (self.max_batch_size, self.max_position_embeddings, self.num_key_value_heads, self.head_dim)
        self.cache_k = torch.zeros(cache_shape, dtype=dtype).cuda()
        self.cache_v = torch.zeros(cache_shape, dtype=dtype).cuda()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_embedding=None,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = positions_embedding

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0), unsqueeze_dim=2
        )

        self.cache_k[:bsz, position_ids] = key_states
        self.cache_v[:bsz, position_ids] = value_states
        key_states, value_states = (
            self.cache_k[:bsz, :, :],
            self.cache_v[:bsz, :, :],
        )

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)

        query_states, key_states, value_states = (x.transpose(1, 2) for x in (query_states, key_states, value_states))

        attn_output = (
            _worldfoundry_scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, bias=config.mlp_bias)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        positions_embedding=None,
    ):
        """
        Forward pass for the LlamaDecoderLayer.

        Args:
            hidden_states (torch.FloatTensor): Input tensor of shape `(batch, seq_len, embed_dim)`.
            attention_mask (torch.FloatTensor, optional): Attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (torch.LongTensor, optional): Positional IDs tensor.


        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: Tuple containing:
                - hidden_states (torch.FloatTensor): Output tensor.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            positions_embedding=positions_embedding,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.max_position_embedding = config.max_position_embeddings
        self.causal_mask = torch.tril(
            torch.ones(self.max_position_embedding, self.max_position_embedding, dtype=torch.bool)
        ).cuda()
        self.post_init()

    def _create_attention_mask(self, input_pos: torch.Tensor | None):
        """
        Creates an attention mask for the transformer layers.

        Args:
            input_pos[torch.Tensor]: The position of input sequence (used for inference only).

        Returns:
            Optional[torch.Tensor]: The attention mask, or None for causal mask.
        """
        mask = self.causal_mask[input_pos]
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        if input_ids is None:
            raise ValueError("decoder_input_ids is None")
        hidden_states = self.embed_tokens(input_ids)

        positions_embedding = self.rotary_emb(hidden_states, seq_len=self.max_position_embedding)

        attention_mask = self._create_attention_mask(input_pos=position_ids)
        for _idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                positions_embedding=positions_embedding,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        logits = self.lm_head(outputs[:, :, :])
        return logits

    def refresh_kvcache(self):
        for i in self.model.layers:
            i.self_attn.init_kv_cache()

    def naive_generate(self, input_ids, max_new_tokens, temperature=1.0, action_all=None, top_p=None, top_k=None):
        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        if action_all is not None:
            input_ids = torch.cat([input_ids, action_all[0]], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        self.decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=True)
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")

        generated_tokens = decode_n_tokens(
            self,
            input_ids=next_token.view(1, -1),
            position_ids=position_ids,
            num_generate_tokens=max_new_tokens - 1,
            temperature=temperature,
            decode_one_token_function=self.decode_one_token,
            action=action_all,
            top_p=top_p,
            top_k=top_k,
        )
        return torch.cat(generated_tokens, dim=1)

    def img_diagd_generate(
        self, input_ids, max_new_tokens, temperature=1.0, action_all=None, windowsize=2, top_p=None, top_k=None
    ):
        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        input_ids = torch.cat([input_ids, action_all[0]], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        self.decode_some_token = torch.compile(decode_some_token, mode="max-autotune", fullgraph=True)
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")

        generated_tokens = img_diagd_decode_n_tokens(
            self,
            input_ids=next_token.view(1, -1),
            position_ids=position_ids,
            num_generate_tokens=max_new_tokens - 1,
            temperature=temperature,
            decode_some_token_function=self.decode_some_token,
            windowsize=windowsize,
            action=action_all,
            prompt=input_ids,
            top_k=top_k,
            top_p=top_p,
        )
        return torch.cat(generated_tokens, dim=1)

    def vid_diagd_generate(self, input_ids, max_new_tokens, windowsize=2, temperature=1.0, action_all=None, **kwargs):
        self.prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        input_ids = torch.cat([input_ids, action_all[0]], dim=-1)
        position_ids = torch.arange(0, input_ids.shape[1], device="cuda")
        next_token = self.prefill(
            self,
            input_ids=input_ids,
            position_ids=position_ids,
            temperature=temperature,
        )

        self.decode_some_token = torch.compile(decode_some_token, mode="max-autotune", fullgraph=True)
        # self.decode_some_token = decode_some_token
        position_ids = torch.tensor([input_ids.shape[1]], dtype=torch.long, device="cuda")

        generated_tokens = video_diagd_decode_n_tokens(
            self,
            input_ids=next_token.view(1, -1),
            position_ids=position_ids,
            num_generate_tokens=max_new_tokens - 1,
            temperature=temperature,
            decode_some_token_function=self.decode_some_token,
            windowsize=windowsize,
            action=action_all,
            prompt=input_ids,
            **kwargs,
        )
        return torch.cat(generated_tokens, dim=1)
