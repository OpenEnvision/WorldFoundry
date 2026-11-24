# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
from typing import Any

import einops
import flax.nnx as nnx
import jax.numpy as jnp

from ...openpi import array_typing as at
from .memory_encoder import FeatureEncoder
from .memory_utils import kernel_init
from .rmt import RMTLayer
from .ttt import TTTLayerLinear


class RecurrentMemory(nnx.Module):
    def __init__(
        self,
        config,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.float32,
    ):
        self.cfg = config

        self.hidden_dim = self.cfg.recurrent_memory.hidden_dim
        self.output_dim = self.cfg.memory_token_dim
        self.recur_type = self.cfg.recurrent_memory.type
        self.budget = self.cfg.budget

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.cfg.memory_feature.img.input_dim,
            pos_input_dim=self.cfg.memory_feature.pos.input_dim,
            state_input_dim=self.cfg.memory_feature.state.input_dim,
            pos_output_dim=self.cfg.memory_feature.pos.hidden_dim,
            state_output_dim=self.cfg.memory_feature.state.hidden_dim,
            ouput_dim_for_recur=self.hidden_dim,
            output_dim_for_percep=None,
            use_pos_emb=self.cfg.use_pos_emb,
            use_state_emb=self.cfg.use_state_emb,
        )
        self.max_seq_len = self.cfg.recurrent_memory.max_input_tokens
        self.mini_batch_size = self.cfg.recurrent_memory.mini_batch_size

        self.pre_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)
        if self.recur_type == "ttt":
            self.recur_layer = TTTLayerLinear(config=self.cfg, rngs=rngs)
        else:
            self.recur_layer = RMTLayer(config=self.cfg, rngs=rngs)

        self.proj = nnx.Linear(
            self.hidden_dim,
            self.output_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(
        self,
        recur_image_emb: at.Float[at.Array, "b t v p d1"],  # left padded
        recur_mask: at.Bool[at.Array, "b t"],  # left padded
        recur_pos_emb: at.Float[at.Array, "b t v p d2"] | None = None,
        recur_state_emb: at.Float[at.Array, "b t d3"] | None = None,
        memory_state: Any | None = None,
    ):
        _memory_state = self.recur_layer.reset() if memory_state is None else memory_state

        hidden_states = self.feature_encoder.encode_recurrent_memory(recur_image_emb, recur_pos_emb, recur_state_emb)
        _, t, v, p, _ = hidden_states.shape
        hidden_states = einops.rearrange(hidden_states, "b t v p d -> b (t v p) d")
        assert self.max_seq_len == t * v * p

        num_mini_batches = self.max_seq_len // self.mini_batch_size

        hidden_states = einops.rearrange(
            hidden_states,
            "b (nm mb) d -> b nm mb d",
            mb=self.mini_batch_size,
            nm=num_mini_batches,
        )

        output, memory_state_new, stats = self.recur_layer(hidden_states, recur_mask, _memory_state)

        final_output = self.proj(output[:, -self.budget :])

        if self.recur_type == "ttt":
            mask_repeat = einops.repeat(recur_mask, "b t -> b (t mb)", mb=self.mini_batch_size)
            final_mask = mask_repeat[:, -self.budget :]
        else:
            final_mask = jnp.ones((hidden_states.shape[0], self.budget), dtype=jnp.bool_)

        return (final_output, final_mask), memory_state_new, stats

    def reset(self):
        # return memory state
        return self.recur_layer.reset()
