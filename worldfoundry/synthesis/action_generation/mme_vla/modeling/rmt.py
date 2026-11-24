# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
from functools import partial

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import vmap

from ...openpi import array_typing as at
from ...openpi import sharding
from .memory_utils import (
    apply_rotary_emb,
    kernel_init,
    kernel_init_out_proj_rmt,
    precompute_freqs_cis,
)


class RMTLayer(nnx.Module):
    def __init__(
        self,
        config,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.mem_slots = config.budget
        self.hidden_dim = config.recurrent_memory.hidden_dim
        self.num_heads = config.recurrent_memory.rmt_config.num_attn_heads
        self.num_kv_heads = config.recurrent_memory.rmt_config.num_kv_heads
        self.mini_batch_size = self.config.recurrent_memory.mini_batch_size
        self.head_dim = self.hidden_dim // self.num_heads

        self.dtype = dtype = jnp.float32

        self.output_stats = config.recurrent_memory.output_stats

        self.q_proj = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.k_proj = nnx.Linear(
            self.hidden_dim,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.v_proj = nnx.Linear(
            self.hidden_dim,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

        self.pre_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)

        self.to_out = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init_out_proj_rmt,
        )

        self.memory_state = nnx.Param(
            jax.random.normal(rngs.params(), (self.mem_slots, self.hidden_dim)) * jnp.sqrt(1 / (self.hidden_dim)),
        )

        self.post_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)
        self.freqs_cis = nnx.Variable(
            precompute_freqs_cis(self.head_dim, (self.mini_batch_size + self.mem_slots) * 2),
        )

    def __call__(
        self,
        hidden_states: at.Float[at.Array, "*b nm mb d"],
        mask: at.Bool[at.Array, "*b nm"],
        mem_state: at.Float[at.Array, "mem_slots d"],
    ) -> tuple[at.Float[at.Array, "*b nm mb d"], at.Float[at.Array, "mem_slots d"], dict]:
        # do cross attention recurrently for each mini-batch
        @partial(vmap, axis_name="batch")
        def update_batch(x, mask):
            def compute_mini_batch(carry, inputs):
                x_mb = inputs["X"]  # (mb, d)
                mask_mb = inputs["mask"]  # (1,)

                carry_new, stats_dict = self.process_mini_batch(
                    x_mb,
                    mask_mb,
                    carry,
                )
                # mem_new = self.post_norm(mem_new)

                return carry_new, stats_dict

            inputs = {
                "X": x,  # (nm, mb, d)
                "mask": mask,  # (nm, )
            }

            mem_tokens, stats_dict = jax.lax.scan(f=compute_mini_batch, init=mem_state, xs=inputs)

            return mem_tokens, stats_dict

        mem_state_new, stats_dict = update_batch(hidden_states, mask)
        return mem_state_new, mem_state_new, stats_dict  # for interface consistency

    def reset(self):
        return jax.tree.map(lambda x: x.value if isinstance(x, nnx.Param) else x, self.memory_state)

    def process_mini_batch(
        self,
        x_mb,  # (mb, d)
        mask_mb,  # (1,)
        mem_state,  # (mem_slots, d)
    ):
        def cross_attention():
            mem_dtype = mem_state.dtype
            mem_slots = mem_state.shape[0]
            input_len = x_mb.shape[0]

            inputs = jnp.concatenate([x_mb, mem_state], axis=0)
            inputs = self.pre_norm(inputs)
            q_vec = self.q_proj(inputs)
            k_vec = self.k_proj(inputs)
            v_vec = self.v_proj(inputs)

            q_vec = einops.rearrange(q_vec, "s (nh d) -> s nh d", nh=self.num_heads)
            k_vec = einops.rearrange(k_vec, "s (nh d) -> s nh d", nh=self.num_kv_heads)
            v_vec = einops.rearrange(v_vec, "s (nh d) -> s nh d", nh=self.num_kv_heads)

            freqs_cis = jnp.take(self.freqs_cis.value, jnp.arange(q_vec.shape[0]), axis=0)  # (ws, d/2)
            q_vec, k_vec = apply_rotary_emb(q_vec[None, :], k_vec[None, :], freqs_cis)

            encoded = jax.nn.dot_product_attention(
                query=q_vec[0].astype(jnp.bfloat16),
                key=k_vec[0].astype(jnp.bfloat16),
                value=v_vec.astype(jnp.bfloat16),
            )

            encoded = einops.rearrange(encoded, "s nh d -> s (nh d)")
            encoded = encoded.astype(self.dtype)

            out = self.to_out(encoded[-mem_slots:])
            out = sharding.activation_sharding_constraint(out)

            if self.output_stats:
                stats_dict = {
                    "x_v_norm": jnp.linalg.norm(v_vec[:input_len], axis=-1).mean().astype(jnp.bfloat16),
                    "mem_v_norm": jnp.linalg.norm(v_vec[-mem_slots:], axis=-1).mean().astype(jnp.bfloat16),
                    "mem_in_norm": jnp.linalg.norm(mem_state, axis=-1).mean().astype(jnp.bfloat16),
                    "mem_out_norm": jnp.linalg.norm(out, axis=-1).mean().astype(jnp.bfloat16),
                    "mask": mask_mb.astype(jnp.bool_),
                }
            else:
                stats_dict = None
            # new_mem_state = out + mem_state # this is unstable
            new_mem_state = self.post_norm(out + mem_state)
            return new_mem_state.astype(mem_dtype), stats_dict

        def keep_original():
            if self.output_stats:
                stats_dict = {
                    "x_v_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "mem_v_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "mem_in_norm": jnp.linalg.norm(mem_state, axis=-1).mean().astype(jnp.bfloat16),
                    "mem_out_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "mask": mask_mb.astype(jnp.bool_),
                }
            else:
                stats_dict = None
            return mem_state, stats_dict

        mem_state_new, stats_dict = jax.lax.cond(mask_mb, cross_attention, keep_original)

        return mem_state_new, stats_dict
