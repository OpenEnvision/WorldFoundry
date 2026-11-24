# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
import jax.numpy as jnp
from flax import nnx

from ...openpi import array_typing as at
from .memory_utils import kernel_init


class FeatureEncoder(nnx.Module):
    def __init__(
        self,
        rngs,
        dtype,
        image_input_dim,
        pos_input_dim,
        pos_output_dim,
        state_input_dim,
        state_output_dim,
        output_dim_for_percep=None,
        ouput_dim_for_recur=None,
        use_pos_emb=True,
        use_state_emb=False,
    ):
        self.use_pos_emb = use_pos_emb
        self.use_state_emb = use_state_emb
        input_dim = image_input_dim
        if use_state_emb:
            self.state_proj = nnx.Linear(
                state_input_dim,
                state_output_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
            input_dim += state_output_dim
        if use_pos_emb:
            self.pos_proj = nnx.Linear(
                pos_input_dim,
                pos_output_dim,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
            input_dim += pos_output_dim

        if output_dim_for_percep is not None:
            self.encoder_static = nnx.Linear(
                input_dim,
                output_dim_for_percep,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
        else:
            self.encoder_static = None
        if ouput_dim_for_recur is not None:
            self.encoder_recur = nnx.Linear(
                input_dim,
                ouput_dim_for_recur,
                rngs=rngs,
                dtype=dtype,
                kernel_init=kernel_init,
            )
        else:
            self.encoder_recur = None

    def _add_pos_emb(
        self,
        base_emb,
        pos_emb,
    ):
        pos_emb = nnx.silu(self.pos_proj(pos_emb))
        base_emb = jnp.concatenate([base_emb, pos_emb], axis=-1)
        return base_emb

    def _add_state_emb(
        self,
        base_emb,
        state_emb,
    ):
        if base_emb.ndim == 5:
            _, _, v, p, _ = base_emb.shape
            state_emb = nnx.silu(self.state_proj(state_emb))
            state_emb = jnp.tile(state_emb[:, :, None, None, :], (1, 1, v, p, 1))
            base_emb = jnp.concatenate([base_emb, state_emb], axis=-1)
        else:
            state_emb = nnx.silu(self.state_proj(state_emb))
            base_emb = jnp.concatenate([base_emb, state_emb], axis=-1)

        return base_emb

    def _encode_memory(
        self,
        image_emb,
        pos_emb,
        state_emb,
        encoder_fn,
    ):
        input_emb = image_emb
        if self.use_pos_emb and pos_emb is not None:
            input_emb = self._add_pos_emb(input_emb, pos_emb)
        if self.use_state_emb and state_emb is not None:
            input_emb = self._add_state_emb(input_emb, state_emb)
        input_emb = encoder_fn(input_emb)
        return input_emb

    def encode_perceptual_memory(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        *args,
        **kwargs,
    ):
        return self._encode_memory(
            static_image_emb,
            static_pos_emb,
            static_state_emb,
            self.encoder_static,
        )

    def encode_recurrent_memory(
        self,
        recurrent_image_emb: at.Float[at.Array, "b t v p d1"],
        recurrent_pos_emb: at.Float[at.Array, "b t v p d2"],
        recurrent_state_emb: at.Float[at.Array, "b t d3"],
        *args,
        **kwargs,
    ):
        return self._encode_memory(
            recurrent_image_emb,
            recurrent_pos_emb,
            recurrent_state_emb,
            self.encoder_recur,
        )
