# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
"""
Adapted from TTT-LM-JAX and LaCT repository.
"""

from functools import partial

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import vmap

from ...openpi import array_typing as at
from ...openpi import sharding
from .memory_utils import (
    apply_layer_norm,
    inv_softplus,
    kernel_init,
    kernel_init_ttt_out_proj,
    kernel_init_ttt_v,
    scan_remat_every_n_iterations_scan,
)


class TTTBase(nnx.Module):
    def __init__(
        self,
        config,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.hidden_dim = config.recurrent_memory.hidden_dim
        self.num_heads = config.recurrent_memory.ttt_config.num_ttt_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.mini_batch_size = config.recurrent_memory.mini_batch_size
        self.mem_slots = config.budget

        self.output_stats = config.recurrent_memory.output_stats

        self.dtype = dtype = jnp.float32  # fix this dtype

        self._setup_lr(rngs, dtype)
        self._setup_qkvo(rngs, dtype)
        self._setup_ttt_weights(rngs, dtype)

        if self.config.recurrent_memory.ttt_config.qk_norm:
            self.q_norm = nnx.LayerNorm(self.head_dim, rngs=rngs, dtype=dtype)
            self.k_norm = nnx.LayerNorm(self.head_dim, rngs=rngs, dtype=dtype)
        if self.config.recurrent_memory.ttt_config.v_norm:
            self.v_norm = nnx.LayerNorm(self.head_dim, rngs=rngs, dtype=dtype)

        self.post_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)
        # self.freqs_cis = nnx.Variable(
        #     precompute_freqs_cis(self.head_dim, self.mini_batch_size * 2),
        # )
        # no rope for ttt, since LaCT already show that it is not necessary
        # and our recurrent embedding already incorporates M-rope

    def _setup_lr(self, rngs: nnx.Rngs, dtype: jnp.dtype):
        self.ttt_base_lr_inv = nnx.Variable(inv_softplus(self.config.recurrent_memory.ttt_config.base_lr))
        self.lr_proj = nnx.Linear(self.hidden_dim, 1 * self.num_heads, rngs=rngs, kernel_init=kernel_init)

    def _setup_qkvo(self, rngs: nnx.Rngs, dtype: jnp.dtype):
        self.wq = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.wk = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.wv = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init_ttt_v,
        )
        self.wo = nnx.Linear(
            self.hidden_dim,
            self.hidden_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init_ttt_out_proj,
        )

    def _setup_ttt_weights(self, rngs: nnx.Rngs, dtype: jnp.dtype):
        raise NotImplementedError

    def get_eta(self, x: at.Float[at.Array, "*b nm mb d"]):
        # Compute learning rate (Adapted from LaCT)
        # We use a simple linear projection since the inputs are bi-directional
        # no need to set causal mask like in TTT-LM-JAX
        lr = self.lr_proj(x.astype(jnp.float32))  # (batch, seq_len, num_heads)
        eta = nnx.softplus(lr + self.ttt_base_lr_inv.value)
        eta = einops.rearrange(eta, "b nm mb nh -> b nh nm mb", mb=self.mini_batch_size)
        eta = eta.astype(jnp.float32)
        eta = sharding.activation_sharding_constraint(eta)
        return eta

    def ttt(
        self,
        xq: at.Float[at.Array, "*b nh nm mb d"],
        xk: at.Float[at.Array, "*b nh nm mb d"],
        xv: at.Float[at.Array, "*b nh nm mb d"],
        eta: at.Float[at.Array, "*b nh nm mb"],
        mask: at.Float[at.Array, "*b nh nm"],
        ttt_params: tuple[jax.Array, ...],
    ):
        _, num_heads, n_mini_batches, mini_batch_size, _ = xq.shape

        @partial(vmap, axis_name="batch")
        def update_batch(xq, xk, xv, eta, mask):
            @partial(vmap, axis_name="head")
            def parallelize_over_heads(xq, xk, xv, eta, mask, ttt_params):
                def compute_mini_batch(ttt_params_carry, inputs):
                    """Process single mini-batch."""
                    xq_mb = inputs["XQ"]  # (mb, head_dim)
                    xk_mb = inputs["XK"]  # (mb, head_dim)
                    xv_mb = inputs["XV"]  # (mb, head_dim)
                    eta_mb = inputs["eta"]  # (mb,)
                    mask_mb = inputs["mask"]  # (1,)

                    ttt_params_new, output = self.process_mini_batch(
                        xq_mb,
                        xk_mb,
                        xv_mb,
                        eta_mb,
                        ttt_params,
                        ttt_params_carry,
                        mask_mb,
                    )

                    return ttt_params_new, output

                inputs = {
                    "XQ": xq,  # (nm, mb, h)
                    "XK": xk,  # (nm, mb, h)
                    "XV": xv,  # (nm, mb, h)
                    "eta": eta,  # (nm, mb)
                    "mask": mask,  # (nm,)
                }

                if mask.shape[0] > 1:
                    ttt_params_new, output = scan_remat_every_n_iterations_scan(
                        f=compute_mini_batch,
                        n=self.config.recurrent_memory.remat_n_mini_batches,
                        carry=ttt_params,
                        x=inputs,
                    )
                else:
                    ttt_params_new, output = jax.lax.scan(
                        f=compute_mini_batch,
                        init=ttt_params,
                        xs=inputs,
                    )

                return ttt_params_new, output

            ttt_params_new, output = parallelize_over_heads(xq, xk, xv, eta, mask, ttt_params)
            return ttt_params_new, output

        ttt_params_new, output = update_batch(xq, xk, xv, eta, mask)

        Z, ttt_stats_dict = output

        Z = einops.rearrange(
            Z,
            "b nh nm mb d -> b (nm mb) (nh d)",
            nh=num_heads,
            nm=n_mini_batches,
            mb=mini_batch_size,
        )  # (batch, seq_len, hidden_dim)

        return (
            Z,
            ttt_params_new,
            ttt_stats_dict,
        )

    def get_ttt_inputs(
        self,
        x: at.Float[at.Array, "*b nm mb d"],
        mask: at.Bool[at.Array, "*b nm"],
    ) -> tuple[
        at.Float[at.Array, "*b nh nm mb d"],
        at.Float[at.Array, "*b nh nm mb d"],
        at.Float[at.Array, "*b nh nm mb d"],
        at.Bool[at.Array, "*b nh nm mb"],
    ]:
        x = sharding.activation_sharding_constraint(x)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = sharding.activation_sharding_constraint(xq)
        xk = sharding.activation_sharding_constraint(xk)
        xv = sharding.activation_sharding_constraint(xv)

        xq = einops.rearrange(xq, "b nm mb (nh d) -> b nh nm mb d", nh=self.num_heads)
        xk = einops.rearrange(xk, "b nm mb (nh d) -> b nh nm mb d", nh=self.num_heads)
        xv = einops.rearrange(xv, "b nm mb (nh d) -> b nh nm mb d", nh=self.num_heads)
        x_mask = einops.repeat(mask, "b nm -> b nh nm", nh=self.num_heads)

        return xq, xk, xv, x_mask

    def reset(self):
        return jax.tree.map(lambda x: x.value if isinstance(x, nnx.Param) else x, self.get_memory_state())

    def __call__(
        self,
        hidden_states: at.Float[at.Array, "*b nm mb d"],
        mask: at.Bool[at.Array, "*b nm"],  # left pad mask
        mem_state: tuple[jax.Array, ...],
    ):
        xq, xk, xv, x_mask = self.get_ttt_inputs(hidden_states, mask)
        if self.config.recurrent_memory.ttt_config.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        if self.config.recurrent_memory.ttt_config.v_norm:
            xv = self.v_norm(xv)
        eta = self.get_eta(hidden_states)
        eta_clipped = jnp.clip(eta, 0, self.config.recurrent_memory.ttt_config.max_lr)

        Z, mem_state_new, stats_dict = self.ttt(xq, xk, xv, eta_clipped, x_mask, mem_state)
        Z = self.wo(self.post_norm(Z))

        # control Z as the residual
        hidden_states_flat = einops.rearrange(hidden_states, "b nm mb d -> b (nm mb) d")

        return hidden_states_flat + Z, mem_state_new, stats_dict


class TTTLayerLinear(TTTBase):
    def _setup_ttt_weights(self, rngs: nnx.Rngs, dtype: jnp.dtype):
        self.W1 = nnx.Param(
            jax.random.normal(rngs.params(), (self.num_heads, self.head_dim, self.head_dim), dtype=self.dtype)
            * jnp.sqrt(1 / self.head_dim),
        )
        self.b1 = nnx.Param(jnp.zeros((self.num_heads, self.head_dim), dtype=self.dtype))
        self.norm = nnx.Param(
            jnp.zeros((self.num_heads, 2, self.head_dim), dtype=self.dtype)  # scale and shift
        )

    def get_memory_state(self):
        return (self.W1, self.b1, self.norm)

    def process_mini_batch(
        self,
        xq_mini_batch,  # (mb, d)
        xk_mini_batch,  # (mb, d)
        xv_mini_batch,  # (mb, d)
        eta_mini_batch,  # (mb)
        memory_state_init,  # very begining ttt params
        memory_state,  # (W1, b1, norm1) in each mb
        mask_mb,  # (1,)
    ):
        """
        Y =  LN(X @ W1 + b1) + X = LN(Z) + X
        loss = 1/2 * ||Y - V||^2
        dH = H - (V - X)

        ttt_norm_out = LN(Z)
        LN is shared across all mini-batches
        """

        def apply_gradients():
            ttt_stats_dict = {}
            (w1, b1, norm) = memory_state
            mem_dtype = w1.dtype
            data_dtype = xq_mini_batch.dtype
            # we use global norm for all ttt inputs
            xi = xk_mini_batch
            z = xi @ w1 + b1
            ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: apply_layer_norm(z, norm), z)
            ssl_target = xv_mini_batch - xk_mini_batch
            error = ttt_norm_out - ssl_target
            dz = ttt_norm_vjp(error)[0]  # (mb,d)

            if self.output_stats:
                ttt_stats_dict["ttt_loss_mse_step_0"] = (error[-1] ** 2).mean()
                ttt_stats_dict["eta_mean"] = jnp.mean(eta_mini_batch)
                w1_init, b1_init, norm_init = memory_state_init
                z_0 = xi[-1:] @ w1_init + b1_init
                ttt_norm_out_0 = apply_layer_norm(z_0, norm_init)
                ttt_stats_dict["ttt_loss_mse_init"] = ((ttt_norm_out_0 - ssl_target[-1:]) ** 2).mean()

            eta_expanded = eta_mini_batch[:, None]  # (mb, 1)
            dw1 = xi.T @ (eta_expanded * dz)  # (d, d)
            db1 = (eta_expanded * dz).sum(axis=0)  # (d,)

            def update_weight(w, w_grad):
                w_grad_norm = jnp.linalg.norm(w_grad)
                threshold = self.config.recurrent_memory.ttt_config.max_grad_norm
                w_new = jnp.where(
                    w_grad_norm > threshold,
                    w - w_grad / (w_grad_norm + 1e-8) * threshold,
                    w - w_grad,
                )
                return w_new

            w1_new = update_weight(w1, dw1)
            b1_new = update_weight(b1, db1)

            if self.output_stats:
                z_new = apply_layer_norm(xi[-1:] @ w1_new + b1_new, norm)
                ttt_stats_dict["ttt_loss_mse_step_1"] = ((z_new - ssl_target[-1:]) ** 2).mean()

                ttt_stats_dict["diff_w_norm"] = jnp.linalg.norm(w1_new - w1)
                ttt_stats_dict["diff_b_norm"] = jnp.linalg.norm(b1_new - b1)
                ttt_stats_dict["grad_w_norm"] = jnp.linalg.norm(dw1)
                ttt_stats_dict["grad_b_norm"] = jnp.linalg.norm(db1)
                ttt_stats_dict["xq"] = jnp.linalg.norm(xq_mini_batch, axis=-1).mean()
                ttt_stats_dict["xk"] = jnp.linalg.norm(xk_mini_batch, axis=-1).mean()
                ttt_stats_dict["xv"] = jnp.linalg.norm(xv_mini_batch, axis=-1).mean()
                ttt_stats_dict["mask"] = mask_mb.astype(jnp.bool_)

            output = apply_layer_norm(xq_mini_batch @ w1_new + b1_new, norm) + xq_mini_batch
            output = sharding.activation_sharding_constraint(output)

            output = output.astype(data_dtype)
            memory_state_new = (w1_new.astype(mem_dtype), b1_new.astype(mem_dtype), norm.astype(mem_dtype))
            return memory_state_new, output, ttt_stats_dict

        def keep_original():
            ttt_stats_dict = {
                "ttt_loss_mse_step_0": jnp.array(0.0),
                "eta_mean": jnp.array(0.0),
                "ttt_loss_mse_init": jnp.array(0.0),
                "ttt_loss_mse_step_1": jnp.array(0.0),
                "diff_w_norm": jnp.array(0.0),
                "grad_w_norm": jnp.array(0.0),
                "diff_b_norm": jnp.array(0.0),
                "grad_b_norm": jnp.array(0.0),
                "xq": jnp.array(0.0, dtype=xq_mini_batch.dtype),
                "xk": jnp.array(0.0, dtype=xk_mini_batch.dtype),
                "xv": jnp.array(0.0, dtype=xv_mini_batch.dtype),
                "mask": jnp.array(False, dtype=jnp.bool_),
            }
            return memory_state, jnp.zeros_like(xq_mini_batch), ttt_stats_dict

        memory_state_new, output, ttt_stats_dict = jax.lax.cond(mask_mb, apply_gradients, keep_original)

        return memory_state_new, (
            output,
            ttt_stats_dict,
        )
