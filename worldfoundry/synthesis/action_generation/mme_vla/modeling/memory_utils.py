from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp

kernel_init = nnx.initializers.normal(stddev=0.02)
kernel_init_ttt_v = nnx.initializers.normal(stddev=0.01)
kernel_init_ttt_out_proj = nnx.initializers.normal(stddev=0.01)
kernel_init_out_proj = nnx.initializers.normal(stddev=0.002)
kernel_init_out_proj_rmt = nnx.initializers.normal(stddev=0.005)


def scan_remat_every_n_iterations_scan(f, n, carry, x):
    """
    Remat every n mini batches - memory efficient scanning.
    Following TTT-LM-JAX implementation.
    """
    x_grouped = jax.tree.map(lambda x: x.reshape((-1, n, *x.shape[1:])), x)
    carry, y_grouped = jax.lax.scan(jax.remat(partial(jax.lax.scan, f)), carry, x_grouped)
    y = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), y_grouped)
    return carry, y


@jax.jit
def inv_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.log(-jnp.expm1(-x))


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> jnp.ndarray:
    """Precompute frequency tensor for rotary embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(seq_len)
    freqs = jnp.outer(t, freqs)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


@jax.jit
def apply_rotary_emb(
    xq: jnp.ndarray,  # (..., seq_len, num_heads, head_dim)
    xk: jnp.ndarray,  # (..., seq_len, num_heads, head_dim)
    freqs_cis: jnp.ndarray,  # (seq_len, head_dim // 2)
):
    dtype = xq.dtype
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])  # (..., seq_len, heads, head_dim // 2)
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])  # (..., seq_len, heads, head_dim // 2)

    for _ in range(len(xq.shape) - 3):  # Add extra dimensions
        freqs_cis = jnp.expand_dims(freqs_cis, axis=0)
    freqs_cis = jnp.expand_dims(freqs_cis, axis=-2)  # Add num_heads dimension
    # freqs_cis: (..., seq_len, 1, head_dim // 2)

    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(
        *xq_out.shape[:-1], -1
    )  # (..., seq_len, heads, head_dim)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(
        *xk_out.shape[:-1], -1
    )  # (..., seq_len, heads, head_dim)

    return xq_out.astype(dtype), xk_out.astype(dtype)


@jax.jit
def apply_layer_norm(x: jnp.ndarray, scale_and_shift: jnp.ndarray) -> jnp.ndarray:
    dtype = x.dtype
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x_normed = x * jax.lax.rsqrt(var + 1e-8) * (1 + scale_and_shift[0]) + scale_and_shift[1]
    return x_normed.astype(dtype)


@jax.jit
def rms_normalize(x):
    dtype = x.dtype
    x = x.astype(jnp.float32)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm_x = x / (norm + 1e-6)
    return norm_x.astype(dtype)
