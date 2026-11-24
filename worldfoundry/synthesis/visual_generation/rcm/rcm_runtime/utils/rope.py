from dataclasses import dataclass

import torch

from worldfoundry.base_models.diffusion_model.models.networks.magi2.preview_layers import apply_rotary_emb_torch

try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
except ImportError:
    flash_apply_rotary_emb = None


@dataclass
class RopeCache:
    """Per-layer RoPE frequency state.

    ``cached_k_rotated``: when True, keys in the KV cache are already rotated
    by RoPE (post-RoPE caching).  The attention operator rotates only the
    query and the *current* key before appending.  When False (pre-RoPE
    caching), RoPE is applied to the full key (cached + current) at
    attention time, enabling flexible temporal re-indexing for extrapolation.

    ``current_key_freqs``: freqs for the current block's key only.  Used in
    post-RoPE mode to rotate the new key before caching.
    """

    query_freqs: torch.Tensor | None = None
    key_freqs: torch.Tensor | None = None
    current_key_freqs: torch.Tensor | None = None
    use_fused: bool = True
    cached_k_rotated: bool = True


def apply_rope(x: torch.Tensor, freqs: torch.Tensor | None, fused: bool = True) -> torch.Tensor:
    if freqs is None:
        return x

    _, seq_len, _, head_dim = x.shape
    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    if fused and flash_apply_rotary_emb is not None:
        rotated = flash_apply_rotary_emb(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)
        return rotated.to(x.dtype)

    rotated = apply_rotary_emb_torch(x.to(torch.float32), cos, sin, interleaved=True)
    return rotated.to(x.dtype)


def rope_time_delta_(k: torch.Tensor, delta_frames: int, dim_t: int, t_ntk_factor: float = 1.0) -> None:
    """Apply a differential temporal RoPE rotation to cached keys **in-place**.

    Shifts the temporal position of ``k`` by ``delta_frames`` without touching
    the spatial (height/width) RoPE components.  This is the equivalent of
    ``_rope_time_delta_mul_`` in the reference DeepForcing / Infinity-RoPE repos.

    Args:
        k: Key tensor of shape ``[B, S, H, D]`` (interleaved RoPE layout).
        delta_frames: Number of frames to shift (positive = forward in time).
        dim_t: Number of temporal RoPE dimensions (= head_dim - 2*dim_h).
        t_ntk_factor: NTK scaling factor for the temporal axis.
    """
    if delta_frames == 0:
        return
    _head_dim = k.shape[-1]
    half_t = dim_t // 2
    t_theta = 10000.0 * t_ntk_factor
    dim_range = torch.arange(0, dim_t, 2, device=k.device, dtype=torch.float32)[:half_t] / dim_t
    freqs = delta_frames / (t_theta**dim_range)
    cos_d = torch.cos(freqs)
    sin_d = torch.sin(freqs)
    cos_d = cos_d.view(1, 1, 1, -1)
    sin_d = sin_d.view(1, 1, 1, -1)
    k_t = k[..., :dim_t].float()
    k_t_even = k_t[..., 0::2]
    k_t_odd = k_t[..., 1::2]
    new_even = k_t_even * cos_d - k_t_odd * sin_d
    new_odd = k_t_even * sin_d + k_t_odd * cos_d
    k[..., :dim_t:2] = new_even.to(k.dtype)
    k[..., 1:dim_t:2] = new_odd.to(k.dtype)


def apply_rope_with_tangent(
    x: torch.Tensor, t_x: torch.Tensor, freqs: torch.Tensor | None, fused: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs is None:
        return x, t_x

    assert fused == False  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
    out, t_out = torch.func.jvp(lambda inp: apply_rope(inp, freqs, fused=False), (x,), (t_x,))
    return out, t_out
