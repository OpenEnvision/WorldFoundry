import torch
import triton
import triton.language as tl

from .utils import calculate_settings, torch_gpu_device

# ------------------------------- replace funtion -------------------------------


def apply_rotary_emb_transposed_flash(x, freqs_cis):
    # Helios inference never needs an autograd graph. Calling the Triton
    # kernel directly also avoids retaining cos/sin tensors for backward.
    batch_size, seq_len, n_heads, head_dim = x.shape
    x_flat = x.reshape(-1, head_dim).contiguous()
    output = torch.empty_like(x_flat)

    freqs_flat = freqs_cis.reshape(batch_size * seq_len, -1).contiguous()
    half_dim = freqs_flat.shape[-1] // 2
    cos = freqs_flat[:, :half_dim].contiguous()
    sin = freqs_flat[:, half_dim:].contiguous()
    block_size, num_warps = calculate_settings(head_dim // 2)

    with torch_gpu_device(x_flat.device):
        _apply_rope_transposed_kernel[(x_flat.shape[0],)](
            x_flat,
            output,
            cos,
            sin,
            n_heads,
            x_flat.stride(0),
            output.stride(0),
            cos.stride(0),
            head_dim,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return output.reshape(batch_size, seq_len, n_heads, head_dim)


def replace_rope_with_flash_rope():
    from .. import model as transformer_helios_diffusers

    transformer_helios_diffusers.apply_rotary_emb_transposed = apply_rotary_emb_transposed_flash
    print("Patched Flash_RoPE globally\n")


# ------------------------------- layer norm -------------------------------


@triton.jit
def _apply_rope_transposed_kernel(
    X,
    Out,
    cos,
    sin,
    n_heads: tl.constexpr,
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    stride_freq: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    freq_row_idx = row_idx // n_heads

    half_head_dim = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    x_ptr = X + row_idx * stride_x
    out_ptr = Out + row_idx * stride_out
    cos_ptr = cos + freq_row_idx * stride_freq
    sin_ptr = sin + freq_row_idx * stride_freq

    x_real = tl.load(x_ptr + col_offsets * 2, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)
    cos_even = tl.load(cos_ptr + col_offsets * 2, mask=mask, other=0.0)
    sin_odd = tl.load(sin_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)

    out_even = x_real * cos_even - x_imag * sin_odd
    out_odd = x_real * sin_odd + x_imag * cos_even

    tl.store(out_ptr + col_offsets * 2, out_even, mask=mask)
    tl.store(out_ptr + col_offsets * 2 + 1, out_odd, mask=mask)
