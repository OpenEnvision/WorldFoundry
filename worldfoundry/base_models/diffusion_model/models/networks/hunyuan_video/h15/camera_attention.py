# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law or agreed to in writing, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""minWM HY15 cached attention; kernels, static KV arena, and sequence parallelism."""

from __future__ import annotations

from typing import Any

import torch

from worldfoundry.core.attention import scaled_dot_product_attention
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.distributed.sequence_parallel.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)

_FP8_ATTN_SCALES: dict[tuple[Any, ...], tuple[float, float, float]] = {}
_FA3_MODULE: Any = None
_FA3_PROBED: bool = False


def hopper_fa3_usable(device: torch.device | str | None = None) -> bool:
    """Return True when ``flash_attn_interface`` is installed and device is SM90 Hopper."""
    global _FA3_MODULE, _FA3_PROBED
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return False
    if torch.cuda.get_device_capability(dev)[0] != 9:
        return False
    if not _FA3_PROBED:
        try:
            _FA3_MODULE = __import__("flash_attn_interface")
        except Exception:
            _FA3_MODULE = None
        _FA3_PROBED = True
    return _FA3_MODULE is not None and callable(getattr(_FA3_MODULE, "flash_attn_func", None))


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _quantize_bhl_d_fp8_kernel(
        x_ptr,
        out_ptr,
        inv_scale,
        H,
        sx_b,
        sx_h,
        sx_l,
        so_b,
        so_h,
        so_l,
        dst_offset,
        BLOCK_D: tl.constexpr,
    ):
        l_idx = tl.program_id(0)
        bh = tl.program_id(1)
        b_idx = bh // H
        h_idx = bh % H
        d_off = tl.arange(0, BLOCK_D)
        x_row = x_ptr + b_idx * sx_b + h_idx * sx_h + l_idx * sx_l + d_off
        val = tl.load(x_row).to(tl.float32) * inv_scale
        val = tl.minimum(tl.maximum(val, -448.0), 448.0)
        out_row = out_ptr + b_idx * so_b + h_idx * so_h + (dst_offset + l_idx) * so_l + d_off
        tl.store(out_row, val.to(tl.float8e4nv))

    _TRITON_FP8_QUANT_AVAILABLE = True
except Exception:
    _TRITON_FP8_QUANT_AVAILABLE = False


def _quantize_static_fp8_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    descale: float,
    dst_offset: int = 0,
) -> None:
    """Quantize ``[B, H, L, D]`` tensor directly into ``dst[:, :, dst_offset:dst_offset+L, :]`` as E4M3."""
    bs, heads, seq_len, head_dim = src.shape
    inv_scale = 1.0 / float(descale)
    if (
        _TRITON_FP8_QUANT_AVAILABLE
        and src.is_cuda
        and dst.is_cuda
        and head_dim in {64, 128}
        and src.stride(-1) == 1
        and dst.stride(-1) == 1
        and seq_len > 0
        and bs * heads <= 65535
    ):
        sx_b, sx_h, sx_l, _ = src.stride()
        so_b, so_h, so_l, _ = dst.stride()
        grid = (seq_len, bs * heads)
        _quantize_bhl_d_fp8_kernel[grid](
            src,
            dst,
            inv_scale,
            heads,
            sx_b,
            sx_h,
            sx_l,
            so_b,
            so_h,
            so_l,
            dst_offset,
            BLOCK_D=head_dim,
            num_warps=2 if head_dim == 64 else 4,
        )
        return
    q = (src * inv_scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    dst[:, :, dst_offset : dst_offset + seq_len, :].copy_(q)


def _quantize_static_fp8(src: torch.Tensor, *, descale: float) -> torch.Tensor:
    """Allocate a contiguous FP8 E4M3 buffer and quantize ``[B, H, L, D]`` in one pass."""
    out = torch.empty(src.shape, dtype=torch.float8_e4m3fn, device=src.device)
    _quantize_static_fp8_into(src, out, descale=descale, dst_offset=0)
    return out


def reset_fp8_attention_scales() -> None:
    """Clear calibrated per-layer cuDNN FP8 attention scales."""
    _FP8_ATTN_SCALES.clear()


def _dispatch_cached_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mode: str | None = None,
    scale_key: tuple[Any, ...] | None = None,
    fp8_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    fp8_prefix_len: int | None = None,
    fp8_chunk_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute B H S D attention via cuDNN/Flash SDPA, calibrated cuDNN FP8, or explicit backend."""
    if attn_mode in (None, "", "auto", "torch", "torch_causal", "flash", "sdpa", "cudnn"):
        return scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    if (
        attn_mode == "flash_attention_3"
        and hopper_fa3_usable(query.device)
        and _FA3_MODULE is not None
        and query.dtype in {torch.bfloat16, torch.float16}
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.shape[-1] <= 256
        and query.shape[-1] % 8 == 0
    ):
        out = _FA3_MODULE.flash_attn_func(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
        )
        out_t = out[0] if isinstance(out, tuple) else out
        return out_t.transpose(1, 2)
    if attn_mode == "cudnn_fp8":
        if not (query.is_cuda and query.shape[-1] in {64, 128} and query.dtype in {torch.bfloat16, torch.float16}):
            return scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        from worldfoundry.core.attention.backends.native_fp8 import native_cudnn_fp8_sdpa

        # Calibrate static per-stream, per-layer E4M3 scales on eager warmup so CUDA Graph
        # capture & replay stay 100% host-sync-free while preserving dynamic range.
        key_id = (query.device, scale_key)
        scales = _FP8_ATTN_SCALES.get(key_id)
        if scales is None:
            if not torch.cuda.is_current_stream_capturing():
                dq = max(float(query.abs().amax().item()) * 1.15 / 448.0, 1.0 / 64.0)
                dk = max(float(key.abs().amax().item()) * 1.15 / 448.0, 1.0 / 64.0)
                dv = max(float(value.abs().amax().item()) * 1.15 / 448.0, 1.0 / 64.0)
                scales = (dq, dk, dv)
                _FP8_ATTN_SCALES[key_id] = scales
            else:
                scales = (1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0)
        dq, dk, dv = scales
        q_fp8 = _quantize_static_fp8(query, descale=dq)
        if fp8_kv_buffer is not None and fp8_prefix_len is not None and fp8_chunk_kv is not None:
            k_fp8_buf, v_fp8_buf = fp8_kv_buffer
            chunk_k, chunk_v = fp8_chunk_kv
            end_len = fp8_prefix_len + chunk_k.shape[2]
            _quantize_static_fp8_into(chunk_k, k_fp8_buf, descale=dk, dst_offset=fp8_prefix_len)
            _quantize_static_fp8_into(chunk_v, v_fp8_buf, descale=dv, dst_offset=fp8_prefix_len)
            k_fp8 = k_fp8_buf[:, :, :end_len]
            v_fp8 = v_fp8_buf[:, :, :end_len]
        else:
            k_fp8 = _quantize_static_fp8(key, descale=dk)
            v_fp8 = _quantize_static_fp8(value, descale=dv)
        return native_cudnn_fp8_sdpa(
            q_fp8,
            k_fp8,
            v_fp8,
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
        ).to(query.dtype)
    from worldfoundry.core.attention.backends.dispatch import attention_forward

    return attention_forward(
        query,
        key,
        value,
        q_pattern="b n s d",
        k_pattern="b n s d",
        v_pattern="b n s d",
        out_pattern="b n s d",
        backend=str(attn_mode),
    )


def prepare_static_kv_arena(
    kv_cache: list[dict[str, Any]],
    chunk_tokens: int,
    *,
    arena: dict[tuple[Any, ...], list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, tuple[Any, ...]]:
    """Populate a static ``[B, H, prefix_len + chunk_tokens, D]`` KV buffer per layer.

    Concatenates ``[k_txt, k_vision]`` once into the prefix slice so denoising steps
    (both RoPE and PRoPE streams) write only the current chunk into the tail slice
    with zero ``torch.cat`` allocations and stable addresses for CUDA Graph replay.
    """
    if not kv_cache:
        raise ValueError("kv_cache must be non-empty")
    first = kv_cache[0]
    k_txt0 = first["k_txt"]
    k_vis0 = first.get("k_vision")
    batch, heads, txt_len, head_dim = k_txt0.shape
    vis_len = 0 if k_vis0 is None else int(k_vis0.shape[2])
    prefix_len = txt_len + vis_len
    chunk_len = int(chunk_tokens)
    total_len = prefix_len + chunk_len
    buf_dtype = k_txt0.dtype if dtype is None else dtype
    device = k_txt0.device
    arena_key = (len(kv_cache), batch, heads, prefix_len, chunk_len, head_dim, buf_dtype, device)

    buffers = None if arena is None else arena.get(arena_key)
    if buffers is None:
        with torch.inference_mode(False):
            buffers = [
                (
                    torch.empty((batch, heads, total_len, head_dim), dtype=buf_dtype, device=device),
                    torch.empty((batch, heads, total_len, head_dim), dtype=buf_dtype, device=device),
                )
                for _ in range(len(kv_cache))
            ]
        if arena is not None:
            arena[arena_key] = buffers

    for idx, slot in enumerate(kv_cache):
        k_buf, v_buf = buffers[idx]
        k_txt = slot["k_txt"]
        v_txt = slot["v_txt"]
        k_buf[:, :, :txt_len].copy_(k_txt)
        v_buf[:, :, :txt_len].copy_(v_txt)
        k_vis = slot.get("k_vision")
        v_vis = slot.get("v_vision")
        if k_vis is not None and v_vis is not None:
            k_buf[:, :, txt_len:prefix_len].copy_(k_vis)
            v_buf[:, :, txt_len:prefix_len].copy_(v_vis)
    return buffers, prefix_len, arena_key


def prepare_static_kv_arena_bshd(
    kv_cache: list[dict[str, Any]],
    chunk_tokens: int,
    *,
    arena: dict[tuple[Any, ...], list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, tuple[Any, ...]]:
    """Populate a static ``[B, prefix_len + chunk_tokens, H, D]`` BSHD KV buffer per layer for Hopper FA3.

    Transposes cached ``[k_txt, k_vision]`` from BHSD -> BSHD *once* per chunk so every
    denoising step writes only the tail chunk slice and invokes ``flash_attn_func``
    with zero prefix transposes or ``einops.rearrange`` copies inside the CUDA Graph.
    """
    if not kv_cache:
        raise ValueError("kv_cache must be non-empty")
    first = kv_cache[0]
    k_txt0 = first["k_txt"]
    k_vis0 = first.get("k_vision")
    batch, heads, txt_len, head_dim = k_txt0.shape
    vis_len = 0 if k_vis0 is None else int(k_vis0.shape[2])
    prefix_len = txt_len + vis_len
    chunk_len = int(chunk_tokens)
    total_len = prefix_len + chunk_len
    buf_dtype = k_txt0.dtype if dtype is None else dtype
    device = k_txt0.device
    arena_key = ("bshd", len(kv_cache), batch, heads, prefix_len, chunk_len, head_dim, buf_dtype, device)

    buffers = None if arena is None else arena.get(arena_key)
    if buffers is None:
        with torch.inference_mode(False):
            buffers = [
                (
                    torch.empty((batch, total_len, heads, head_dim), dtype=buf_dtype, device=device),
                    torch.empty((batch, total_len, heads, head_dim), dtype=buf_dtype, device=device),
                )
                for _ in range(len(kv_cache))
            ]
        if arena is not None:
            arena[arena_key] = buffers

    for idx, slot in enumerate(kv_cache):
        k_buf, v_buf = buffers[idx]
        k_buf[:, :txt_len].copy_(slot["k_txt"].transpose(1, 2))
        v_buf[:, :txt_len].copy_(slot["v_txt"].transpose(1, 2))
        k_vis = slot.get("k_vision")
        v_vis = slot.get("v_vision")
        if k_vis is not None and v_vis is not None:
            k_buf[:, txt_len:prefix_len].copy_(k_vis.transpose(1, 2))
            v_buf[:, txt_len:prefix_len].copy_(v_vis.transpose(1, 2))
    return buffers, prefix_len, arena_key


def prepare_static_fp8_kv_arena(
    kv_cache: list[dict[str, Any]],
    chunk_tokens: int,
    *,
    arena: dict[tuple[Any, ...], dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]]] | None = None,
) -> tuple[dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]] | None, int, tuple[Any, ...] | None]:
    """Pre-quantize ``[k_txt, k_vision]`` once per chunk into FP8 E4M3 static arenas per stream.

    Returns ``(stream_buffers, prefix_len, fp8_arena_key)`` where ``stream_buffers`` maps
    ``"rope"`` and ``"prope"`` to per-layer ``(k_fp8_buf, v_fp8_buf)`` pairs for every
    interior block whose calibrated FP8 scales are already populated. Returns ``(None, 0, None)``
    when scales have not been calibrated yet (first eager warmup step).
    """
    if not kv_cache:
        return None, 0, None
    first = kv_cache[0]
    k_txt0 = first["k_txt"]
    if not k_txt0.is_cuda or k_txt0.shape[-1] not in {64, 128}:
        return None, 0, None
    num_blocks = len(kv_cache)
    device = k_txt0.device
    interior_indices = [
        idx for idx in range(num_blocks) if not (num_blocks >= 2 and (idx == 0 or idx == num_blocks - 1))
    ]
    if not interior_indices:
        return None, 0, None

    scale_snapshot: list[tuple[str, int, float, float]] = []
    for stream_tag in ("rope", "prope"):
        for idx in interior_indices:
            scales = _FP8_ATTN_SCALES.get((device, ("vision", stream_tag, idx)))
            if scales is None:
                return None, 0, None
            scale_snapshot.append((stream_tag, idx, scales[1], scales[2]))

    k_vis0 = first.get("k_vision")
    batch, heads, txt_len, head_dim = k_txt0.shape
    vis_len = 0 if k_vis0 is None else int(k_vis0.shape[2])
    prefix_len = txt_len + vis_len
    chunk_len = int(chunk_tokens)
    total_len = prefix_len + chunk_len
    fp8_key = ("fp8", num_blocks, batch, heads, prefix_len, chunk_len, head_dim, device)

    stream_buffers = None if arena is None else arena.get(fp8_key)
    if stream_buffers is None:
        stream_buffers = {"rope": [None] * num_blocks, "prope": [None] * num_blocks}
        with torch.inference_mode(False):
            for stream_tag in ("rope", "prope"):
                for idx in interior_indices:
                    stream_buffers[stream_tag][idx] = (
                        torch.empty((batch, heads, total_len, head_dim), dtype=torch.float8_e4m3fn, device=device),
                        torch.empty((batch, heads, total_len, head_dim), dtype=torch.float8_e4m3fn, device=device),
                    )
        if arena is not None:
            arena[fp8_key] = stream_buffers

    for stream_tag, idx, dk, dv in scale_snapshot:
        pair = stream_buffers[stream_tag][idx]
        assert pair is not None
        k_fp8_buf, v_fp8_buf = pair
        slot = kv_cache[idx]
        _quantize_static_fp8_into(slot["k_txt"], k_fp8_buf, descale=dk, dst_offset=0)
        _quantize_static_fp8_into(slot["v_txt"], v_fp8_buf, descale=dv, dst_offset=0)
        k_vis = slot.get("k_vision")
        v_vis = slot.get("v_vision")
        if k_vis is not None and v_vis is not None:
            _quantize_static_fp8_into(k_vis, k_fp8_buf, descale=dk, dst_offset=txt_len)
            _quantize_static_fp8_into(v_vis, v_fp8_buf, descale=dv, dst_offset=txt_len)

    return stream_buffers, prefix_len, fp8_key


def _tensor_version(tensor: torch.Tensor | None) -> int | None:
    if tensor is None or tensor.is_inference():
        return None
    return tensor._version


def _slot_sources_match(
    slot: dict[str, Any],
    refs_key: str,
    versions_key: str,
    encoder_key: torch.Tensor,
    encoder_value: torch.Tensor,
    cache_vision_key: torch.Tensor | None,
    cache_vision_value: torch.Tensor | None,
) -> bool:
    prev_refs = slot.get(refs_key)
    prev_vers = slot.get(versions_key)
    if prev_refs is None or prev_vers is None:
        return False
    p_ek, p_ev, p_vk, p_vv = prev_refs
    if (
        p_ek is not encoder_key
        or p_ev is not encoder_value
        or p_vk is not cache_vision_key
        or p_vv is not cache_vision_value
    ):
        return False
    cur_vers = (
        _tensor_version(encoder_key),
        _tensor_version(encoder_value),
        _tensor_version(cache_vision_key),
        _tensor_version(cache_vision_value),
    )
    return prev_vers == cur_vers


def _get_or_build_slot_kv_buffer(
    slot: dict[str, Any],
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily reuse a pre-concatenated prefix buffer inside ``kv_cache[block_idx]``."""
    encoder_key = slot["k_txt"]
    encoder_value = slot["v_txt"]
    cache_vision_key = slot.get("k_vision")
    cache_vision_value = slot.get("v_vision")

    txt_len = encoder_key.shape[2]
    vis_len = 0 if cache_vision_key is None else cache_vision_key.shape[2]
    prefix_len = txt_len + vis_len
    chunk_len = key.shape[2]
    total_len = prefix_len + chunk_len
    shape_sig = (
        txt_len,
        vis_len,
        chunk_len,
        key.shape[0],
        key.shape[1],
        key.shape[3],
        key.dtype,
        key.device,
    )
    if slot.get("_static_kv_sig") != shape_sig or not _slot_sources_match(
        slot,
        "_static_kv_refs",
        "_static_kv_vers",
        encoder_key,
        encoder_value,
        cache_vision_key,
        cache_vision_value,
    ):
        k_buf = torch.empty((key.shape[0], key.shape[1], total_len, key.shape[3]), dtype=key.dtype, device=key.device)
        v_buf = torch.empty(
            (value.shape[0], value.shape[1], total_len, value.shape[3]), dtype=value.dtype, device=value.device
        )
        k_buf[:, :, :txt_len].copy_(encoder_key)
        v_buf[:, :, :txt_len].copy_(encoder_value)
        if cache_vision_key is not None and cache_vision_value is not None:
            k_buf[:, :, txt_len:prefix_len].copy_(cache_vision_key)
            v_buf[:, :, txt_len:prefix_len].copy_(cache_vision_value)
        slot["_static_kv_sig"] = shape_sig
        slot["_static_kv_refs"] = (encoder_key, encoder_value, cache_vision_key, cache_vision_value)
        slot["_static_kv_vers"] = (
            _tensor_version(encoder_key),
            _tensor_version(encoder_value),
            _tensor_version(cache_vision_key),
            _tensor_version(cache_vision_value),
        )
        slot["_static_k"] = k_buf
        slot["_static_v"] = v_buf
        slot["_static_prefix_len"] = prefix_len
    else:
        k_buf = slot["_static_k"]
        v_buf = slot["_static_v"]
    k_buf[:, :, prefix_len:total_len].copy_(key)
    v_buf[:, :, prefix_len:total_len].copy_(value)
    return k_buf, v_buf


def _get_or_build_slot_fp8_kv_buffer(
    slot: dict[str, Any],
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    stream_tag: str,
    dk: float,
    dv: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Lazily reuse a pre-quantized FP8 E4M3 prefix buffer inside ``kv_cache[block_idx]``."""
    encoder_key = slot["k_txt"]
    encoder_value = slot["v_txt"]
    cache_vision_key = slot.get("k_vision")
    cache_vision_value = slot.get("v_vision")

    txt_len = encoder_key.shape[2]
    vis_len = 0 if cache_vision_key is None else cache_vision_key.shape[2]
    prefix_len = txt_len + vis_len
    chunk_len = key.shape[2]
    total_len = prefix_len + chunk_len
    sig_key = f"_static_fp8_sig_{stream_tag}"
    refs_key = f"_static_fp8_refs_{stream_tag}"
    vers_key = f"_static_fp8_vers_{stream_tag}"
    buf_key = f"_static_fp8_buf_{stream_tag}"
    shape_sig = (txt_len, vis_len, chunk_len, key.shape[0], key.shape[1], key.shape[3], key.device, dk, dv)

    if slot.get(sig_key) != shape_sig or not _slot_sources_match(
        slot,
        refs_key,
        vers_key,
        encoder_key,
        encoder_value,
        cache_vision_key,
        cache_vision_value,
    ):
        existing = slot.get(buf_key)
        if (
            existing is not None
            and existing[0].shape == (key.shape[0], key.shape[1], total_len, key.shape[3])
            and existing[0].device == key.device
        ):
            k_buf, v_buf = existing
        else:
            k_buf = torch.empty(
                (key.shape[0], key.shape[1], total_len, key.shape[3]),
                dtype=torch.float8_e4m3fn,
                device=key.device,
            )
            v_buf = torch.empty(
                (value.shape[0], value.shape[1], total_len, value.shape[3]),
                dtype=torch.float8_e4m3fn,
                device=value.device,
            )
            slot[buf_key] = (k_buf, v_buf)
        _quantize_static_fp8_into(encoder_key, k_buf, descale=dk, dst_offset=0)
        _quantize_static_fp8_into(encoder_value, v_buf, descale=dv, dst_offset=0)
        if cache_vision_key is not None and cache_vision_value is not None:
            _quantize_static_fp8_into(cache_vision_key, k_buf, descale=dk, dst_offset=txt_len)
            _quantize_static_fp8_into(cache_vision_value, v_buf, descale=dv, dst_offset=txt_len)
        slot[sig_key] = shape_sig
        slot[refs_key] = (encoder_key, encoder_value, cache_vision_key, cache_vision_value)
        slot[vers_key] = (
            _tensor_version(encoder_key),
            _tensor_version(encoder_value),
            _tensor_version(cache_vision_key),
            _tensor_version(cache_vision_value),
        )
    else:
        k_buf, v_buf = slot[buf_key]
    return k_buf, v_buf, prefix_len


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
    del img_q_len, img_kv_len, text_mask, attn_param, kv_cache
    encoder_query = q
    encoder_key = k
    encoder_value = v
    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank

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
        t_kv["k_txt"] = encoder_key.contiguous()
        t_kv["v_txt"] = encoder_value.contiguous()
    encoder_hidden_states = _dispatch_cached_attention(
        encoder_query,
        encoder_key,
        encoder_value,
        attn_mode=attn_mode,
        scale_key=("txt", block_idx),
    )
    encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
    if enable_sp:
        encoder_hidden_states = sequence_model_parallel_all_gather(encoder_hidden_states, dim=2).contiguous()
        encoder_hidden_states = encoder_hidden_states.to(q.dtype)
    (b, s, a, d) = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.reshape(b, s, -1)
    return (encoder_hidden_states, t_kv)


@torch.compiler.disable
def sequence_parallel_attention_vision(
    q,
    k,
    v,
    block_idx=None,
    kv_cache=None,
    cache_vision=False,
    *,
    attn_mode: str | None = None,
    static_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    static_prefix_len: int | None = None,
    static_fp8_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    static_bshd_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    stream_tag: str = "rope",
    num_blocks: int | None = None,
):
    query = q
    key = k
    value = v
    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    # Direct zero-transpose BSHD Hopper FlashAttention-3 fast path for cached single-GPU AR steps.
    if (
        not enable_sp
        and not cache_vision
        and attn_mode == "flash_attention_3"
        and static_bshd_kv_buffer is not None
        and static_prefix_len is not None
        and query.dtype in {torch.bfloat16, torch.float16}
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.shape[-1] <= 256
        and query.shape[-1] % 8 == 0
        and hopper_fa3_usable(query.device)
        and _FA3_MODULE is not None
    ):
        k_buf, v_buf = static_bshd_kv_buffer
        end_len = static_prefix_len + key.shape[1]
        k_buf[:, static_prefix_len:end_len].copy_(key)
        v_buf[:, static_prefix_len:end_len].copy_(value)
        q_c = query if query.is_contiguous() else query.contiguous()
        out = _FA3_MODULE.flash_attn_func(q_c, k_buf[:, :end_len], v_buf[:, :end_len])
        hidden_states = out[0] if isinstance(out, tuple) else out
        return (hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1), {})
    if enable_sp:
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    vision_kv = {}
    if cache_vision:
        vision_kv["k_vision"] = key.contiguous()
        vision_kv["v_vision"] = value.contiguous()

    effective_mode = attn_mode
    if (
        effective_mode == "cudnn_fp8"
        and block_idx is not None
        and num_blocks is not None
        and num_blocks >= 2
        and (block_idx == 0 or block_idx == num_blocks - 1)
    ):
        # Preserve BF16 attention precision on sensitive boundary blocks.
        effective_mode = "torch"

    scale_key = ("vision", stream_tag, block_idx)
    fp8_kv_buf: tuple[torch.Tensor, torch.Tensor] | None = None
    fp8_pref_len: int | None = None
    fp8_chunk_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    full_key: torch.Tensor
    full_value: torch.Tensor

    if (
        effective_mode == "cudnn_fp8"
        and not cache_vision
        and query.is_cuda
        and query.shape[-1] in {64, 128}
        and (query.device, scale_key) in _FP8_ATTN_SCALES
    ):
        if static_fp8_kv_buffer is not None and static_prefix_len is not None:
            fp8_kv_buf = static_fp8_kv_buffer
            fp8_pref_len = static_prefix_len
            fp8_chunk_kv = (key, value)
            full_key = key
            full_value = value
        elif kv_cache is not None and block_idx is not None:
            _, dk, dv = _FP8_ATTN_SCALES[(query.device, scale_key)]
            k_fp8_buf, v_fp8_buf, pref_len = _get_or_build_slot_fp8_kv_buffer(
                kv_cache[block_idx],
                key,
                value,
                stream_tag=stream_tag,
                dk=dk,
                dv=dv,
            )
            fp8_kv_buf = (k_fp8_buf, v_fp8_buf)
            fp8_pref_len = pref_len
            fp8_chunk_kv = (key, value)
            full_key = key
            full_value = value

    if fp8_kv_buf is None:
        if static_kv_buffer is not None and static_prefix_len is not None and not cache_vision:
            k_buf, v_buf = static_kv_buffer
            end_len = static_prefix_len + key.shape[2]
            k_buf[:, :, static_prefix_len:end_len].copy_(key)
            v_buf[:, :, static_prefix_len:end_len].copy_(value)
            full_key = k_buf[:, :, :end_len]
            full_value = v_buf[:, :, :end_len]
        else:
            assert kv_cache is not None
            slot = kv_cache[block_idx]
            if not cache_vision:
                full_key, full_value = _get_or_build_slot_kv_buffer(slot, key, value)
            else:
                cache_vision_key = slot.get("k_vision")
                cache_vision_value = slot.get("v_vision")
                encoder_key = slot["k_txt"]
                encoder_value = slot["v_txt"]
                if cache_vision_key is not None:
                    full_key = torch.cat([encoder_key, cache_vision_key, key], dim=2)
                    full_value = torch.cat([encoder_value, cache_vision_value, value], dim=2)
                else:
                    full_key = torch.cat([encoder_key, key], dim=2)
                    full_value = torch.cat([encoder_value, value], dim=2)

    hidden_states = _dispatch_cached_attention(
        query,
        full_key,
        full_value,
        attn_mode=effective_mode,
        scale_key=scale_key,
        fp8_kv_buffer=fp8_kv_buf,
        fp8_prefix_len=fp8_pref_len,
        fp8_chunk_kv=fp8_chunk_kv,
    )
    hidden_states = hidden_states.transpose(1, 2)
    if enable_sp:
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states, scatter_dim=1, gather_dim=2)
        hidden_states = hidden_states.to(query.dtype)
    (b, s, a, d) = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)
    return (hidden_states, vision_kv)


@torch.compiler.disable
def sequence_parallel_attention_bi_fp8(
    q: tuple[torch.Tensor, torch.Tensor],
    k: tuple[torch.Tensor, torch.Tensor],
    v: tuple[torch.Tensor, torch.Tensor],
    *,
    block_idx: int | None = None,
    num_blocks: int | None = None,
    stream_tag: str = "rope",
) -> torch.Tensor:
    """Joint multimodal (img + txt) attention with calibrated cuDNN FP8 SDPA and SP support.

    Used when ``attn_mode == "cudnn_fp8"`` and ``text_mask is None``. Preserves
    first/last blocks in BF16 and runs calibrated per-stream FP8 SDPA on interior
    blocks with single-pass Triton quantization.
    """
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def _shrink_head(tensor: torch.Tensor) -> torch.Tensor:
            local_heads = tensor.shape[2] // sp_size
            return tensor.narrow(2, sp_rank * local_heads, local_heads)

        encoder_query = _shrink_head(encoder_query)
        encoder_key = _shrink_head(encoder_key)
        encoder_value = _shrink_head(encoder_value)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)
    q_bhsd = torch.cat([query, encoder_query], dim=1).transpose(1, 2).contiguous()
    k_bhsd = torch.cat([key, encoder_key], dim=1).transpose(1, 2).contiguous()
    v_bhsd = torch.cat([value, encoder_value], dim=1).transpose(1, 2).contiguous()

    is_edge_block = (
        block_idx is not None
        and num_blocks is not None
        and num_blocks >= 2
        and (block_idx == 0 or block_idx == num_blocks - 1)
    )
    eff_mode = "torch" if is_edge_block else "cudnn_fp8"
    out_bhsd = _dispatch_cached_attention(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        attn_mode=eff_mode,
        scale_key=("bi", stream_tag, block_idx),
    )
    hidden_states = out_bhsd.transpose(1, 2)
    if enable_sp:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states.contiguous(), scatter_dim=1, gather_dim=2)
        encoder_hidden_states = sequence_model_parallel_all_gather(
            encoder_hidden_states.contiguous(), dim=2
        ).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    return hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)


@torch.compiler.disable
def sequence_parallel_attention_bi_fa3(
    q: tuple[torch.Tensor, torch.Tensor],
    k: tuple[torch.Tensor, torch.Tensor],
    v: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Direct BSHD joint multimodal (img + txt) Hopper FlashAttention-3 with zero BHSD transposes."""
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    parallel_dims = get_parallel_state()
    enable_sp = parallel_dims.sp_enabled
    if enable_sp:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

        def _shrink_head(tensor: torch.Tensor) -> torch.Tensor:
            local_heads = tensor.shape[2] // sp_size
            return tensor.narrow(2, sp_rank * local_heads, local_heads)

        encoder_query = _shrink_head(encoder_query)
        encoder_key = _shrink_head(encoder_key)
        encoder_value = _shrink_head(encoder_value)

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)
    q_bshd = torch.cat([query, encoder_query], dim=1).contiguous()
    k_bshd = torch.cat([key, encoder_key], dim=1).contiguous()
    v_bshd = torch.cat([value, encoder_value], dim=1).contiguous()

    if (
        hopper_fa3_usable(q_bshd.device)
        and _FA3_MODULE is not None
        and q_bshd.dtype in {torch.bfloat16, torch.float16}
        and k_bshd.dtype == q_bshd.dtype
        and v_bshd.dtype == q_bshd.dtype
        and q_bshd.shape[-1] <= 256
        and q_bshd.shape[-1] % 8 == 0
    ):
        out = _FA3_MODULE.flash_attn_func(q_bshd, k_bshd, v_bshd)
        hidden_states = out[0] if isinstance(out, tuple) else out
    else:
        hidden_states = scaled_dot_product_attention(
            q_bshd.transpose(1, 2),
            k_bshd.transpose(1, 2),
            v_bshd.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)

    if enable_sp:
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )
        hidden_states = sequence_model_parallel_all_to_all_4D(hidden_states.contiguous(), scatter_dim=1, gather_dim=2)
        encoder_hidden_states = sequence_model_parallel_all_gather(
            encoder_hidden_states.contiguous(), dim=2
        ).contiguous()
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states = encoder_hidden_states.to(query.dtype)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    return hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
