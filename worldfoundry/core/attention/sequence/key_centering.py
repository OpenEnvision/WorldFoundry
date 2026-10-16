"""Request-owned finalized-key reductions for streaming quantized attention."""


def streaming_key_mean(cache, *, current_start, local_start, local_end, maximum):
    """Reuse only history that remains fixed across denoising calls of a chunk.

    The visible window is resolved after KV eviction. Current-chunk keys are
    reduced on every call; finalized history is reduced once per window/chunk.
    The caller owns and resets the KV cache dictionary with its request.
    """
    visible_start = max(0, local_end - maximum)
    history_end = max(visible_start, local_start)
    window = (current_start, visible_start, history_end)
    if cache.get("key_history_window") != window:
        cache["key_history_sum"] = cache["k"][:, visible_start:history_end].float().sum(dim=1, keepdim=True)
        cache["key_history_window"] = window
    current_sum = cache["k"][:, history_end:local_end].float().sum(dim=1, keepdim=True)
    mean = (cache["key_history_sum"] + current_sum) / (local_end - visible_start)
    return mean.to(cache["k"].dtype)


def streaming_scaled_attention(query, key, value, cache, *, current_start, local_start, local_end, maximum):
    """Execute the shared scaled provider with a request-owned history reduction."""
    from worldfoundry.core.attention.backends.dispatch import attention_forward

    mean = streaming_key_mean(
        cache, current_start=current_start, local_start=local_start, local_end=local_end, maximum=maximum
    )
    return attention_forward(
        query,
        key,
        value,
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        backend="scaled_int8_fp8",
        backend_options={"key_mean": mean},
    )
