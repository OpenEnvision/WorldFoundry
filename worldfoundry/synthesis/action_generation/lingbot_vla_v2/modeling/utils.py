"""Tensor and attention-mask utilities for LingBot-VLA v2 inference."""


# from xformers.ops import memory_efficient_attention


def prefix_query_segments(
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return prefix segment order after the image block.

    Task-specific query tokens are always placed after language tokens. Current
    task queries precede future task queries; future-depth remains the last
    query segment so the existing suffix-to-future-depth blocking can keep using
    the tail span.
    """
    segments = ["language"]
    if not use_depth_align:
        return tuple(segments)

    segments.append("current_depth")
    if use_future_video:
        if use_future_video_cls:
            segments.append("future_video_cls")
        if use_future_video_patch and not future_video_share_future_depth_query:
            segments.append("future_video")
    if use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def prefix_query_token_spans(
    prefix_len,
    num_task_tokens,
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
    """Return [start, end) spans for non-language task query segments."""
    counts = {
        "current_depth": num_task_tokens,
        "future_video_cls": 1,
        "future_video": num_task_tokens,
        "future_depth": num_task_tokens,
    }
    ordered = prefix_query_segments(
        use_depth_align=use_depth_align,
        use_future_depth=use_future_depth,
        use_future_video=use_future_video,
        use_future_video_cls=use_future_video_cls,
        use_future_video_patch=use_future_video_patch,
        future_video_share_future_depth_query=future_video_share_future_depth_query,
    )
    query_segments = [name for name in ordered if name != "language"]
    cursor = prefix_len - sum(counts[name] for name in query_segments)
    spans = {}
    for name in query_segments:
        count = counts[name]
        spans[name] = (cursor, cursor + count)
        cursor += count
    return spans


def fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch):
    """Return [start, end) of a tail query block inside the prefix.

    This legacy helper is still used for future-depth tail blocking in V2.
    New prefix layout code should prefer prefix_query_token_spans(), which also
    handles current-depth and separate future-video spans.
    """
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len


def block_suffix_to_fv_(
    att_2d_masks, suffix_row_start, prefix_len, num_task_tokens, use_cls=False, use_patch=True, drop_mask=None
):
    """In-place mask out the suffix-to-future-video attention edge.

    `make_att_2d_masks`' cumsum scheme cannot express "a query cannot see a
    segment that precedes it", so we zero the rectangular [suffix rows, FV cols]
    block on the already-built 2D mask instead of touching mask_ar.

    att_2d_masks: bool[B, Q, K], True == visible. `suffix_row_start` is the first
    query row belonging to the suffix: prefix_len in the square training mask,
    0 in the suffix-only inference mask. Leaves FV -> img/lang rows untouched so
    the distillation query still reads the current observation.

    `drop_mask`: optional bool[B], True where this sample's suffix must NOT see
    FV. None == block every sample (hard mask). Used for per-sample stochastic
    masking (FV-attention dropout): keep = visible iff not dropped, applied via
    broadcast multiply so it stays a static graph under torch.compile.
    """
    fv_start, fv_end = fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch)
    if fv_end <= fv_start:
        return att_2d_masks
    if drop_mask is None:
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = False
    else:
        # keep[b] = True where the sample is NOT dropped -> AND keeps those rows
        # visible and zeros the dropped ones, with no data-dependent indexing.
        keep = (~drop_mask).view(-1, 1, 1)
        block = att_2d_masks[:, suffix_row_start:, fv_start:fv_end]
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = block & keep
    return att_2d_masks


# @torch.jit.script
