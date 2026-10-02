from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.core.attention.causal_cache import (  # noqa: E402
    CausalVideoCacheGeometry,
    allocate_causal_video_cache,
    begin_causal_video_cache_block,
    causal_video_cache_state,
    commit_causal_video_cache_block,
    finish_causal_video_cache_call,
)


def _cache(*, local_attention_frames: int = -1):
    return allocate_causal_video_cache(
        CausalVideoCacheGeometry(
            batch_size=1,
            total_frames=9,
            frame_tokens=2,
            frames_per_block=3,
            num_layers=2,
            num_heads=1,
            head_dim=4,
            local_attention_frames=local_attention_frames,
        ),
        device="cpu",
        dtype=torch.float32,
    )


def _finish_graph_call(cache, *, start_frame: int, frame_count: int) -> None:
    global_end = (start_frame + frame_count) * int(cache["frame_tokens"])
    local_end = min(
        global_end,
        int(cache["local_attention_frames"]) * int(cache["frame_tokens"])
        if int(cache["local_attention_frames"]) > 0
        else global_end,
    )
    for layer in cache["kv_cache"]:
        layer["global_end_index"].fill_(global_end)
        layer["local_end_index"].fill_(local_end)
    finish_causal_video_cache_call(
        cache,
        start_frame=start_frame,
        frame_count=frame_count,
    )


def test_causal_video_cache_overwrites_then_advances_only_after_clean_commit() -> None:
    cache = _cache()
    assert causal_video_cache_state(cache).current_block_idx == -1

    begin_causal_video_cache_block(cache, block_index=0, start_frame=0, frame_count=3)
    _finish_graph_call(cache, start_frame=0, frame_count=3)
    begin_causal_video_cache_block(cache, block_index=0, start_frame=0, frame_count=3)
    _finish_graph_call(cache, start_frame=0, frame_count=3)
    assert causal_video_cache_state(cache).blocks == ()

    state = commit_causal_video_cache_block(
        cache,
        block_index=0,
        start_frame=0,
        frame_count=3,
    )
    assert [(block.block_idx, block.frame_start, block.frame_count) for block in state.blocks] == [(0, 0, 3)]

    begin_causal_video_cache_block(cache, block_index=1, start_frame=3, frame_count=3)
    _finish_graph_call(cache, start_frame=3, frame_count=3)
    state = commit_causal_video_cache_block(
        cache,
        block_index=1,
        start_frame=3,
        frame_count=3,
    )
    assert state.cached_frames == 6
    assert state.current_block_idx == 1


def test_causal_video_cache_audits_local_rollover_and_rejects_position_drift() -> None:
    cache = _cache(local_attention_frames=3)
    for block_index, start_frame in enumerate((0, 3, 6)):
        begin_causal_video_cache_block(
            cache,
            block_index=block_index,
            start_frame=start_frame,
            frame_count=3,
        )
        _finish_graph_call(cache, start_frame=start_frame, frame_count=3)
        commit_causal_video_cache_block(
            cache,
            block_index=block_index,
            start_frame=start_frame,
            frame_count=3,
        )
    assert causal_video_cache_state(cache).cached_frames == 9
    for layer in cache["kv_cache"]:
        assert layer["global_end_index"].item() == 18
        assert layer["local_end_index"].item() == 6

    fresh = _cache()
    with pytest.raises(ValueError, match="start_frame differs"):
        begin_causal_video_cache_block(
            fresh,
            block_index=0,
            start_frame=3,
            frame_count=3,
        )
    assert fresh["active_block"] == -1
    assert fresh["committed_blocks"] == 0
