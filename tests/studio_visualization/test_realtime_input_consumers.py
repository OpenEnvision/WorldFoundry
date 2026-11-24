from __future__ import annotations

from types import SimpleNamespace

import pytest

from worldfoundry.studio.serving.realtime.input import RealtimeControlResampler


@pytest.mark.parametrize("consumer", ["dreamx", "matrix2"])
@pytest.mark.parametrize("chunk_frames", [9, 12])
def test_realtime_input_short_taps_reach_native_control_samples(consumer: str, chunk_frames: int) -> None:
    import torch

    from worldfoundry.synthesis.visual_generation.dreamx_world.realtime import _frame_keys
    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
        MatrixGame2RealtimeSession,
    )

    resampler = RealtimeControlResampler(fps=12, start_time=100)
    resampler.sample_chunk(chunk_frames, wall_time=100)
    for stamp, event, key in (
        (100.01, "keydown", "d"),
        (100.011, "keyup", "d"),
        (100.02, "keydown", "a"),
        (100.021, "keyup", "a"),
    ):
        assert resampler.on_edge(arrival_time=stamp, event=event, key=key)
    segments = [
        {"duration": end - start, "keys": sorted(keys)}
        for start, end, keys in resampler.sample_chunk(chunk_frames, wall_time=101)
    ]
    if consumer == "dreamx":
        keys = _frame_keys(segments, [], frame_count=chunk_frames, fps=12)
        assert frozenset({"d"}) in keys
        assert frozenset({"a"}) in keys
        assert keys.index(frozenset({"d"})) < keys.index(frozenset({"a"}))
    else:
        session = object.__new__(MatrixGame2RealtimeSession)
        session.runtime = SimpleNamespace(mode="universal")
        session.device = torch.device("cpu")
        session.dtype = torch.float32
        session._keyboard_history = None
        session._mouse_history = None
        actions = session._actions_for_frames([], segments, num_frames=chunk_frames)
        assert ["right"] in actions
        assert ["left"] in actions
        assert actions.index(["right"]) < actions.index(["left"])
        session._append_actions(actions)
        from worldfoundry.operators.matrix_game_2_operator import encode_actions

        expected = torch.stack([encode_actions(action, "universal")[0] for action in actions])
        torch.testing.assert_close(session._keyboard_history, expected, rtol=0, atol=0)
        assert isinstance(session._keyboard_history, torch.Tensor)
        assert session._keyboard_history.shape[0] == chunk_frames


def test_matrix_game_2_control_history_terminal_row_enters_next_causal_window() -> None:
    import torch

    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.action_conditioning_21 import (
        ActionModule,
    )

    action = ActionModule(
        hidden_size=6,
        img_hidden_size=8,
        keyboard_hidden_dim=64,
        heads_num=1,
        enable_mouse=False,
        windows_size=3,
    )
    action.keyboard_embed = torch.nn.Sequential(torch.nn.Identity())
    groups: list[torch.Tensor] = []

    class WindowCaptured(Exception):
        pass

    def capture_window(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        groups.append(args[0].detach().clone())
        raise WindowCaptured

    hook = action.keyboard_attn_kv.register_forward_pre_hook(capture_window)
    history = torch.zeros(1, 21, 6)
    history[:, 8, 0] = 1  # Final pixel control row of the first nine-frame block.
    try:
        for start, rows in ((0, 9), (3, 21)):
            with pytest.raises(WindowCaptured):
                action(
                    torch.zeros(1, 3, 8),
                    tt=3,
                    th=1,
                    tw=1,
                    keyboard_condition=history[:, :rows],
                    is_causal=True,
                    kv_cache_keyboard={},
                    start_frame=start,
                    num_frame_per_block=3,
                )
    finally:
        hook.remove()
    assert torch.count_nonzero(groups[0]).item() == 0
    assert torch.count_nonzero(groups[1]).item() > 0


def test_realtime_mouse_axis_reaches_matrix_game_2_mouse_condition() -> None:
    import torch

    from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_2_runtime.realtime import (
        MatrixGame2RealtimeSession,
    )

    session = object.__new__(MatrixGame2RealtimeSession)
    session.runtime = SimpleNamespace(mode="universal")
    session.device = torch.device("cpu")
    session.dtype = torch.float32
    session._keyboard_history = None
    session._mouse_history = None
    spec = session.realtime_spec()
    resampler = RealtimeControlResampler(fps=12, start_time=100, realtime_keys=spec.keys, axes=spec.axes)
    resampler.sample_input_chunk(9, wall_time=100)
    assert resampler.on_edge(arrival_time=100.70, event="keydown", key="i")
    # 12 px right on frame 0, 80 px right on frame 1 (beyond the limit), 20 px down on frame 3.
    for stamp, samples in (
        (100.76, {"mouse_x": 12.0}),
        (100.84, {"mouse_x": 80.0}),
        (101.01, {"mouse_y": 20.0}),
    ):
        assert resampler.on_axis(arrival_time=stamp, samples=samples)
    chunk = resampler.sample_input_chunk(9, wall_time=101)
    segments = [{"duration": end - start, "keys": sorted(keys)} for start, end, keys in chunk.segments]

    session._append_actions(
        session._actions_for_frames([], segments, num_frames=9),
        session._camera_offsets(chunk.axes, num_frames=9),
    )

    expected = torch.zeros(9, 2)
    expected[:, 0] = 0.1  # The held I key looks up on every frame.
    expected[3, 0] = 0.05  # Moving the mouse down looks down.
    expected[:2, 1] = torch.tensor([0.03, 0.1])
    torch.testing.assert_close(session._mouse_history, expected)
