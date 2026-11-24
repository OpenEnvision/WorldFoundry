"""Golden records of the default eight-key realtime control resampler.

The default ``RealtimeSpec`` keys must reproduce these segments, interactions
and live key states exactly.
"""

from __future__ import annotations

from typing import Any

import pytest

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.studio.serving.realtime.input import RealtimeControlResampler, interactions_from_segments

Operation = tuple[Any, ...]
Record = tuple[Any, tuple[str, ...], bool]

SEQUENCES: dict[str, tuple[float, float, list[Operation]]] = {
    "sub_frame_taps": (
        20,
        100.0,
        [
            ("edge", 100.01, "keydown", "d"),
            ("edge", 100.02, "keyup", "d"),
            ("edge", 100.021, "keydown", "w"),
            ("edge", 100.022, "keyup", "w"),
            ("sample", 4, 100.0),
            ("sample", 4, 100.2),
            ("sample", 4, 100.4),
        ],
    ),
    "overlapping_opposites": (
        20,
        100.0,
        [
            ("edge", 100.0, "keydown", "w"),
            ("edge", 100.1, "keydown", "s"),
            ("edge", 100.12, "keydown", "a"),
            ("edge", 100.15, "keydown", "d"),
            ("edge", 100.3, "keyup", "s"),
            ("edge", 100.32, "keyup", "d"),
            ("edge", 100.4, "keyup", "w"),
            ("edge", 100.45, "keyup", "a"),
            ("sample", 16, 101.0),
            ("sample", 4, 101.8),
        ],
    ),
    "late_releases": (
        16,
        50.0,
        [
            ("edge", 50.0, "keydown", "w"),
            ("edge", 50.0, "keydown", "j"),
            ("sample", 9, 50.0),
            ("edge", 50.2, "keyup", "w"),
            ("edge", 50.25, "keydown", "w"),
            ("edge", 50.6, "keyup", "j"),
            ("edge", 50.61, "keyup", "w"),
            ("sample", 9, 51.5),
            ("sample", 9, 52.1),
        ],
    ),
    "multi_chunk_hold": (
        12,
        10.0,
        [
            ("edge", 10.03, "keydown", "d"),
            ("edge", 10.05, "keydown", "i"),
            ("sample", 3, 10.0),
            ("sample", 3, 10.25),
            ("sample", 3, 10.5),
            ("edge", 10.6, "keydown", "k"),
            ("sample", 3, 10.75),
            ("edge", 10.95, "keyup", "k"),
            ("edge", 10.96, "keyup", "i"),
            ("sample", 3, 11.0),
            ("edge", 11.2, "keyup", "d"),
            ("sample", 3, 11.25),
            ("sample", 3, 11.5),
        ],
    ),
    "arrow_aliases": (
        20,
        0.0,
        [
            ("edge", 1.0, "keydown", "ArrowUp"),
            ("edge", 1.01, "keydown", "ArrowLeft"),
            ("edge", 1.02, "keydown", "q"),
            ("edge", 1.03, "keydown", "ArrowRight"),
            ("edge", 1.04, "keypress", "w"),
            ("edge", 1.05, "keydown", " W "),
            ("sample", 4, 1.0),
            ("edge", 1.25, "keyup", "arrowup"),
            ("edge", 1.26, "keyup", "ArrowLeft"),
            ("edge", 1.27, "keyup", "ArrowRight"),
            ("edge", 1.28, "keydown", "ArrowDown"),
            ("edge", 1.29, "keyup", "W"),
            ("sample", 4, 1.2),
            ("edge", 1.5, "keyup", "ArrowDown"),
            ("sample", 4, 1.4),
            ("sample", 4, 1.6),
        ],
    ),
    "reset": (
        20,
        100.0,
        [
            ("edge", 100.0, "keydown", "w"),
            ("edge", 100.0, "keydown", "ArrowRight"),
            ("sample", 4, 100.0),
            ("edge", 100.3, "keydown", "d"),
            ("edge", 100.31, "keyup", "d"),
            ("reset", 200.0),
            ("sample", 4, 200.0),
            ("edge", 200.21, "keydown", "a"),
            ("sample", 4, 200.25),
            ("edge", 200.6, "keyup", "a"),
            ("sample", 4, 200.45),
            ("sample", 4, 200.65),
        ],
    ),
}

EXPECTED: dict[str, list[Record]] = {
    "sub_frame_taps": [
        (True, ("d",), True),
        (True, (), True),
        (True, ("w",), True),
        (True, (), True),
        (
            (
                [
                    (100.0, 100.01, ()),
                    (100.01, 100.06, ("d",)),
                    (100.06, 100.11, ()),
                    (100.11, 100.16, ("w",)),
                    (100.16, 100.2, ()),
                ],
                ["forward"],
            ),
            (),
            False,
        ),
        (([(100.2, 100.4, ())], []), (), False),
        (([(100.4, 100.60000000000001, ())], []), (), False),
    ],
    "overlapping_opposites": [
        (True, ("w",), True),
        (True, ("s",), True),
        (True, ("a", "s"), True),
        (True, ("d", "s"), True),
        (True, ("d", "w"), True),
        (True, ("a", "w"), True),
        (True, ("a",), True),
        (True, (), True),
        (
            (
                [
                    (101.0, 101.1, ("w",)),
                    (101.1, 101.14999999999999, ("s",)),
                    (101.14999999999999, 101.19999999999999, ("a", "s")),
                    (101.19999999999999, 101.34999999999998, ("d", "s")),
                    (101.34999999999998, 101.39999999999998, ("d", "w")),
                    (101.39999999999998, 101.47999999999999, ("a", "w")),
                    (101.47999999999999, 101.52999999999999, ("a",)),
                    (101.52999999999999, 101.8, ()),
                ],
                ["left"],
            ),
            (),
            False,
        ),
        (([(101.8, 102.0, ())], []), (), False),
    ],
    "late_releases": [
        (True, ("w",), True),
        (True, ("j", "w"), True),
        (([(50.0, 50.5625, ("j", "w"))], ["forward", "camera_l"]), ("j", "w"), True),
        (True, ("j",), True),
        (True, ("j", "w"), True),
        (True, ("w",), True),
        (True, (), True),
        (
            (
                [
                    (51.5, 51.5625, ("j",)),
                    (51.5625, 51.9125, ("j", "w")),
                    (51.9125, 51.975, ("w",)),
                    (51.975, 52.0625, ()),
                ],
                ["forward"],
            ),
            (),
            False,
        ),
        (([(52.0625, 52.625, ())], []), (), False),
    ],
    "multi_chunk_hold": [
        (True, ("d",), True),
        (True, ("d", "i"), True),
        (
            (
                [(10.0, 10.03, ()), (10.03, 10.113333333333333, ("d",)), (10.113333333333333, 10.25, ("d", "i"))],
                ["right", "camera_up"],
            ),
            ("d", "i"),
            True,
        ),
        (([(10.25, 10.5, ("d", "i"))], ["right", "camera_up"]), ("d", "i"), True),
        (([(10.5, 10.75, ("d", "i"))], ["right", "camera_up"]), ("d", "i"), True),
        (True, ("d", "k"), True),
        (([(10.75, 11.0, ("d", "k"))], ["right", "camera_down"]), ("d", "k"), True),
        (True, ("d", "i"), True),
        (True, ("d",), True),
        (([(11.0, 11.083333333333334, ("d", "i")), (11.083333333333334, 11.25, ("d",))], ["right"]), ("d",), True),
        (True, (), True),
        (([(11.25, 11.5, ())], []), (), False),
        (([(11.5, 11.75, ())], []), (), False),
    ],
    "arrow_aliases": [
        (True, ("i",), True),
        (True, ("i", "j"), True),
        (False, ("i", "j"), True),
        (True, ("i", "l"), True),
        (False, ("i", "l"), True),
        (True, ("i", "l", "w"), True),
        (
            (
                [
                    (1.0, 1.05, ("i",)),
                    (1.05, 1.1, ("i", "j")),
                    (1.1, 1.1500000000000001, ("i", "l")),
                    (1.1500000000000001, 1.2, ("i", "l", "w")),
                ],
                ["forward", "camera_up", "camera_r"],
            ),
            ("i", "l", "w"),
            True,
        ),
        (True, ("l", "w"), True),
        (True, ("l", "w"), True),
        (True, ("w",), True),
        (True, ("k", "w"), True),
        (True, ("k",), True),
        (
            (
                [(1.2, 1.25, ("i", "l", "w")), (1.25, 1.3, ("l", "w")), (1.3, 1.35, ("w",)), (1.35, 1.4, ("k", "w"))],
                ["forward", "camera_down"],
            ),
            ("k",),
            True,
        ),
        (True, (), True),
        (
            (
                [(1.4, 1.4000000000000001, ("k", "w")), (1.4000000000000001, 1.5999999999999999, ("k",))],
                ["camera_down"],
            ),
            (),
            True,
        ),
        (([(1.5999999999999999, 1.61, ("k",)), (1.61, 1.7999999999999998, ())], ["camera_down"]), (), False),
    ],
    "reset": [
        (True, ("w",), True),
        (True, ("l", "w"), True),
        (([(100.0, 100.2, ("l", "w"))], ["forward", "camera_r"]), ("l", "w"), True),
        (True, ("d", "l", "w"), True),
        (True, ("l", "w"), True),
        (None, (), False),
        (([(200.0, 200.2, ())], []), (), False),
        (True, ("a",), True),
        (([(200.2, 200.21, ()), (200.21, 200.39999999999998, ("a",))], ["left"]), ("a",), True),
        (True, (), True),
        (([(200.39999999999998, 200.59999999999997, ("a",))], ["left"]), (), True),
        (([(200.59999999999997, 200.6, ("a",)), (200.6, 200.79999999999995, ())], ["left"]), (), False),
    ],
}


def _replay(resampler: RealtimeControlResampler, operations: list[Operation], *, spec_keys: bool) -> list[Record]:
    """Apply edges, samples and resets; record each result with the live key state.

    With ``spec_keys`` the replay samples through ``sample_input_chunk`` and maps
    keys through the resampler's declared bindings, as the realtime workers do.
    """

    records: list[Record] = []
    for operation in operations:
        result: Any
        if operation[0] == "edge":
            _, stamp, event, key = operation
            result = resampler.on_edge(arrival_time=stamp, event=event, key=key)
        elif operation[0] == "sample":
            _, frames, wall_time = operation
            if spec_keys:
                chunk = resampler.sample_input_chunk(frames, wall_time=wall_time)
                assert chunk.axes == {}
                segments = chunk.segments
                interactions = interactions_from_segments(segments, resampler.realtime_keys)
            else:
                segments = resampler.sample_chunk(frames, wall_time=wall_time)
                interactions = interactions_from_segments(segments)
            result = ([(start, end, tuple(sorted(keys))) for start, end, keys in segments], interactions)
        else:
            resampler.reset(start_time=operation[1])
            result = None
        records.append((result, tuple(sorted(resampler.effective_keys)), resampler.has_pending_input))
    return records


@pytest.mark.parametrize("spec_keys", [False, True])
@pytest.mark.parametrize("name", list(SEQUENCES))
def test_realtime_input_default_keys_match_golden_records(name: str, spec_keys: bool) -> None:
    fps, start_time, operations = SEQUENCES[name]
    if spec_keys:
        spec = RealtimeSpec()
        resampler = RealtimeControlResampler(
            fps=fps,
            start_time=start_time,
            realtime_keys=spec.keys,
            axes=spec.axes,
        )
    else:
        resampler = RealtimeControlResampler(fps=fps, start_time=start_time)
    assert _replay(resampler, operations, spec_keys=spec_keys) == EXPECTED[name]
