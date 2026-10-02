from __future__ import annotations

import pytest

from worldfoundry.training.post_training.rl.rollout_strategies.contracts import (
    FlowSDEIndexResolver,
)
from worldfoundry.training.post_training.rl.rollout_strategies.window_sde_steps import (
    FlowSDEWindowSchedule,
)


def test_sliding_window_is_a_pure_function_of_committed_rollout_id() -> None:
    schedule = FlowSDEWindowSchedule(
        transition_count=25,
        window_size=4,
        iterations_per_window=25,
        stride=1,
        rollback=True,
    )

    assert isinstance(schedule, FlowSDEIndexResolver)
    assert schedule.window_count == 22
    assert schedule.resolve(0) == (0, 1, 2, 3)
    assert schedule.resolve(24) == (0, 1, 2, 3)
    assert schedule.resolve(25) == (1, 2, 3, 4)
    assert schedule.resolve(25 * 21) == (21, 22, 23, 24)
    assert schedule.resolve(25 * 22) == (0, 1, 2, 3)
    assert schedule.resolve(25) == schedule.resolve(25)


def test_non_rollback_window_stays_on_the_last_complete_window() -> None:
    schedule = FlowSDEWindowSchedule(
        transition_count=10,
        window_size=4,
        iterations_per_window=2,
        stride=3,
        initial_index=1,
        rollback=False,
    )

    assert schedule.window_count == 2
    assert schedule.resolve(0) == (1, 2, 3, 4)
    assert schedule.resolve(2) == (4, 5, 6, 7)
    assert schedule.resolve(10_000) == (4, 5, 6, 7)
    assert dict(schedule.identity) == {
        "schema": "worldfoundry-flow-sde-window-schedule",
        "transition_count": 10,
        "window_size": 4,
        "iterations_per_window": 2,
        "stride": 3,
        "initial_index": 1,
        "rollback": False,
    }


def test_omitted_window_options_match_unirl_defaults() -> None:
    schedule = FlowSDEWindowSchedule(
        transition_count=12,
        window_size=4,
        iterations_per_window=2,
    )

    assert schedule.stride == 4
    assert schedule.rollback is False
    assert schedule.resolve(0) == (0, 1, 2, 3)
    assert schedule.resolve(2) == (4, 5, 6, 7)
    assert schedule.resolve(4) == (8, 9, 10, 11)
    assert schedule.resolve(10_000) == (8, 9, 10, 11)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"transition_count": 0, "window_size": 1, "iterations_per_window": 1},
        {"transition_count": 4, "window_size": 5, "iterations_per_window": 1},
        {"transition_count": 8, "window_size": 2, "iterations_per_window": 1, "stride": 3},
    ),
)
def test_invalid_window_geometry_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        FlowSDEWindowSchedule(**kwargs)
