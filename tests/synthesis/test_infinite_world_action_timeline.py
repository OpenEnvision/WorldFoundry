from __future__ import annotations

import torch

from worldfoundry.pipelines.infinite_world.pipeline_infinite_world import (
    InfiniteWorldPipeline,
)


def test_short_interactions_expand_across_the_generated_timeline() -> None:
    move = torch.tensor([0, 0, 1, 3])
    view = torch.tensor([0, 0, 0, 4])

    expanded_move, expanded_view = InfiniteWorldPipeline._expand_action_ids(
        move,
        view,
        prefix_length=2,
        target_length=8,
    )

    assert expanded_move.tolist() == [0, 0, 1, 1, 1, 3, 3, 3]
    assert expanded_view.tolist() == [0, 0, 0, 0, 0, 4, 4, 4]


def test_frame_level_action_timeline_is_not_rewritten() -> None:
    move = torch.tensor([1, 1, 3, 3])
    view = torch.tensor([0, 0, 4, 4])

    expanded_move, expanded_view = InfiniteWorldPipeline._expand_action_ids(
        move,
        view,
        prefix_length=0,
        target_length=4,
    )

    assert expanded_move is move
    assert expanded_view is view
