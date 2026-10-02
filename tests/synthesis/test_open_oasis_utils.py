from __future__ import annotations

import torch

from worldfoundry.synthesis.visual_generation.open_oasis.utils import (
    _fit_actions_to_frames,
)


def test_oasis_holds_last_action_for_full_requested_clip() -> None:
    actions = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    fitted = _fit_actions_to_frames(actions, 5)

    assert fitted.shape == (5, 2)
    assert torch.equal(fitted[:2], actions)
    assert torch.equal(fitted[2:], actions[-1:].expand(3, -1))


def test_oasis_trims_actions_to_requested_clip() -> None:
    actions = torch.arange(12).reshape(6, 2)

    assert torch.equal(_fit_actions_to_frames(actions, 3), actions[:3])
