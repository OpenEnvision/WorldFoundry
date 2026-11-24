"""Action-token masks used by the OpenVLA-OFT inference model."""

from __future__ import annotations

import torch

from ..constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = (cumsum >= 1) & (cumsum <= ACTION_DIM)
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only_mask * mask


def get_next_actions_mask(token_ids):
    newline_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(newline_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only_mask * mask
