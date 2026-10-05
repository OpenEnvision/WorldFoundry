from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan import WanPrompter
from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel


def test_wan_bfloat16_per_token_timestep_matches_embedding_dtype() -> None:
    model = WanModel(
        dim=12,
        in_dim=2,
        ffn_dim=24,
        out_dim=2,
        text_dim=8,
        freq_dim=4,
        patch_size=(1, 1, 1),
        num_heads=1,
        num_layers=0,
        eps=1e-6,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
        per_token_timestep=True,
    ).to(torch.bfloat16)

    output = model(
        torch.zeros((1, 2, 1, 1, 1), dtype=torch.bfloat16),
        torch.ones((1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1, 8), dtype=torch.bfloat16),
    )

    assert output.shape == (1, 2, 1, 1, 1)
    assert output.dtype == torch.bfloat16


def test_wan_prompter_masks_padding_without_mutating_multi_view_output() -> None:
    class _Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]])

    class _Encoder:
        def __init__(self) -> None:
            self.source = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).requires_grad_()

        def __call__(self, _ids: torch.Tensor, _mask: torch.Tensor) -> torch.Tensor:
            # ``chunk`` returns sibling views that PyTorch forbids modifying
            # in place, matching the Transformers output that Astra exposed.
            return self.source.chunk(2, dim=-1)[0]

    prompter = WanPrompter()
    encoder = _Encoder()
    prompter.tokenizer = _Tokenizer()
    prompter.text_encoder = encoder

    output = prompter.encode_prompt("test", device="cpu")
    output.sum().backward()

    assert torch.count_nonzero(output[:, :2]).item() > 0
    assert torch.count_nonzero(output[:, 2:]).item() == 0
    assert encoder.source.grad is not None
