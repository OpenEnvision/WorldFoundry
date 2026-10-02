"""Fullgraph compile eligibility contract for the canonical Wan inference graph."""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel


def test_tiny_wan_forward_is_cpu_fullgraph_compilable() -> None:
    torch.manual_seed(0)
    model = WanModel(
        dim=96,
        in_dim=4,
        ffn_dim=192,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        require_vae_embedding=False,
    ).eval()
    model.set_attention_compatibility_mode(True)
    x = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([10.0])
    context = torch.randn(1, 5, 32)

    with torch.no_grad():
        expected = model(x=x, timestep=timestep, context=context)
        compiled = torch.compile(
            model.forward,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(x=x, timestep=timestep, context=context)

    torch.testing.assert_close(actual, expected)
