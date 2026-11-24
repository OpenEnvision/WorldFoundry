from __future__ import annotations

import torch

from worldfoundry.studio.inference.catalog.models.video_generation import _cameractrl_default_load_kwargs
from worldfoundry.synthesis.visual_generation.cameractrl.cameractrl_runtime.inference import (
    fuse_motion_adapter,
)


class _ToyUNet:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(3, 2))

    def named_parameters(self):
        yield "block.attn1.to_q.weight", self.weight


def test_fuse_motion_adapter_applies_paired_lora_weights(tmp_path) -> None:
    checkpoint = tmp_path / "adapter.ckpt"
    torch.save(
        {
            "block.attn1.processor.to_q_lora.down.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "block.attn1.processor.to_q_lora.up.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        },
        checkpoint,
    )
    unet = _ToyUNet()

    assert fuse_motion_adapter(unet, checkpoint, scale=0.5) == 1
    torch.testing.assert_close(
        unet.weight,
        torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.0, 3.0]]),
    )


def test_cameractrl_catalog_supplies_the_required_v3_adapter(tmp_path, monkeypatch) -> None:
    adapter = tmp_path / "ckpts" / "guoyww--animatediff" / "v3_sd15_adapter.ckpt"
    adapter.parent.mkdir(parents=True)
    adapter.touch()
    monkeypatch.setenv("WORLDFOUNDRY_CKPT_DIR", str(tmp_path / "ckpts"))

    load_kwargs = _cameractrl_default_load_kwargs()

    assert load_kwargs["motion_adapter_ckpt"] == str(adapter)
    assert load_kwargs["unet_subfolder"] == "unet"


def test_attn_processor_handles_plain_linear_without_peft(monkeypatch) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from worldfoundry.synthesis.visual_generation.cameractrl.cameractrl_runtime.models import (
        attention_processor,
    )

    monkeypatch.setattr(attention_processor, "USE_PEFT_BACKEND", False)
    attn = SimpleNamespace(
        spatial_norm=None,
        group_norm=None,
        norm_cross=False,
        residual_connection=False,
        rescale_output_factor=1.0,
        prepare_attention_mask=lambda mask, seq_len, batch: mask,
        to_q=torch.nn.Linear(4, 4),
        to_k=torch.nn.Linear(4, 4),
        to_v=torch.nn.Linear(4, 4),
        to_out=torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Identity()]),
        head_to_batch_dim=lambda x: x,
        get_attention_scores=lambda q, k, mask: torch.bmm(q, k.transpose(-1, -2)),
        batch_to_head_dim=lambda x: x,
    )

    out = attention_processor.AttnProcessor()(cast(Any, attn), cast(Any, torch.randn(1, 3, 4)), scale=0.5)

    assert out.shape == (1, 3, 4)
