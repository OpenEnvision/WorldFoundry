from __future__ import annotations

from pathlib import Path

import torch
from torch.distributed.checkpoint import save

from worldfoundry.synthesis.action_generation.dreamdojo.convert_distcp import convert_distcp


def test_convert_distcp_exports_only_bf16_ema_weights(tmp_path: Path) -> None:
    distcp_dir = tmp_path / "model"
    save(
        {
            "net.weight": torch.ones(2, dtype=torch.float32),
            "net_ema.weight": torch.arange(2, dtype=torch.float32),
            "net_ema.counter": torch.tensor(3, dtype=torch.int64),
        },
        checkpoint_id=distcp_dir,
    )

    target = convert_distcp(distcp_dir, tmp_path)
    result = torch.load(target, map_location="cpu", weights_only=False)

    assert set(result) == {"net.weight", "net.counter"}
    assert result["net.weight"].dtype == torch.bfloat16
    assert result["net.counter"].dtype == torch.int64
    assert not (tmp_path / ".worldfoundry_model.pt").exists()


def test_convert_distcp_reuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "model_ema_bf16.pt"
    torch.save({"sentinel": True}, target)

    assert convert_distcp(tmp_path / "missing", tmp_path) == target
