"""The in-tree DINO torch.hub entry point loads local checkpoints the way VBench calls it."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worldfoundry.base_models.capabilities import vbench_dino_hub_config  # noqa: E402
from worldfoundry.base_models.perception_core.general_perception.dino import vit_base  # noqa: E402


@pytest.mark.unit
def test_vbench_dino_config_loads_the_local_checkpoint_through_torch_hub(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = vit_base(patch_size=16, num_classes=0).eval()
    checkpoint = tmp_path / "dino_vitbase16_pretrain.pth"
    torch.save(reference.state_dict(), checkpoint)

    config = vbench_dino_hub_config(env={"WORLDFOUNDRY_VBENCH_DINO_VITB16_CKPT": str(checkpoint)}, read_frame=False)
    model = torch.hub.load(**config).eval()

    assert config["source"] == "local" and config["model"] == "dino_vitb16"
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(sample), reference(sample))


@pytest.mark.unit
def test_pretrained_dino_without_a_checkpoint_path_fails_instead_of_downloading() -> None:
    from worldfoundry.base_models.perception_core.general_perception.dino.hub import hubconf

    with pytest.raises(RuntimeError, match="does not download weights"):
        hubconf.dino_vitb16()
