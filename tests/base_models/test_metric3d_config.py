# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Metric3D configuration files are data, not runtime Python entrypoints."""

from pathlib import Path

import pytest

from worldfoundry.base_models.three_dimensions.depth.metric3d.model_fn import MODEL_TYPE, config_from_py_file

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model_type", "backbone"),
    [
        ("ConvNeXt-Tiny", "convnext_tiny"),
        ("ConvNeXt-Large", "convnext_large"),
        ("ViT-Small", "vit_small_reg"),
        ("ViT-Large", "vit_large_reg"),
        ("ViT-giant2", "vit_giant2_reg"),
    ],
)
def test_metric3d_config_preserves_attribute_access(model_type: str, backbone: str) -> None:
    config = config_from_py_file(MODEL_TYPE[model_type]["cfg_file"])
    assert config.model.backbone.type == backbone
    assert config.data_basic.canonical_space.focal_length == 1000.0


def test_metric3d_config_does_not_execute_file(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    config_path = tmp_path / "config.py"
    config_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "config = DictConfig({'model': {'backbone': 'small'}})\n"
    )
    config = config_from_py_file(str(config_path))
    assert config.model.backbone == "small"
    assert not marker.exists()


def test_metric3d_config_rejects_nonliteral_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.py"
    config_path.write_text("config = DictConfig({'model': make_model()})\n")
    with pytest.raises(ValueError):
        config_from_py_file(str(config_path))
