"""Path helpers for the in-tree ViT human anatomy detector."""

from __future__ import annotations

from pathlib import Path

from worldfoundry.core.io.paths import package_data_path


def config_path() -> Path:
    return package_data_path('models', 'runtime', 'configs', 'vit_detector', 'simmim_finetune__vit_base__img224__800ep.yaml')
