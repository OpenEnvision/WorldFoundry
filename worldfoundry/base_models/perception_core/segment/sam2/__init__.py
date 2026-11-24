# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES
from worldfoundry.core.io.paths import resolve_data_path

from .video_tracker import SAM2MaskTracker, stage_video_frames

if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(
        config_dir=str(resolve_data_path("models", "runtime", "configs", "sam2")),
        version_base="1.2",
    )


def checkpoint_path() -> Path:
    asset = BASE_MODEL_CAPABILITIES["sam2"].assets[0]
    status = asset.check()
    return Path(status["matched_path"] or status["local_path"])


def config_name() -> str:
    return "configs/sam2.1/sam2.1_hiera_b+.yaml"


__all__ = ["SAM2MaskTracker", "checkpoint_path", "config_name", "stage_video_frames"]
