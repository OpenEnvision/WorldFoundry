# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration class for the Hy-VLA policy.

The on-disk format of ``config.json`` is a flat JSON object whose keys
are the dataclass fields (plus a ``"type"`` discriminator).

Modified by WorldFoundry to retain inference fields and checkpoint loading
only; optimizer, scheduler, fine-tuning, and checkpoint-writing helpers were
removed. Unknown training keys in an official config remain forward-compatible
and are ignored by :meth:`HyVLAConfig.from_pretrained`.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from worldfoundry.core.io.paths import resolve_local_hf_model_path

CONFIG_NAME = "config.json"


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = FeatureType(self.type)
        self.shape = tuple(self.shape)


T = TypeVar("T", bound="HyVLAConfig")


@dataclass
class HyVLAConfig:
    """Configuration for the Hy-VLA policy."""

    # --- Core I/O structure ------------------------------------------------
    chunk_size: int = 50
    n_action_steps: int = 50

    # --- Image features --------------------------------------------------
    image_features: dict | None = field(default=None)

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Number of all-(-1) padding cameras to append when ``image_features``
    # expects more cameras than the batch provides. Default 0 = no padding.
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 64

    # Action-expert projector width
    proj_width: int = 1024

    # Flow-matching steps
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # one of: eager, fa2, flex

    vis_attn: bool = False

    # Path to the upstream Hy-Embodied VLM (HF repo id or local dir).
    # Used as a fallback when the VLA ckpt is not self-contained, and as
    # the source for raw-VLM bootstrap (``pretrain_source`` in
    # {``scratch``, ``vlm``}).
    vlm_model_path: str = "tencent/HY-Embodied-0.5"

    # Embedded upstream VLM AutoConfig payload. When set, the VLA ckpt is
    # self-contained and builds its inner VLM tower without contacting
    # any external Hy-Embodied repo. ``None`` means fall back to
    # ``vlm_model_path``.
    vlm_config_dict: dict | None = None

    # --- MEM video-encoder switch (paper arXiv:2603.03596v1) --------------
    use_video_encoder: bool = False
    spacetime_layer_stride: int = 4
    past_drop_layer: int | None = None

    # Visual-segment attention mask scope. See
    # ``HyVLA._apply_visual_segment_mask``:
    #   * False -- patch-only: clear cross-image visibility between
    #     image-patch tokens; split rows stay on the causal pathway.
    #   * True  -- full-segment isolation: clear the whole visual
    #     segment's outward visibility and re-enable bidirectional
    #     visibility within each segment. Required to reproduce the
    #     released RoboTwin post-training ckpt.
    visual_segment_isolation: bool = False

    # Class-level constant (no type annotation, so dataclasses treats it
    # as a class variable and does NOT serialize it).
    action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(20,))

    # Path the config was loaded from, populated by from_pretrained.
    pretrained_path: str | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        # Default image_features when the user / ckpt did not provide them.
        if self.image_features is None:
            self.image_features = {
                "observation.images.top_head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.hand_left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.hand_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            }

    @property
    def type(self) -> str:
        # Discriminator used by the on-disk ``config.json`` format.
        return "hy"

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
        token: str | bool | None = None,
        **overrides: Any,
    ) -> T:
        if not local_files_only or force_download:
            raise ValueError("HyVLAConfig loading is local-only; stage Hub assets with hfd first")
        del cache_dir, revision, token
        model_id = str(pretrained_name_or_path)
        model_dir = resolve_local_hf_model_path(model_id, required_files=(CONFIG_NAME,))
        config_file = os.path.join(model_dir, CONFIG_NAME)

        with open(config_file, encoding="utf-8") as fp:
            data = json.load(fp)

        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k == "type":
                continue
            if k not in allowed:
                # Forward compat: silently drop fields the current code
                # version doesn't know about.
                continue
            if k == "image_features" and isinstance(v, dict):
                v = {kk: PolicyFeature(**vv) if isinstance(vv, dict) else vv for kk, vv in v.items()}
            elif k == "resize_imgs_with_padding" and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        kwargs.update(overrides)

        instance = cls(**kwargs)
        instance.pretrained_path = str(model_dir)
        return instance


__all__ = [
    "FeatureType",
    "HyVLAConfig",
    "PolicyFeature",
]
