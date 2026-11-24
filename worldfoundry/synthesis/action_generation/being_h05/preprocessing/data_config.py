# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import random
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import (
    ComposedModalityTransform,
    ModalityTransform,
)
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.schema import RotationType
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

from .constants import (
    TARGET_ACTION_ROTATION_DIM,
    TARGET_ACTION_ROTATION_TYPE,
    TARGET_STATE_ROTATION_DIM,
    TARGET_STATE_ROTATION_TYPE,
)


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column (exclusive)")
    absolute: bool = True

    rotation_type: RotationType | None = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = Field(True, description="Whether the data is continuous (floating point)")


class BaseDataConfig(ABC):
    def __init__(
        self,
        embodiment_tag,
        use_fixed_view,
        max_view_num,
        obs_indices=[0],  # noqa: B006 - Keeps the upstream default object and signature.
        action_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # noqa: B006 - Keeps the upstream default object and signature.
    ):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices
        self.action_indices = action_indices

    @abstractmethod
    def define_modalities(self) -> dict[str, ModalityDef]:
        """
        Define how to extract and name new modalities from raw Parquet columns.
        Returns: {'modality.key': ModalityDef(...), ...}
        """

    def get_sampling_indices(self) -> dict[str, list[int]]:
        """Define sampling indices"""
        sampling_map = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        """
        Define a complete, ordered data transformation pipeline.
        Returns a ComposedModalityTransform object.
        """

    def add_video_modality(self, modalities):
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
            # rand_view_num = random.randint(1, len(self.VIDEO_SOURCE_COLUMNS))
            # video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), rand_view_num)
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), max_view_num)

        for video_key in video_keys:
            modalities[video_key] = ModalityDef(source_column=self.VIDEO_SOURCE_COLUMNS[video_key], start=0, end=0)

        return modalities


class LiberoOriginDataConfig(BaseDataConfig):
    VIDEO_KEYS: ClassVar[list] = ["video.top_view"]
    VIDEO_SOURCE_COLUMNS: ClassVar[dict] = {"video.top_view": "observation.images.image"}
    STATE_KEYS: ClassVar[list] = ["state.state"]
    ACTION_KEYS: ClassVar[list] = ["action.action"]

    LANGUAGE_KEYS: ClassVar[list] = ["language.instruction"]

    state_normalization_modes: ClassVar[dict] = {"state.state": "min_max"}
    action_normalization_modes: ClassVar[dict] = {"action.action": "min_max"}

    state_action_type: ClassVar[dict] = {
        "state.state": "7-d absolute state (xyz,roll,pitch,yaw,pad) + 1-d gripper pos",
        "action.action": "6-d relative action (xyz,roll,pitch,yaw) + 1-d gripper pos",
    }

    def define_modalities(self) -> dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            "language.instruction": ModalityDef(source_column="task_index", start=0, end=0),
            "state.state": ModalityDef(source_column="observation.state", start=0, end=8),
            "action.action": ModalityDef(source_column="action", start=0, end=7, absolute=False),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(apply_to=self.STATE_KEYS, normalization_modes=self.state_normalization_modes),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(apply_to=self.ACTION_KEYS, normalization_modes=self.action_normalization_modes),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoNoNormDataConfig(LiberoOriginDataConfig):
    VIDEO_KEYS: ClassVar[list] = ["video.top_view", "video.wrist_view"]
    VIDEO_SOURCE_COLUMNS: ClassVar[dict] = {
        "video.top_view": "observation.images.image",
        "video.wrist_view": "observation.images.wrist_image",
    }
    STATE_KEYS: ClassVar[list] = ["state.eef_position", "state.eef_rotation", "state.libero_gripper_position"]
    ACTION_KEYS: ClassVar[list] = ["action.eef_position", "action.eef_rotation", "action.gripper_position"]

    UNIFIED_MAPPING: dict[str, tuple[int, int]] = {  # noqa: RUF012 - Shared class-level value, as upstream.
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 6),
        "state.libero_gripper_position": (44, 46),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 6),
        "action.gripper_position": (18, 19),
    }

    state_normalization_modes: ClassVar[dict] = {}

    action_normalization_modes: ClassVar[dict] = {}

    def get_feature_meta(self):
        return {
            "state.eef_position": ("3-d absolute eef position (xyz)", 3),
            "state.eef_rotation": (
                f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})",
                TARGET_STATE_ROTATION_DIM,
            ),
            "state.libero_gripper_position": ("2-d gripper position", 2),
            "action.eef_position": ("3-d relative eef position (xyz)", 3),
            "action.eef_rotation": (
                f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})",
                TARGET_ACTION_ROTATION_DIM,
            ),
            "action.gripper_position": ("1-d gripper position"),
        }

    def define_modalities(self) -> dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            "language.instruction": ModalityDef(source_column="task_index", start=0, end=0),
            "state.eef_position": ModalityDef(source_column="observation.state", start=0, end=3),
            "state.eef_rotation": ModalityDef(
                source_column="observation.state", start=3, end=6, rotation_type="axis_angle"
            ),
            "state.libero_gripper_position": ModalityDef(source_column="observation.state", start=6, end=8),
            "action.eef_position": ModalityDef(source_column="action", start=0, end=3, absolute=False),
            "action.eef_rotation": ModalityDef(
                source_column="action", start=3, end=6, absolute=False, rotation_type="axis_angle"
            ),
            "action.gripper_position": ModalityDef(source_column="action", start=6, end=7),
        }
        modalities = self.add_video_modality(modalities)

        return modalities


class RobocasaHumanDataConfig(BaseDataConfig):
    VIDEO_KEYS: ClassVar[list] = ["video.left_view", "video.right_view", "video.wrist_view"]
    VIDEO_SOURCE_COLUMNS: ClassVar[dict] = {
        "video.left_view": "observation.images.left_view",
        "video.right_view": "observation.images.right_view",
        "video.wrist_view": "observation.images.wrist_view",
    }
    STATE_KEYS: ClassVar[list] = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    ACTION_KEYS: ClassVar[list] = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
        "action.base_motion",
        "action.control_mode",
    ]

    UNIFIED_MAPPING: dict[str, tuple[int, int]] = {  # noqa: RUF012 - Shared class-level value, as upstream.
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 6),
        "state.gripper_qpos": (44, 46),
        "state.base_position": (70, 73),
        "state.base_rotation": (73, 76),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 6),
        "action.gripper_position": (18, 19),
        "action.base_motion": (70, 74),
        "action.control_mode": (74, 75),
    }

    LANGUAGE_KEYS: ClassVar[list] = ["language.instruction"]

    state_normalization_modes: ClassVar[dict] = {}
    # action_normalization_modes = {}

    action_normalization_modes: ClassVar[dict] = {
        # "action.end_effector_position": "min_max",
        # "action.end_effector_rotation": "min_max",
        "action.gripper_position": "binary",
        # "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def get_feature_meta(self):
        return {
            "state.eef_position": ("3-d absolute eef position (xyz)", 3),
            "state.eef_rotation": (
                f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})",
                TARGET_STATE_ROTATION_DIM,
            ),
            "state.gripper_qpos": ("2-d gripper position", 2),
            "action.eef_position": ("3-d relative eef position (xyz)", 3),
            "action.eef_rotation": (
                f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})",
                TARGET_ACTION_ROTATION_DIM,
            ),
            "action.gripper_position": ("1-d gripper position"),
        }

    def define_modalities(self) -> dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            "language.instruction": ModalityDef(source_column="task_index", start=0, end=0),
            "state.eef_position": ModalityDef(source_column="world_abs_state", start=0, end=3),
            "state.eef_rotation": ModalityDef(
                source_column="world_abs_state", start=3, end=6, rotation_type="axis_angle"
            ),
            "state.gripper_qpos": ModalityDef(source_column="world_abs_state", start=6, end=8),
            "state.base_position": ModalityDef(source_column="observation.state", start=0, end=3),
            "state.base_rotation": ModalityDef(
                source_column="observation.state", start=3, end=7, rotation_type="quaternion"
            ),
            "action.eef_position": ModalityDef(source_column="world_delta_action", start=0, end=3, absolute=False),
            "action.eef_rotation": ModalityDef(
                source_column="world_delta_action", start=3, end=6, absolute=False, rotation_type="axis_angle"
            ),
            "action.gripper_position": ModalityDef(source_column="world_delta_action", start=6, end=7),
            "action.base_motion": ModalityDef(source_column="action", start=7, end=11, absolute=False),
            "action.control_mode": ModalityDef(source_column="action", start=11, end=12),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={
                    # "state.eef_rotation": TARGET_STATE_ROTATION_TYPE,
                    "state.base_rotation": TARGET_STATE_ROTATION_TYPE
                },
                # normalization_modes=self.action_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                # target_rotations={"action.eef_rotation": TARGET_ACTION_ROTATION_TYPE},
                normalization_modes=self.action_normalization_modes,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    "libero_nonorm": LiberoNoNormDataConfig,
    "robocasa_human": RobocasaHumanDataConfig,
}
