# Inference-only Being-H0.5 runtime retained in-tree.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

try:
    from numpydantic import NDArray
except ImportError:  # pragma: no cover - optional typing dependency for lightweight imports.
    NDArray = Any
from pydantic import BaseModel, Field, field_serializer

# ==============================================================================
# Rotation Representation
# ==============================================================================
# ==============================================================================
# LeRobot Schema Compatibility
# ==============================================================================
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.schema import DatasetModalities, RotationType

from .constants import EmbodimentTag


class LeRobotModalityField(BaseModel):
    """Base metadata for LeRobot modality field."""

    original_key: str | None = Field(default=None, description="Original key in LeRobot dataset")


class LeRobotStateActionMetadata(LeRobotModalityField):
    """Metadata for LeRobot state or action modality."""

    start: int = Field(
        ...,
        description="Start index of the modality in concatenated state/action vector",
    )
    end: int = Field(
        ...,
        description="End index of the modality in concatenated state/action vector",
    )
    rotation_type: RotationType | None = Field(default=None, description="The type of rotation for the modality")
    absolute: bool = Field(default=True, description="Whether the modality is absolute")
    dtype: str = Field(
        default="float64",
        description="The data type of the modality. Defaults to float64.",
    )
    range: tuple[float, float] | None = Field(
        default=None,
        description="The range of the modality, if applicable. Defaults to None.",
    )
    original_key: str | None = Field(
        default=None,
        description="The original key of the modality in the LeRobot dataset.",
    )


class LeRobotStateMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot state modality."""

    original_key: str | None = Field(
        default="observation.state",  # LeRobot convention for states
        description="The original key of the state modality in the LeRobot dataset",
    )


class LeRobotActionMetadata(LeRobotStateActionMetadata):
    """Metadata for a LeRobot action modality."""

    original_key: str | None = Field(
        default="action",  # LeRobot convention for actions
        description="The original key of the action modality in the LeRobot dataset",
    )


class LeRobotModalityMetadata(BaseModel):
    """Metadata for a LeRobot modality."""

    state: dict[str, LeRobotStateMetadata] = Field(
        ...,
        description="The metadata for the state modality. The keys are the names of each split of the state vector.",
    )
    action: dict[str, LeRobotActionMetadata] = Field(
        ...,
        description="The metadata for the action modality. The keys are the names of each split of the action vector.",
    )
    video: dict[str, LeRobotModalityField] = Field(
        ...,
        description="The metadata for the video modality. The keys are the new names of each video modality.",
    )
    annotation: dict[str, LeRobotModalityField] | None = Field(
        default=None,
        description="The metadata for the annotation modality. The keys are the new names of each annotation modality.",
    )

    def get_key_meta(self, key: str) -> LeRobotModalityField:
        """Get the metadata for a key in the LeRobot modality metadata.

        Args:
            key (str): The key to get the metadata for.

        Returns:
            LeRobotModalityField: The metadata for the key.

        Example:
            lerobot_modality_meta = LeRobotModalityMetadata.model_validate(U.load_json(modality_meta_path))
            lerobot_modality_meta.get_key_meta("state.joint_shoulder_y")
            lerobot_modality_meta.get_key_meta("video.main_camera")
            lerobot_modality_meta.get_key_meta("annotation.human.action.task_description")
        """
        split_key = key.split(".")
        modality = split_key[0]
        subkey = ".".join(split_key[1:])
        if modality == "state":
            if subkey not in self.state:
                raise ValueError(
                    f"Key: {key}, state key {subkey} not found in metadata, available state keys: {self.state.keys()}"
                )
            return self.state[subkey]
        elif modality == "action":
            if subkey not in self.action:
                raise ValueError(
                    f"Key: {key}, action key {subkey} not found in metadata, available action keys: {self.action.keys()}"
                )
            return self.action[subkey]
        elif modality == "video":
            if subkey not in self.video:
                raise ValueError(
                    f"Key: {key}, video key {subkey} not found in metadata, available video keys: {self.video.keys()}"
                )
            return self.video[subkey]
        elif modality == "annotation":
            assert self.annotation is not None, "Trying to get annotation metadata for a dataset with no annotations"
            if subkey not in self.annotation:
                raise ValueError(
                    f"Key: {key}, annotation key {subkey} not found in metadata, available annotation keys: {self.annotation.keys()}"
                )
            return self.annotation[subkey]
        else:
            raise ValueError(f"Key: {key}, unexpected modality: {modality}")


# ==============================================================================
# Dataset Statistics
# ==============================================================================


class DatasetStatisticalValues(BaseModel):
    """Statistical values for dataset normalization."""

    max: NDArray = Field(..., description="Maximum values")
    min: NDArray = Field(..., description="Minimum values")
    mean: NDArray = Field(..., description="Mean values")
    std: NDArray = Field(..., description="Standard deviation")
    q01: NDArray = Field(..., description="1st percentile values")
    q99: NDArray = Field(..., description="99th percentile values")

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: NDArray) -> list[float]:
        if hasattr(v, "tolist"):
            return v.tolist()  # type: ignore[no-any-return]
        return list(v)


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the state")
    action: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the action")


# ==============================================================================
# Modality Metadata
# ==============================================================================


class DatasetMetadata(BaseModel):
    """Metadata of the trainable dataset

    Changes:
        - Update to use the new RawCommitHashMetadataMetadata_V1_2
    """

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")
