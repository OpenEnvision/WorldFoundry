# Shared GR00T-derived inference transforms.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .schema import NormalizationMetadata


class ModalityTransform(BaseModel, ABC):
    """
    Abstract class for transforming data modalities, e.g. video frame augmentation or action normalization.
    """

    apply_to: list[str] = Field(..., description="The keys to apply the transform to.")
    training: bool = Field(default=False, description="Inference mode flag.")
    _dataset_metadata: NormalizationMetadata | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def dataset_metadata(self) -> NormalizationMetadata:
        assert self._dataset_metadata is not None, (
            "Dataset metadata is not set. Please call set_metadata() before calling apply()."
        )
        return self._dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: NormalizationMetadata):
        self._dataset_metadata = value

    def set_metadata(self, dataset_metadata: NormalizationMetadata):
        """
        Set the dataset metadata. This is useful for transforms that need to know the dataset metadata, e.g. to normalize actions.
        Subclasses can override this method if they need to do something more complex.
        """
        self.dataset_metadata = dataset_metadata

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transformation to the data corresponding to target_keys and return the processed data.

        Args:
            data (dict[str, Any]): The data to transform.
                example: data = {
                    "video.image_side_0": np.ndarray,
                    "action.eef_position": np.ndarray,
                    ...
                }

        Returns:
            dict[str, Any]: The transformed data.
                example: transformed_data = {
                    "video.image_side_0": np.ndarray,
                    "action.eef_position": torch.Tensor,  # Normalized and converted to tensor
                    ...
                }
        """
        return self.apply(data)

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply the transformation to the data corresponding to keys matching the `apply_to` regular expression and return the processed data."""

    def eval(self) -> "ModalityTransform":
        self.training = False
        return self


class InvertibleModalityTransform(ModalityTransform):
    @abstractmethod
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Reverse the transformation to the data corresponding to keys matching the `apply_to` regular expression and return the processed data."""


class ComposedModalityTransform(ModalityTransform):
    """Compose multiple modality transforms."""

    transforms: list[ModalityTransform] = Field(..., description="The transforms to compose.")
    apply_to: list[str] = Field(default_factory=list, description="Will be ignored for composed transforms.")
    training: bool = Field(default=False, description="Inference mode flag.")

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def set_metadata(self, dataset_metadata: NormalizationMetadata):
        for transform in self.transforms:
            transform.set_metadata(dataset_metadata)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                raise ValueError(f"Error applying transform {i} to data: {e}") from e
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(reversed(self.transforms)):
            if isinstance(transform, InvertibleModalityTransform):
                try:
                    data = transform.unapply(data)
                except Exception as e:
                    step = len(self.transforms) - i - 1
                    raise ValueError(f"Error unapplying transform {step} to data: {e}") from e
        return data

    def eval(self) -> "ComposedModalityTransform":
        for transform in self.transforms:
            transform.eval()
        return self
