# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
"""DreamZero metadata propagation for the shared inference transform chain."""

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import (
    ComposedModalityTransform as GR00TComposedModalityTransform,
)

from .schema import DatasetMetadata


class ComposedModalityTransform(GR00TComposedModalityTransform):
    def set_metadata(self, dataset_metadata: DatasetMetadata):
        for transform in self.transforms:
            transform.set_metadata(dataset_metadata)
            # this is used to pass the list of transforms to concat transform
            # concat transform needs needs to know what transforms were applied
            # because it needs to compute the correct dimension of features
            # post transform (during unapply).
            # this attribute can also be used by other transforms to know what
            # transforms were applied before it in the pipeline.
            if hasattr(transform, "set_transform_pipeline"):
                transform.set_transform_pipeline(self.transforms)

    def set_per_horizon_statistics(self, per_horizon_stats: dict[str, dict[str, list]]):
        """Set per-horizon statistics for transforms that support it (e.g., PerHorizonActionTransform).

        Args:
            per_horizon_stats: Dict from dataset.lerobot_relative_horizon_stats_meta
                Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
        """
        for transform in self.transforms:
            if hasattr(transform, "set_per_horizon_statistics"):
                transform.set_per_horizon_statistics(per_horizon_stats)
