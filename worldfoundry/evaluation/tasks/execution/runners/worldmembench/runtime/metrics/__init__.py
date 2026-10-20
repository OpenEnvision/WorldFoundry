# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""WorldMemBench metric interfaces and built-in registry."""

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.base import Metric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.camera import (
    CameraRotationMetric,
    CameraTranslationMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.dynamic import DynamicMemoryMetric
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.environment import (
    EnvironmentDepthFidelityMetric,
    EnvironmentSceneReconstructionMetric,
    EnvironmentVisibleMemoryMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.general_video_quality import (
    AestheticQualityMetric,
    ImagingQualityMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.human import (
    HumanAppearanceMetric,
    HumanIdentityMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.object import (
    ObjectAppearanceMetric,
    ObjectGeometryMetric,
)
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics.registry import MetricRegistry

registry = MetricRegistry()
registry.register(DynamicMemoryMetric())
registry.register(CameraRotationMetric())
registry.register(CameraTranslationMetric())
registry.register(ImagingQualityMetric())
registry.register(AestheticQualityMetric())
registry.register(HumanIdentityMetric())
registry.register(HumanAppearanceMetric())
registry.register(ObjectAppearanceMetric())
registry.register(ObjectGeometryMetric())
registry.register(EnvironmentVisibleMemoryMetric())
registry.register(EnvironmentDepthFidelityMetric())
registry.register(EnvironmentSceneReconstructionMetric())

__all__ = ["Metric", "MetricRegistry", "registry"]
