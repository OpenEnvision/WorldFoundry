"""WorldScore image metrics backed by the shared IQA implementation."""

from worldfoundry.evaluation.tasks.execution.runners.worldscore.runtime.worldscore.worldscore.common.gpu_utils import (
    get_torch_device,
)
from worldfoundry.evaluation.tasks.execution.runners.worldscore.runtime.worldscore.worldscore.common.images_utils import (
    open_image,
)
from worldfoundry.evaluation.tasks.metrics._shared.pyiqa import ImageQualityMetric, PyIQAMetric


class BaseMetric(ImageQualityMetric):
    """Retain WorldScore's device and local/URL image loading policy."""

    def __init__(self) -> None:
        super().__init__(device=get_torch_device(), image_loader=open_image)


class IQAPytorchMetric(PyIQAMetric):
    """Apply WorldScore's input policy to the selected PyIQA metric."""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name, device=get_torch_device(), image_loader=open_image)
