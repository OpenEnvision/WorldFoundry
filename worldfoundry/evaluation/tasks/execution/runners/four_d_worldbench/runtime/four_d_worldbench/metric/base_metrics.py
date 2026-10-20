"""4DWorldBench image metrics backed by the shared IQA implementation."""

from typing import Any

from worldfoundry.evaluation.tasks.metrics._shared.pyiqa import ImageQualityMetric, PyIQAMetric


def get_torch_device() -> Any:
    """Retain the benchmark's CUDA, MPS, then CPU device preference."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseMetric(ImageQualityMetric):
    """Use the benchmark's device preference for image comparisons."""

    def __init__(self) -> None:
        super().__init__(device=get_torch_device())


class IQAPytorchMetric(PyIQAMetric):
    """Apply 4DWorldBench's input policy to the selected PyIQA metric."""

    def __init__(self, metric_name: str) -> None:
        super().__init__(metric_name, device=get_torch_device())
