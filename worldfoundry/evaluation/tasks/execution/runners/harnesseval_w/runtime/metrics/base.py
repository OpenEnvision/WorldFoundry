from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, device="cuda"):
        self.device = device

    @property
    @abstractmethod
    def name(self):
        """Return the unique name of this metric."""
        pass

    @abstractmethod
    def compute(self, frames, first_frame=None, prompt=None, **kwargs):
        """
        Core computation logic.

        Args:
            frames: List of PIL images (decoded video frames)
            first_frame: PIL image (reference first frame, optional)
            prompt: str (text prompt, optional)
            **kwargs: Extra parameters (e.g., actions, perspective)
        """
        pass
