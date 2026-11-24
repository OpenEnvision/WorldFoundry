"""DreamZero observation transforms required for checkpoint inference."""

from pydantic import BaseModel

from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import (
    InvertibleModalityTransform as InvertibleModalityTransform,
)
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.base import (
    ModalityTransform as ModalityTransform,
)
from worldfoundry.base_models.llm_mllm_core.action.gr00t.preprocessing.state_action import StateActionToTensor
from worldfoundry.synthesis.action_generation.dreamzero.preprocessing.transform_state_action import (
    PerHorizonActionTransform,
    StateActionTransform,
)

from .dream_transform import DreamTransform
from .embodiment import EmbodimentTag
from .schema import DatasetMetadata
from .transform_base import ComposedModalityTransform
from .transform_concat import ConcatTransform
from .transform_video import (
    VideoColorJitter,
    VideoCrop,
    VideoFocusRect,
    VideoGrayscale,
    VideoHorizontalFlip,
    VideoNormalize,
    VideoRandomErasing,
    VideoRandomGrayscale,
    VideoRandomPosterize,
    VideoRandomRotation,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class ModalityConfig(BaseModel):
    delta_indices: list[int]
    eval_delta_indices: list[int] | None = None
    modality_keys: list[str]

    def model_post_init(self, __context) -> None:
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


__all__ = [
    "ComposedModalityTransform",
    "ConcatTransform",
    "DatasetMetadata",
    "DreamTransform",
    "EmbodimentTag",
    "ModalityConfig",
    "PerHorizonActionTransform",
    "StateActionToTensor",
    "StateActionTransform",
    "VideoColorJitter",
    "VideoCrop",
    "VideoFocusRect",
    "VideoGrayscale",
    "VideoHorizontalFlip",
    "VideoNormalize",
    "VideoRandomErasing",
    "VideoRandomGrayscale",
    "VideoRandomPosterize",
    "VideoRandomRotation",
    "VideoResize",
    "VideoToNumpy",
    "VideoToTensor",
]
