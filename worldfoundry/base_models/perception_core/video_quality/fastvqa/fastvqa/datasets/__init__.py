## Version 0.0 Dataset API, includes FAST-VQA and its variants
from .basic_datasets import (
    CroppedImageDataset,
    CroppedVideoDataset,
    FragmentImageDataset,
    FragmentSampleFrames,
    FragmentVideoDataset,
    ResizedImageDataset,
    ResizedVideoDataset,
    SampleFrames,
)
from .basic_datasets import (
    FastVQAPlusPlusDataset as FastVQAPlusPlusDataset,
)

## Version 1.0 Dataset API, includes DiViDe VQA and its variants
from .fusion_datasets import FusionDataset, FusionDatasetK400, LSVQPatchDataset, SimpleDataset, get_spatial_fragments

__all__ = [
    "CroppedImageDataset",
    "CroppedVideoDataset",
    "FragmentImageDataset",
    "FragmentSampleFrames",
    "FragmentVideoDataset",
    "FusionDataset",
    "FusionDatasetK400",
    "LSVQPatchDataset",
    "ResizedImageDataset",
    "ResizedVideoDataset",
    "SampleFrames",
    "SimpleDataset",
    "get_spatial_fragments",
]
