import dataclasses

import numpy as np

from ..utils import get_all_keys

TOKEN_POOLING_KEYS = ["token_pooling", "low_res_token_pooling"]


@dataclasses.dataclass
class TokenizedVisionData:
    """Data for visual input that has been converted into tokens"""

    tokens: np.ndarray
    """Token ids to use in the LLM"""

    images: np.ndarray
    """Images in [n_images, n_patches, patch_size] format"""

    image_masks: np.ndarray | None = None
    """Images in [n_images, n_patches] format"""

    token_pooling: np.ndarray | None = None
    """[n_visual_tokens, pooling_dim] array of of how to pool ViT patches for patch_id tokens"""

    low_res_token_pooling: np.ndarray | None = None
    """[n_visual_tokens, pooling_dim] array of of how to pool ViT patches for low_res tokens"""

    position_ids: np.ndarray | None = None
    """Position id for each token if not sequential"""

    other_data: dict[str, np.ndarray] | None = None
    """Any extra data that needs to be passed to the model"""


@dataclasses.dataclass
class TensorSpec:
    """The expected shape and dtype of a tensor"""

    shape: tuple[int]
    dtype: np.dtype

    def __post_init__(self):
        assert all(x >= 0 for x in self.shape if x is not None)

    def extend(self, n, dim=0):
        shape = list(self.shape)
        shape[dim] += n
        return TensorSpec(tuple(shape), self.dtype)

    def __mul__(self, other):
        assert isinstance(other, int)
        return TensorSpec(tuple(s * other for s in self.shape), dtype=self.dtype)

    @classmethod
    def build(cls, data):
        if data is None:
            return None
        return TensorSpec(data.shape, data.dtype)

    @classmethod
    def get_spec(cls, src: TokenizedVisionData) -> dict[str, "TensorSpec"]:
        spec = {} if src.other_data is None else src.other_data
        for k in ["tokens", "images", "image_masks", "token_pooling", "low_res_token_pooling"]:
            v = getattr(src, k)
            if v is not None:
                spec[k] = TensorSpec.build(getattr(src, k))
        return spec

    @staticmethod
    def max(*other: "TensorSpec"):
        other = [o for o in other if o is not None]
        if not other:
            return None
        rank = len(other[0].shape)
        dtype = other[0].dtype
        assert all(rank == len(o.shape) for o in other)
        assert all(dtype == o.dtype for o in other)
        return TensorSpec([max([o.shape[i] for o in other]) for i in range(rank)], dtype)

    def max_dictionaries(*other: dict[str, "TensorSpec"]):
        return {k: TensorSpec.max(*(o[k] for o in other if k in o)) for k in get_all_keys(other)}


@dataclasses.dataclass
class VariablePaddingSpec:
    """Indicates a key can have a variable shape after collation"""

    shape: tuple[int]
    """Shape of padding tensor if key is not in batch and padding is required"""

    dtype: np.dtype
    """dtype of padding tensor if the key is not in batch and padding is required"""
