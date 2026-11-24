from collections.abc import Callable
from typing import Union

from torch import Tensor

Tree = Union[dict[str, "Tree"], Tensor]  # noqa: UP007 - Runtime type alias kept as typing.Union.


def collate(trees: list[Tree], merge_fn: Callable[[list[Tensor]], Tensor]) -> Tree:
    """Merge nested dictionaries of tensors."""
    if isinstance(trees[0], Tensor):
        return merge_fn(trees)
    else:
        return {key: collate([tree[key] for tree in trees], merge_fn) for key in trees[0]}
