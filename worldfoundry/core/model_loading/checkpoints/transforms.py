# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Lazy checkpoint tensor transforms for split, fused and quantized weights.

The key plan is built without reading payloads. A lookup materializes only the
source tensors needed for that destination, so disk offload can use the same
converter as a resident load without retaining a second checkpoint in RAM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, dtype


def checkpoint_tensor_metadata(
    state_dict: Mapping[str, Tensor],
    key: str,
) -> tuple[dtype, tuple[int, ...]]:
    """Use lazy dtype/shape metadata when available, otherwise inspect the tensor."""
    metadata = getattr(state_dict, "tensor_metadata", None)
    if metadata is not None:
        return metadata(key)
    tensor = state_dict[key]
    return tensor.dtype, tuple(tensor.shape)


@dataclass(frozen=True)
class TensorTransform:
    """A destination tensor computed from explicitly named source tensors."""

    sources: tuple[str, ...]
    function: Callable[..., Tensor]


@dataclass(frozen=True)
class TensorGroupTransform:
    """One output of a shared transform, such as packed weights and their scales."""

    sources: tuple[str, ...]
    function: Callable[..., tuple[Tensor, ...]]
    index: int


class TransformedStateDict(Mapping[str, "Tensor"]):
    """Read-through mapping retaining at most one partially consumed output group.

    Ordinary transforms are uncached. Grouped transforms release their cached
    outputs after every member is read, or when another group is requested.
    """

    def __init__(
        self,
        source: Mapping[str, Tensor],
        keys: Mapping[str, str | TensorTransform | TensorGroupTransform],
    ) -> None:
        self.source = source
        self.keys_map = dict(keys)
        self._group_key: tuple[object, ...] | None = None
        self._group_values: tuple[Tensor, ...] = ()
        self._group_pending: set[int] = set()

    def __getitem__(self, key: str) -> Tensor:
        transform = self.keys_map[key]
        if isinstance(transform, str):
            return self.source[transform]
        if isinstance(transform, TensorGroupTransform):
            group_key = (transform.sources, transform.function)
            if group_key != self._group_key:
                self._group_values = transform.function(*(self.source[name] for name in transform.sources))
                self._group_pending = set(range(len(self._group_values)))
                self._group_key = group_key
            value = self._group_values[transform.index]
            self._group_pending.discard(transform.index)
            if not self._group_pending:
                self._group_key, self._group_values = None, ()
            return value
        return transform.function(*(self.source[name] for name in transform.sources))

    def tensor_metadata(self, key: str) -> tuple[dtype, tuple[int, ...]]:
        """Keep renamed source metadata lazy; computed transforms need evaluation."""
        transform = self.keys_map[key]
        if isinstance(transform, str):
            return checkpoint_tensor_metadata(self.source, transform)
        tensor = self[key]
        return tensor.dtype, tuple(tensor.shape)

    def __contains__(self, key: object) -> bool:
        return key in self.keys_map

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys_map)

    def __len__(self) -> int:
        return len(self.keys_map)
