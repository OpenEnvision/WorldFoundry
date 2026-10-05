"""Batched mask data container used by SAM automatic mask generation.

Responsibility
    Keep masks, scores, and boxes aligned when filtering or concatenating
    SAM proposal batches.

Boundaries
    SAM-specific. Not a generic attention mask, not a segmentation dataset
    type, and not a pytree. Values must be ``list`` / ``ndarray`` /
    ``Tensor``; other types fail at set / filter / cat.

Public surface
    :class:`MaskData`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ItemsView

import numpy as np
import torch


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.

    Originally from SAM 2 (``sam2/utils/amg.py``).  The SAM v1 version lacked
    the ``.float()`` call in ``to_numpy``; SAM 2's version is safer because it
    avoids dtype issues when converting half-precision tensors.
    """

    def __init__(self, **kwargs) -> None:
        """Store aligned batched fields; every value must be list / ndarray / Tensor."""
        for v in kwargs.values():
            assert isinstance(v, (list, np.ndarray, torch.Tensor)), (
                "MaskData only supports list, numpy arrays, and torch tensors."
            )
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        """Replace one field; rejects types that cannot be filtered or concatenated."""
        assert isinstance(item, (list, np.ndarray, torch.Tensor)), (
            "MaskData only supports list, numpy arrays, and torch tensors."
        )
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        """Drop one field; missing keys raise :exc:`KeyError`."""
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        """Return one field; missing keys raise :exc:`KeyError`."""
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        """Iterate ``(name, batched_value)`` pairs without copying."""
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        """In-place row subset so every field stays aligned with ``keep``.

        ``keep`` is a 1-D index or bool mask. Tensors are indexed on their own
        device; ndarrays move ``keep`` to CPU. Bool ``keep`` on a list uses
        Python filtering; integer ``keep`` uses positional gather. Unsupported
        value types raise :exc:`TypeError`.
        """
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        """Append ``new_stats`` along the batch axis, deep-copying missing keys.

        Tensor / ndarray concat uses dim 0. Lists are concatenated with a
        deepcopy of the incoming values so later in-place edits do not alias.
        """
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        """Cast tensor fields to ``float32`` CPU ndarrays in place.

        ``.float()`` before ``.numpy()`` avoids a half-precision export that
        some SAM post-process steps cannot consume.
        """
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()
