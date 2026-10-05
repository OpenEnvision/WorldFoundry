"""Base for memories scoped to a single model's artifacts."""

from __future__ import annotations

from typing import Any, Optional

from worldfoundry.core.memory import BaseMemory


class ModelScopedMemory(BaseMemory):
    """Memory that tags every record with the model that produced it.

    Subclasses decide what a record looks like and how one is selected; the
    model tag and the reset/evict controls are the same for all of them.

    Attributes:
        MODEL_ID: Class-level default model identifier, used when the caller
            does not pass one.
    """

    MODEL_ID: str | None = None

    def __init__(self, capacity: Optional[int] = None, model_id: str | None = None, **kwargs: Any):
        """Initialize the memory.

        Args:
            capacity: Maximum number of retained entries; unbounded when None.
            model_id: Identifier of the producing model; falls back to
                :attr:`MODEL_ID`.
            kwargs: Further :class:`BaseMemory` options.
        """
        super().__init__(capacity=capacity, **kwargs)
        self.model_id = model_id or self.MODEL_ID

    def manage(self, action: str = "reset", **kwargs: Any):
        """Reset the memory, or evict down to capacity.

        Args:
            action: ``"reset"`` clears every record; ``"evict"`` drops the
                oldest records, and is a no-op when no capacity is set.
            kwargs: Ignored.
        """
        del kwargs
        if action == "reset":
            self.reset_records()
            return
        if action == "evict" and self.capacity is not None:
            self._store.evict()


__all__ = ["ModelScopedMemory"]
