"""State store for a model's runtime artifacts."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._model_scoped import ModelScopedMemory


class RuntimeMemory(ModelScopedMemory):
    """Small state store for model-specific runtime artifacts."""

    def record(self, data: Any, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Store ``data``, tagged with this memory's model.

        Args:
            data: Content to store.
            metadata: Extra metadata; its ``type`` names the record kind and
                defaults to ``"runtime_result"``.
            kwargs: Ignored.
        """
        del kwargs
        entry_metadata = dict(metadata or {})
        if self.model_id is not None:
            entry_metadata = {"model_id": self.model_id, **entry_metadata}
        self.append_record(
            data,
            kind=str(entry_metadata.get("type", "runtime_result")),
            metadata=entry_metadata,
        )

    def select(self, context_query: Any = None, prefer_type: str | None = None, **kwargs: Any):
        """Return the newest record's content, or None when the memory is empty.

        Args:
            context_query: Ignored.
            prefer_type: Restricts the search to records of this kind.
            kwargs: Ignored.
        """
        del context_query, kwargs
        record = self.latest_record(prefer_type=prefer_type)
        return record["content"] if record is not None else None


__all__ = ["RuntimeMemory"]
