"""Worldlabs Marble 1P1 visual generation pipeline module."""

from __future__ import annotations

from typing import Any

from ..worldlabs.pipeline_worldlabs import WorldLabsPipeline


class WorldLabsMarble11Pipeline(WorldLabsPipeline):
    """Model-specific World Labs Marble 1.1 API pipeline."""

    MODEL_ID = "worldlabs-marble-1.1"
    DEFAULT_MODEL = "marble-1.1"

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> WorldLabsMarble11Pipeline:
        """Load the pipeline from pretrained checkpoints and configurations."""
        del model_id
        return super().from_pretrained(
            model_path=model_path,
            required_components=required_components,
            device=device,
            **kwargs,
        )

    def __call__(self, *args: Any, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Execute the complete pipeline generation flow."""
        return super().__call__(*args, model=model or self.DEFAULT_MODEL, **kwargs)
