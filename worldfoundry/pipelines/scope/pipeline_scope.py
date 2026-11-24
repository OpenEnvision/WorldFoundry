"""Scope visual generation pipeline module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...synthesis.visual_generation.scope import SCOPESynthesis
from ..pipeline_utils import PipelineABC


class SCOPEPipeline(PipelineABC):
    """WorldFoundry pipeline wrapper for the in-tree SCOPE runtime."""

    MODEL_ID = "scope"
    SYNTHESIS_CLS = SCOPESynthesis

    def __init__(
        self,
        *,
        model_id: str | None = None,
        synthesis_model: SCOPESynthesis | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.model_name = "SCOPE"
        self.generation_type = "action_conditioned_i2v"
        self.synthesis_model = synthesis_model or self._component("SYNTHESIS_CLS").from_pretrained(device=device)
        self.device = device
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> SCOPEPipeline:
        """Create a SCOPE pipeline without importing the heavyweight runtime."""

        options: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            options.update(model_path)
        elif model_path is not None:
            options["model_dir"] = model_path
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        synthesis_model = cls._component("SYNTHESIS_CLS").from_pretrained(
            {
                "model_dir": (
                    options.pop("model_dir", None)
                    or options.pop("checkpoint_root", None)
                    or options.pop("checkpoint_dir", None)
                    or options.pop("model_path", None)
                ),
                **options,
            },
            device=device,
        )
        return cls(model_id=resolved_model_id, synthesis_model=synthesis_model, device=device)

    @staticmethod
    def _action_path_from(
        interactions: Any = None, action_path: Any = None, operator_kwargs: Mapping[str, Any] | None = None
    ) -> Any:
        """Action path from for SCOPEPipeline."""
        if action_path is not None:
            return action_path
        operator_kwargs = dict(operator_kwargs or {})
        for key in ("action_path", "actions_path", "scope_action_path"):
            if key in operator_kwargs:
                return operator_kwargs[key]
        if isinstance(interactions, Mapping):
            return interactions.get("action_path") or interactions.get("actions_path")
        if isinstance(interactions, (str, Path)):
            return interactions
        if isinstance(interactions, (list, tuple)) and len(interactions) == 1:
            return interactions[0]
        return None

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        execute: bool = False,
        action_path: Any = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """Run the in-tree SCOPE subprocess when explicit execution is requested."""

        resolved_action_path = self._action_path_from(
            interactions=interactions,
            action_path=action_path,
            operator_kwargs=operator_kwargs,
        )
        result = self.synthesis_model.predict(
            prompt=prompt or "",
            images=images,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            execute=execute,
            action_path=resolved_action_path,
            **kwargs,
        )
        self.history.append(result)
        if return_dict:
            return result
        return result.get("artifact_path") or result

    def stream(self, *args: Any, **kwargs: Any) -> dict[str, Any] | str:
        """Expose SCOPE through the generic streaming API."""

        return self(*args, **kwargs)


__all__ = ["SCOPEPipeline"]
