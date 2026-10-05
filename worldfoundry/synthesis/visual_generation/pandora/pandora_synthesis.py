"""
Module for the PandoraSynthesis class, an in-tree wrapper for the Pandora runtime plan.

This module provides an interface to prepare and generate runtime plans for the Pandora
world generation model, facilitating its use within the evaluation framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...base_synthesis import BaseSynthesis


class PandoraSynthesis(BaseSynthesis):
    """In-tree Pandora runtime plan wrapper with vendored model code."""

    MODEL_ID = "pandora"
    DISPLAY_NAME = "Pandora"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """
        Initializes the PandoraSynthesis wrapper.

        Args:
            model_id: Identifier for the model, defaults to "pandora".
            device: The torch device to use for the model, e.g., "cuda" or "cpu".
            checkpoint_path: Optional path to the model checkpoint.
        """
        super().__init__()
        self.model_id = model_id
        self.model_name = self.DISPLAY_NAME
        self.generation_type = "world_generation"
        self.device = device
        # Resolve the checkpoint path to an absolute string path if provided.
        self.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser())

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "PandoraSynthesis":
        """
        Build a lazy Pandora runtime plan wrapper.

        Args:
            pretrained_model_path: Optional checkpoint path or mapping with runtime options.
            args: Unused compatibility argument for pipeline factories.
            device: Target torch device.
            model_id: Runtime profile id, defaults to ``pandora``.
            **kwargs: Optional ``checkpoint_path`` value.
        """
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as the checkpoint_path.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options.
        options.update(kwargs)
        return cls(
            # Determine model_id with fallback hierarchy: options.model_id, options.profile_id,
            # explicit model_id, or class's default MODEL_ID.
            model_id=str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID),
            # Determine device with fallback hierarchy: explicit device, options.device, or "cuda".
            device=str(device or options.get("device") or "cuda"),
            # Determine checkpoint_path with fallback hierarchy: options.checkpoint_path or options.ckpt_path.
            checkpoint_path=options.get("checkpoint_path") or options.get("ckpt_path"),
        )

    @staticmethod
    def _runtime_root() -> Path:
        """
        Calculates the absolute path to the Pandora runtime assets.

        The vendored runtime is stored beside this synthesis module.

        Returns:
            The absolute Path object pointing to the Pandora runtime directory.
        """
        # Resolve the vendored runtime relative to this module, independent of cwd.
        return Path(__file__).resolve().parent / "pandora_runtime"

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare a Pandora in-tree runtime plan.

        This method generates a JSON file representing a runtime plan for the Pandora
        model, detailing its configuration and inputs.

        Args:
            prompt: Text control action prompt.
            images: Optional initial image metadata.
            video: Optional video context metadata.
            interactions: Optional action history metadata.
            output_path: Target JSON plan path. If None, defaults to `pandora_plan.json`
                         in the current working directory.
            fps: Optional target FPS metadata.
            **kwargs: Additional runtime options to be included in the 'extra' field.

        Returns:
            A dictionary summarizing the prepared plan, including its status, model ID,
            artifact kind, and the path to the generated plan file.
        """
        # Determine the target output path for the JSON plan.
        # If output_path is not provided, default to "pandora_plan.json" in the current working directory.
        target = Path(output_path) if output_path is not None else Path.cwd() / "pandora_plan.json"
        # Resolve the absolute path and expand any user home directory references.
        target = target.expanduser().resolve()
        # Ensure the parent directories for the target output file exist, creating them if necessary.
        target.parent.mkdir(parents=True, exist_ok=True)
        # Construct the payload dictionary for the runtime plan.
        payload = {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "backend": "worldfoundry.pandora.in_tree_runtime",
            "backend_quality": "in_tree_runtime_plan",
            "runtime_root": str(self._runtime_root()),
            "checkpoint_path": self.checkpoint_path,
            "prompt": prompt,
            "has_images": images is not None,
            "has_video": video is not None,
            "interactions": list(interactions),
            "fps": fps,
            # Convert all extra kwargs values to strings for consistent JSON serialization.
            "extra": {key: str(value) for key, value in kwargs.items()},
            "note": (
                "Pandora source is vendored in-tree. Full generation remains gated on the official "
                "checkpoint availability and runtime assets."
            ),
        }
        # Write the JSON payload to the target file with proper formatting.
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Return a summary dictionary describing the outcome of the plan preparation.
        return {
            "status": "prepared",
            "model_id": self.model_id,
            "artifact_kind": "generated_world",
            "artifact_path": str(target),
            "runtime": payload["backend"],
            "backend_quality": payload["backend_quality"],
        }
