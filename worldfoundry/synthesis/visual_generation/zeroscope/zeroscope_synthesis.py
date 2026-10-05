"""ZeroScope synthesis adapter for worldfoundry, enabling video generation from prompts.

This module provides an integration layer to use ZeroScope models within the
worldfoundry evaluation framework, specifically for synthesis tasks like
text-to-video generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from worldfoundry.evaluation.models.runtime.profiles import RuntimeProfileSynthesis
from worldfoundry.runtime.assets import expand_worldfoundry_path

from ..runtime_facade import DelegatedPredictMixin
from .worldfoundry_runtime import ZeroScopeRuntime


class ZeroScopeSynthesis(DelegatedPredictMixin, RuntimeProfileSynthesis):
    """ZeroScope synthesis adapter, integrating with the worldfoundry runtime.

    This class extends `RuntimeProfileSynthesis` to provide a standardized
    interface for loading and interacting with ZeroScope models for video
    synthesis tasks. It handles model loading, configuration, and delegates
    prediction calls to the underlying ZeroScope runtime.
    """

    MODEL_ID = "zeroscope"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "ZeroScopeSynthesis":
        """Loads a pre-trained ZeroScope model and initializes the synthesis adapter.

        This factory method is responsible for parsing model configuration,
        initializing the base `RuntimeProfileSynthesis`, and setting up the
        ZeroScope-specific runtime for predictions.

        Args:
            pretrained_model_path: Path to the pre-trained model, or a dictionary
                of options. Can be a string, Path object, or a Mapping.
            args: Additional arguments to pass to the model runtime.
            device: The device (e.g., 'cpu', 'cuda') on which to load the model.
            model_id: An optional identifier for the model. Defaults to `cls.MODEL_ID`.
            **kwargs: Arbitrary keyword arguments to pass as model options.

        Returns:
            An instance of `ZeroScopeSynthesis` configured with the loaded model.
        """
        # Initialize options dictionary, allowing `pretrained_model_path` to be a mapping
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If `pretrained_model_path` is a string or Path, set it as 'model_path' in options
        if isinstance(pretrained_model_path, (str, Path)):
            options["model_path"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options dictionary
        options.update(kwargs)

        # Call the parent class's `from_pretrained` method to handle common initialization
        instance = super().from_pretrained(
            options,
            args=args,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

        # Determine the actual model path, prioritizing 'model_path' then 'pretrained_model_path' from options
        model_path = options.get("model_path") or options.get("pretrained_model_path")
        # If no explicit model_path is found, try to get it from the first checkpoint in the profile
        if not model_path and instance.profile.checkpoints:
            model_path = dict(instance.profile.checkpoints[0]).get("local_dir")

        # Resolve the model path using `expand_worldfoundry_path` for consistent path handling
        instance.model_path = str(expand_worldfoundry_path(str(model_path))) if model_path else ""
        # Initialize the ZeroScope runtime with the current synthesis instance
        instance.runtime = ZeroScopeRuntime.from_synthesis(instance)
        return instance
