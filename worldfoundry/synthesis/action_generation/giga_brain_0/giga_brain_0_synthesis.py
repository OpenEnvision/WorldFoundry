"""Module for GigaBrain-0 action model synthesis.

This module provides the GigaBrain0Synthesis class, which wraps the GigaBrain-0
runtime for generating actions based on prompts, images, and other contextual information.
It handles configuration, runtime setup, and execution of the GigaBrain-0 model.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.giga_brain_0.runtime import (
    GigaBrain0Runtime,
    GigaBrain0RuntimeConfig,
    select_giga_brain_0_paths,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class GigaBrain0Synthesis(ActionModelSynthesis):
    """GigaBrain0Synthesis class for generating actions using the GigaBrain-0 model.

    This class extends ActionModelSynthesis to provide an interface for
    configuring and running the GigaBrain-0 model. It manages model loading,
    runtime configuration, and the prediction process, including planning and
    in-tree inference execution.
    """

    MODEL_ID = "giga-brain-0"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the GigaBrain0Synthesis instance.

        Args:
            profile: The runtime profile containing model metadata and schema.
            device: The device to run the model on (e.g., "cuda", "cpu").
            command_template: An optional sequence of strings to form the command
                              for model execution in a subprocess.
            env: An optional mapping of environment variables for the model's subprocess.
            runtime_options: Additional runtime options to be merged with profile settings.
        """
        super().__init__(profile, device=device, command_template=command_template, env=env)
        self.runtime_options = dict(runtime_options or {})
        self._runtime: GigaBrain0Runtime | None = None
        self._runtime_key: tuple[Any, ...] | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> GigaBrain0Synthesis:
        """Create a lazy GigaBrain-0 synthesis wrapper.

        Args:
            pretrained_model_path: Optional checkpoint path or option mapping. If a mapping,
                                   it's merged into options. If a path, it sets "model_path".
            args: Unused compatibility argument (will be deleted).
            device: Runtime device string (e.g., "cuda", "cpu").
            model_id: Runtime profile ID. Overrides `cls.MODEL_ID` if provided.
            profile_path: Optional path to an override runtime profile JSON file.
            manifest_path: Optional path to an acquisition manifest JSON file.
            acquisition_root: Optional root directory for acquired models.
            hf_models_root: Optional root directory for Hugging Face models cache.
            command_template: Optional command template override for subprocess execution.
            **kwargs: Additional keyword arguments to be passed as runtime options.
                      These are merged into the final options dictionary.
        """
        del args
        # Parse `pretrained_model_path` which can be a path string or a dictionary of options.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["model_path"] = str(pretrained_model_path)
        options.update(kwargs)  # Merge additional kwargs into options.
        # Resolve the model ID from options, explicit argument, or class default.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile based on the resolved model ID and path overrides.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        # Instantiate GigaBrain0Synthesis with the loaded profile and resolved options.
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any], *, require_existing: bool = True) -> GigaBrain0RuntimeConfig:
        """Resolves runtime paths and required GigaBrain inference settings.

        This method merges instance-level `runtime_options` with call-specific `options`,
        loads default runtime configurations, and then resolves all necessary paths
        (model, normalization statistics, tokenizers) and parameters for
        GigaBrain-0's runtime configuration.

        Args:
            options: Per-call runtime options that override instance and default settings.
            require_existing: If True, raises an error if model and norm-stat paths
                              or critical parameters are not found or do not exist locally.

        Returns:
            A GigaBrain0RuntimeConfig object populated with resolved settings.

        Raises:
            ValueError: If `require_existing` is True and a critical path or parameter
                        (like `delta_mask`, `original_action_dim`, `embodiment_id`)
                        is missing or invalid.
        """
        # Merge instance runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default VLA/VA/WAM runtime configurations for GigaBrain-0.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options; explicit options take precedence.
        merged = {**runtime_defaults, **explicit_options}

        # Select and validate model-related paths, potentially using checkpoints from the profile.
        paths = select_giga_brain_0_paths(
            model_path=merged.get("model_path") or merged.get("checkpoint_dir") or merged.get("ckpt_path"),
            norm_stats_path=(
                merged.get("norm_stats_path")
                or merged.get("stats_path")
                or self.profile.input_schema.get("norm_stats_path")
            ),
            tokenizer_model_path=merged.get("tokenizer_model_path"),
            fast_tokenizer_path=merged.get("fast_tokenizer_path"),
            variant_id=merged.get("variant_id") or merged.get("variant") or merged.get("model_variant"),
            checkpoints=self.profile.checkpoints,
            require_existing=require_existing,
        )

        # Process the `delta_mask` parameter.
        delta_mask = merged.get("delta_mask") or self.profile.input_schema.get("delta_mask")
        if isinstance(delta_mask, str):
            # Convert comma-separated string to a list of booleans.
            delta_mask = [
                item.strip().lower() in {"1", "true", "yes"} for item in delta_mask.split(",") if item.strip()
            ]
        if not delta_mask:
            if require_existing:
                raise ValueError("GigaBrain-0 requires delta_mask.")
            # Fallback to a default delta_mask if not required to exist, based on action dimension.
            delta_mask = [True] * int(
                merged.get("original_action_dim") or merged.get("action_dim") or runtime_defaults["fallback_action_dim"]
            )

        # Process the `original_action_dim` parameter.
        original_action_dim = merged.get("original_action_dim") or merged.get("action_dim")
        if original_action_dim in (None, ""):
            if require_existing:
                raise ValueError("GigaBrain-0 requires original_action_dim/action_dim.")
            # Fallback to a default if not required to exist, potentially derived from delta_mask length.
            original_action_dim = len(delta_mask) or runtime_defaults["fallback_action_dim"]

        # Process the `embodiment_id` parameter.
        embodiment_id = merged.get("embodiment_id")
        if embodiment_id in (None, ""):
            if require_existing:
                raise ValueError("GigaBrain-0 requires embodiment_id.")
            # Fallback to a default if not required to exist.
            embodiment_id = runtime_defaults["fallback_embodiment_id"]

        # Construct and return the GigaBrain0RuntimeConfig object.
        return GigaBrain0RuntimeConfig(
            model_path=paths["model_path"],
            norm_stats_path=paths["norm_stats_path"],
            tokenizer_model_path=str(paths["tokenizer_model_path"]),
            fast_tokenizer_path=str(paths["fast_tokenizer_path"]),
            embodiment_id=int(embodiment_id),
            delta_mask=tuple(bool(item) for item in delta_mask),
            original_action_dim=int(original_action_dim),
            action_chunk=int(merged["action_chunk"]),
            device=str(merged.get("device") or self.device),
            compile_policy=bool(merged["compile_policy"]),
            torch_dtype=merged.get("torch_dtype") or merged.get("dtype"),
            autoregressive_mode_only=bool(merged["autoregressive_mode_only"]),
            enable_2d_traj_output=bool(merged["enable_2d_traj_output"]),
            depth_img_prefix_name=merged.get("depth_img_prefix_name"),
        )

    def _runtime_for(self, config: GigaBrain0RuntimeConfig) -> GigaBrain0Runtime:
        """Retrieves or creates a GigaBrain0Runtime instance for the given configuration.

        This method implements a caching mechanism: if a runtime for the exact
        same configuration already exists, it is returned. Otherwise, a new
        runtime is created and cached.

        Args:
            config: The GigaBrain0RuntimeConfig object specifying the runtime settings.

        Returns:
            An instance of GigaBrain0Runtime configured as specified.
        """
        # Create a unique key from the configuration values for caching.
        key = tuple(config.__dict__.values())
        # If no runtime is cached or the cached runtime does not match the current config, create a new one.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = GigaBrain0Runtime(config)
            self._runtime_key = key
        return self._runtime

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepares a GigaBrain-0 plan and optionally runs in-tree inference.

        This method orchestrates the prediction process: it sets up a run directory,
        generates a runtime plan, and if `plan_only` is false, it executes the
        GigaBrain-0 model using the specified inputs.

        Args:
            prompt: The text prompt for the action generation.
            images: Image data (e.g., NumPy array, PyTorch tensor) as input to the model.
            video: Video data (e.g., NumPy array, PyTorch tensor) as input to the model.
            interactions: A sequence of interaction strings, typically for dialogue or action history.
            output_path: Optional path to save artifacts, overrides the profile's output path.
            fps: Frames per second, relevant if video input is provided.
            timeout_seconds: Maximum time allowed for the prediction.
            **kwargs: Additional keyword arguments, including model-specific options or
                      overrides (e.g., `plan_only`, `run_dir`, `state`, `giga_brain_0_observation`).

        Returns:
            A dictionary containing the prediction results, status, run directory,
            plan path, and profile information.
            If `plan_only` is True, returns a dictionary with preparation status and plan details.
        """
        del timeout_seconds  # This parameter is currently unused in this implementation.
        # Determine if only a plan should be generated without running inference.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Create a temporary run directory for outputs if not explicitly provided.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="giga_brain_0_")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the context dictionary, including input data and output path.
        context = self._context(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )

        # Resolve the GigaBrain-0 runtime configuration. If plan_only, existing paths are not strictly required.
        runtime_config = self._runtime_config(
            {**kwargs, "device": self.device},
            require_existing=not plan_only,
        )

        # Define the path for the runtime plan file.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, containing profile and runtime settings.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-giga-brain-0-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.giga_brain_0.in_tree_runtime",
                "model_path": str(runtime_config.model_path),
                "norm_stats_path": str(runtime_config.norm_stats_path),
                "tokenizer_model_path": runtime_config.tokenizer_model_path,
                "fast_tokenizer_path": runtime_config.fast_tokenizer_path,
                "device": runtime_config.device,
                "compile_policy": runtime_config.compile_policy,
                "torch_dtype": runtime_config.torch_dtype,
            },
        }
        # Write the plan payload to the plan file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If only a plan is requested, return the planning details and exit.
        if plan_only:
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # Extract robot state from kwargs or a provided GigaBrain-0 observation.
        state = kwargs.get("state")
        observation = kwargs.get("giga_brain_0_observation")
        if state is None and isinstance(observation, Mapping):
            state = observation.get("robot_state")

        # Get or create the GigaBrain0Runtime instance and perform the action prediction.
        result = self._runtime_for(runtime_config).predict_action(
            prompt=prompt,
            images=images,
            state=state,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
            },
        )
        # Return the prediction result augmented with run directory, plan path, and profile.
        return {**result, "run_dir": str(run_dir), "plan_path": str(plan_path), "profile": self.profile.to_dict()}
