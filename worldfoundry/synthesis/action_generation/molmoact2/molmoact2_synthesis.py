"""
This module defines the MolmoAct2Synthesis class, which integrates the MolmoAct2 action model
into the WorldFoundry evaluation framework. It handles model loading, configuration, and
action prediction based on various input modalities.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.molmoact2.runtime import (
    MolmoAct2Runtime,
    MolmoAct2RuntimeConfig,
    select_molmoact2_checkpoint,
)
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config, variant_defaults
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class MolmoAct2Synthesis(ActionModelSynthesis):
    """
    A synthesis class for the MolmoAct2 action generation model.

    This class extends ActionModelSynthesis to provide an interface for loading,
    configuring, and running the MolmoAct2 model for action prediction.
    It manages runtime configuration, checkpoint selection, and data preparation.
    """

    MODEL_ID = "molmoact2"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda:0",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initializes the MolmoAct2Synthesis instance.

        Args:
            profile: The runtime profile for the model, containing configuration details.
            device: The device to run the model on (e.g., "cuda:0", "cpu"). Defaults to "cuda:0".
            command_template: An optional template for commands, not directly used by MolmoAct2
                              but passed through from base class.
            env: An optional mapping of environment variables for the model process.
            runtime_options: A dictionary of additional runtime options that can override defaults.
        """
        super().__init__(profile, device=device, command_template=command_template, env=env)
        self.runtime_options = dict(runtime_options or {})
        self._runtime: MolmoAct2Runtime | None = None
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
    ) -> MolmoAct2Synthesis:
        """
        Loads a pretrained MolmoAct2 model from a specified path or configuration.

        This class method is the primary entry point for loading the model, resolving
        paths, and consolidating various configuration options.

        Args:
            pretrained_model_path: Path to the pretrained model or a dictionary of options.
                                   If a dict, it can contain 'checkpoint_dir', 'model_id', etc.
            args: Legacy argument, currently unused.
            device: The device to run the model on (e.g., "cuda:0", "cpu").
            model_id: Explicit model ID to use.
            profile_path: Path to the runtime profile JSON file.
            manifest_path: Path to the model manifest JSON file.
            acquisition_root: Root directory for model acquisition.
            hf_models_root: Root directory for Hugging Face models.
            command_template: Optional command template for the model.
            **kwargs: Additional keyword arguments that can override or extend configuration.

        Returns:
            An instance of MolmoAct2Synthesis configured with the loaded model.
        """
        del args  # This argument is not used.
        # Initialize options from pretrained_model_path if it's a mapping, otherwise an empty dict.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is provided and not a mapping, treat it as 'checkpoint_dir'.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        options.update(kwargs)  # Incorporate additional keyword arguments.

        # Resolve the final model ID, prioritizing explicit options, then manifest values, then class default.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile using the resolved model ID and other paths.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda:0"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> MolmoAct2RuntimeConfig:
        """
        Constructs a MolmoAct2RuntimeConfig object based on various configuration sources.

        This method merges options from instance defaults, runtime defaults, variant defaults,
        the model profile, and explicit call-time options, resolving conflicts and providing fallbacks.

        Args:
            options: Explicit options provided during the predict call or other runtime context.

        Returns:
            A MolmoAct2RuntimeConfig instance ready for model initialization.
        """
        # Combine instance-level runtime options with explicit options provided to this method.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load general runtime defaults for the VLA/VA/WAM categories, specific to MODEL_ID.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Initial merge: runtime_defaults < explicit_options
        initial = {**runtime_defaults, **explicit_options}
        # Determine the model variant, checking multiple keys for flexibility.
        variant = (
            initial.get("variant_id")
            or initial.get("variant")
            or initial.get("model_variant")
            or initial.get("default_variant")
        )
        # Load variant-specific defaults, potentially overriding general runtime defaults.
        variant_config = variant_defaults(runtime_defaults, variant)
        # Final merge of all configuration sources: runtime_defaults < variant_config < explicit_options
        merged = {**runtime_defaults, **variant_config, **explicit_options}

        # Extract 'molmoact2_observation' to potentially influence checkpoint selection.
        observation = merged.get("molmoact2_observation")
        observation_map = observation if isinstance(observation, Mapping) else {}

        # Select the appropriate MolmoAct2 checkpoint based on various criteria.
        checkpoint = select_molmoact2_checkpoint(
            repo_id=explicit_options.get("repo_id"),
            checkpoint_dir=(
                explicit_options.get("checkpoint_dir")
                or explicit_options.get("ckpt_path")
                or explicit_options.get("pretrained_model_path")
            ),
            # Determine embodiment, checking explicit options, observation map, variant config, and profile defaults.
            embodiment=(
                explicit_options.get("embodiment")
                or explicit_options.get("variant_id")
                or explicit_options.get("variant")
                or explicit_options.get("model_variant")
                or observation_map.get("embodiment")
                or variant_config.get("embodiment")
                or self.profile.input_schema.get("default_embodiment")
            ),
            variant_id=explicit_options.get("variant_id")
            or explicit_options.get("variant")
            or explicit_options.get("model_variant"),
            checkpoints=self.profile.checkpoints,
        )

        # Re-evaluate defaults based on the *selected* checkpoint's embodiment.
        defaults = variant_defaults(runtime_defaults, checkpoint["embodiment"]) or variant_config

        # Determine the normalization tag using a precedence rule.
        explicit_norm_tag = explicit_options.get("norm_tag") or explicit_options.get("normalization_tag")
        observation_norm_tag = observation_map.get("norm_tag")
        profile_norm_tag = self.profile.input_schema.get("norm_tag")
        if explicit_norm_tag:
            norm_tag = explicit_norm_tag
        # Use observation_norm_tag if it's present and not identical to the profile's default
        # when the selected embodiment differs from the profile's default.
        elif observation_norm_tag and not (
            checkpoint["embodiment"] != self.profile.input_schema.get("default_embodiment")
            and observation_norm_tag == profile_norm_tag
        ):
            norm_tag = observation_norm_tag
        else:
            norm_tag = defaults["norm_tag"]

        # Resolve camera keys from explicit options, observation map, defaults, or profile schema.
        camera_keys = tuple(
            str(item)
            for item in (
                explicit_options.get("camera_keys")
                or observation_map.get("camera_keys")
                or defaults["camera_keys"]
                or self.profile.input_schema.get("camera_keys")
            )
        )
        if not camera_keys:
            camera_keys = tuple(defaults["camera_keys"])

        # Resolve state dimension from explicit options, defaults, or profile schema.
        state_dim = int(
            explicit_options.get("state_dim") or defaults["state_dim"] or self.profile.input_schema.get("state_dim")
        )

        return MolmoAct2RuntimeConfig(
            repo_id=str(checkpoint["repo_id"]),
            local_dir=checkpoint["local_dir"],
            embodiment=str(checkpoint["embodiment"]),
            norm_tag=str(norm_tag),
            camera_keys=camera_keys,
            state_dim=state_dim,
            action_mode_key=str(explicit_options.get("action_mode_key") or defaults["action_mode_key"]),
            device=str(merged.get("device") or self.device),
            torch_dtype=str(
                explicit_options.get("torch_dtype") or explicit_options.get("dtype") or runtime_defaults["torch_dtype"]
            ),
            num_steps=int(explicit_options.get("num_steps") or runtime_defaults["num_steps"]),
            enable_cuda_graph=bool(explicit_options.get("enable_cuda_graph", runtime_defaults["enable_cuda_graph"])),
            enable_depth_reasoning=bool(
                merged.get("enable_depth_reasoning", runtime_defaults["enable_depth_reasoning"])
            ),
            enable_adaptive_depth=bool(merged.get("enable_adaptive_depth", False)),
            normalize_language=bool(explicit_options.get("normalize_language", runtime_defaults["normalize_language"])),
        )

    def _runtime_for(self, config: MolmoAct2RuntimeConfig) -> MolmoAct2Runtime:
        """
        Retrieves or creates a MolmoAct2Runtime instance for the given configuration.

        This method implements a caching mechanism to avoid re-initializing the runtime
        if the configuration has not changed.

        Args:
            config: The MolmoAct2RuntimeConfig object to use for the runtime.

        Returns:
            A MolmoAct2Runtime instance.
        """
        # Create a key from the config's dictionary values to check for changes.
        key = tuple(config.__dict__.values())
        # If no runtime exists or the configuration has changed, create a new one.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = MolmoAct2Runtime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_images(images: Any, kwargs: Mapping[str, Any], camera_keys: Sequence[str]) -> Any:
        """
        Selects image data from various input sources.

        This static method prioritizes images explicitly passed, then from a 'molmoact2_observation'
        dictionary, then direct kwargs using camera_keys, and finally attempts to resolve aliases.

        Args:
            images: Explicit image data, if provided.
            kwargs: A dictionary of additional keyword arguments, potentially containing image data.
            camera_keys: A sequence of expected camera keys for gathering images.

        Returns:
            A dictionary of selected images keyed by camera name, or None if no images are found.
        """
        if images is not None:
            return images

        # Check for images within a 'molmoact2_observation' dictionary in kwargs.
        observation = kwargs.get("molmoact2_observation")
        if isinstance(observation, Mapping) and observation.get("images") is not None:
            return observation["images"]

        # Gather images directly from kwargs using the provided camera keys.
        gathered = {key: kwargs[key] for key in camera_keys if key in kwargs and kwargs[key] is not None}
        if gathered:
            return gathered

        # Define aliases for common camera key names.
        aliases = {
            "top_cam": ("front_camera_rgb", "front_cam"),
            "left_cam": ("left_camera_rgb",),
            "right_cam": ("right_camera_rgb",),
            "external_cam": ("scene_cam", "third_person_cam", "image"),
            "external_cam_2": ("exterior_2_cam", "scene_cam_2", "third_person_cam_2"),
            "side_cam": ("side_camera_rgb",),
            "agentview_cam": ("agentview_rgb", "front_rgb", "agent_view"),
            "wrist_cam": ("wrist_image",),
        }
        # Attempt to resolve images using aliases if direct keys failed.
        for key in camera_keys:
            for alias in aliases.get(str(key), ()):
                if alias in kwargs and kwargs[alias] is not None:
                    gathered[str(key)] = kwargs[alias]
                    break
        return gathered or None

    @staticmethod
    def _select_state(kwargs: Mapping[str, Any]) -> Any:
        """
        Selects state data from various input sources.

        This static method prioritizes state from a 'molmoact2_observation' dictionary
        and then checks common state-related keys directly in kwargs.

        Args:
            kwargs: A dictionary of additional keyword arguments, potentially containing state data.

        Returns:
            The selected state data, or None if no state is found.
        """
        # Check for state within a 'molmoact2_observation' dictionary in kwargs.
        observation = kwargs.get("molmoact2_observation")
        if isinstance(observation, Mapping) and observation.get("state") is not None:
            return observation["state"]

        # Iterate through common keys to find state information.
        for key in ("state", "robot_state", "joint_state", "proprio", "joint_positions"):
            if key in kwargs and kwargs[key] is not None:
                return kwargs[key]
        return None

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
        """
        Generates an action prediction using the MolmoAct2 model.

        This method orchestrates the prediction process, including setting up the runtime
        configuration, preparing input data (images, state), executing the model,
        and managing output artifacts. It also supports a 'plan_only' mode to
        dump the runtime plan without executing the model.

        Args:
            prompt: The text prompt for guiding action generation.
            images: Input images for the model (e.g., a dictionary of camera views).
            video: Input video for the model (currently not directly used by MolmoAct2).
            interactions: A sequence of previous interactions or turns, for multi-turn tasks.
            output_path: The desired path to save any generated artifacts.
            fps: Frames per second, relevant if video input were used.
            timeout_seconds: Timeout for the prediction process (not directly used in MolmoAct2Runtime).
            **kwargs: Additional keyword arguments for configuration or input data.
                      Can include 'plan_only', 'run_dir', and various model-specific options.

        Returns:
            A dictionary containing the prediction results, including generated actions
            and metadata like 'run_dir' and 'plan_path'.
        """
        del timeout_seconds  # This argument is not used in the current implementation flow.
        # Determine if only a plan should be generated without executing the model.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Create a temporary directory for this prediction run, resolving its path.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="molmoact2_")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the context dictionary, which will be stored in the plan.
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
        # Generate the MolmoAct2 specific runtime configuration.
        runtime_config = self._runtime_config({**kwargs, "device": self.device})

        # Define the path for the runtime plan file.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, detailing the model and its configuration.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-molmoact2-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.molmoact2.in_tree_hf_predict_action",
                "repo_id": runtime_config.repo_id,
                "local_dir": "" if runtime_config.local_dir is None else str(runtime_config.local_dir),
                "embodiment": runtime_config.embodiment,
                "norm_tag": runtime_config.norm_tag,
                "camera_keys": list(runtime_config.camera_keys),
                "state_dim": runtime_config.state_dim,
                "device": runtime_config.device,
                "torch_dtype": runtime_config.torch_dtype,
                "num_steps": runtime_config.num_steps,
                "enable_depth_reasoning": runtime_config.enable_depth_reasoning,
                "enable_adaptive_depth": runtime_config.enable_adaptive_depth,
            },
        }
        # Write the plan payload to the designated plan file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # If only planning is requested, return the plan details without running the model.
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

        # Select the actual image data from inputs using the resolved camera keys.
        selected_images = self._select_images(images, kwargs, runtime_config.camera_keys)
        # Select the state data from inputs.
        selected_state = self._select_state(kwargs)

        # Get the MolmoAct2 runtime instance and perform the action prediction.
        result = self._runtime_for(runtime_config).predict_action(
            prompt=prompt,
            images=selected_images,
            state=selected_state,
            output_path=context["output_path"],
            # Include extra metadata for logging or debugging within the runtime.
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
            },
        )
        # Merge the runtime's prediction result with run directory and plan path information.
        return {**result, "run_dir": str(run_dir), "plan_path": str(plan_path), "profile": self.profile.to_dict()}
