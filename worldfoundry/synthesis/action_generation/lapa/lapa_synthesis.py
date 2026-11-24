"""Module for LAPA (Latent Action Prediction for Agents) action model synthesis.

This module provides a wrapper around the LAPA runtime, enabling the
synthesis of latent actions based on visual inputs and instructions.
It supports lazy loading of the LAPA model and manages runtime
configurations and asset resolution.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.synthesis.action_generation.lapa.runtime import LAPARuntime, LAPARuntimeConfig, select_lapa_assets
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class LAPASynthesis(ActionModelSynthesis):
    """A synthesis wrapper for the LAPA (Latent Action Prediction for Agents) model.

    This class extends `ActionModelSynthesis` to provide a convenient interface
    for generating latent actions using the LAPA model. It handles model
    initialization, configuration, asset loading, and interaction with the
    underlying LAPA runtime.
    """

    MODEL_ID = "lapa"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a lazy LAPA synthesis wrapper.

        Args:
            profile: Runtime profile for LAPA.
            device: Requested runtime device label.
            command_template: Unused legacy command template slot.
            env: Optional environment values kept for profile compatibility.
            runtime_options: Checkpoint and LAPA generation settings.
        """
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        self._runtime: LAPARuntime | None = None
        # Stores the key used to identify the currently loaded runtime,
        # enabling efficient caching and reuse of LAPA models.
        self._runtime_key: tuple[str, str, int, str, int, int] | None = None

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
    ) -> LAPASynthesis:
        """Create a profile-backed LAPA wrapper without loading JAX dependencies.

        This factory method loads the appropriate runtime profile and initializes
        the `LAPASynthesis` class. It allows specifying model assets and
        runtime configurations through various parameters.

        Args:
            pretrained_model_path: Optional checkpoint directory or options mapping.
            args: Reserved compatibility parameter.
            device: Requested runtime device label.
            model_id: Optional runtime profile id.
            profile_path: Optional runtime profile override path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional source checkout root.
            hf_models_root: Optional Hugging Face model cache root.
            command_template: Unused legacy command template slot.
            **kwargs: Additional runtime options.
        """
        del args
        # Initialize options from pretrained_model_path if it's a mapping, otherwise empty.
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        # If pretrained_model_path is a string (path-like), set it as 'checkpoint_dir'.
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_dir"] = str(pretrained_model_path)
        # Merge any additional keyword arguments into the options.
        options.update(kwargs)
        # Resolve the model ID from multiple possible sources, prioritizing explicit args.
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        # Load the runtime profile based on the resolved model ID and paths.
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> LAPARuntimeConfig:
        """Resolve LAPA assets and generation settings.

        This method merges instance-level runtime options with call-specific options
        and a base runtime configuration to produce a complete `LAPARuntimeConfig`.
        It also selects the appropriate LAPA model assets.

        Args:
            options: Per-call runtime options.
        """
        # Combine instance-level runtime options with call-specific options.
        explicit_options = {**self.runtime_options, **dict(options)}
        # Load default runtime configuration for the model ID.
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        # Merge defaults with explicit options, explicit options take precedence.
        merged = {**runtime_defaults, **explicit_options}
        # Determine the checkpoint directory from various possible keys.
        checkpoint_dir = (
            explicit_options.get("checkpoint_dir")
            or explicit_options.get("ckpt_path")
            or explicit_options.get("pretrained_model_path")
        )
        # Select LAPA assets (e.g., params, tokenizer, VQGAN paths) based on the checkpoint.
        assets = select_lapa_assets(
            checkpoint_dir=checkpoint_dir,
            checkpoints=self.profile.checkpoints,
        )
        return LAPARuntimeConfig(
            assets=assets,
            dtype=str(merged["dtype"]),
            image_size=int(merged["image_size"]),
            mesh_dim=str(merged["mesh_dim"]),
            seed=int(merged["seed"]),
            tokens_per_delta=int(merged["tokens_per_delta"]),
        )

    def _runtime_for(self, config: LAPARuntimeConfig) -> LAPARuntime:
        """Return a cached runtime for the selected LAPA asset/config tuple.

        If a runtime with the exact configuration is already loaded, it is reused.
        Otherwise, a new `LAPARuntime` is instantiated and cached.

        Args:
            config: Resolved LAPA runtime configuration.
        """
        # Create a unique key representing the current runtime configuration.
        # This key is used to determine if an existing runtime can be reused.
        key = (
            str(config.assets.checkpoint_dir),
            config.dtype,
            config.image_size,
            config.mesh_dim,
            config.seed,
            config.tokens_per_delta,
        )
        # If no runtime is loaded or the configuration key has changed,
        # instantiate a new LAPARuntime and update the cache.
        if self._runtime is None or self._runtime_key != key:
            self._runtime = LAPARuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_image(images: Any, kwargs: Mapping[str, Any]) -> Any:
        """Select the image payload used for LAPA latent-token inference.

        This method checks various input sources for an image, prioritizing direct
        image inputs, then keyword arguments like 'image', 'image_path', and
        finally 'observation' or 'ref_image_path'.

        Args:
            images: Primary image input from the WorldFoundry pipeline.
            kwargs: Additional model-specific inputs.
        """
        if images is not None:
            return images
        if kwargs.get("image") is not None:
            return kwargs["image"]
        if kwargs.get("image_path") is not None:
            return kwargs["image_path"]
        # Check for image within an observation dictionary, specifically "frames".
        observation = kwargs.get("lapa_observation") or kwargs.get("observation")
        if isinstance(observation, Mapping) and observation.get("frames") is not None:
            return observation["frames"]
        return kwargs.get("ref_image_path")

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
        """Prepare a LAPA run plan and optionally execute in-tree token inference.

        This method generates a plan for a LAPA inference run, including model
        configuration and context. If `plan_only` is false, it proceeds to
        execute the inference and returns the results.

        Args:
            prompt: Task instruction.
            images: Image input or image path.
            video: Optional video context recorded in the plan.
            interactions: Prior latent-action interactions.
            output_path: Optional action-token artifact path.
            fps: Optional video frame rate metadata.
            timeout_seconds: Reserved compatibility parameter.
            **kwargs: Runtime and operator context options.
        """
        del timeout_seconds

        # Determine if only a plan should be generated or if inference should also run.
        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        # Set up a temporary directory for the run artifacts if not specified.
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="lapa_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Prepare the context dictionary, which encapsulates inputs and metadata for the run.
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
        # Resolve the full LAPA runtime configuration for this prediction call.
        runtime_config = self._runtime_config(kwargs)
        # Define the path for the runtime plan JSON file.
        plan_path = run_dir / "runtime_profile_plan.json"
        # Construct the payload for the runtime plan, detailing the model, context, and runtime parameters.
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-lapa-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": {
                "backend": "worldfoundry.lapa.in_tree_runtime.predict_tokens",
                "checkpoint_dir": str(runtime_config.assets.checkpoint_dir),
                "params_path": str(runtime_config.assets.params_path),
                "tokenizer_path": str(runtime_config.assets.tokenizer_path),
                "vqgan_path": str(runtime_config.assets.vqgan_path),
                "dtype": runtime_config.dtype,
                "image_size": runtime_config.image_size,
                "mesh_dim": runtime_config.mesh_dim,
                "seed": runtime_config.seed,
                "tokens_per_delta": runtime_config.tokens_per_delta,
            },
        }
        # Write the generated plan to a JSON file.
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if plan_only:
            # If only a plan is requested, return the plan details without executing inference.
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": str(context["output_path"]),
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        # Select the primary image input from various possible sources.
        image = self._select_image(images, kwargs)
        # Fallback to image_path from context if no direct image was selected.
        if image is None and context.get("image_path"):
            image = context["image_path"]
        # Raise an error if no valid image input could be found.
        if image is None:
            raise ValueError("LAPA predict requires an image input for latent action inference.")
        # Get or create the LAPA runtime instance based on the resolved configuration.
        # Then, execute the token prediction with the provided instruction and image.
        result = self._runtime_for(runtime_config).predict_tokens(
            instruction=prompt,
            image=image,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                    "observation_keys": sorted((kwargs.get("lapa_observation") or {}).keys())
                    if isinstance(kwargs.get("lapa_observation"), Mapping)
                    else [],
                },
            },
        )
        # Merge the prediction result with additional run metadata.
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
