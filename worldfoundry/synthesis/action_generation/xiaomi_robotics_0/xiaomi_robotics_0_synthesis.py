"""Runtime-profile synthesis bridge for Xiaomi-Robotics-0."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.action_generation._native_policy_runtime import option_bool
from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis

from .runtime import MODEL_ID, XiaomiRobotics0RuntimeConfig, runtime_for

_MODELS_DATA_ROOT = resolve_data_path("models")
_DEFAULT_CATALOG_PATH = _MODELS_DATA_ROOT / "catalog" / "vla_va_wam" / "xiaomi-robotics-0.yaml"
_DEFAULT_PROFILE_PATH = _MODELS_DATA_ROOT / "runtime" / "profiles" / "xiaomi-robotics-0.yaml"
_DEFAULT_ENVIRONMENT_PATH = _MODELS_DATA_ROOT / "runtime" / "environments" / "action" / "xiaomi-robotics-0.yaml"


class XiaomiRobotics0Synthesis(ActionModelSynthesis):
    """Profile-backed, lazy in-tree Xiaomi-Robotics-0 action synthesis."""

    MODEL_ID = MODEL_ID

    def __init__(
        self,
        profile: Any,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(profile, device=device, command_template=command_template, env=env)
        self.runtime_options = dict(runtime_options or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Any = None,
        args: Any = None,
        device: str | None = None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        target_profile_path: str | Path | None = None,
        conda_env_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "XiaomiRobotics0Synthesis":
        """Create a lazy wrapper without importing torch or transformers."""

        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path") or _DEFAULT_CATALOG_PATH,
            profile_path=profile_path or options.get("profile_path"),
            target_profile_path=target_profile_path or options.get("target_profile_path") or _DEFAULT_PROFILE_PATH,
            conda_env_path=conda_env_path or options.get("conda_env_path") or _DEFAULT_ENVIRONMENT_PATH,
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        return cls(
            profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=command_template or options.get("command_template"),
            env=options.get("env"),
            runtime_options=options,
        )

    def _runtime_config(self, options: Mapping[str, Any]) -> XiaomiRobotics0RuntimeConfig:
        merged = {**self.runtime_options, **dict(options)}
        return XiaomiRobotics0RuntimeConfig.from_options(merged, device=str(merged.get("device") or self.device))

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[Any] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        timeout_seconds: int = 21600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan or execute one checkpoint-backed action-chunk prediction."""

        del timeout_seconds
        if video is not None:
            raise ValueError("Xiaomi-Robotics-0 is an image/state VLA policy; video input is unsupported")
        plan_only = option_bool(kwargs.pop("plan_only", None), option_bool(self.runtime_options.get("plan_only")))
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="xiaomi_robotics_0_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        context = self._context(
            prompt=prompt,
            images=images,
            video=None,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )
        runtime_config = self._runtime_config(kwargs)
        plan_path = write_json(
            run_dir / "runtime_profile_plan.json",
            {
                "schema_version": "worldfoundry-xiaomi-robotics-0-runtime-plan",
                "profile": self.profile.to_dict(),
                "context": context,
                "runtime": {
                    "backend": "worldfoundry.xiaomi_robotics_0.in_tree_hf_runtime",
                    "checkpoint": runtime_config.checkpoint,
                    "revision": runtime_config.revision,
                    "variant": runtime_config.variant,
                    "robot_type": runtime_config.robot_type,
                    "device": runtime_config.device,
                    "torch_dtype": runtime_config.torch_dtype,
                    "attn_implementation": runtime_config.attn_implementation,
                    "num_steps": runtime_config.num_steps,
                    "seed": runtime_config.seed,
                    "expected_action_horizon": runtime_config.expected_action_horizon,
                    "camera_keys": runtime_config.camera_keys,
                    "view_labels": runtime_config.view_labels,
                    "trust_remote_code": False,
                },
            },
        )
        if plan_only:
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": self.profile.artifact_kind,
                "artifact_path": context["output_path"],
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "profile": self.profile.to_dict(),
            }

        observation_value = kwargs.pop("observation", None) or kwargs.pop("xiaomi_robotics_0_observation", None) or {}
        if not isinstance(observation_value, Mapping):
            raise TypeError("observation must be a mapping")
        observation = dict(observation_value)
        for key in (
            "state",
            "proprio_state",
            "proprio",
            "robot_state",
            "images",
            "base",
            "wrist_left",
            "base_view",
            "left_wrist",
            "wrist_image",
            "left_wrist_view",
            "ego_view",
        ):
            if key in kwargs and key not in observation:
                observation[key] = kwargs[key]

        result = runtime_for(runtime_config).predict(
            instruction=prompt,
            image=images,
            observation=observation,
            output_path=context["output_path"],
            seed=kwargs.get("seed"),
            num_steps=kwargs.get("num_steps"),
            prompt_is_formatted=option_bool(kwargs.get("prompt_is_formatted")),
        )
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }


__all__ = ["XiaomiRobotics0Synthesis"]
