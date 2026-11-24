from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import resolve_worldfoundry_path
from worldfoundry.synthesis.action_generation.act.runtime import ACTRuntime, ACTRuntimeConfig, select_act_checkpoint
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config
from worldfoundry.zoo.runtime.profiles import load_runtime_profile

from ..base_action_synthesis import ActionModelSynthesis


class ACTSynthesis(ActionModelSynthesis):
    MODEL_ID = "act"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            profile,
            device=device,
            command_template=command_template,
            env=env,
        )
        self.runtime_options = dict(runtime_options or {})
        self._runtime: ACTRuntime | None = None
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
    ) -> ACTSynthesis:
        """Create a lazy ACT synthesis wrapper from profile/runtime options.

        Args:
            pretrained_model_path: Optional checkpoint file, checkpoint dir, or options mapping.
            args: Reserved for compatibility with pipeline loaders.
            device: Runtime device string.
            model_id: Optional model profile id override.
            profile_path: Optional runtime profile JSON path.
            manifest_path: Optional acquisition manifest path.
            acquisition_root: Optional local acquisition cache root.
            hf_models_root: Optional shared checkpoint cache root.
            command_template: Optional in-tree command template.
            kwargs: Additional ACT runtime options.
        """
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            path = resolve_worldfoundry_path(pretrained_model_path)
            options["checkpoint_dir" if path.is_dir() else "checkpoint_path"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID)
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

    def _runtime_options_with_defaults(self, options: Mapping[str, Any]) -> dict[str, Any]:
        explicit_options = {**self.runtime_options, **dict(options)}
        runtime_defaults = load_vla_va_wam_runtime_config(
            self.MODEL_ID,
            explicit_options.get("runtime_config_path"),
        )
        return {**runtime_defaults, **explicit_options}

    def _runtime_config(self, options: Mapping[str, Any], *, require_checkpoint: bool) -> ACTRuntimeConfig | None:
        """Resolve ACT runtime settings without importing torch dependencies.

        Args:
            options: Call-time runtime overrides.
            require_checkpoint: Whether a missing checkpoint should fail immediately.
        """
        merged = self._runtime_options_with_defaults(options)
        checkpoint_path = (
            merged.get("checkpoint_path") or merged.get("ckpt_path") or merged.get("pretrained_model_path")
        )
        checkpoint_dir = merged.get("checkpoint_dir")
        checkpoint: Path | None = None
        if require_checkpoint or checkpoint_path or checkpoint_dir:
            checkpoint = select_act_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_dir=checkpoint_dir,
                checkpoints=self.profile.checkpoints,
            )
        if checkpoint is None:
            return None
        camera_names = tuple(str(item) for item in merged["camera_names"])
        return ACTRuntimeConfig(
            checkpoint_path=checkpoint,
            device=str(merged.get("device") or self.device),
            camera_names=camera_names,
            state_dim=int(merged["state_dim"]),
            chunk_size=int(merged["chunk_size"]),
            temporal_agg=bool(merged["temporal_agg"]),
            backbone=str(merged["backbone"]),
            dilation=bool(merged["dilation"]),
            position_embedding=str(merged["position_embedding"]),
            enc_layers=int(merged["enc_layers"]),
            dec_layers=int(merged["dec_layers"]),
            dim_feedforward=int(merged["dim_feedforward"]),
            hidden_dim=int(merged["hidden_dim"]),
            dropout=float(merged["dropout"]),
            nheads=int(merged["nheads"]),
            pre_norm=bool(merged["pre_norm"]),
            masks=bool(merged["masks"]),
            require_dataset_stats=bool(merged.get("require_dataset_stats", True)),
        )

    def _runtime_for(self, config: ACTRuntimeConfig) -> ACTRuntime:
        key = (
            str(config.checkpoint_path),
            config.device,
            config.camera_names,
            config.state_dim,
            config.chunk_size,
            config.temporal_agg,
        )
        if self._runtime is None or self._runtime_key != key:
            self._runtime = ACTRuntime(config)
            self._runtime_key = key
        return self._runtime

    @staticmethod
    def _select_observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        observation = kwargs.get("act_observation")
        if isinstance(observation, Mapping):
            return observation
        observation = kwargs.get("observation")
        if isinstance(observation, Mapping):
            return observation
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
        """Prepare an ACT run plan and lazily execute in-tree policy inference.

        Args:
            prompt: Optional task label.
            images: Optional camera observations.
            video: Unused video input placeholder.
            interactions: Operator interaction payloads.
            output_path: Optional action trace output path.
            fps: Unused video framerate placeholder.
            timeout_seconds: Reserved compatibility parameter.
            kwargs: ACT runtime and observation options.
        """
        del timeout_seconds

        plan_only = bool(kwargs.pop("plan_only", False)) or bool(self.runtime_options.get("plan_only"))
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="act_"))
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
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
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        merged_runtime_options = self._runtime_options_with_defaults(runtime_options)
        runtime_config = self._runtime_config(runtime_options, require_checkpoint=not plan_only)
        plan_path = run_dir / "runtime_profile_plan.json"
        runtime_payload = {
            "backend": "worldfoundry.act.in_tree_runtime.ACTRuntime.predict_action",
            "checkpoint_path": str(runtime_config.checkpoint_path) if runtime_config is not None else None,
            "device": runtime_config.device if runtime_config is not None else str(merged_runtime_options["device"]),
            "camera_names": list(runtime_config.camera_names)
            if runtime_config is not None
            else list(merged_runtime_options["camera_names"]),
            "state_dim": runtime_config.state_dim
            if runtime_config is not None
            else int(merged_runtime_options["state_dim"]),
            "chunk_size": runtime_config.chunk_size
            if runtime_config is not None
            else int(merged_runtime_options["chunk_size"]),
            "temporal_agg": runtime_config.temporal_agg
            if runtime_config is not None
            else bool(merged_runtime_options["temporal_agg"]),
        }
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-act-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "runtime": runtime_payload,
        }
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

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

        assert runtime_config is not None
        observation = self._select_observation(kwargs)
        if observation is None:
            raise ValueError(
                "ACT predict requires act_observation or observation with qpos and configured camera arrays."
            )
        result = self._runtime_for(runtime_config).predict_action(
            observation=observation,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "profile": self.profile.to_dict(),
                "interactions": list(interactions),
                "operator_context": {
                    "instruction": prompt,
                    "action_space": kwargs.get("action_space"),
                    "policy_controls": kwargs.get("policy_controls"),
                },
            },
        )
        return {
            **result,
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "profile": self.profile.to_dict(),
        }
