"""WorldFoundry synthesis surface for the in-tree Hy-Embodied-0.5-VLA runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.formats.serialization import write_json
from worldfoundry.evaluation.models.runtime.profiles import load_runtime_profile
from worldfoundry.synthesis.action_generation.base_action_synthesis import ActionModelSynthesis

from .runtime import (
    DEFAULT_ALLOW_CHECKPOINT_DOWNLOAD,
    DEFAULT_BLEND_MODE,
    DEFAULT_LOCAL_FILES_ONLY,
    DEFAULT_REPLICATE_SINGLE_IMAGE,
    DEFAULT_STATE_FORMAT,
    DEFAULT_STRICT_CHECKPOINT,
    DEFAULT_TORCH_DTYPE,
    DEFAULT_VARIANT,
    HyEmbodiedVLARuntime,
    HyEmbodiedVLARuntimeConfig,
    build_plan_payload,
    select_hy_embodied_vla_checkpoint,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


class HyEmbodiedVLASynthesis(ActionModelSynthesis):
    """Profile-backed adapter around Tencent's official Hy-VLA inference path."""

    MODEL_ID = "hy-embodied-vla"

    def __init__(
        self,
        profile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(profile, device=device, command_template=command_template, env=env)
        self.runtime_options = dict(runtime_options or {})
        self._runtime: HyEmbodiedVLARuntime | None = None
        self._runtime_config_cache: HyEmbodiedVLARuntimeConfig | None = None

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
    ) -> "HyEmbodiedVLASynthesis":
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        if pretrained_model_path is not None and not isinstance(pretrained_model_path, Mapping):
            options["checkpoint"] = str(pretrained_model_path)
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or model_id or cls.MODEL_ID)
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
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

    def _runtime_config(self, options: Mapping[str, Any]) -> HyEmbodiedVLARuntimeConfig:
        merged = {**self.runtime_options, **dict(options)}
        variant = str(merged.get("variant") or merged.get("variant_id") or DEFAULT_VARIANT)
        checkpoint = select_hy_embodied_vla_checkpoint(
            checkpoint=merged.get("checkpoint")
            or merged.get("checkpoint_dir")
            or merged.get("ckpt_path"),
            checkpoints=self.profile.checkpoints,
            variant=variant,
        )
        history_size = merged.get("history_size", merged.get("img_history_size"))
        return HyEmbodiedVLARuntimeConfig(
            checkpoint=checkpoint,
            variant=variant,
            revision=str(merged["revision"]) if merged.get("revision") else None,
            device=str(merged.get("device") or self.device),
            torch_dtype=str(merged.get("torch_dtype") or DEFAULT_TORCH_DTYPE),
            local_files_only=_as_bool(merged.get("local_files_only"), DEFAULT_LOCAL_FILES_ONLY),
            allow_checkpoint_download=_as_bool(
                merged.get("allow_checkpoint_download"), DEFAULT_ALLOW_CHECKPOINT_DOWNLOAD
            ),
            history_size=int(history_size) if history_size is not None else None,
            replicate_single_image=_as_bool(
                merged.get("replicate_single_image"), DEFAULT_REPLICATE_SINGLE_IMAGE
            ),
            state_format=str(merged.get("state_format") or DEFAULT_STATE_FORMAT),
            blend_mode=str(merged.get("blend_mode") or DEFAULT_BLEND_MODE),
            strict_checkpoint=_as_bool(
                merged.get("strict_checkpoint"), DEFAULT_STRICT_CHECKPOINT
            ),
        )

    def _runtime_for(self, config: HyEmbodiedVLARuntimeConfig) -> HyEmbodiedVLARuntime:
        if self._runtime is None or self._runtime_config_cache != config:
            self._runtime = HyEmbodiedVLARuntime(config)
            self._runtime_config_cache = config
        return self._runtime

    @staticmethod
    def _observation(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        for key in ("hy_embodied_vla_observation", "observation", "official_policy_observation"):
            value = kwargs.get(key)
            if isinstance(value, Mapping):
                return value
        return None

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
        del video, timeout_seconds
        plan_only = _as_bool(kwargs.pop("plan_only", None), _as_bool(self.runtime_options.get("plan_only"), False))
        run_dir = Path(kwargs.pop("run_dir", "") or tempfile.mkdtemp(prefix="hy_embodied_vla_"))
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
        runtime_options = dict(kwargs)
        runtime_options.setdefault("device", self.device)
        config = self._runtime_config(runtime_options)
        plan_path = run_dir / "runtime_profile_plan.json"
        plan = build_plan_payload(
            config=config,
            context=context,
            profile=self.profile.to_dict(),
        )
        write_json(plan_path, plan)
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

        observation = self._observation(kwargs)
        state = kwargs.get("state")
        if state is None and observation is not None:
            state = observation.get("observation.state", observation.get("state"))
        if images is None and observation is not None:
            images = observation
        result = self._runtime_for(config).predict_action(
            instruction=prompt,
            images=images,
            state=state,
            observation=observation,
            output_path=context["output_path"],
            extra_metadata={
                "run_dir": str(run_dir),
                "plan_path": str(plan_path),
                "interactions": list(interactions),
            },
        )
        result["run_dir"] = str(run_dir)
        result["plan_path"] = str(plan_path)
        result["profile"] = self.profile.to_dict()
        return result


__all__ = ["HyEmbodiedVLASynthesis"]
