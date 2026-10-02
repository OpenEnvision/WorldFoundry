# Inference-only RoboFlamingo source retained in-tree.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core import jsonable
from worldfoundry.core.io import file_sha256
from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path
from worldfoundry.core.model_loading.checkpoints.file import load_state_dict
from worldfoundry.synthesis.action_generation.runtime_config import load_vla_va_wam_runtime_config

from .configuration import RoboFlamingoArchitectureConfig, action_trace_contract


@dataclass(frozen=True)
class RoboFlamingoRuntimeConfig:
    """Resolved RoboFlamingo runtime paths and options.

    Args:
        policy_checkpoint_path: Fine-tuned RoboFlamingo policy checkpoint.
        openflamingo_checkpoint_path: Base OpenFlamingo checkpoint.
        lang_encoder_path: Local MPT/LLaMA language model directory.
        tokenizer_path: Local tokenizer directory.
        device: Torch device for execution.
        precision: Runtime precision string.
        architecture: RoboFlamingo architecture options.
    """

    policy_checkpoint_path: Path | None
    openflamingo_checkpoint_path: Path | None
    lang_encoder_path: Path | None
    tokenizer_path: Path | None
    device: str
    precision: str
    architecture: RoboFlamingoArchitectureConfig
    source: Mapping[str, Any]

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "backend": "worldfoundry.roboflamingo.in_tree_runtime.predict_action",
            "runtime_package": "worldfoundry.synthesis.action_generation.roboflamingo",
            "source": dict(self.source),
            "policy_checkpoint_path": str(self.policy_checkpoint_path or ""),
            "openflamingo_checkpoint_path": str(self.openflamingo_checkpoint_path or ""),
            "lang_encoder_path": str(self.lang_encoder_path or ""),
            "tokenizer_path": str(self.tokenizer_path or ""),
            "device": self.device,
            "precision": self.precision,
            "architecture": self.architecture.to_dict(),
            "action_trace_contract": action_trace_contract(self.architecture),
        }


def _bool_option(options: Mapping[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _checkpoint_by_role(checkpoints: Sequence[Mapping[str, Any]], role_fragment: str) -> str | None:
    for item in checkpoints:
        role = str(item.get("role") or "").lower()
        if role_fragment in role:
            return str(item.get("local_dir") or "")
    return None


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = resolve_worldfoundry_path(str(value))
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def _resolve_checkpoint_file(path: Path | None, *, require_existing: bool, label: str) -> Path | None:
    if path is None:
        if require_existing:
            raise FileNotFoundError(f"RoboFlamingo {label} path is required for real inference.")
        return None
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.pth")) + sorted(path.glob("*.pt")) + sorted(path.glob("*.bin"))
        if candidates:
            return candidates[0].resolve()
    if require_existing:
        raise FileNotFoundError(f"RoboFlamingo {label} file was not found: {path}")
    return path


def _resolve_existing_dir(path: Path | None, *, require_existing: bool, label: str) -> Path | None:
    if path is None:
        if require_existing:
            raise FileNotFoundError(f"RoboFlamingo {label} directory is required for real inference.")
        return None
    if require_existing and not path.is_dir():
        raise FileNotFoundError(f"RoboFlamingo {label} directory was not found: {path}")
    return path


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _adjacent_released_checkpoint(path: Path | None, *, explicit: bool, relative_path: str) -> Path | None:
    """Use the released sibling checkpoint store when a profile default is absent."""
    if explicit or (path is not None and path.exists()):
        return path
    candidate = project_root().parent / "ckpts" / relative_path
    return candidate.resolve() if candidate.exists() else path


def _load_rgb_image(value: Any):
    from PIL import Image

    if value is None:
        raise ValueError("RoboFlamingo observation requires current_image/image_path.")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = _path_or_none(value)
    if path is not None and path.is_file():
        return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"RoboFlamingo image input was not found: {value}")


def _as_image_sequence(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _prepare_visual_tensor(
    *,
    observation: Mapping[str, Any],
    image_processor: Any,
    device: str,
    window_size: int,
) -> tuple[Any, Any]:
    import torch

    current = _first_present(observation, "current_image", "image", "image_path", "ref_image_path")
    history_values = _as_image_sequence(_first_present(observation, "visual_history", "history", "images"))
    if not history_values:
        history_values = [current]
    frames = [_load_rgb_image(item) for item in history_values if item not in (None, "")]
    if not frames:
        frames = [_load_rgb_image(current)]
    if len(frames) < window_size:
        frames = [frames[0]] * (window_size - len(frames)) + frames
    frames = frames[-window_size:]
    rgb = torch.stack([image_processor(frame) for frame in frames], dim=0).to(device=device)

    gripper_value = _first_present(observation, "gripper_image", "gripper", "wrist_image", "current_image", "image", "image_path")
    gripper_frames = [_load_rgb_image(item) for item in _as_image_sequence(gripper_value)]
    if not gripper_frames:
        gripper_frames = frames
    if len(gripper_frames) < window_size:
        gripper_frames = [gripper_frames[0]] * (window_size - len(gripper_frames)) + gripper_frames
    gripper_frames = gripper_frames[-window_size:]
    gripper = torch.stack([image_processor(frame) for frame in gripper_frames], dim=0).to(device=device)
    return rgb.unsqueeze(1).unsqueeze(2), gripper.unsqueeze(1).unsqueeze(2)


def _prepare_language_tensor(*, instruction: str, tokenizer: Any, device: str, window_size: int) -> tuple[Any, Any]:
    prompt = "<image> " + (instruction or "follow the instruction") + " <|endofchunk|>"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].repeat(max(window_size, 1), 1).to(device=device)
    attention_mask = encoded["attention_mask"].repeat(max(window_size, 1), 1).to(device=device)
    return input_ids, attention_mask


def _coerce_action_trace(logits: Any) -> dict[str, Any]:
    import torch

    if isinstance(logits, (list, tuple)):
        actions = logits[0] if logits else None
        gripper = logits[1] if len(logits) > 1 else None
    else:
        actions = logits
        gripper = None
    if actions is None:
        raise RuntimeError("RoboFlamingo model forward returned no action tensor.")
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions)
    actions = actions.detach().float().cpu()
    if gripper is not None:
        if not torch.is_tensor(gripper):
            gripper = torch.as_tensor(gripper)
        gripper = (gripper.detach().float().cpu() > 0.5).to(actions.dtype) * 2 - 1
        while gripper.ndim < actions.ndim:
            gripper = gripper.unsqueeze(1)
        if gripper.shape[:-1] == actions.shape[:-1]:
            actions = torch.cat([actions, gripper], dim=-1)
    return {
        "action": jsonable(actions),
        "action_shape": list(actions.shape),
    }


def select_roboflamingo_runtime_config(
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    require_existing: bool,
) -> RoboFlamingoRuntimeConfig:
    """Select RoboFlamingo checkpoint paths and architecture options.

    Args:
        checkpoints: Runtime-profile checkpoint records.
        options: Runtime options from constructor and predict call.
        require_existing: Whether paths must already exist.
    """

    defaults = load_vla_va_wam_runtime_config("roboflamingo")
    architecture_defaults = defaults.get("architecture")
    if not isinstance(architecture_defaults, Mapping):
        raise TypeError("roboflamingo data config requires an architecture mapping")
    source = defaults.get("source")
    if not isinstance(source, Mapping):
        raise TypeError("roboflamingo data config requires a source mapping")

    policy_override = _first_present(
        options, "policy_checkpoint_path", "checkpoint_path", "ckpt_path", "evaluate_from_checkpoint"
    )
    policy_path = _adjacent_released_checkpoint(
        _path_or_none(policy_override or _checkpoint_by_role(checkpoints, "released_calvin")),
        explicit=policy_override is not None,
        relative_path="robovlms--RoboFlamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth",
    )
    openflamingo_override = _first_present(options, "openflamingo_checkpoint_path", "openflamingo_checkpoint")
    openflamingo_path = _adjacent_released_checkpoint(
        _path_or_none(openflamingo_override or _checkpoint_by_role(checkpoints, "openflamingo")),
        explicit=openflamingo_override is not None,
        relative_path="openflamingo--OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt",
    )
    language_override = _first_present(options, "lang_encoder_path", "lm_path")
    lang_encoder_path = _adjacent_released_checkpoint(
        _path_or_none(
            language_override
            or _checkpoint_by_role(checkpoints, "language")
            or _checkpoint_by_role(checkpoints, "mpt")
        ),
        explicit=language_override is not None,
        relative_path="anas-awadalla--mpt-1b-redpajama-200b-dolly",
    )
    tokenizer_path = _path_or_none(options.get("tokenizer_path") or lang_encoder_path)
    vision_override = options.get("vision_encoder_pretrained")
    vision_pretrained = str(
        vision_override if vision_override is not None else architecture_defaults["vision_encoder_pretrained"]
    )
    if vision_override is None:
        vision_pretrained = str(
            _adjacent_released_checkpoint(
                _path_or_none(vision_pretrained),
                explicit=False,
                relative_path="open_clip/ViT-L-14.pt",
            ) or vision_pretrained
        )
    architecture = RoboFlamingoArchitectureConfig(
        llm_name=str(options.get("llm_name") or architecture_defaults["llm_name"]),
        vision_encoder_path=str(
            options.get("vision_encoder_path") or architecture_defaults["vision_encoder_path"]
        ),
        vision_encoder_pretrained=vision_pretrained,
        cross_attn_every_n_layers=int(
            options.get("cross_attn_every_n_layers")
            or architecture_defaults["cross_attn_every_n_layers"]
        ),
        window_size=int(
            options.get("window_size")
            or options.get("eval_hist_size")
            or architecture_defaults["window_size"]
        ),
        fusion_mode=str(options.get("fusion_mode") or architecture_defaults["fusion_mode"]),
        use_gripper=_bool_option(
            options, "use_gripper", bool(architecture_defaults["use_gripper"])
        ),
        use_state=_bool_option(options, "use_state", bool(architecture_defaults["use_state"])),
        decoder_type=str(options.get("decoder_type") or architecture_defaults["decoder_type"]),
        head_type=str(options.get("head_type") or architecture_defaults["head_type"]),
        n_obs_steps=max(1, int(options.get("n_obs_steps") or architecture_defaults["n_obs_steps"])),
        precision=str(options.get("precision") or architecture_defaults["precision"]),
    )
    return RoboFlamingoRuntimeConfig(
        policy_checkpoint_path=_resolve_checkpoint_file(policy_path, require_existing=require_existing, label="policy checkpoint"),
        openflamingo_checkpoint_path=_resolve_checkpoint_file(
            openflamingo_path,
            require_existing=require_existing,
            label="OpenFlamingo checkpoint",
        ),
        lang_encoder_path=_resolve_existing_dir(lang_encoder_path, require_existing=require_existing, label="language backbone"),
        tokenizer_path=_resolve_existing_dir(tokenizer_path, require_existing=require_existing, label="tokenizer"),
        device=str(options.get("device") or "cuda"),
        precision=architecture.precision,
        architecture=architecture,
        source=dict(source),
    )


class RoboFlamingoRuntime:
    def __init__(self, config: RoboFlamingoRuntimeConfig) -> None:
        """Create a lazy RoboFlamingo runtime.

        Args:
            config: Checkpoint, device, and architecture settings.
        """

        self.config = config
        self._model_bundle: tuple[Any, Any, Any] | None = None
        self._diffusion_history: list[Any] = []

    def _sample_diffusion_actions(self, model: Any, features: Any, observation: Mapping[str, Any]) -> Any:
        """Run the official conditioned diffusion action sampler."""
        import numpy as np
        import torch

        diffusion = model.diffusion_model
        history_len = min(self.config.architecture.n_obs_steps, diffusion.horizon - 1)
        supplied = observation.get("action_history")
        if supplied is not None:
            history = np.asarray(supplied, dtype=np.float32).reshape(-1, diffusion.data_dim)
        elif self._diffusion_history:
            history = np.asarray(self._diffusion_history, dtype=np.float32).reshape(-1, diffusion.data_dim)
        else:
            history = np.zeros((0, diffusion.data_dim), dtype=np.float32)
        history = history[-history_len:]
        if len(history) < history_len:
            history = np.concatenate(
                [np.zeros((history_len - len(history), diffusion.data_dim), dtype=np.float32), history],
                axis=0,
            )

        action_history = torch.as_tensor(history, device=features.device, dtype=torch.float32).unsqueeze(0)
        action_history = diffusion.normalizer.normalize(action_history)
        conditioned = torch.zeros(
            (action_history.shape[0], diffusion.horizon, diffusion.data_dim),
            device=action_history.device,
            dtype=action_history.dtype,
        )
        conditioned[:, :history_len] = action_history
        condition_mask = torch.zeros_like(conditioned, dtype=torch.bool)
        condition_mask[:, :history_len] = True
        sampled = diffusion.conditional_sample(
            cond_data=conditioned,
            cond_mask=condition_mask,
            global_cond=features,
        )
        actions = diffusion.normalizer.unnormalize(sampled)[:, history_len:]
        actions[..., -1] = (actions[..., -1] > 0.5).to(actions.dtype) * 2 - 1
        first_action = actions[0, 0].detach().float().cpu().numpy()
        self._diffusion_history = (self._diffusion_history + [first_action])[-history_len:]
        return actions

    def _load_model_bundle(self) -> tuple[Any, Any, Any]:
        if self._model_bundle is not None:
            return self._model_bundle

        import torch

        if not self.config.device.startswith("cuda"):
            raise RuntimeError(
                "RoboFlamingo upstream CALVIN action execution is CUDA-only in the official eval wrapper; "
                f"requested device={self.config.device!r}. Use plan_only=True on CPU hosts."
            )
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("RoboFlamingo runtime requested CUDA, but torch.cuda.is_available() is false.")
        if self.config.policy_checkpoint_path is None or not self.config.policy_checkpoint_path.is_file():
            raise FileNotFoundError(f"RoboFlamingo policy checkpoint is missing: {self.config.policy_checkpoint_path}")
        if self.config.openflamingo_checkpoint_path is None or not self.config.openflamingo_checkpoint_path.is_file():
            raise FileNotFoundError(f"RoboFlamingo OpenFlamingo checkpoint is missing: {self.config.openflamingo_checkpoint_path}")
        if self.config.lang_encoder_path is None or not self.config.lang_encoder_path.is_dir():
            raise FileNotFoundError(f"RoboFlamingo language backbone directory is missing: {self.config.lang_encoder_path}")
        if self.config.tokenizer_path is None or not self.config.tokenizer_path.is_dir():
            raise FileNotFoundError(f"RoboFlamingo tokenizer directory is missing: {self.config.tokenizer_path}")
        vision_checkpoint = resolve_worldfoundry_path(
            self.config.architecture.vision_encoder_pretrained
        ).resolve()
        if not vision_checkpoint.is_file():
            raise FileNotFoundError(
                "RoboFlamingo requires the OpenCLIP vision checkpoint as a staged local file: "
                f"{vision_checkpoint}"
            )

        from .imports import ensure_runtime_import_paths

        ensure_runtime_import_paths()
        from .modeling.factory import create_model_and_transforms

        model, image_processor, tokenizer = create_model_and_transforms(
            self.config.architecture.vision_encoder_path,
            str(vision_checkpoint),
            str(self.config.lang_encoder_path),
            str(self.config.tokenizer_path),
            cross_attn_every_n_layers=self.config.architecture.cross_attn_every_n_layers,
            use_local_files=True,
            window_size=self.config.architecture.window_size,
            fusion_mode=self.config.architecture.fusion_mode,
            use_gripper=self.config.architecture.use_gripper,
            use_state=self.config.architecture.use_state,
            decoder_type=self.config.architecture.decoder_type,
            use_diff=self.config.architecture.head_type == "diffusion",
            llm_name=self.config.architecture.llm_name,
            return_feature=True,
        )
        openflamingo_state = load_state_dict(
            self.config.openflamingo_checkpoint_path,
            device="cpu",
        )
        model.load_state_dict(openflamingo_state, strict=False)
        policy_state = load_state_dict(self.config.policy_checkpoint_path, device="cpu")
        policy_weights = policy_state["model_state_dict"]
        policy_keys = tuple(policy_weights)
        if policy_keys and all(key.startswith("module.") for key in policy_keys):
            # Released CALVIN checkpoints were saved from DistributedDataParallel.
            policy_weights = {key.removeprefix("module."): value for key, value in policy_weights.items()}
        elif any(key.startswith("module.") for key in policy_keys):
            raise ValueError("RoboFlamingo policy checkpoint mixes DDP and unprefixed state-dict keys.")
        model_keys = set(model.state_dict())
        # The released LSTM checkpoint has two names for the same decoder tensors.
        # This model exposes only lm_head; discard action_head aliases only when
        # every corresponding tensor is present and byte-for-byte identical.
        aliases = [key for key in policy_weights if key.startswith("action_head.") and key not in model_keys]
        for alias in aliases:
            decoder_key = "lm_head." + alias.removeprefix("action_head.")
            if (
                decoder_key not in model_keys
                or decoder_key not in policy_weights
                or not torch.equal(policy_weights[alias], policy_weights[decoder_key])
            ):
                raise RuntimeError(f"RoboFlamingo action-head alias does not match its decoder: {alias}")
        if aliases:
            policy_weights = {key: value for key, value in policy_weights.items() if key not in aliases}
        loaded = model.load_state_dict(policy_weights, strict=False)
        action_head_keys = {key for key in policy_weights if key.startswith("lm_head.")}
        if loaded.unexpected_keys:
            raise RuntimeError(
                f"RoboFlamingo policy has {len(loaded.unexpected_keys)} unmatched keys; "
                f"first: {loaded.unexpected_keys[:5]}"
            )
        if not action_head_keys or action_head_keys.intersection(loaded.missing_keys):
            raise RuntimeError(
                "RoboFlamingo policy action head did not load; refusing inference with an uninitialized head."
            )
        model = model.to(torch.device(self.config.device)).eval()
        self._model_bundle = (model, image_processor, tokenizer)
        return self._model_bundle

    def predict_action(
        self,
        *,
        instruction: str,
        observation: Mapping[str, Any],
        output_path: str | Path,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run RoboFlamingo policy inference and write a WorldFoundry trace.

        Args:
            instruction: Natural-language task instruction.
            observation: RoboFlamingo observation mapping.
            output_path: JSON artifact path for the action trace.
            extra_metadata: Additional WorldFoundry metadata.
        """

        import torch

        model, image_processor, tokenizer = self._load_model_bundle()
        vision_x, vision_gripper = _prepare_visual_tensor(
            observation=observation,
            image_processor=image_processor,
            device=self.config.device,
            window_size=self.config.architecture.window_size,
        )
        lang_x, attention_mask = _prepare_language_tensor(
            instruction=instruction,
            tokenizer=tokenizer,
            device=self.config.device,
            window_size=self.config.architecture.window_size,
        )
        with torch.no_grad():
            output = model(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                vision_gripper=vision_gripper,
                return_feature=True,
            )
            logits = getattr(output, "logits", output)
            if self.config.architecture.head_type == "diffusion":
                logits = self._sample_diffusion_actions(model, logits, observation)
        prediction = _coerce_action_trace(logits)
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "model_id": "roboflamingo",
            "artifact_kind": "action_trace",
            "runtime": "worldfoundry.roboflamingo.in_tree_runtime.predict_action",
            "runtime_config": self.config.to_plan_dict(),
            "status": "success",
            "action": prediction["action"],
            "action_shape": prediction["action_shape"],
            "metadata": jsonable(dict(extra_metadata or {})),
        }
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "status": "success",
            "model_id": "roboflamingo",
            "artifact_kind": "action_trace",
            "artifact_path": str(target),
            "artifact_sha256": file_sha256(target),
            "runtime": trace["runtime"],
            "action_shape": prediction["action_shape"],
            "action": prediction["action"],
        }
