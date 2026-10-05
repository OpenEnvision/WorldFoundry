from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.synthesis.action_generation._native_policy_runtime import (
    collect_images,
    completed_action_result,
    first_present,
    option_bool,
    option_int,
    runtime_options_cache_key,
    to_numpy_image,
)


_RUNTIME_CACHE: dict[tuple[Any, ...], Any] = {}


def clear_runtime_cache() -> None:
    from worldfoundry.core.execution.runtime_cache import clear_inference_runtime_cache

    clear_inference_runtime_cache(_RUNTIME_CACHE)


def _is_pi0(location: str, options: Mapping[str, Any]) -> bool:
    selector = options.get("variant") or options.get("variant_id") or options.get("runtime_variant")
    return "pi0" in str(selector or "").lower() or "pi0" in location.lower()


def _pi0_camera_keys(options: Mapping[str, Any], observation: Mapping[str, Any] | None = None) -> tuple[str, str]:
    keys = tuple(str(key) for key in (options.get("camera_keys") or (observation or {}).get("camera_keys") or ("exo_camera_1", "wrist_camera")))
    if len(keys) != 2 or keys[0] == keys[1]:
        raise ValueError("MolmoBot-Pi0 requires two distinct camera keys: exterior and wrist.")
    return keys


def _pi0_runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    if device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("MolmoBot-Pi0's official PyTorch policy requires a CUDA device.")
    checkpoint = Path(location).expanduser()
    config_file = checkpoint / "assets" / "train_config.pkl"
    if not checkpoint.is_dir() or not config_file.is_file():
        raise FileNotFoundError(f"MolmoBot-Pi0 checkpoint is missing assets/train_config.pkl: {checkpoint}")
    # The official loader unpickles train_config.pkl.  Require a vetted digest
    # before importing the policy, including when a custom checkpoint is used.
    expected = str(options.get("train_config_sha256") or "").lower()
    if not expected or hashlib.sha256(config_file.read_bytes()).hexdigest() != expected:
        raise ValueError(
            "MolmoBot-Pi0 train_config.pkl has no matching trusted SHA-256; "
            "provide the digest for a reviewed checkpoint via train_config_sha256."
        )
    tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "big_vision" / "paligemma_tokenizer.model"
    tokenizer_hash = str(options.get("paligemma_tokenizer_sha256") or "").lower()
    if not tokenizer.is_file() or not tokenizer_hash or hashlib.sha256(tokenizer.read_bytes()).hexdigest() != tokenizer_hash:
        raise FileNotFoundError(
            "MolmoBot-Pi0 requires the validated PaliGemma tokenizer at "
            f"{tokenizer}; stage gs://big_vision/paligemma_tokenizer.model "
            "(https://storage.googleapis.com/big_vision/paligemma_tokenizer.model) there."
        )
    camera_keys = _pi0_camera_keys(options)
    key = ("pi0", str(checkpoint.resolve()), device, runtime_options_cache_key(options))
    agent = _RUNTIME_CACHE.get(key)
    if agent is not None:
        return agent
    from .pi0_policy import Pi0JointPositionPolicy

    try:
        agent = Pi0JointPositionPolicy(
            checkpoint_dir=str(checkpoint),
            device=device,
            camera_keys=camera_keys,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MolmoBot-Pi0 needs the pinned official molmobot-pi0 and OpenPI "
            "dependencies in a dedicated environment."
        ) from exc
    agent.prepare_model()
    _RUNTIME_CACHE[key] = agent
    return agent


def _runtime_for(location: str, device: str, options: Mapping[str, Any]) -> Any:
    if _is_pi0(location, options):
        return _pi0_runtime_for(location, device, options)
    key = (
        location,
        device,
        options.get("num_flow_steps"),
        options.get("max_seq_len"),
        options.get("norm_repo_id"),
        options.get("states_mode"),
        options.get("use_bfloat16"),
        options.get("compile_model"),
    )
    agent = _RUNTIME_CACHE.get(key)
    if agent is not None:
        return agent

    from .policy import SynthManipMolmoInferenceWrapper

    agent = SynthManipMolmoInferenceWrapper(
        checkpoint_path=location,
        device=device,
        num_flow_steps=None if options.get("num_flow_steps") in (None, "") else option_int(options.get("num_flow_steps"), 10),
        max_seq_len=None if options.get("max_seq_len") in (None, "") else option_int(options.get("max_seq_len"), 0),
        norm_repo_id=str(options.get("norm_repo_id") or "synthmanip"),
        use_bfloat16=option_bool(options.get("use_bfloat16"), True),
        compile_model=option_bool(options.get("compile_model"), False),
        states_mode=options.get("states_mode"),
    )
    _RUNTIME_CACHE[key] = agent
    return agent


def _pi0_observation(
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    camera_keys: tuple[str, str],
) -> dict[str, Any]:
    import numpy as np

    sources = [observation]
    if isinstance(observation.get("images"), Mapping):
        sources.append(observation["images"])
    if isinstance(image, Mapping):
        sources.append(image)
    cameras: dict[str, Any] = {}
    for key in camera_keys:
        value = next((source[key] for source in sources if key in source and source[key] is not None), None)
        if value is None:
            raise ValueError(f"MolmoBot-Pi0 requires camera image {key!r}.")
        array = np.asarray(to_numpy_image(value))
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"MolmoBot-Pi0 camera {key!r} must be an HWC uint8 RGB image.")
        cameras[key] = array

    qpos = first_present(observation, "qpos", "state")
    if isinstance(qpos, Mapping):
        arm = np.asarray(qpos.get("arm"), dtype=np.float32)
        gripper = np.asarray(qpos.get("gripper"), dtype=np.float32)
    elif qpos is not None:
        state = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if state.shape != (9,):
            raise ValueError("MolmoBot-Pi0 state must be qpos.arm (7) plus qpos.gripper (2).")
        arm, gripper = state[:7], state[7:]
    else:
        raise ValueError("MolmoBot-Pi0 requires qpos.arm (7) and qpos.gripper (2).")
    if arm.shape != (7,) or gripper.shape != (2,) or not np.isfinite(arm).all() or not np.isfinite(gripper).all():
        raise ValueError("MolmoBot-Pi0 requires finite qpos.arm (7) and qpos.gripper (2).")
    return {"task": str(observation.get("task") or instruction), "qpos": {"arm": arm, "gripper": gripper}, **cameras}


def predict_action(
    *,
    instruction: str,
    image: Any,
    observation: Mapping[str, Any],
    action_context: Sequence[Any],
    checkpoint_path: str,
    device: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del action_context
    options = dict(runtime_options or {})
    location = checkpoint_path or str(options.get("checkpoint_ref") or "allenai/MolmoBot-DROID")
    if _is_pi0(location, options):
        camera_keys = _pi0_camera_keys(options, observation)
        agent = _runtime_for(location, device, {**options, "camera_keys": camera_keys})
        agent.reset()  # A new call is a fresh observation, not a buffered prior episode.
        raw = agent.get_action(_pi0_observation(instruction, image, observation, camera_keys))
        import numpy as np

        actions = {key: np.asarray(value).tolist() for key, value in raw.items()}
        if set(actions) != {"arm", "gripper"} or len(actions["arm"]) != 7 or len(actions["gripper"]) != 1:
            raise RuntimeError("MolmoBot-Pi0 returned an unexpected joint-position action shape.")
        if not all(np.isfinite(value).all() for value in raw.values()):
            raise RuntimeError("MolmoBot-Pi0 returned a non-finite action.")
        return completed_action_result(
            model_id="molmobot",
            instruction=instruction,
            actions=actions,
            raw_output=actions,
            checkpoint_path=location,
            device=device,
            runtime="worldfoundry.molmobot.pi0_openpi_in_process",
            metadata={
                "entrypoint": "worldfoundry.synthesis.action_generation.molmobot.pi0_policy:Pi0JointPositionPolicy.get_action",
                "official_reference": "molmobot_pi0.eval.policies.pi:PiJointPosPolicy.get_action",
                "camera_keys": list(camera_keys),
                "variant": "pi0_droid",
            },
        )
    agent = _runtime_for(location, device, options)
    camera_keys = tuple(str(item) for item in options.get("camera_keys") or observation.get("camera_keys") or ("exo_camera_1", "wrist_camera"))
    images = collect_images(observation, image, camera_keys)
    if not images:
        raise ValueError("MolmoBot requires one or more camera images for get_action_chunk.")
    state = first_present(observation, "state", "qpos", "robot_state", "joint_state")
    raw = agent.get_action_chunk(
        images=images,
        task_description=str(first_present(observation, "task", "task_description") or instruction),
        state=state,
    )
    return completed_action_result(
        model_id="molmobot",
        instruction=instruction,
        actions=raw,
        raw_output=raw,
        checkpoint_path=checkpoint_path,
        device=device,
        runtime="worldfoundry.molmobot.native_in_process",
        metadata={
            "entrypoint": "worldfoundry.synthesis.action_generation.molmobot.policy:SynthManipMolmoInferenceWrapper.get_action_chunk",
            "camera_keys": list(camera_keys),
            "agent_config": getattr(agent, "config", {}),
        },
    )


__all__ = ["clear_runtime_cache", "predict_action"]
