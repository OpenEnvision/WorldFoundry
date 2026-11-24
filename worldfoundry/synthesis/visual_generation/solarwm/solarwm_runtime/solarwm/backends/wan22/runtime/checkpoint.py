"""Exact Wan full-state checkpoint commit and restore."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....checkpoint import CheckpointContract, assert_resume_compatible, verify_checkpoint
from ....errors import BackendContractError

_MODEL_PREFIX = "model."


@dataclass(frozen=True)
class RestoredWanCheckpoint:
    step: int
    identity: str
    path: Path
    standalone: bool
    optimizer_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class RestoredWanWeights:
    """Resolved identity of a weights-only Stage0.5 EMA initialization."""

    source_step: int
    identity: str
    path: Path
    standalone: bool


def _standalone_checkpoint_id(path: Path) -> str:
    size = path.stat().st_size
    if size <= 0:
        raise BackendContractError(f"Wan standalone checkpoint is empty: {path}")
    return f"inventory:file={path.name}:bytes={size}"


def checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    """Bind every algorithm-bearing Stage0.5 field while allowing corpus refreshes."""

    model = config["model"]
    data = config["data"]
    train = config["train"]
    distributed = config["distributed"]
    optimizer = train["optimizer"]
    ema = train["ema"]
    fsdp = train["fsdp"]
    data_generation = (
        f"{data['encoding']}:{data.get('preencode_schema', 'online')}:"
        f"{data['pixel_frames']}f:{data['height']}x{data['width']}"
    )
    extras: dict[str, Any] = {
        "model_name": str(model["name"]),
        "timestep_shift": float(model["timestep_shift"]),
        "latent_channels": int(model["latent_channels"]),
        "frame_sequence_length": int(model["frame_sequence_length"]),
        "num_output_frames": int(model["num_output_frames"]),
        "local_attn_size": int(model["local_attn_size"]),
        "raw_world_size": int(distributed["world_size"]),
        "global_batch_size": int(train["global_batch_size"]),
        "max_steps": int(train["max_steps"]),
        "warmup_steps": int(train["warmup_steps"]),
        "num_train_timesteps": int(train["num_train_timesteps"]),
        "timestep_mode": str(train["timestep_mode"]),
        "i2v_image_condition_dropout": float(train["i2v_image_condition_dropout"]),
        "optimizer": {
            "lr": float(optimizer["lr"]),
            "betas": [float(value) for value in optimizer["betas"]],
            "eps": float(optimizer["eps"]),
            "weight_decay": float(optimizer["weight_decay"]),
        },
        "ema": {
            "decay": float(ema["decay"]),
            "start_step": int(ema["start_step"]),
            "update_every": int(ema.get("update_every", 1)),
        },
        "activation_checkpointing": bool(fsdp["activation_checkpointing"]),
    }
    if str(train["objective"]) == "anyflow_forward_map":
        anyflow_contract = {
            "variant": str(train["anyflow_variant"]),
            "weight_type": str(train["anyflow_weight_type"]),
            "gate": float(train["anyflow_gate"]),
            "deltatime_type": str(train["anyflow_deltatime_type"]),
            "epsilon": float(train["anyflow_epsilon"]),
            "diffusion_ratio": float(train["anyflow_diffusion_ratio"]),
            "consistency_ratio": float(train["anyflow_consistency_ratio"]),
            "fuse_guidance_scale": float(train["anyflow_fuse_guidance_scale"]),
            "negative_embedding": str(train["anyflow_negative_embedding"]),
        }
        extras["anyflow_v1_5"] = anyflow_contract
    return CheckpointContract(
        family=str(model["family"]),
        stage=str(train["stage"]),
        causal_mode=str(train["causal_mode"]),
        objective=str(train["objective"]),
        objective_variant=str(train.get("objective_variant", "standard")),
        camera_translation_transform=str(model["camera_translation_transform"]),
        parameterization="full-parameter",
        sp_size=int(distributed["sequence_parallel_size"]),
        data_generation=data_generation,
        extras=extras,
    )


def _strip_model_prefix_key(key: str) -> str:
    return key[len(_MODEL_PREFIX) :] if key.startswith(_MODEL_PREFIX) else key


def normalize_model_state(
    state: Mapping[str, Any],
    *,
    field: str,
) -> OrderedDict[str, Any]:
    """Normalize the optional root ``model.`` namespace."""

    if not isinstance(state, Mapping) or not state:
        raise BackendContractError(f"Wan checkpoint {field} must be a non-empty mapping")
    keys = tuple(state)
    if not all(isinstance(key, str) and key for key in keys):
        raise BackendContractError(f"Wan checkpoint {field} keys must be strings")
    prefixed = tuple(key.startswith(_MODEL_PREFIX) for key in keys)
    if any(prefixed) and not all(prefixed):
        raise BackendContractError(f"Wan checkpoint {field} mixes model-prefixed and unprefixed keys")
    normalized = OrderedDict((_strip_model_prefix_key(key), value) for key, value in state.items())
    if len(normalized) != len(state):
        raise BackendContractError(f"Wan checkpoint {field} key normalization produced duplicates")
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        normalized._metadata = {  # type: ignore[attr-defined]
            _strip_model_prefix_key(str(key)): value for key, value in metadata.items()
        }
    return normalized


def _distributed_context() -> tuple[Any, int, int]:
    import torch.distributed as dist

    if dist.is_initialized():
        return dist, dist.get_rank(), dist.get_world_size()
    return dist, 0, 1


def _is_fsdp(module: Any) -> bool:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, FullyShardedDataParallel)


def _collective_error(local_error: str | None, *, phase: str) -> None:
    dist, rank, world_size = _distributed_context()
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, (rank, local_error))
        failures = [item for item in gathered if item[1] is not None]
    else:
        failures = [(rank, local_error)] if local_error is not None else []
    if failures:
        rendered = "; ".join(f"rank={item[0]}: {item[1]}" for item in failures)
        raise BackendContractError(f"Wan checkpoint {phase} failed collectively: {rendered}")


def _broadcast_text(value: str) -> str:
    dist, _, world_size = _distributed_context()
    payload = [value]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def _load_full_model_state(module: Any, state: Mapping[str, Any]) -> None:
    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        options = FullStateDictConfig(rank0_only=False, offload_to_cpu=False)
        with FullyShardedDataParallel.state_dict_type(module, StateDictType.FULL_STATE_DICT, options):
            result = module.load_state_dict(state, strict=True)
    else:
        result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise BackendContractError(
            "Wan full-state load was not exact: "
            f"missing={result.missing_keys[:8]} unexpected={result.unexpected_keys[:8]}"
        )


def _load_model_state_result(
    module: Any,
    state: Mapping[str, Any],
    *,
    strict: bool,
) -> Any:
    """Run one root-aware full-state load and return PyTorch's key report."""

    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        options = FullStateDictConfig(rank0_only=False, offload_to_cpu=False)
        with FullyShardedDataParallel.state_dict_type(module, StateDictType.FULL_STATE_DICT, options):
            return module.load_state_dict(state, strict=bool(strict))
    return module.load_state_dict(state, strict=bool(strict))


def _load_stage0p5_state_into_anyflow(module: Any, state: Mapping[str, Any]) -> tuple[str, ...]:
    """Upgrade FM Stage0.5 weights for AnyFlow-v1.5.

    Every shared tensor must match.  The only missing tensors are the four
    objective-only ``delta_embedding`` values, each initialized from its
    corresponding loaded ``time_embedding`` tensor.
    """

    time_keys = tuple(sorted(str(key) for key in state if str(key).startswith("time_embedding.")))
    delta = tuple(key.replace("time_embedding.", "delta_embedding.", 1) for key in time_keys)
    if len(delta) != 4 or any(key in state for key in delta):
        raise BackendContractError(
            "Wan FM -> AnyFlow initialization permits exactly four missing "
            "delta_embedding tensors and no pre-existing delta state"
        )
    upgraded = OrderedDict(state.items())
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        upgraded._metadata = metadata  # type: ignore[attr-defined]
    for time_key, delta_key in zip(time_keys, delta, strict=True):
        upgraded[delta_key] = state[time_key].detach().clone()
    try:
        exact = _load_model_state_result(module, upgraded, strict=True)
    except Exception as exc:
        raise BackendContractError(
            "Wan FM -> AnyFlow initialization permits exactly four missing "
            "delta_embedding tensors and no other mismatch"
        ) from exc
    if exact.missing_keys or exact.unexpected_keys:
        raise BackendContractError("Wan upgraded AnyFlow state was not exact after delta initialization")
    return tuple(sorted(delta))


def _named_parameters(module: Any) -> list[tuple[str, Any]]:
    root = getattr(module, "module", module)
    return list(root.named_parameters())


def _load_full_ema_state(
    module: Any,
    ema: Any,
    state: Mapping[str, Any],
    *,
    num_updates: int,
) -> None:
    live = {name: parameter.detach().clone() for name, parameter in _named_parameters(module)}
    try:
        _load_full_model_state(module, state)
        local_ema = {name: parameter.detach().clone() for name, parameter in _named_parameters(module)}
        ema.load_state_dict(local_ema, num_updates=int(num_updates))
    finally:
        import torch

        with torch.no_grad():
            for name, parameter in _named_parameters(module):
                if name not in live:
                    raise BackendContractError(f"Wan live parameter {name!r} disappeared during EMA restore")
                parameter.copy_(live[name].to(device=parameter.device, dtype=parameter.dtype))


def _payload_config_contract(payload: Mapping[str, Any]) -> dict[str, str]:
    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        raise BackendContractError("Wan checkpoint config must be a mapping")
    model = saved.get("model")
    train = saved.get("train")
    if not isinstance(model, Mapping) or not isinstance(train, Mapping):
        raise BackendContractError("Wan checkpoint config lacks model or train")
    return {
        "family": str(model.get("family", "")),
        "stage": str(train.get("stage", "")),
        "objective": str(train.get("objective", "")),
        "camera_translation_transform": str(model.get("camera_translation_transform", "linear")),
    }


def _assert_payload_contract(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, str],
    label: str,
) -> None:
    actual = _payload_config_contract(payload)
    drift = {
        field: {"actual": actual.get(field), "expected": value}
        for field, value in expected.items()
        if actual.get(field) != value
    }
    if drift:
        raise BackendContractError(f"{label} contract mismatch: {drift}")


def _resolve_input_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, bool]:
    source = Path(path).expanduser().resolve()
    dist, rank, _ = _distributed_context()
    result = ["", "", False]
    error = [""]
    if rank == 0:
        try:
            if source.is_dir():
                verified = verify_checkpoint(source)
                assert_resume_compatible(checkpoint_contract(config), verified.contract)
                result = [
                    str(verified.path / "model.pt"),
                    f"digest:{verified.manifest_digest}",
                    False,
                ]
            elif source.is_file():
                raise BackendContractError("Wan full resume requires the complete checkpoint transaction directory")
            else:
                raise BackendContractError(f"Wan checkpoint is missing: {source}")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
        dist.broadcast_object_list(result, src=0)
    if error[0]:
        raise BackendContractError(f"Wan checkpoint resolution failed: {error[0]}")
    resolved = Path(str(result[0]))
    local_error = None if resolved.is_file() else f"checkpoint is not visible: {resolved}"
    _collective_error(local_error, phase="visibility")
    return resolved, str(result[1]), bool(result[2])


def _weights_source_contract_matches(actual: CheckpointContract, config: Mapping[str, Any]) -> None:
    source = config["checkpoint"]["source_contract"]
    expected = {
        "family": str(config["model"]["family"]),
        "stage": str(source["stage"]),
        "causal_mode": "bidirectional",
        "objective": str(source["objective"]),
        "camera_translation_transform": str(source["camera_translation_transform"]),
        "parameterization": "full-parameter",
    }
    drift = {
        field: {"actual": getattr(actual, field), "expected": value}
        for field, value in expected.items()
        if getattr(actual, field) != value
    }
    if drift:
        raise BackendContractError(f"Wan weights-only source contract mismatch: {drift}")


def _resolve_weights_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, bool, int]:
    """Resolve a Stage0.5 source without comparing it to the Stage1 contract."""

    source = Path(path).expanduser().resolve()
    dist, rank, _ = _distributed_context()
    result: list[Any] = ["", "", False, 0]
    error = [""]
    if rank == 0:
        try:
            expected_step = int(config["checkpoint"]["source_step"])
            if source.is_dir():
                verified = verify_checkpoint(source)
                _weights_source_contract_matches(verified.contract, config)
                if int(verified.step) != expected_step:
                    raise BackendContractError(
                        "Wan weights-only source step differs from checkpoint.source_step: "
                        f"{verified.step} != {expected_step}"
                    )
                result = [
                    str(verified.path / "model.pt"),
                    f"digest:{verified.manifest_digest}",
                    False,
                    int(verified.step),
                ]
            elif source.is_file():
                result = [
                    str(source),
                    _standalone_checkpoint_id(source),
                    True,
                    expected_step,
                ]
            else:
                raise BackendContractError(f"Wan weights source is missing: {source}")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
        dist.broadcast_object_list(result, src=0)
    if error[0]:
        raise BackendContractError(f"Wan weights source resolution failed: {error[0]}")
    resolved = Path(str(result[0]))
    local_error = None if resolved.is_file() else f"weights source is not visible: {resolved}"
    _collective_error(local_error, phase="weights-source visibility")
    return resolved, str(result[1]), bool(result[2]), int(result[3])


def _validate_standalone_weights_source(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    source_step: int,
) -> Mapping[str, Any]:
    try:
        actual_step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan weights source has no valid global_step") from exc
    if actual_step != int(source_step):
        raise BackendContractError(f"Wan weights source step {actual_step} != {source_step}")
    source = config["checkpoint"]["source_contract"]
    _assert_payload_contract(
        payload,
        expected={
            "family": str(config["model"]["family"]),
            "stage": str(source["stage"]),
            "objective": str(source["objective"]),
            "camera_translation_transform": str(source["camera_translation_transform"]),
        },
        label="Wan standalone weights source",
    )
    state = payload.get("generator_ema")
    if not isinstance(state, Mapping) or not state:
        raise BackendContractError("Wan weights-only initialization requires generator_ema")
    try:
        ema_updates = int(payload["ema_num_updates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan weights source has no EMA update count") from exc
    if ema_updates != actual_step:
        raise BackendContractError("Wan weights source EMA update count differs from global step")
    return state


def load_weights_only_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
) -> RestoredWanWeights:
    """Load only Stage0.5 EMA weights, leaving all Stage1 state fresh."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan weights source payload must be a mapping")
        if standalone:
            raw_state = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError("Wan transactional weights source step differs from its manifest")
            raw_state = loaded.get("generator_ema")
            if not isinstance(raw_state, Mapping) or not raw_state:
                raise BackendContractError("Wan transactional weights source has no generator_ema")
        state = normalize_model_state(raw_state, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="weights-only preload/contract")
    assert state is not None
    _load_full_model_state(diffusion.module, state)
    return RestoredWanWeights(source_step, identity, resolved, standalone)


def load_anyflow_weights_only_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
) -> tuple[RestoredWanWeights, tuple[str, ...]]:
    """Load Stage0.5 EMA into AnyFlow and initialize only its four deltas."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan AnyFlow source payload must be a mapping")
        if standalone:
            raw_state = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError("Wan AnyFlow source step differs from its transaction manifest")
            raw_state = loaded.get("generator_ema")
            if not isinstance(raw_state, Mapping) or not raw_state:
                raise BackendContractError("Wan AnyFlow source has no generator_ema")
        state = normalize_model_state(raw_state, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="AnyFlow weights-only preload/contract")
    assert state is not None
    delta_keys: tuple[str, ...] = ()
    try:
        delta_keys = _load_stage0p5_state_into_anyflow(diffusion.module, state)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    else:
        local_error = None
    _collective_error(local_error, phase="AnyFlow objective upgrade")
    return (
        RestoredWanWeights(source_step, identity, resolved, standalone),
        delta_keys,
    )


def load_live_and_ema_weights_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
    ema: Any,
) -> RestoredWanWeights:
    """Initialize Stage0.5 LIVE/EMA selections while resetting training state."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    live: Mapping[str, Any] | None = None
    ema_state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan dual weights source payload must be a mapping")
        if standalone:
            raw_ema = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError("Wan transactional dual source step differs from its manifest")
            raw_ema = loaded.get("generator_ema")
            if not isinstance(raw_ema, Mapping) or not raw_ema:
                raise BackendContractError("Wan dual source has no generator_ema")
        selections = list(config["checkpoint"].get("weights", []))
        if selections not in (["live", "ema"], ["ema", "ema"]):
            raise BackendContractError("Wan Stage0.5 source must select LIVE/EMA or EMA/EMA weights")
        raw_live = loaded.get("generator" if selections[0] == "live" else "generator_ema")
        if not isinstance(raw_live, Mapping) or not raw_live:
            raise BackendContractError("Wan Stage0.5 source has no selected LIVE weights")
        live = normalize_model_state(raw_live, field=selections[0])
        ema_state = normalize_model_state(raw_ema, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="dual weights preload/contract")
    assert live is not None and ema_state is not None
    _load_full_model_state(diffusion.module, live)
    _load_full_ema_state(diffusion.module, ema, ema_state, num_updates=0)
    return RestoredWanWeights(source_step, identity, resolved, standalone)


__all__ = [
    "RestoredWanCheckpoint",
    "RestoredWanWeights",
    "checkpoint_contract",
    "load_anyflow_weights_only_checkpoint",
    "load_live_and_ema_weights_checkpoint",
    "load_weights_only_checkpoint",
    "normalize_model_state",
]
