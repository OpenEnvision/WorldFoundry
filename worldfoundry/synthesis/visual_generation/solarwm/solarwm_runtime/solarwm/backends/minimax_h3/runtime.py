"""MiniMax-H3 training, transactional checkpoints and shared validation lifecycle."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...checkpoint import CheckpointContract, assert_resume_compatible, verify_checkpoint
from ...data import resolve_index_path
from ...errors import BackendContractError
from ...inference import (
    InferenceCase,
    InferenceEngine,
    publish_comparison_complete,
    publish_comparison_partition,
)
from ...runtime import Topology, rng_identity
from ...runtime.distributed import gather_and_assert_sp_identity
from ...runtime.randomness import seed_process
from .artifacts import H3PreencodedStream, h3_silence_profile, load_silence_latents
from .distributed import get_sp_group, get_sp_rank, get_sp_size
from .inference import camera_fingerprint, package_generated
from .optional import load_conditioners, load_transformer, require_h3_runtime
from .stage0p5 import H3Stage0p5Core


def _base_model_profile(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Portable, readable identity for a compatible MiniMax-H3 base model."""

    profile = {
        "schema": "solarwm.minimax-h3-base-profile.v1",
        "family": "minimax_h3",
        "architecture": str(model_cfg.get("architecture", "minimax-h3-33b")),
        "component": str(model_cfg.get("transformer_subfolder", "transformer")),
        "parameter_dtype_profile": "bfloat16-blocks+float32-io-timestep-v1",
    }
    revision = str(model_cfg.get("revision") or "").strip()
    if revision:
        profile["revision"] = revision
    return profile


def _base_model_load_receipt(model_cfg: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """Describe the strictly loaded model with portable semantic fields."""

    parameters = tuple(model.named_parameters())
    if not parameters or len({name for name, _ in parameters}) != len(parameters):
        raise BackendContractError("H3 base model parameter inventory is empty or duplicated")
    dtype_inventory: dict[str, dict[str, int]] = {}
    for _name, parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        entry = dtype_inventory.setdefault(dtype, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(parameter.numel())
    return {
        "schema": "solarwm.minimax-h3-base-load.v1",
        "profile": _base_model_profile(model_cfg),
        "strict_state_load": {
            "missing_keys": 0,
            "unexpected_keys": 0,
            "mismatched_shapes": 0,
            "load_errors": 0,
        },
        "parameter_inventory": {
            "tensors": len(parameters),
            "parameters": sum(int(parameter.numel()) for _, parameter in parameters),
            "dtypes": dict(sorted(dtype_inventory.items())),
        },
    }


def _base_weights_label(model_cfg: Mapping[str, Any]) -> str:
    profile = _base_model_profile(model_cfg)
    revision = f"@{profile['revision']}" if profile.get("revision") else ""
    return f"{profile['architecture']}/{profile['component']}{revision}"


def _topology(*, sp_size: int, require_torchrun: bool) -> Topology:
    required = {"WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"}
    if required <= set(os.environ):
        return Topology.from_environ(sp_size)
    if require_torchrun:
        raise BackendContractError("H3 SP2 training must be launched with torchrun rank variables")
    return Topology(1, 0, 1, 0, sp_size=1)


def _reader(
    config: Mapping[str, Any],
    topology: Topology,
    *,
    index_field: str = "train_index",
    fixed_validation: bool = False,
    fixed_validation_sample_count: int | None = None,
    fixed_validation_sample_ids: Sequence[str] | None = None,
) -> Any:
    if fixed_validation and config["validation"].get("prepared_plan"):
        from .validation_inputs import H3PreparedValidationStream

        return H3PreparedValidationStream(config, topology)
    if fixed_validation and config["train"]["stage"] in {"stage1", "stage2"}:
        from .validation_stream import H3IndexedValidationStream

        return H3IndexedValidationStream(config, topology, frozen_ids=fixed_validation_sample_ids)
    data = config["data"]
    transport = data["transport"]
    common = {
        "root": str(transport["root"]),
        "index": str(resolve_index_path(data, index_field)),
        "topology": topology,
        "seed": int(data.get("seed", 42)),
        "cache_dir": transport.get("cache_dir"),
        "cache_max_gib": float(transport.get("cache_max_gib", 256)),
        "gcs_prefetch_shards": (0 if fixed_validation else data.get("gcs_prefetch_shards", 0)),
        "shuffle_buffer": int(data.get("shuffle_buffer", 4096)),
        "num_workers": 1 if fixed_validation else int(data.get("num_workers", 1)),
    }
    if fixed_validation:
        validation = config["validation"]
        common.update(
            fixed_validation_selection_seed=int(validation["selection_seed"]),
            fixed_validation_noise_seed=int(validation["noise_seed"]),
        )
    return H3PreencodedStream(
        **common,
        encoder_contract_path=str(data["encoder_contract_path"]),
        fixed_validation=fixed_validation,
        fixed_validation_sample_count=fixed_validation_sample_count,
        fixed_validation_sample_ids=fixed_validation_sample_ids,
        camera_audit_latents=47 if fixed_validation else int(data["train_target_latents"]),
    )


def _validation_schedule(
    validation: Mapping[str, Any],
    topology: Topology,
) -> tuple[int, int]:
    raw_count = validation.get("sample_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count <= 0:
        raise BackendContractError("H3 validation.sample_count must be a positive integer")
    dp_world_size = int(topology.dp_world_size)
    if raw_count % dp_world_size:
        raise BackendContractError(
            "H3 validation.sample_count must form complete logical-DP waves: "
            f"sample_count={raw_count} dp_world_size={dp_world_size}"
        )
    return raw_count, raw_count // dp_world_size


def _publish_h3_validation_complete(
    destination: Path,
    *,
    step: int,
    pass_name: str,
    expected_local_slots: Sequence[int],
    global_slots: int,
) -> None:
    compare_paths = tuple(sorted((destination / "compare").glob("rank*.mp4")))
    manifest_paths = tuple(sorted((destination / "manifests").glob("rank_*.json")))
    local_slots = tuple(sorted(int(slot) for slot in expected_local_slots))
    if len(compare_paths) != len(local_slots) or len(manifest_paths) != len(local_slots):
        raise BackendContractError(
            "H3 validation output count differs before completion: "
            f"compare={len(compare_paths)} manifests={len(manifest_paths)} "
            f"expected_local={len(local_slots)} global={global_slots}"
        )
    slots: list[int] = []
    for path in manifest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            slots.append(int(payload["logical_validation_rank"]))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise BackendContractError(f"H3 validation manifest has no usable logical slot: {path}") from exc
    if sorted(slots) != list(local_slots):
        raise BackendContractError("H3 validation completion requires each node-local logical slot exactly once")
    publish_comparison_complete(
        destination,
        step=int(step),
        pass_name=pass_name,
        local_slots=len(local_slots),
        global_slots=int(global_slots),
    )


def _checkpoint_contract(
    *,
    encoder_profile: Mapping[str, Any],
    silence_profile: Mapping[str, Any],
    base_model: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> CheckpointContract:
    train = config["train"] if config is not None else {}
    stage = str(train.get("stage", "stage0p5"))
    stage_extras = {}
    if stage != "stage0p5":
        stage_extras = {
            "chunk_latents": 5,
            "window_chunks": 6,
            "target_latents": 45 if stage == "stage1" else 47,
            "rollout_latents": 50,
            "student_rope_mode": "sliding_local" if stage == "stage2" else "native_absolute",
            "training_profile": {
                key: value for key, value in train.items() if key not in {"max_steps", "global_batch_size"}
            },
        }
    return CheckpointContract(
        family="minimax_h3",
        stage=stage,
        causal_mode=str(train.get("causal_mode", "bidirectional")),
        objective=str(train.get("objective", "flow_matching")),
        objective_variant="v1_5" if stage == "stage1" else "data_ward_velocity",
        camera_translation_transform="logd4",
        parameterization="peft-lora-r384-alpha384",
        sp_size=4 if stage == "stage2" else 2,
        data_generation="h3.158f.v1",
        extras={
            "encoder_profile": json.loads(json.dumps(dict(encoder_profile), sort_keys=True)),
            "silence_profile": json.loads(json.dumps(dict(silence_profile), sort_keys=True)),
            "base_model": json.loads(json.dumps(dict(base_model), sort_keys=True)),
            "lora_target_count": 312,
            "lora_trainable_parameters": 2_075_394_048,
            "optimizer": "fp32_master_adamw",
            "ema": "rank_local_fp32_student_start39" if stage == "stage2" else "rank_local_fp32_start0",
            **stage_extras,
        },
    )


def _assert_encoder_silence_identity(encoder_values: Mapping[str, Any], silence_profile: Mapping[str, Any]) -> None:
    """Reject readable silence semantics outside the selected encoder profile."""

    encoder_extras = encoder_values.get("extras", {})
    expected_silence = encoder_extras.get("silence_artifact_profile") if isinstance(encoder_extras, Mapping) else None
    compatible_duration = expected_silence == h3_silence_profile() and any(
        dict(silence_profile) == h3_silence_profile(pixel_frames=frames) for frames in (153, 158, 170)
    )
    if expected_silence is not None and expected_silence != dict(silence_profile) and not compatible_duration:
        raise BackendContractError("H3 encoder contract was produced with a different silence artifact")


def _collective_failures(
    dist: Any,
    topology: Topology,
    local_error: str,
) -> list[str]:
    errors = [local_error]
    if dist.is_initialized():
        errors = [""] * topology.raw_world_size
        dist.all_gather_object(errors, local_error)
    return [error for error in errors if error]


def _collective_call(
    call: Any,
    *,
    dist: Any,
    topology: Topology,
    label: str,
) -> Any:
    """Run rank-local work and report failures before the next collective phase."""

    result = None
    local_error = ""
    try:
        result = call()
    except Exception as exc:
        local_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    failures = _collective_failures(dist, topology, local_error)
    if failures:
        raise BackendContractError(f"H3 distributed {label} failed: " + " | ".join(failures))
    if result is None:
        raise BackendContractError(f"H3 distributed {label} produced no result")
    return result


def _load_adapter_checkpoint(
    path: str,
    lora: Any,
    *,
    torch: Any,
    expected_contract: CheckpointContract,
    broadcast: bool = True,
) -> str:
    verified = verify_checkpoint(path)
    assert_resume_compatible(expected_contract, verified.contract)
    adapter = torch.load(verified.path / "adapter.pt", map_location="cpu", weights_only=True)
    if adapter.get("metadata") != lora.metadata():
        raise BackendContractError("inference LoRA metadata differs")
    lora.load_state_dict(adapter["state"], broadcast=broadcast)
    return verified.manifest_digest


def _load_lora_checkpoint(
    path: str,
    lora: Any,
    *,
    weight_source: str,
    broadcast: bool = False,
) -> str:
    """Load a Stage0.5 LoRA package without inheriting training state."""

    source = str(weight_source).strip().lower()
    if source not in {"live", "ema"}:
        raise BackendContractError("H3 inference weight_source must be live or ema")
    checkpoint = Path(path)
    root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if (root / "checkpoint-manifest.json").is_file():
        from .weights import load_initial_weights

        identity = load_initial_weights({"path": str(root), "stage": "stage0p5", "weight_source": source}, lora)
        if broadcast:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                for parameter in lora.parameter_by_key.values():
                    dist.broadcast(parameter.detach(), src=0)
        return identity
    sidecar = root / ("adapter_model.safetensors" if source == "live" else "adapter_model_ema.safetensors")
    if not sidecar.is_file():
        raise BackendContractError(f"H3 {source} sidecar is missing: {sidecar}")
    size = sidecar.stat().st_size
    if size <= 0:
        raise BackendContractError(f"H3 {source} sidecar is empty: {sidecar}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional H3 dependency
        raise BackendContractError("H3 inference requires safetensors") from exc
    values = load_file(str(sidecar), device="cpu")
    lora.load_state_dict(values, broadcast=broadcast)
    if sidecar.stat().st_size != size:
        raise BackendContractError(f"H3 {source} sidecar changed while it was being loaded")
    return f"inventory:file={sidecar.name}:bytes={size}:role={source}"


def run_inference(config: Mapping[str, Any]) -> int:
    if config.get("inference", {}).get("length_policy") == "source":
        from .full_inference import run_source_length_inference

        return run_source_length_inference(config)
    torch, _diffusers, _transformers = require_h3_runtime()
    import torch.distributed as dist

    from .fsdp import initialize_distributed, wrap_h3_fsdp
    from .lora import inject_h3_lora

    stage = str(config["train"]["stage"])
    distributed = config.get("distributed", {})
    configured_sp = int(distributed.get("sequence_parallel_size", 1))
    topology = _topology(sp_size=configured_sp, require_torchrun=True)
    configured_world = int(distributed["world_size"])
    if topology.raw_world_size != configured_world:
        raise BackendContractError(
            f"torchrun WORLD_SIZE={topology.raw_world_size} differs from configured "
            f"H3 inference world_size={configured_world}"
        )
    initialize_distributed(sp_size=topology.sp_size, local_rank=topology.local_rank)
    topology = (
        Topology.from_environ(topology.sp_size)
        if {"WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK"} <= set(os.environ)
        else topology
    )
    device = torch.device("cuda", topology.local_rank)
    identity = rng_identity("minimax_h3", int(config["data"].get("seed", 42)), topology)
    seed_process(identity.model_init_seed)
    modules = load_transformer(config["model"], device=device)
    if stage == "stage1":
        modules.transformer.enable_anyflow(float(config["train"]["anyflow_gate"]))
        modules.fp32_fsdp_units = (
            *modules.fp32_fsdp_units,
            modules.transformer.delta_embedding.linear_1,
            modules.transformer.delta_embedding.linear_2,
        )
    modules.transformer.eval().requires_grad_(False)
    base_model = _base_model_load_receipt(config["model"], modules.transformer)
    model, lora = inject_h3_lora(
        modules.transformer,
        config["model"]["adapter"],
        base_identity=base_model,
    )
    model = wrap_h3_fsdp(
        model,
        local_rank=topology.local_rank,
        transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units,
        ignored_parameters=lora.parameters,
        activation_checkpointing=False,
    )
    sample_count, num_waves = _validation_schedule(config["validation"], topology)
    stream = _reader(
        config,
        topology,
        index_field="test_index",
        fixed_validation=True,
        fixed_validation_sample_count=sample_count,
    )
    fixed_cases: list[tuple[int, Any, InferenceCase]] = []
    try:
        silence, silence_profile = load_silence_latents(
            str(config["data"]["silence_latents_path"]),
            pixel_frames=153 if stage == "stage1" else 158,
        )
        _assert_encoder_silence_identity(stream.encoder_profile, silence_profile)
        expected_contract = _checkpoint_contract(
            encoder_profile=stream.encoder_profile,
            silence_profile=silence_profile,
            base_model=base_model,
        )
        checkpoint_path = str(
            config.get("checkpoint", {}).get("resume_from") or config["runtime"].get("weights_checkpoint") or ""
        ).strip()
        checkpoint_cfg = config.get("checkpoint", {})
        checkpoint_format = str(checkpoint_cfg.get("format", "solarwm")).strip().lower()
        if checkpoint_path and stage != "stage0p5":
            from .weights import load_initial_weights

            weights_id = _collective_call(
                lambda: load_initial_weights(
                    {
                        "path": checkpoint_path,
                        "stage": stage,
                        "weight_source": checkpoint_cfg.get("weight_source", "live"),
                    },
                    lora,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        elif checkpoint_path and checkpoint_format == "solarwm_lora":
            weights_id = _collective_call(
                lambda: _load_lora_checkpoint(
                    checkpoint_path,
                    lora,
                    weight_source=str(checkpoint_cfg.get("weight_source", "")),
                    broadcast=False,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        elif checkpoint_path:
            weights_id = _collective_call(
                lambda: _load_adapter_checkpoint(
                    checkpoint_path,
                    lora,
                    torch=torch,
                    expected_contract=expected_contract,
                    broadcast=False,
                ),
                dist=dist,
                topology=topology,
                label="inference checkpoint restore",
            )
        else:
            weights_id = _base_weights_label(config["model"])
        if hasattr(stream, "prepare"):
            stream.prepare(dist)
        for wave_index in range(num_waves):
            batch = _collective_call(
                stream.next,
                dist=dist,
                topology=topology,
                label=f"inference data wave {wave_index}",
            )
            expected_slot = wave_index * int(topology.dp_world_size) + int(topology.dp_rank)
            identity_error = ""
            if batch.validation_noise_seed is None or batch.validation_slot is None:
                identity_error = f"rank {topology.raw_rank}: H3 inference batch lacks selected slot/noise"
            elif int(batch.validation_slot) != expected_slot:
                identity_error = (
                    f"rank {topology.raw_rank}: H3 inference wave {wave_index} "
                    f"expected slot {expected_slot}, got {batch.validation_slot}"
                )
            identity_failures = _collective_failures(dist, topology, identity_error)
            if identity_failures:
                raise BackendContractError("H3 inference selection failed: " + " | ".join(identity_failures))
            seed = int(batch.validation_noise_seed)
            case = InferenceCase(
                slot=expected_slot,
                sample_id=batch.sample_id,
                prompt="",
                start_frame=batch.start_frame,
                noise_seed=seed,
                camera_fingerprint=camera_fingerprint(batch),
                metadata={
                    "key": batch.sample_id,
                    "dataset": batch.dataset_source or batch.sample_id.split("/")[0],
                    "plan_fingerprint": batch.plan_fingerprint,
                    "source_pixel_frames": 158,
                    "output_pixel_frames": 158,
                    "train_latent_frames": 45 if stage == "stage1" else 47,
                    "rollout_latent_frames": 47 if stage == "stage0p5" else 50,
                    "generation_mode": "bidirectional" if stage == "stage0p5" else "autoregressive",
                    "sample_solver": {
                        "stage0p5": "shifted-euler-data-ward",
                        "stage1": "flowmap",
                        "stage2": "self_forcing",
                    }[stage],
                    "camera_translation_transform": "logd4",
                    "artifact_valid": True,
                },
            )
            gather_and_assert_sp_identity(
                {
                    "sample_id": batch.sample_id,
                    "start_frame": batch.start_frame,
                    "noise_seed": seed,
                    "plan_fingerprint": batch.plan_fingerprint,
                },
                sp_size=get_sp_size(),
                group=get_sp_group(),
            )
            fixed_cases.append((wave_index, batch, case))
    finally:
        stream.close()
    if stage == "stage1":
        from .stage1 import H3Stage1Core

        validation_silence, _ = load_silence_latents(str(config["data"]["silence_latents_path"]), pixel_frames=170)
        core = H3Stage1Core(model, silence, device, config["train"], validation_silence)
    elif stage == "stage2":
        from .stage2 import H3SGFCore

        core = H3SGFCore(model, silence, device)
    else:
        core = H3Stage0p5Core(model, silence, device)
    conditioners = None
    comparison_destination = Path(str(config["runtime"]["output_dir"])) / "inference"
    for wave_index, batch, case in fixed_cases:
        generated_latents = _collective_call(
            lambda batch=batch, case=case: core.generate(
                batch,
                noise_seed=case.noise_seed,
                num_inference_steps=int(config["validation"]["num_inference_steps"]),
            ),
            dist=dist,
            topology=topology,
            label=f"inference wave {wave_index} generation",
        )
        output_error = ""
        if get_sp_rank() == 0:
            try:
                if conditioners is None:
                    conditioners = load_conditioners(
                        config["model"],
                        device=device,
                        qwen=False,
                        video_vae=True,
                        audio_vae=False,
                        schedulers=False,
                    )
                packaged = package_generated(
                    generated_latents,
                    video_vae=conditioners.video_vae,
                    device=device,
                    weights_id=weights_id,
                    num_inference_steps=int(config["validation"]["num_inference_steps"]),
                    reference_latents=batch.target_latents,
                    sample_solver=str(case.metadata["sample_solver"]),
                )

                class CachedAdapter:
                    family = "minimax_h3"

                    def __init__(self, sample: Any) -> None:
                        self.sample = sample

                    def generate(self, _case: Any, *, weights_id: str) -> Any:
                        del _case, weights_id
                        return self.sample

                destination = (
                    Path(str(config["runtime"]["output_dir"]))
                    / "inference-parts"
                    / f"wave-{wave_index:03d}"
                    / f"dp-rank-{topology.dp_rank:05d}"
                )
                summary = InferenceEngine(CachedAdapter(packaged)).run(
                    [case],
                    weights_id=weights_id,
                    output_dir=destination,
                )
                publish_comparison_partition(
                    summary.output_dir,
                    comparison_destination,
                    step=0,
                    pass_name="inference",
                    cases=sample_count,
                    dp_world_size=int(topology.dp_world_size),
                    sp_size=int(topology.sp_size),
                    run_root=Path(str(config["runtime"]["output_dir"])),
                    logical_world_size_per_round=int(topology.dp_world_size),
                )
            except Exception as exc:
                output_error = (
                    f"rank {topology.raw_rank}/dp {topology.dp_rank} wave {wave_index}: {type(exc).__name__}: {exc}"
                )
        output_failures = _collective_failures(dist, topology, output_error)
        if output_failures:
            raise BackendContractError("H3 distributed inference output failed: " + " | ".join(output_failures))
    complete_error = ""
    if int(topology.local_rank) == 0:
        try:
            expected_local_slots = tuple(
                wave_index * int(topology.dp_world_size)
                + int(topology.node_id) * int(topology.local_dp_world_size)
                + local_dp_rank
                for wave_index in range(num_waves)
                for local_dp_rank in range(int(topology.local_dp_world_size))
            )
            _publish_h3_validation_complete(
                comparison_destination,
                step=0,
                pass_name="inference",
                expected_local_slots=expected_local_slots,
                global_slots=sample_count,
            )
        except Exception as exc:
            complete_error = f"rank {topology.raw_rank}: {type(exc).__name__}: {exc}"
    complete_failures = _collective_failures(dist, topology, complete_error)
    if complete_failures:
        raise BackendContractError("H3 distributed inference commit failed: " + " | ".join(complete_failures))
    return 0


__all__ = ["run_inference"]
