"""Wan2.2 TI2V-5B Stage2 checkpoint loading and self-forcing inference."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ....checkpoint import verify_checkpoint
from ....errors import BackendContractError
from ..sgf import (
    warp_denoising_steps,
)
from .assets import WanAssetLayout
from .components import (
    Wan5BVAE,
    WanTextEncoder,
    build_diffusion_architecture,
)
from .distributed import initialize_torchrun
from .stage0p5 import expand_timesteps_to_tokens

_STREAMING_VAE_LATENT_CHUNK = 60
_FILE_STREAMING_PIXEL_THRESHOLD = 4096
_FILE_STREAMING_VAE_LATENT_CHUNK = 12


def _slice_camera(
    camera: Mapping[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    frame_sequence_length: int,
) -> dict[str, Any]:
    start = int(start_frame) * int(frame_sequence_length)
    end = int(end_frame) * int(frame_sequence_length)
    return {
        "viewmats": camera["viewmats"][:, start:end],
        "K": camera["K"][:, start:end],
    }


def _restore_first(value: Any, first: Any, *, start_frame: int) -> Any:
    if int(start_frame) != 0:
        return value
    result = value.clone()
    result[:, :1] = first
    return result


def _generation_steps(provider: Any) -> tuple[Any, ...]:
    configured = getattr(provider, "denoising_steps", None)
    if configured is not None:
        return tuple(configured)
    train = provider.config["train"]
    return warp_denoising_steps(
        train["denoising_step_list"],
        provider.diffusion.scheduler.timesteps.to(provider.device),
        num_train_timesteps=int(train["num_train_timesteps"]),
    )


def _stage2_self_forcing_latents(
    provider: Any,
    generation_pass: Any,
    first_latent: Any,
    condition: Mapping[str, Any],
    camera: Mapping[str, Any],
    generator: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Run the NFE4 sampler over one persistent rolling six-chunk cache."""

    import torch

    if str(generation_pass.solver) != "self_forcing" or int(generation_pass.num_inference_steps) != 4:
        raise BackendContractError("Stage2 generation supports self_forcing NFE4 only")
    latent_frames = int(generation_pass.rollout_latent_frames)
    chunk = int(provider.config["model"]["num_frame_per_block"])
    if chunk != 3 or latent_frames % chunk:
        raise BackendContractError("Stage2 generation horizon must divide into three-latent chunks")
    channels = int(provider.config["model"]["latent_channels"])
    latent_height = int(provider.config["data"]["latent_shape"][-2])
    latent_width = int(provider.config["data"]["latent_shape"][-1])
    frame_tokens = int(provider.config["model"]["frame_sequence_length"])
    output = torch.zeros(
        1,
        latent_frames,
        channels,
        latent_height,
        latent_width,
        device=provider.device,
        dtype=torch.bfloat16,
    )
    output[:, 0] = first_latent[:, 0]
    initial_noise = provider._noise(tuple(output.shape), generator)
    initial_noise[:, 0] = first_latent[:, 0]
    steps = _generation_steps(provider)
    if len(steps) != 4:
        raise BackendContractError("Stage2 shifted generation schedule must have four steps")
    kv_cache = provider.allocate_kv_cache(1, dtype=output.dtype, device=provider.device)
    crossattn_cache = provider.allocate_crossattn_cache(1, dtype=output.dtype, device=provider.device)
    if kv_cache and "_fused_prope_camera_metadata" not in kv_cache[0]:
        raise BackendContractError("Stage2 generation requires preallocated fused camera metadata")
    try:
        for start in range(0, latent_frames, chunk):
            end = start + chunk
            latents = initial_noise[:, start:end].clone()
            camera_chunk = _slice_camera(
                camera,
                start_frame=start,
                end_frame=end,
                frame_sequence_length=frame_tokens,
            )
            for step_index, step in enumerate(steps):
                timestep = torch.full(
                    (1, chunk),
                    float(step.item()),
                    device=provider.device,
                    dtype=output.dtype,
                )
                if start == 0:
                    timestep[:, 0] = 0.0
                with torch.autocast(
                    device_type=provider.device.type,
                    dtype=torch.bfloat16,
                    enabled=provider.device.type == "cuda",
                ):
                    flow = provider.diffusion(
                        latents,
                        condition,
                        camera_chunk,
                        expand_timesteps_to_tokens(timestep, frame_tokens),
                        sequence_length=chunk * frame_tokens,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start * frame_tokens,
                        cache_start=0,
                        cache_update_policy="none",
                    )
                    x0 = provider.diffusion.flow_to_x0(latents, flow, timestep)
                if start == 0:
                    x0 = _restore_first(x0, first_latent, start_frame=0)
                if step_index + 1 < len(steps):
                    next_timestep = torch.full(
                        (1, chunk),
                        float(steps[step_index + 1].item()),
                        device=provider.device,
                        dtype=torch.float32,
                    )
                    if start == 0:
                        next_timestep[:, 0] = 0.0
                    renoise = provider._noise(tuple(x0.shape), generator)
                    latents = (
                        provider.diffusion.scheduler.add_noise(
                            x0.flatten(0, 1).float(),
                            renoise.flatten(0, 1).float(),
                            next_timestep.flatten(0, 1),
                        )
                        .unflatten(0, (1, chunk))
                        .to(output.dtype)
                    )
                    if start == 0:
                        latents = _restore_first(latents, first_latent, start_frame=0)
                else:
                    latents = x0
            if not bool(torch.isfinite(latents).all().item()):
                raise BackendContractError(f"Stage2 generation chunk {start // chunk} is non-finite")
            output[:, start:end] = latents
            commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=provider.device.type,
                    dtype=torch.bfloat16,
                    enabled=provider.device.type == "cuda",
                ),
            ):
                provider.diffusion(
                    latents,
                    condition,
                    camera_chunk,
                    expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=0,
                    cache_update_policy="commit_detached",
                )
    finally:
        del kv_cache, crossattn_cache
    return output, {
        "schema": "solarwm.wan22-stage2-self-forcing-schedule.v1",
        "solver": "self_forcing",
        "timesteps": [float(value.item()) for value in steps],
        "chunk_latent_frames": chunk,
        "persistent_kv_cache": True,
        "cache_update_policy": "commit_detached",
        "camera_attention_mode": "fused_prope",
    }


def _stage2_generated_sample(
    provider: Any,
    case: Any,
    *,
    weights_id: str,
) -> Any:
    """Materialize one Stage2 sample for either validation or standalone infer."""

    import torch

    from ....config.loader import canonical_json
    from ....inference import GeneratedSample
    from ..generation import GenerationPass
    from .inference import _encode_compare_mp4, _encode_mp4, _encode_stage2_streaming

    metadata = case.metadata.get("generation_pass")
    if not isinstance(metadata, Mapping):
        raise BackendContractError("Stage2 case lacks generation_pass metadata")
    generation_pass = GenerationPass(
        name=str(metadata["name"]),
        weights=str(metadata["weights"]),
        mode=str(metadata["mode"]),
        solver=str(metadata["solver"]),
        num_inference_steps=int(metadata["num_inference_steps"]),
        rollout_latent_frames=int(metadata["rollout_latent_frames"]),
        min_rollout_latent_frames=int(
            metadata.get(
                "min_rollout_latent_frames",
                metadata["rollout_latent_frames"],
            )
        ),
        fixed_plan_pixel_frames=int(
            metadata.get(
                "fixed_plan_pixel_frames",
                1 + 4 * (int(metadata["rollout_latent_frames"]) - 1),
            )
        ),
        variable_rollout_by_source=bool(metadata.get("variable_rollout_by_source", False)),
    )
    if generation_pass.mode != "autoregressive":
        raise BackendContractError("Stage2 self forcing requires autoregressive mode")
    rng = torch.Generator(device=provider.device)
    rng.manual_seed(int(case.noise_seed))
    first, condition, camera, model_y = provider._conditions(case, latent_frames=generation_pass.rollout_latent_frames)
    if model_y is not None:
        raise BackendContractError("Stage2 TI2V generation may not receive I2V y")
    with torch.no_grad():
        latents, schedule = _stage2_self_forcing_latents(
            provider,
            generation_pass,
            first,
            condition,
            camera,
            rng,
        )
        output_latent_frames = int(
            metadata.get(
                "output_rollout_latent_frames",
                generation_pass.rollout_latent_frames,
            )
        )
        output_latents = latents[:, :output_latent_frames].contiguous()
        prepared = provider._prepared.get(case.slot)
        if prepared is None:
            raise BackendContractError(f"Stage2 adapter has no comparison input for slot {case.slot}")
        model_output_pixel_frames = 1 + 4 * (output_latent_frames - 1)
        publication_pixel_frames = getattr(prepared, "publication_pixel_frames", None)
        target_pixel_frames = (
            int(publication_pixel_frames) if publication_pixel_frames is not None else model_output_pixel_frames
        )
        if (
            publication_pixel_frames is not None
            and not bool(getattr(prepared, "repeat_first_frame", False))
            and int(prepared.pixels.shape[0]) != target_pixel_frames
        ):
            raise BackendContractError("camera publication GT length differs from the requested output length")

        camera_artifact: bytes | None = None
        publication_provenance: dict[str, Any] = {}
        if publication_pixel_frames is not None:
            c2w = getattr(prepared, "publication_c2w", None)
            if (
                c2w is None
                or c2w.shape != (target_pixel_frames, 4, 4)
                or c2w.dtype != np.float64
                or not np.isfinite(c2w).all()
            ):
                raise BackendContractError(
                    "camera publication requires finite FP64 absolute C2W for every output frame"
                )
            buffer = io.BytesIO()
            np.save(buffer, c2w, allow_pickle=False)
            camera_artifact = buffer.getvalue()
            publication_provenance = {
                "camera_convention": "authoritative_absolute_c2w",
                "model_output_pixel_frames": model_output_pixel_frames,
                "published_pixel_frames": target_pixel_frames,
                "trim_frames": max(model_output_pixel_frames - target_pixel_frames, 0),
                "tail_pad_frames": max(target_pixel_frames - model_output_pixel_frames, 0),
                "tail_padding": (
                    "repeat_last_generated_frame" if target_pixel_frames > model_output_pixel_frames else "none"
                ),
            }

        encoder = getattr(provider, "video_encoder", None)
        if target_pixel_frames > _FILE_STREAMING_PIXEL_THRESHOLD:
            if callable(encoder):
                raise BackendContractError("hour-scale Stage2 generation requires the released streaming encoder")
            runtime_output = Path(str(provider.config["runtime"]["output_dir"])).expanduser()
            streamed = _encode_stage2_streaming(
                provider.vae,
                output_latents,
                prepared,
                target_pixel_frames=target_pixel_frames,
                fps=float(provider.config["data"].get("fps", 16.0)),
                temporary_root=runtime_output.resolve().parent,
                chunk_latent_frames=_FILE_STREAMING_VAE_LATENT_CHUNK,
            )
            artifacts: dict[str, Any] = {
                "compare.mp4": streamed.compare,
                "video.mp4": streamed.video,
                "schedule.json": canonical_json(dict(schedule)),
            }
            shape = streamed.shape
            finite_fraction = 1.0
            vae_decode = {
                "mode": "continuous_cached_file_tiles",
                "chunk_latent_frames": _FILE_STREAMING_VAE_LATENT_CHUNK,
            }
        else:
            if output_latent_frames > _STREAMING_VAE_LATENT_CHUNK:
                decoded = provider.vae.decode_streaming(
                    output_latents,
                    chunk_latent_frames=_STREAMING_VAE_LATENT_CHUNK,
                )
                vae_decode = {
                    "mode": "continuous_cached_tiles",
                    "chunk_latent_frames": _STREAMING_VAE_LATENT_CHUNK,
                }
            else:
                decoded = provider.vae.decode(output_latents, use_cache=False)
                vae_decode = {
                    "mode": "direct",
                    "chunk_latent_frames": output_latent_frames,
                }
            if int(decoded.shape[1]) != model_output_pixel_frames:
                raise BackendContractError(
                    "Stage2 VAE frame count differs from latent-aligned output: "
                    f"{int(decoded.shape[1])} != {model_output_pixel_frames}"
                )
            finite_fraction = float(torch.isfinite(decoded).float().mean().item())
            if finite_fraction != 1.0:
                raise BackendContractError("Stage2 VAE decode produced non-finite pixels")
            trim_frames = max(int(decoded.shape[1]) - target_pixel_frames, 0)
            tail_pad_frames = max(target_pixel_frames - int(decoded.shape[1]), 0)
            if trim_frames:
                decoded = decoded[:, :target_pixel_frames].contiguous()
            elif tail_pad_frames:
                tail = decoded[:, -1:].expand(
                    -1,
                    tail_pad_frames,
                    -1,
                    -1,
                    -1,
                )
                decoded = torch.cat((decoded, tail), dim=1).contiguous()
            video = (
                encoder(decoded, fps=float(provider.config["data"].get("fps", 16.0)))
                if callable(encoder)
                else _encode_mp4(decoded, fps=float(provider.config["data"].get("fps", 16.0)))
            )
            compare = _encode_compare_mp4(
                decoded,
                prepared,
                fps=float(provider.config["data"].get("fps", 16.0)),
            )
            artifacts = {
                "compare.mp4": compare,
                "video.mp4": video,
                "schedule.json": canonical_json(dict(schedule)),
            }
            shape = tuple(int(value) for value in decoded.shape)

    if camera_artifact is not None:
        artifacts["camera.npy"] = camera_artifact
    return GeneratedSample(
        artifacts=artifacts,
        shape=shape,
        dtype="float32",
        metrics={"finite_fraction": finite_fraction},
        provenance={
            "generation_pass": asdict(generation_pass),
            "weights_id": str(weights_id),
            "denoising_step_list": [int(value) for value in provider.config["train"]["denoising_step_list"]],
            "camera_translation_transform": str(provider.config["model"]["camera_translation_transform"]),
            "resolved_weights_role": str(getattr(provider, "_model_weight_role", generation_pass.weights)),
            "vae_decode": vae_decode,
            **publication_provenance,
        },
    )


def _verified_stage2_inference_checkpoint(
    config: Mapping[str, Any],
    source: str | Path,
) -> tuple[Path, str, int]:
    """Verify the portable Stage2 inference transaction before allocation."""

    requested = Path(source).expanduser().resolve()
    if requested.is_dir():
        root = requested
        requested = root / "model.pt"
    else:
        root = requested.parent
    if requested.name != "model.pt":
        raise BackendContractError("Stage2 inference checkpoint must name model.pt or its transaction directory")
    try:
        verified = verify_checkpoint(root)
    except Exception as exc:
        raise BackendContractError(
            f"Stage2 inference checkpoint verification failed: {type(exc).__name__}: {exc}"
        ) from exc
    members = {record.path for record in verified.files}
    if members != {"model.pt"}:
        raise BackendContractError("Stage2 inference checkpoint must contain exactly model.pt")
    metadata = verified.metadata
    if (
        metadata.get("schema") != "solarwm.wan22-stage2-inference.v1"
        or metadata.get("available_weights") != ["live", "ema"]
        or metadata.get("ema_present") is not True
    ):
        raise BackendContractError("Stage2 inference checkpoint metadata differs")
    contract = verified.contract
    model = config["model"]
    train = config["train"]
    expected_header = {
        "family": str(model["family"]),
        "stage": "stage2",
        "causal_mode": "self_gradient_forcing",
        "objective": "flow_matching",
        "objective_variant": "sgf_v1",
        "camera_translation_transform": str(model["camera_translation_transform"]),
        "parameterization": "full-parameter-live-ema",
        "sp_size": 1,
        "data_generation": "inference-portable:v1",
    }
    header_drift = {
        name: {"actual": getattr(contract, name), "expected": expected}
        for name, expected in expected_header.items()
        if getattr(contract, name) != expected
    }
    extras = contract.extras
    expected_extras = {
        "denoising_step_list": list(train["denoising_step_list"]),
        "attention": {
            "local_attn_size": int(model["local_attn_size"]),
            "max_prior_clean_chunks": int(model["max_prior_clean_chunks"]),
            "sink_size": int(model["sink_size"]),
            "rope_train_frames": int(model["rope_train_frames"]),
            "use_echorope": bool(model["use_echorope"]),
        },
    }
    extras_drift = {
        name: {"actual": extras.get(name), "expected": expected}
        for name, expected in expected_extras.items()
        if extras.get(name) != expected
    }
    if header_drift or extras_drift:
        raise BackendContractError(
            f"Stage2 inference checkpoint contract differs: header={header_drift} extras={extras_drift}"
        )
    if not requested.is_file():
        raise BackendContractError("Stage2 verified transaction has no visible model.pt")
    return requested, verified.manifest_digest, int(verified.step)


def _published_default_weight_role(source: str | Path) -> str:
    """Resolve the release-selected live/EMA role without exposing a CLI choice."""

    requested = Path(source).expanduser().resolve()
    root = requested if requested.is_dir() else requested.parent
    manifest_path = root / "release-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackendContractError(f"Stage2 camera-length checkpoint lacks a readable release manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise BackendContractError("Stage2 release manifest must be a mapping")
    load = manifest.get("load")
    identity = manifest.get("identity")
    model = identity.get("model") if isinstance(identity, Mapping) else None
    if (
        manifest.get("schema") != "solarwm.public-weight-manifest.v1"
        or not isinstance(load, Mapping)
        or load.get("format") != "solarwm_wan_stage2_transaction_v1"
        or load.get("entrypoint") != "."
        or not isinstance(model, Mapping)
    ):
        raise BackendContractError("Stage2 release manifest contract differs")
    role = str(load.get("default_weights", "")).strip().lower()
    available = str(model.get("weight_role", "")).strip().lower().split("+")
    if role not in {"live", "ema"} or role not in available:
        raise BackendContractError("Stage2 release default weight role is invalid")
    return role


class Wan5BStage2Runtime:
    """Stage2 cache allocators shared by the standalone CUDA generation adapter."""

    @staticmethod
    def _root(module: Any) -> Any:
        return getattr(module, "module", module)

    def allocate_kv_cache(self, batch_size: int, *, dtype: Any, device: Any) -> list[dict[str, Any]]:
        """Allocate the six-chunk no-sink cache and camera metadata up front."""

        if self._kv_cache_allocator is not None:
            return self._kv_cache_allocator(batch_size, dtype=dtype, device=device)
        import torch

        root = self._root(self.student.module)
        blocks = tuple(getattr(root, "blocks", ()))
        num_heads = int(getattr(root, "num_heads", 0))
        dim = int(getattr(root, "dim", 0))
        if not blocks or num_heads <= 0 or dim <= 0 or dim % num_heads:
            raise BackendContractError("Stage2 cannot infer the Wan KV-cache geometry")
        head_dim = dim // num_heads
        cache_frames = int(self.config["model"]["local_attn_size"])
        cache_tokens = cache_frames * int(self.config["model"]["frame_sequence_length"])
        caches = [
            {
                "k": torch.zeros(
                    batch_size,
                    cache_tokens,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    cache_tokens,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
            }
            for _ in blocks
        ]
        caches[0]["_fused_prope_camera_metadata"] = {
            "viewmats": torch.zeros(
                batch_size,
                cache_tokens,
                4,
                4,
                dtype=torch.float32,
                device=device,
            ),
            "K": torch.zeros(
                batch_size,
                cache_tokens,
                3,
                3,
                dtype=torch.float32,
                device=device,
            ),
        }
        return caches

    def allocate_crossattn_cache(self, batch_size: int, *, dtype: Any, device: Any) -> list[dict[str, Any]]:
        if self._crossattn_cache_allocator is not None:
            return self._crossattn_cache_allocator(batch_size, dtype=dtype, device=device)
        import torch

        root = self._root(self.student.module)
        blocks = tuple(getattr(root, "blocks", ()))
        num_heads = int(getattr(root, "num_heads", 0))
        dim = int(getattr(root, "dim", 0))
        if not blocks or num_heads <= 0 or dim % num_heads:
            raise BackendContractError("Stage2 cannot infer the Wan cross-attention cache geometry")
        head_dim = dim // num_heads
        return [
            {
                "k": torch.zeros(
                    batch_size,
                    512,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    512,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False,
            }
            for _ in blocks
        ]


class CudaWanStage2GenerationAdapter:
    """Lazy facade over the common CUDA adapter with the Stage2 sampler."""

    def __new__(cls, config: Mapping[str, Any], plan: Any) -> Any:
        # Delaying the subclass definition avoids pulling CUDA/model imports
        # into CPU-only contract tests that merely import this module.
        from .inference import CudaWanGenerationAdapter

        class _Adapter(CudaWanGenerationAdapter):
            video_encoder = None

            def __init__(self, values: Mapping[str, Any], generation_plan: Any) -> None:
                import torch

                if not torch.cuda.is_available():
                    raise BackendContractError("Wan Stage2 inference requires CUDA")
                self.config = values
                self.plan = generation_plan
                self.family = str(values["model"]["family"])
                self._direct_model = str(values.get("inference", {}).get("length", "fixed")).strip().lower() == "camera"
                self.topology = initialize_torchrun(int(values["distributed"]["sequence_parallel_size"]))
                self.is_writer = int(self.topology.sp_rank) == 0
                self.device = torch.device("cuda", int(self.topology.local_rank))
                (
                    self.checkpoint_path,
                    checkpoint_manifest_id,
                    self.checkpoint_step,
                ) = _verified_stage2_inference_checkpoint(
                    values,
                    str(values["checkpoint"]["path"]),
                )
                self._model_weight_role = (
                    _published_default_weight_role(values["checkpoint"]["path"]) if self._direct_model else None
                )
                self.checkpoint_id = f"manifest:{checkpoint_manifest_id}"
                layout = WanAssetLayout.from_config(values)
                self.diffusion = build_diffusion_architecture(values)
                self.text_encoder = WanTextEncoder(layout.text_encoder, layout.tokenizer)
                self.vae = Wan5BVAE(layout.vae)
                inference_dtype = torch.bfloat16
                self.diffusion.module.eval().requires_grad_(False).to(device=self.device, dtype=inference_dtype)
                self.text_encoder.to(self.device, dtype=inference_dtype)
                self.vae.to(self.device, dtype=inference_dtype)
                self._loaded_role: str | None = None
                self._prepared: dict[int, Any] = {}
                self._deferred_camera_inputs = None

            @staticmethod
            def _root(module: Any) -> Any:
                return getattr(module, "module", module)

            def weight_id(self, role: str) -> str:
                if role == "model" and self._direct_model:
                    return f"{self.checkpoint_id}#release-default:{self._model_weight_role}"
                return super().weight_id(role)

            def _checkpoint_state_field(self, role: str) -> str:
                if role == "model" and self._direct_model:
                    return super()._checkpoint_state_field(str(self._model_weight_role))
                return super()._checkpoint_state_field(role)

            def allocate_kv_cache(self, batch_size: int, *, dtype: Any, device: Any) -> list[dict[str, Any]]:
                self.student = self.diffusion
                self._kv_cache_allocator = None
                return Wan5BStage2Runtime.allocate_kv_cache(self, batch_size, dtype=dtype, device=device)

            def allocate_crossattn_cache(self, batch_size: int, *, dtype: Any, device: Any) -> list[dict[str, Any]]:
                self.student = self.diffusion
                self._crossattn_cache_allocator = None
                return Wan5BStage2Runtime.allocate_crossattn_cache(self, batch_size, dtype=dtype, device=device)

            def generate(self, case: Any, *, weights_id: str) -> Any:
                metadata = case.metadata.get("generation_pass")
                if not isinstance(metadata, Mapping):
                    raise BackendContractError("Stage2 generation case lacks pass metadata")
                role = str(metadata.get("weights", "")).strip().lower()
                expected = self.weight_id(role)
                if str(weights_id) != expected:
                    raise BackendContractError("Stage2 standalone generation weights identity drift")
                self._load_role(role)
                camera_length = str(self.config.get("inference", {}).get("length", "fixed")).strip().lower() == "camera"
                deferred = self._deferred_camera_inputs
                if camera_length and deferred is None:
                    raise BackendContractError("Stage2 camera-length inputs were not prepared")
                if not camera_length:
                    return _stage2_generated_sample(self, case, weights_id=weights_id)
                self._materialize_deferred_camera_case(case)
                try:
                    return _stage2_generated_sample(self, case, weights_id=weights_id)
                finally:
                    self._prepared.pop(case.slot, None)

        return _Adapter(config, plan)


def build_stage2_generation_provider(config: Mapping[str, Any]) -> Any:
    """Build the standalone Stage2 provider used by unified Wan inference."""

    from ..generation import resolve_generation_plan

    if (
        str(config.get("model", {}).get("family", "")) != "wan22_ti2v_5b"
        or str(config.get("train", {}).get("stage", "")) != "stage2"
        or str(config.get("train", {}).get("objective", "")) != "flow_matching"
    ):
        raise BackendContractError("Stage2 inference provider contract differs")
    plan = resolve_generation_plan(config)
    if any(
        item.mode != "autoregressive" or item.solver != "self_forcing" or item.num_inference_steps != 4
        for item in plan.passes
    ):
        raise BackendContractError("Stage2 inference requires autoregressive self_forcing NFE4")
    return CudaWanStage2GenerationAdapter(config, plan)


def run_stage2_inference(config: Mapping[str, Any]) -> Any:
    """Run standalone Stage2 inference through the validation-identical core."""

    from .inference import run_wan_inference

    provider = build_stage2_generation_provider(config)
    try:
        return run_wan_inference(config, provider=provider)
    finally:
        provider.close()


__all__ = [
    "CudaWanStage2GenerationAdapter",
    "Wan5BStage2Runtime",
    "build_stage2_generation_provider",
    "run_stage2_inference",
]
