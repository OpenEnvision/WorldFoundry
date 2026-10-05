# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from transformers import AutoTokenizer, UMT5EncoderModel

from .output import sha256, save_chunk_state, assemble_resumed_video
from worldfoundry.base_models.diffusion_model.runners.worldcrafter import WorldCrafterPipeline
from worldfoundry.base_models.diffusion_model.schedulers.worldcrafter import WorldCrafterScheduler
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.transformer import WorldCrafterTransformer3DModel
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.repencoder import (
    RepEncoder,
    RepEncoderInferenceMemoryProvider,
    RepEncoderInferenceProviderConfig,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.bridge import (
    enable_ucpe_inference_sdpa_attention,
    load_ucpe_camera_adapter_weights,
    patch_worldcrafter_transformer_ucpe,
)


CAMERA_CHUNK_FRAMES = 33
MODEL_HEIGHT = 384
MODEL_WIDTH = 640


@dataclass(frozen=True)
class InferenceResult:
    video_path: Path
    summary_path: Path
    summary: dict[str, object]


def load_camera(path: Path, num_chunks: int | None = None) -> np.ndarray:
    pose = np.load(path, allow_pickle=False)
    if pose.ndim == 3:
        pose = pose[None]
    if pose.ndim != 4 or pose.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"camera must be [B,T,3,4] or [B,T,4,4], got {pose.shape}")
    if pose.shape[0] != 1 or pose.shape[1] == 0 or pose.shape[1] % CAMERA_CHUNK_FRAMES:
        raise ValueError(
            "WorldCrafter requires one camera trajectory containing complete 33-frame chunks"
        )
    if not np.issubdtype(pose.dtype, np.floating) or not np.isfinite(pose).all():
        raise ValueError("camera must contain finite floating-point c2w matrices")
    if pose.shape[-2:] == (4, 4) and not np.allclose(pose[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("camera homogeneous bottom row must be [0,0,0,1]")
    rotation = np.asarray(pose[..., :3, :3], dtype=np.float64)
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    if not np.allclose(gram, np.eye(3), atol=5e-3, rtol=0.0):
        raise ValueError("camera rotations are not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1.0, atol=5e-3, rtol=0.0):
        raise ValueError("camera rotations must have determinant +1")
    if num_chunks is not None:
        if isinstance(num_chunks, bool) or int(num_chunks) != num_chunks or num_chunks < 1:
            raise ValueError("num_chunks must be a positive integer")
        frames = int(num_chunks) * CAMERA_CHUNK_FRAMES
        if pose.shape[1] < frames:
            raise ValueError(
                f"camera has {pose.shape[1] // CAMERA_CHUNK_FRAMES} chunks, "
                f"but {num_chunks} were requested"
            )
        pose = pose[:, :frames]
    return np.ascontiguousarray(pose)


def validate_weights(model_path: Path, shared_model_path: Path | None = None) -> dict[str, Path]:
    root = model_path.expanduser().resolve()
    config_path = root / "inference_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    shared = (root / config.get("shared_components", ".")).resolve()
    if shared_model_path is not None:
        shared = Path(shared_model_path).expanduser().resolve()
    elif not (shared / "repencoder").is_dir():
        from worldfoundry.core.io.paths import resolve_local_hf_model_path
        shared = Path(resolve_local_hf_model_path("TencentARC/WorldCrafter-Fast", required_files=("repencoder/config.json",)))

    required = {
        "root": root,
        "transformer": root / "transformer",
        "adapter": root / "adapter",
        "repencoder": shared / "repencoder",
        "vae": shared / "vae",
        "scheduler": shared / "scheduler",
        "text_encoder": shared / "text_encoder",
        "tokenizer": shared / "tokenizer",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    for filename in (
        required["adapter"] / "camera_adapter.pth",
        required["adapter"] / "pytorch_lora_weights.safetensors",
        required["repencoder"] / "model.safetensors",
        required["repencoder"] / "config.json",
        required["repencoder"] / "manifest.json",
    ):
        if not filename.is_file():
            missing.append(str(filename))
    if missing:
        raise FileNotFoundError(
            "WorldCrafter-Base is incomplete: " + ", ".join(missing)
        )
    return required


def configure_attention(
    transformer: WorldCrafterTransformer3DModel, backend: str
) -> str:
    if backend != "auto":
        transformer.set_attention_backend(backend)
        return backend
    for candidate in ("native", "_flash_3_hub", "flash_hub"):
        try:
            transformer.set_attention_backend(candidate)
            return candidate
        except (ImportError, RuntimeError, ValueError):
            continue
    raise RuntimeError("no supported attention backend is available")


def load_model_adapter(
    pipe: WorldCrafterPipeline, adapter_path: Path
) -> dict[str, object]:
    from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

    _SET_ADAPTER_SCALE_FN_MAPPING.setdefault(
        WorldCrafterTransformer3DModel.__name__, lambda _model_class, weights: weights
    )
    if adapter_path.is_dir():
        adapter_path = adapter_path / "pytorch_lora_weights.safetensors"
    state = WorldCrafterPipeline.lora_state_dict(
        str(adapter_path.parent), weight_name=adapter_path.name, local_files_only=True
    )
    transformer_keys = [key for key in state if key.startswith("transformer.")]
    if not transformer_keys:
        raise RuntimeError("adapter does not contain transformer low-rank weights")
    name = "worldcrafter"
    pipe.load_lora_weights(state, adapter_name=name)
    pipe.set_adapters([name], adapter_weights=[1.0])
    return {"name": name, "tensor_keys": len(transformer_keys)}


class WorldCrafter:
    def __init__(
        self,
        *,
        pipeline: WorldCrafterPipeline,
        memory_provider: RepEncoderInferenceMemoryProvider,
        model_path: Path,
        device: torch.device,
        attention_backend: str,
        adapter_load: dict[str, object],
        height: int,
        width: int,
    ) -> None:
        self.model_type = "base"
        self.pipeline = pipeline
        self.memory_provider = memory_provider
        self.model_path = model_path
        self.device = device
        self.attention_backend = attention_backend
        self.adapter_load = adapter_load
        self.height = height
        self.width = width

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *,
        model_type: str = "base",
        device: str = "cuda:0",
        height: int = MODEL_HEIGHT,
        width: int = MODEL_WIDTH,
        seed: int = 42,
        memory_fov_h_deg: float = 100.0,
        memory_fov_v_deg: float = 71.13349068444832,
        memory_fov_samples_per_axis: int = 10,
        attention_backend: str = "native",
        enable_compile: bool = False,
        shared_model_path: Path | None = None,
    ) -> "WorldCrafter":
        if model_type == "fast":
            from .model_loading import load_fast

            return load_fast(
                cls,
                model_path,
                device=device,
                height=height,
                width=width,
                seed=seed,
                memory_fov_h_deg=memory_fov_h_deg,
                memory_fov_v_deg=memory_fov_v_deg,
                memory_fov_samples_per_axis=memory_fov_samples_per_axis,
                attention_backend=attention_backend,
                enable_compile=enable_compile,
                shared_model_path=shared_model_path,
            )
        if model_type != "base":
            raise ValueError(f"Unknown model type: {model_type}")
        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("WorldCrafter inference requires CUDA")
        if (height, width) != (MODEL_HEIGHT, MODEL_WIDTH):
            raise ValueError("WorldCrafter-Base is fixed to 384x640 inference")
        torch.cuda.set_device(torch_device)
        paths = validate_weights(model_path, shared_model_path)

        enable_ucpe_inference_sdpa_attention()
        repencoder = RepEncoder.from_pretrained(
            paths["repencoder"],
            device=torch_device,
            compute_dtype="bf16",
            target_microbatch=4,
        )
        memory_provider = RepEncoderInferenceMemoryProvider(
            repencoder,
            RepEncoderInferenceProviderConfig(
                seed=seed,
                trajectory_fov_horizontal_fov_degrees=memory_fov_h_deg,
                trajectory_fov_vertical_fov_degrees=memory_fov_v_deg,
                trajectory_fov_samples_per_axis=memory_fov_samples_per_axis,
            ),
        )

        transformer = WorldCrafterTransformer3DModel.from_pretrained(
            paths["transformer"], torch_dtype=torch.bfloat16
        )
        patch_worldcrafter_transformer_ucpe(
            transformer=transformer,
            method="relray_absmap",
            height=height,
            width=width,
            attn_compress=8,
            adaptation_method="parallel",
        )
        camera_adapter = load_ucpe_camera_adapter_weights(transformer, paths["adapter"])
        if (
            camera_adapter["loaded_tensor_keys"]
            != camera_adapter["expected_tensor_keys"]
        ):
            raise RuntimeError(f"camera adapter load is incomplete: {camera_adapter}")
        adapter_dtypes = {
            parameter.dtype
            for block in transformer.blocks
            for parameter in block.cam_self_attn.parameters()
        }
        if adapter_dtypes != {torch.float32}:
            raise RuntimeError(f"camera adapter dtypes are invalid: {adapter_dtypes}")

        if not enable_compile:
            transformer = replace_rmsnorm_with_fp32(transformer)
            transformer = replace_all_norms_with_flash_norms(transformer)
            replace_rope_with_flash_rope()
        resolved_backend = configure_attention(transformer, attention_backend)
        pipeline = WorldCrafterPipeline(
            tokenizer=AutoTokenizer.from_pretrained(paths["tokenizer"]),
            text_encoder=UMT5EncoderModel.from_pretrained(
                paths["text_encoder"], torch_dtype=torch.bfloat16
            ),
            transformer=transformer,
            vae=AutoencoderKLWan.from_pretrained(
                paths["vae"], torch_dtype=torch.float32
            ),
            scheduler=WorldCrafterScheduler.from_pretrained(paths["scheduler"]),
        )
        adapter_load = load_model_adapter(pipeline, paths["adapter"])
        pipeline = pipeline.to(torch_device)
        if enable_compile:
            torch.backends.cudnn.benchmark = True
            pipeline.text_encoder.compile(
                mode="max-autotune-no-cudagraphs", dynamic=False
            )
            pipeline.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
            pipeline.transformer.compile(
                mode="max-autotune-no-cudagraphs", dynamic=False
            )
        return cls(
            pipeline=pipeline,
            memory_provider=memory_provider,
            model_path=paths["root"],
            device=torch_device,
            attention_backend=resolved_backend,
            adapter_load=adapter_load,
            height=height,
            width=width,
        )

    def generate(
        self,
        *,
        mode: str,
        camera_path: Path,
        output_path: Path,
        prompt: str,
        negative_prompt: str,
        image_path: Path | None = None,
        num_chunks: int | None = None,
        chunk_output_dir: Path | None = None,
        state_output_dir: Path | None = None,
        resume_from: Path | None = None,
        on_chunk_saved: Callable[[int, Path], None] | None = None,
        stop_after_chunk: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int = 42,
        fps: int = 16,
        image_noise_sigma_min: float = 0.111,
        image_noise_sigma_max: float = 0.135,
        camera_x_fov: float = 100.0,
        camera_xi: float = 0.0,
        local_camera_path: Path | None = None,
    ) -> InferenceResult:
        if on_chunk_saved is not None and chunk_output_dir is None:
            raise ValueError("on_chunk_saved requires chunk_output_dir")
        is_fast = self.model_type == "fast"
        if num_inference_steps is None:
            num_inference_steps = 6 if is_fast else 50
        if guidance_scale is None:
            guidance_scale = 1.0 if is_fast else 5.0
        if is_fast:
            if guidance_scale != 1.0 or num_inference_steps != 6:
                raise ValueError(
                    "Fast requires CFG=1 and six regular steps; the first T2V chunk uses twelve steps"
                )
            self.pipeline.resident_branches.switch("equal")
            self.pipeline.resident_branches.switches = 0
            self.pipeline.stage_model_trace.clear()
            self.pipeline.fast_inference_mode = mode
        elif local_camera_path is not None:
            raise ValueError("--local-camera-path is only used by fast inference")
        if mode not in {"i2v", "t2v"}:
            raise ValueError(f"unsupported mode: {mode}")
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        if mode == "i2v":
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = load_image(str(image_path)).resize((self.width, self.height))
        else:
            if image_path is not None:
                raise ValueError(
                    "text-to-video inference does not accept an input image"
                )
            image = None

        camera_c2w = load_camera(camera_path, num_chunks=num_chunks)
        num_frames = int(camera_c2w.shape[1])
        total_chunks = num_frames // CAMERA_CHUNK_FRAMES
        camera = {
            "c2w": camera_c2w,
            "x_fov": torch.full(
                (1,), camera_x_fov, device=self.device, dtype=torch.float32
            ),
            "xi": torch.full((1,), camera_xi, device=self.device, dtype=torch.float32),
        }
        if is_fast:
            if local_camera_path is not None:
                local_c2w = load_camera(local_camera_path, num_chunks=total_chunks)
            else:
                from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.bridge import _relative_pose_chunk

                local_c2w = torch.cat(
                    [
                        _relative_pose_chunk(
                            camera_c2w,
                            chunk_index=k,
                            window_num_frames=33,
                            device=self.device,
                        )
                        for k in range(total_chunks)
                    ],
                    dim=1,
                )
            camera["pose"] = torch.as_tensor(
                local_c2w, device=self.device, dtype=torch.float32
            )
            camera["c2w"] = torch.as_tensor(
                camera_c2w, device=self.device, dtype=torch.float32
            )
        if chunk_output_dir is not None:
            chunk_output_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None and chunk_output_dir is None:
            raise ValueError(
                "resuming requires --chunk-output-dir with completed prefix chunks"
            )
        if state_output_dir is not None:
            state_output_dir.mkdir(parents=True, exist_ok=True)
        final_chunk_index = total_chunks - 1
        if stop_after_chunk is not None:
            if stop_after_chunk < 0 or stop_after_chunk >= total_chunks:
                raise ValueError("stop_after_chunk must identify a generated chunk")
            final_chunk_index = int(stop_after_chunk)

        run_contract = {
            "mode": mode,
            "camera_sha256": sha256(camera_path),
            "image_sha256": sha256(image_path) if image_path is not None else None,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(
                negative_prompt.encode("utf-8")
            ).hexdigest(),
            "num_chunks": total_chunks,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "image_noise_sigma_min": image_noise_sigma_min,
            "image_noise_sigma_max": image_noise_sigma_max,
            "camera_x_fov": camera_x_fov,
            "camera_xi": camera_xi,
            "repencoder_model_sha256": self.memory_provider.runtime.report[
                "model_sha256"
            ],
        }
        if is_fast:
            run_contract["model_type"] = "fast"
        resume_state: dict[str, object] | None = None
        prior_history_selection: list[dict[str, object]] = []
        if resume_from is not None:
            if not resume_from.is_file():
                raise FileNotFoundError(resume_from)
            resume_state = torch.load(
                resume_from, map_location="cpu", weights_only=True
            )
            if resume_state.get("run_contract") != run_contract:
                raise ValueError(
                    "resume checkpoint does not match this inference run contract"
                )
            prior_history_selection = list(resume_state.get("history_selection", []))
            if final_chunk_index < int(resume_state["next_chunk_index"]):
                raise ValueError("stop_after_chunk precedes the resume point")

        def save_chunk(chunk_index: int, current_video: torch.Tensor) -> None:
            if chunk_output_dir is None:
                return
            frames = self.pipeline.video_processor.postprocess_video(
                current_video, output_type="np"
            )[0]
            path = chunk_output_dir / f"chunk_{chunk_index:03d}_33f.mp4"
            export_to_video(frames, str(path), fps=fps)
            if on_chunk_saved is not None:
                on_chunk_saved(chunk_index, path)
            print(f"[worldcrafter] completed {path}", flush=True)

        def save_state(chunk_index: int, state: dict[str, object]) -> None:
            if state_output_dir is not None:
                save_chunk_state(
                    chunk_index,
                    state,
                    state_output_dir=state_output_dir,
                    run_contract=run_contract,
                    history_selection=[
                        *prior_history_selection,
                        *[
                            record.to_jsonable()
                            for record in self.memory_provider.render_records
                        ],
                    ],
                )

        self.memory_provider.reset_sequence()
        with torch.inference_mode():
            frames = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                memory_size=4,
                history_sizes=[2, 1],
                num_latent_frames_per_chunk=9,
                keep_first_frame=True,
                is_enable_stage2=is_fast,
                pyramid_num_inference_steps_list=None if is_fast else [2, 2, 2],
                is_skip_first_chunk=False,
                is_amplify_first_chunk=False,
                use_zero_init=False,
                zero_steps=1,
                image=image,
                image_noise_sigma_min=image_noise_sigma_min,
                image_noise_sigma_max=image_noise_sigma_max,
                video=None,
                video_noise_sigma_min=0.111,
                video_noise_sigma_max=0.135,
                camera_trajectory=camera,
                memory_provider=self.memory_provider,
                callback_on_chunk_end=(
                    save_chunk if chunk_output_dir is not None else None
                ),
                callback_on_chunk_state=(
                    save_state if state_output_dir is not None else None
                ),
                resume_state=resume_state,
                stop_after_chunk=final_chunk_index,
            ).frames[0]

        start_chunk = (
            int(resume_state["next_chunk_index"]) if resume_state is not None else 0
        )
        expected_records = max(0, final_chunk_index - max(1, start_chunk) + 1)
        if len(self.memory_provider.render_records) != expected_records:
            raise RuntimeError(
                f"expected {expected_records} RepEncoder calls, "
                f"got {len(self.memory_provider.render_records)}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if is_fast and len(frames) != (final_chunk_index - start_chunk + 1) * CAMERA_CHUNK_FRAMES:
            raise RuntimeError("Fast output must preserve every decoded RGB frame")
        if resume_state is None:
            export_to_video(frames, str(output_path), fps=fps)
        else:
            assemble_resumed_video(
                output_path, chunk_output_dir, final_chunk_index, fps,
                overlap_frames=0 if is_fast else 1,
            )
        history_selection = [
            *prior_history_selection,
            *[record.to_jsonable() for record in self.memory_provider.render_records],
        ]
        summary = {
            "format": "worldcrafter_inference_v2",
            "mode": mode,
            "model_path": str(self.model_path),
            "image_path": str(image_path) if image_path is not None else None,
            "image_sha256": sha256(image_path) if image_path is not None else None,
            "camera_path": str(camera_path),
            "camera_sha256": sha256(camera_path),
            "camera_semantics": "global metric c2w; UCPE chunk-relative poses are derived internally",
            "output_path": str(output_path),
            "output_sha256": sha256(output_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "completed_through_chunk": final_chunk_index,
            "fps": fps,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "attention_backend": self.attention_backend,
            "adapter": self.adapter_load,
            "repencoder_model_sha256": self.memory_provider.runtime.report[
                "model_sha256"
            ],
            "resumed_from": str(resume_from) if resume_from is not None else None,
            "history_selection": history_selection,
        }
        if is_fast:
            first_chunk_steps = 12 if mode == "t2v" else 6
            expected_calls = (
                first_chunk_steps + final_chunk_index * 6
                if resume_state is None
                else (final_chunk_index - start_chunk + 1) * 6
            )
            if len(self.pipeline.stage_model_trace) != expected_calls:
                raise RuntimeError(
                    "Fast forward count differs from the mode-specific DMD contract"
                )
            summary.update(
                model_type="fast",
                local_camera_path=str(local_camera_path),
                local_camera_sha256=(
                    sha256(local_camera_path) if local_camera_path else None
                ),
                first_chunk_inference_steps=first_chunk_steps,
                first_chunk_routing="4+8" if mode == "t2v" else "5+1",
                subsequent_chunk_routing="2+4" if mode == "t2v" else "5+1",
                fast=self.fast_report,
                stage_model_trace=self.pipeline.stage_model_trace,
                resident_branch_switches=self.pipeline.resident_branches.switches,
            )
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"[worldcrafter] saved {output_path}", flush=True)
        return InferenceResult(output_path, summary_path, summary)


__all__ = ["InferenceResult", "WorldCrafter", "load_camera", "sha256"]
