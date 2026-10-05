"""Run v1.1 generation without trainer setup, optimizers, critics or training data.

The inference-only ViGeo rollout is extracted from the pinned upstream.
No trainer class, setup method, optimizer or training dataloader is imported. Shared
WorldFoundry loaders, history encoder, LTX VAE, text encoding and LoRA merger
own the common components. This is an external-runtime bridge, not a claim of
checkpoint-backed parity for a fully native v1.1 port.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def build_config(request):
    from alaya.config import schema

    # Only inference sections are instantiated; no TrainConfig or optimization defaults.
    sections = {
        "run": schema.RunConfig,
        "paths": schema.PathsConfig,
        "sample": schema.SampleConfig,
        "layout": schema.LayoutConfig,
        "memory": schema.MemoryConfig,
        "spatial_memory": schema.SpatialMemoryConfig,
        "control": schema.ControlConfig,
        "validation": schema.ValidationConfig,
        "runtime": schema.RuntimeConfig,
    }

    raw = yaml.safe_load(Path(request["config"]).read_text())
    raw.pop("student_adapter")
    shift = raw.pop("sigma_shift")
    output = Path(request["output_dir"])
    inputs = output / "inputs"
    raw["run"] = {"seed": request["seed"], "output_dir": str(output / "artifacts"), "name": "worldfoundry-inference"}
    raw["paths"] = {
        "base_transformer": request["base_checkpoint"],
        "vae": request["base_checkpoint"],
        "gemma": request["gemma_path"],
    }
    raw["spatial_memory"].update(
        vigeo_checkpoint=request["vigeo_checkpoint"], vigeo_repo_path=request["vigeo_source_root"]
    )
    raw["runtime"].update(
        fsdp=False, gradient_checkpointing=False, text_embed_cache_dir=None, vae_latent_cache_dir=None
    )
    mode = raw["validation"]["modes"]["custom_i2v"]
    mode["rollout_rounds"] = request["rounds"]
    mode["dataset"].update(
        image_dir=str(inputs / "images"),
        captions_json=str(inputs / "captions.json"),
        pose_jsonl=str(inputs / "pose.jsonl"),
    )
    cfg = SimpleNamespace(**{key: factory() for key, factory in sections.items()})
    schema._update_dataclass(cfg, raw)
    schema._validate_control(cfg.control)
    schema._validate_validation(cfg.validation, allow_empty_modes=False)
    cfg.sigma_shift = SimpleNamespace(**shift)
    return cfg


def _prepare_input(request):
    command = [
        sys.executable,
        str(Path(request["source_root"]) / "scripts/infer/prepare_i2v_inputs.py"),
        "--image",
        request["image"],
        "--prompt",
        request["prompt"],
        "--out",
        str(Path(request["output_dir"]) / "inputs"),
        "--rounds",
        str(request["rounds"]),
    ]
    for key, flag in (
        ("camera_path", "--extrinsics"),
        ("forward", "--forward"),
        ("yaw", "--yaw"),
        ("pitch", "--pitch"),
    ):
        if key in request:
            command.extend((flag, str(request[key])))
    if "intrinsic" in request:
        command.extend(("--intrinsic", *(str(value) for value in request["intrinsic"])))
    subprocess.run(command, check=True, cwd=request["source_root"])


# Inference configuration adapters from the pinned Alaya loader.
def _read_transformer_config(checkpoint_path) -> dict:
    import safetensors

    if not checkpoint_path or not Path(checkpoint_path).exists() or (not checkpoint_path.endswith(".safetensors")):
        return {}
    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        metadata = handle.metadata() or {}
        return json.loads(metadata.get("config", "{}")).get("transformer", {})


def _runtime_transformer_overrides(cfg) -> dict:
    from ltx2.modules.attention import AttentionFunction
    from ltx2.modules.rope import LTXRopeType

    attention_map = {
        "flash_attention_3": AttentionFunction.FLASH_ATTENTION_3,
        "xformers": AttentionFunction.XFORMERS,
        "pytorch": AttentionFunction.PYTORCH,
    }
    use_action = cfg.control.uses("action")
    return {
        "attention_type": attention_map.get(cfg.runtime.attention_type, AttentionFunction.FLASH_ATTENTION_3),
        "rope_type": LTXRopeType.SPLIT,
        "normalize_time_by_fps": cfg.runtime.norm_by_fps,
        "normalize_rope_positions": cfg.runtime.norm_by_max_frames,
        "positional_embedding_max_pos": [int(x.strip()) for x in cfg.runtime.positional_embedding_max_pos.split(",")],
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "caption_proj_before_connector": True,
        "enable_action_control": use_action,
        "compact_spatial_tokens": cfg.runtime.compact_spatial_tokens,
    }


def _configure_control_env(cfg) -> None:
    use_action = cfg.control.uses("action")
    os.environ["LTX_USE_ACTION_CONTROL"] = "1" if use_action else "0"
    os.environ["LTX_ACTION_SCALE"] = cfg.control.action_scale
    os.environ["LTX_ACTION_FREQ_SCALE"] = str(cfg.control.action_freq_scale)
    os.environ["LTX_ACTION_FREQ_DIM_PER_AXIS"] = str(cfg.control.action_freq_dim_per_axis)


def run(request):
    from ltx2.modules.model_ltx_2_3 import LTX23Model
    from safetensors.torch import load_file

    from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec, ModuleLoadSpec, NativeModuleLoader
    from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.component import (
        load_ltx_video_decoder_checkpoint,
        load_ltx_video_encoder_checkpoint,
    )
    from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
    from worldfoundry.core.model_loading.lora import merge_named_lora_
    from worldfoundry.synthesis.visual_generation.alayaworld.history_encoder import AlayaHistoryEncoder
    from worldfoundry.synthesis.visual_generation.alayaworld.runtime import AlayaWorldRuntime
    from worldfoundry.synthesis.visual_generation.alayaworld_v11.rollout import AlayaV11Rollout

    if not torch.cuda.is_available():
        raise RuntimeError("AlayaWorld v1.1 requires CUDA")
    _prepare_input(request)
    cfg = build_config(request)
    # This session contains only inference methods and has no trainer lifecycle.
    session = AlayaV11Rollout(cfg)
    device, dtype = session.dist.device, session.dtype
    _configure_control_env(cfg)
    model_config = _read_transformer_config(request["base_checkpoint"])
    model_config.update(_runtime_transformer_overrides(cfg))
    parameters = inspect.signature(LTX23Model.__init__).parameters
    loader = NativeModuleLoader()
    policy = RuntimePolicy(device=device, dtype=dtype)
    transformer = loader.load(
        ModuleLoadSpec(module_class=LTX23Model, config={k: v for k, v in model_config.items() if k in parameters}),
        CheckpointSpec(source=str(Path(request["checkpoint_dir"]) / "transformer.pt")),
        policy,
    )
    if request["variant"] == "dmd":
        state = load_file(str(Path(request["student_dir"]) / "lora.safetensors"))
        state = {key.replace("transformer_blocks.", "blocks."): value for key, value in state.items()}
        # Released rank=alpha=256; upstream serialization already includes its scaling in A.
        expected = sum(key.endswith(".lora_B.weight") for key in state)
        if not expected or len(state) != expected * 2:
            raise ValueError("Invalid AlayaWorld v1.1 LoRA tensor pairs")
        merge_named_lora_(transformer, state, expected_modules=expected, owner="AlayaWorld v1.1", device=device)
        del state
    transformer.eval().requires_grad_(False)
    history_options = {
        key: getattr(cfg.memory, key)
        for key in (
            "compress_t",
            "compress_h",
            "compress_w",
            "lr_compress_t",
            "lr_compress_h",
            "lr_compress_w",
            "gate_init",
            "use_self_attn",
            "use_lr_branch",
        )
    }
    history = loader.load(
        ModuleLoadSpec(
            module_class=AlayaHistoryEncoder,
            config={
                "in_channels": transformer.patchify_proj.in_features,
                "out_channels": transformer.patchify_proj.out_features,
                **history_options,
            },
        ),
        CheckpointSpec(source=str(Path(request["checkpoint_dir"]) / "history_encoder.pt")),
        policy,
    )
    history.setup_lr_proj_from_patchify(transformer.patchify_proj)
    history.eval().requires_grad_(False)
    checkpoint = CheckpointSpec(source=request["base_checkpoint"])
    encoder = load_ltx_video_encoder_checkpoint(checkpoint, policy).encoder.eval().requires_grad_(False)
    decoder = load_ltx_video_decoder_checkpoint(checkpoint, policy).decoder.eval().requires_grad_(False)

    class EncoderAdapter:
        def encode(self, pixels, *, chunk_size=None, verbose=False):
            # Use the shared full-window encoder for conditioning windows.
            # Rollout supplies short conditioning windows; no long source video is encoded here.
            return encoder(pixels)

    prompt_runtime = AlayaWorldRuntime(
        checkpoint_path=request["base_checkpoint"],
        gemma_root=request["gemma_path"],
        device=device,
        dtype=dtype,
        spatial_enabled=False,
        compile_mode="none",
    )
    session.components = SimpleNamespace(
        transformer=transformer,
        vae_encoder=EncoderAdapter(),
        vae_decoder=decoder,
        text_encoder=prompt_runtime,
        encode_text=lambda runtime, prompts: [runtime._encode_prompt(p) for p in prompts],
    )
    session.history_encoder = history
    with torch.inference_mode():
        generate_case(session, request)
    (Path(request["output_dir"]) / "INFERENCE_COMPLETE.json").write_text(
        json.dumps({"status": "complete", "variant": request["variant"]}) + "\n"
    )


def generate_case(session, request):
    import imageio.v3 as iio
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    cfg = session.cfg
    mode = cfg.validation.modes["custom_i2v"]
    history_frames, chunk_frames, rounds = (
        int(mode.layout.history_latent_frames),
        int(mode.layout.output_latent_frames),
        request["rounds"],
    )
    prefix_frames = session._vigeo_target_prefix_pixel_frames(history_latent_frames=history_frames)
    target_start = cfg.layout.sink_latent_frames + history_frames
    needed_frames = prefix_frames + rounds * chunk_frames * cfg.sample.temporal_stride
    with Image.open(request["image"]) as image:
        pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float().div(255)
    pixels = F.interpolate(
        pixels[None], size=(cfg.sample.height, cfg.sample.width), mode="bicubic", align_corners=False, antialias=True
    )[0]
    video_pixels = (pixels * 2 - 1)[None].repeat(prefix_frames, 1, 1, 1).contiguous()
    with np.load(Path(request["output_dir"]) / "inputs/cam_c2w.npz", allow_pickle=False) as poses:
        camera = torch.from_numpy(poses["cam_c2w"].copy()).float()
        real_intrinsic = "intrinsic" in poses.files
        intrinsic = (
            torch.from_numpy(poses["intrinsic"].copy()).float()
            if real_intrinsic
            else torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1.0]])
        )
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or len(camera) == 0 or not torch.isfinite(camera).all():
        raise ValueError("Invalid camera trajectory; expected finite C2W [F,4,4]")
    if len(camera) < needed_frames:
        raise ValueError(f"Camera trajectory has {len(camera)} frames; this rollout requires {needed_frames}")
    camera = camera[:needed_frames]
    metadata = {
        "intrinsic": intrinsic,
        "cam_c2w": camera,
        "intrinsic_raw": intrinsic.clone(),
        "cam_c2w_raw": camera.clone(),
        "has_camera": True,
        "has_real_intrinsic": real_intrinsic,
        "source": "custom_i2v",
        "caption_type": "custom",
        "video_id": "worldfoundry",
        "pose_orig_w": float(cfg.sample.width),
        "pose_orig_h": float(cfg.sample.height),
        "frame_start": 0,
        "frame_end": needed_frames - 1,
    }
    latent = session._build_vigeo_validation_latent_full(
        video_pixels=video_pixels,
        metadata=metadata,
        required_latents=target_start + rounds * chunk_frames,
        target_base_start=target_start,
        history_latent_frames=history_frames,
        allow_short=True,
        allow_empty_target=True,
    )
    context = session._encode_caption(request["prompt"], sync=False)
    negative = (
        session._encode_caption(cfg.validation.negative_prompt, sync=False)
        if session._validation_cfg_scale() > 1
        else None
    )
    metrics, chunks, nearby, *_ = session._validate_rollout_sample(
        video_pixels=video_pixels,
        latent_full=latent,
        context=context,
        scheduled_contexts=None,
        scheduled_prompt_captions=None,
        negative_context=negative,
        metadata=metadata,
        mode_cfg=mode,
        K=chunk_frames,
        rounds=rounds,
        N=history_frames,
        gap_steps=0,
        cond_end=1,
    )
    frames = session._decode_i2v_rollout_chunks_to_video_frames(nearby_latents=nearby, chunks=chunks)
    output = Path(request["output_dir"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output / "result.mp4", frames.cpu().numpy(), fps=cfg.sample.fps)
    (output / "result.json").write_text(
        json.dumps(
            {
                "variant": request["variant"],
                "rounds": rounds,
                "frames": len(frames),
                "fps": cfg.sample.fps,
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
