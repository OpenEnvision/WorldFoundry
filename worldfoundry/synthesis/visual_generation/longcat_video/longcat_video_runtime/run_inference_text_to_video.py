import os
import argparse
import datetime
import json
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

from transformers import UMT5EncoderModel
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.video_io import write_video
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.tokenizer_loading import (
    load_longcat_tokenizer,
)

from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.scheduling_flow_match_euler import FlowMatchEulerDiscreteScheduler
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from worldfoundry.core.distributed.model_parallel import context_state as context_parallel_util
from worldfoundry.core.distributed.model_parallel.context_state import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _load_worldfoundry_request() -> dict:
    request_path = os.environ.get("WORLD_EVALS_LONGCAT_REQUEST_PATH", "").strip()
    if not request_path:
        return {}
    try:
        with open(request_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _request_int(request: dict, key: str, default: int) -> int:
    value = request.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _request_float(request: dict, key: str, default: float) -> float:
    value = request.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _request_bool(request: dict, key: str, default: bool) -> bool:
    value = request.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def generate(args):
    # case setup
    request = _load_worldfoundry_request()
    prompt = request.get("prompt") or "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    negative_prompt = request.get("negative_prompt") or "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    spatial_refine_only = bool(request.get("spatial_refine_only", False))
    height = _request_int(request, "height", 480)
    width = _request_int(request, "width", 832)
    num_frames = _request_int(request, "num_frames", 93)
    num_inference_steps = _request_int(request, "num_inference_steps", 50)
    distill_steps = _request_int(request, "distill_num_inference_steps", min(16, num_inference_steps))
    guidance_scale = _request_float(request, "guidance_scale", 4.0)
    seed_override = request.get("seed")
    fps = _request_int(request, "fps", 15)
    cpu_offload = _request_bool(request, "cpu_offload", True)
    run_base = _request_bool(request, "run_base", True)
    run_distill = _request_bool(request, "run_distill", True)
    run_refiner = _request_bool(request, "run_refiner", True)
    max_sequence_length = _request_int(request, "max_sequence_length", 512)
    if not run_base and not run_distill:
        raise ValueError("LongCat-Video requires at least one of run_base or run_distill.")
    if run_refiner and not run_distill:
        raise ValueError("LongCat-Video refinement requires run_distill=True.")

    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile

    # prepare distributed environment
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
    global_rank    = dist.get_rank()
    num_processes  = dist.get_world_size()

    # initialize context parallel before loading models
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = load_longcat_tokenizer(checkpoint_dir)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer = tokenizer,
        text_encoder = text_encoder,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
    )
    if cpu_offload:
        pipe.enable_t2v_cpu_offload(local_rank)
    else:
        pipe.to(local_rank)

    global_seed = 42 if seed_override is None else _request_int(request, "seed", 42)
    seed = global_seed + global_rank

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    if run_base:
        ### t2v (480p)
        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )[0]

        if local_rank == 0:
            output_tensor = torch.from_numpy(np.array(output))
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            write_video("output_t2v.mp4", output_tensor, fps=fps, video_codec="libx264", options={"crf": f"{18}"})
        del output
        torch_gc()

    output_distill = None
    if run_distill:
        ### t2v distill (480p)
        cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
        pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')
        pipe.dit.enable_loras(['cfg_step_lora'])

        if enable_compile:
            dit = torch.compile(dit)

        output_distill = pipe.generate_t2v(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=distill_steps,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )[0]
        pipe.dit.disable_all_loras()

        if local_rank == 0:
            output_processed_tensor = torch.from_numpy(np.array(output_distill))
            output_processed_tensor = (output_processed_tensor * 255).clamp(0, 255).to(torch.uint8)
            write_video("output_t2v_distill.mp4", output_processed_tensor, fps=fps, video_codec="libx264", options={"crf": f"{18}"})

    if run_refiner:
        ### t2v refinement (720p)
        refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
        pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
        pipe.dit.enable_loras(['refinement_lora'])
        pipe.dit.enable_bsa()

        if enable_compile:
            dit = torch.compile(dit)

        stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
        stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
        del output_distill
        torch_gc()

        output_refine = pipe.generate_refine(
            prompt=prompt,
            stage1_video=stage1_video,
            num_inference_steps=num_inference_steps,
            generator=generator,
            spatial_refine_only=spatial_refine_only,
        )[0]

        pipe.dit.disable_all_loras()
        pipe.dit.disable_bsa()

        if local_rank == 0:
            output_tensor = torch.from_numpy(output_refine)
            output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
            refine_fps = fps if spatial_refine_only else fps * 2
            write_video("output_t2v_refine.mp4", output_tensor, fps=refine_fps, video_codec="libx264", options={"crf": f"{10}"})


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--enable_compile',
        action='store_true',
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
