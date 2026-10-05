import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.core.io import VideoData, save_video
from worldfoundry.synthesis.visual_generation.spatia.native_pipeline import load_spatia_pipeline
from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime.utils.camera_io import (
    read_intrinsics_from_txt,
    read_w2cs_from_txt,
)
from worldfoundry.synthesis.visual_generation.spatia.spatia_runtime.utils.frustum_culling import (
    find_mask_matches,
    frustum_frontmost_mask_triton,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
UTILS_DIR = SCRIPT_DIR / "utils"

NEGATIVE_PROMPT = (
    "画面切换，画面切出，镜头切换，镜头渐黑，镜头渐亮，最差质量, 节目横幅, 低质量, 正常质量, "
    "JPEG压缩残留, 噪点, 模糊, 细节缺失, 马赛克, 水印, 文字, 签名, 商标, 丑陋的, 畸形的, 毁容的, "
    "解剖学错误, 形态糟糕, 多余的肢体, 残缺的肢体, 融合的手指, 画得不好的手, 画得不好的脸, "
    "僵硬的表情, 杂乱的构图, 不协调的元素, 空旷的画面, 透视错误, 比例失调, 变形的物体, "
    "融合的物体, 物体穿模, 漂浮的物体, 平淡的光线, 不自然的光源, 单调的色彩, 奇怪的颜色, "
    "刺眼的颜色, 画面发灰"
)


def resolve_path(path_str, base_dir=PROJECT_ROOT):
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def run_command(command, cwd=None, quiet=False, env=None):
    print(f"[INFO] Running command: {' '.join(str(arg) for arg in command)}")
    run_kwargs = {
        "cwd": str(cwd) if cwd is not None else None,
    }
    if env is not None:
        run_kwargs["env"] = env
    if quiet:
        completed = subprocess.run(command, capture_output=True, text=True, **run_kwargs)
        if completed.returncode:
            raise RuntimeError(
                f"Spatia subprocess exited with code {completed.returncode}: "
                f"{' '.join(str(arg) for arg in command)}\n"
                f"stdout (tail):\n{completed.stdout[-4000:]}\n"
                f"stderr (tail):\n{completed.stderr[-4000:]}"
            )
    else:
        subprocess.run(command, check=True, **run_kwargs)
def save_extrinsic_and_intrinsics(extrinsic=None, intrinsic=None, extrinsic_path=None, intrinsics_path=None):
    if extrinsic_path is not None:
        os.makedirs(os.path.dirname(extrinsic_path), exist_ok=True)
    if intrinsics_path is not None:
        os.makedirs(os.path.dirname(intrinsics_path), exist_ok=True)
    if extrinsic is not None:
        with open(extrinsic_path, "w", encoding="utf-8") as f:
            for i in range(extrinsic.shape[0]):
                f.write(f"{extrinsic[i].cpu().numpy().tolist()}\n")
    if intrinsic is not None:
        with open(intrinsics_path, "w", encoding="utf-8") as f:
            f.write(f"[{intrinsic[0].item()} {intrinsic[1].item()} {intrinsic[2].item()} {intrinsic[3].item()}]")


def trim_camera_w2c(camera_w2c_path: Path, max_frames: int, work_dir: Path):
    with camera_w2c_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if max_frames > 0:
        lines = lines[:max_frames]
    trimmed_path = work_dir / "camera_w2c_used.txt"
    ensure_parent(trimmed_path)
    with trimmed_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    return trimmed_path, read_w2cs_from_txt(trimmed_path)


def get_prompt_list(args, total_rounds):
    prompt_list = []
    prompt_path = resolve_path(args.prompt_path) if args.prompt_path else None
    if prompt_path is not None:
        with prompt_path.open("r", encoding="utf-8") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
    elif args.prompt:
        prompt_list = [args.prompt]
    if not prompt_list:
        raise ValueError("Please provide `--prompt` or `--prompt_path`.")
    if len(prompt_list) == 1 and total_rounds > 1:
        prompt_list = prompt_list * total_rounds
    elif len(prompt_list) < total_rounds:
        prompt_list.extend([prompt_list[-1]] * (total_rounds - len(prompt_list)))
    elif len(prompt_list) > total_rounds:
        prompt_list = prompt_list[:total_rounds]
    return prompt_list


def build_pipe(vace_path, lora_path, hist_frames, model_root=None):
    del hist_frames
    model_root = resolve_path(model_root) if model_root else PROJECT_ROOT / "model_weights" / "Wan2.2-TI2V-5B"
    pipe = load_spatia_pipeline(
        model_root=model_root,
        vace_path=vace_path,
        lora_path=lora_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.eval()
    pipe.enable_vram_management()
    return pipe, pipe.lora_state_dict


def set_pipe_units(pipe, hist_frames, use_history):
    del pipe, hist_frames, use_history


def compute_total_rounds(total_frames, first_round_frames, round_frames, hist_frames):
    if total_frames <= 0:
        raise ValueError("No valid camera poses found.")
    if total_frames <= first_round_frames:
        return 1
    effective_new_frames = round_frames - hist_frames
    if effective_new_frames <= 0:
        raise ValueError("`round_frames` must be larger than `hist_frames`.")
    return 1 + math.ceil((total_frames - first_round_frames) / effective_new_frames)


def get_round_range(round_idx, total_frames, first_round_frames, round_frames, hist_frames):
    if round_idx == 0:
        round_start_idx = 0
        target_frames = first_round_frames
    else:
        round_start_idx = max(
            0,
            first_round_frames + (round_idx - 1) * round_frames - hist_frames * round_idx,
        )
        target_frames = round_frames
    round_end_idx = min(round_start_idx + target_frames, total_frames)
    return round_start_idx, round_end_idx, target_frames


def load_video_frames(video_path, height, width):
    return [frame for frame in VideoData(str(video_path), height=height, width=width)]


def run_mapanything(args, round_idx, map_frames, map_extrinsics, intrinsic_path, work_dir):
    round_dir = work_dir / f"round_{round_idx:02d}"
    map_output_dir = round_dir / "map_output"
    map_output_dir.mkdir(parents=True, exist_ok=True)
    map_video_path = round_dir / "map_hist_video.mp4"
    map_extrinsic_path = round_dir / "map_hist_extrinsic.txt"

    save_video(map_frames, str(map_video_path), fps=args.fps, quality=5)
    save_extrinsic_and_intrinsics(map_extrinsics, None, str(map_extrinsic_path), None)

    points_path = map_output_dir / "points.ply"
    intrinsic_output_path = map_output_dir / "intrinsics.txt"
    need_map = args.force_rebuild_intermediate or (not points_path.exists()) or (not intrinsic_output_path.exists())
    if need_map:
        map_cmd = [
            args.map_python,
            str(UTILS_DIR / "map_anything_inference.py"),
            "--model_path",
            str(args.map_model_path),
            "--vid_path",
            str(map_video_path),
            "--extrinsics_path",
            str(map_extrinsic_path),
            "--output_path",
            str(map_output_dir),
            "--conf_percentile",
            str(args.map_conf_percentile),
            "--voxel_size",
            str(args.map_voxel_size),
            "--device",
            args.map_device,
        ]
        if intrinsic_path is not None:
            map_cmd.extend(["--intrinsic_path", str(intrinsic_path)])
        if args.mask_path:
            map_cmd.extend(["--mask_path", *[str(path) for path in args.mask_path]])
        if args.per_frame_mask:
            map_cmd.append("--per_frame_mask")
        run_command(
            map_cmd,
            cwd=SCRIPT_DIR,
            quiet=not args.verbose_subprocess,
        )
    else:
        print(f"[INFO] Reusing MapAnything outputs for round {round_idx}.")

    return {
        "round_dir": round_dir,
        "map_output_dir": map_output_dir,
        "map_video_path": map_video_path,
        "map_intrinsic_path": intrinsic_output_path,
        "points_path": points_path,
    }


def run_render(args, round_idx, points_path, map_video_path, round_extrinsic, intrinsic, work_dir):
    round_dir = work_dir / f"round_{round_idx:02d}"
    render_input_dir = round_dir / "render_input"
    render_output_dir = round_dir / "render_output"
    render_input_dir.mkdir(parents=True, exist_ok=True)
    render_output_dir.mkdir(parents=True, exist_ok=True)

    round_extrinsic_path = render_input_dir / "cam-extrinsic.txt"
    round_intrinsics_path = render_input_dir / "cam-intrinsics.txt"
    save_extrinsic_and_intrinsics(
        round_extrinsic,
        intrinsic,
        str(round_extrinsic_path),
        str(round_intrinsics_path),
    )

    render_video_path = render_output_dir / "render.mp4"
    render_score_path = render_output_dir / "render-score.mp4"
    need_render = (
        args.force_rebuild_intermediate
        or (not render_video_path.exists())
        or (not render_score_path.exists())
    )
    if need_render:
        render_cmd = [
            args.render_python,
            str(UTILS_DIR / "render_point.py"),
            "--data_path",
            str(points_path),
            "--w2c_path",
            str(round_extrinsic_path),
            "--normed_intrinsics",
            str(round_intrinsics_path),
            "--img_vid_path",
            str(map_video_path),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--score_render",
            "--dummy_score",
            "--point_coordinate",
            "opencv",
            "--downsample",
            "--voxel_size",
            str(args.render_voxel_size + round_idx * args.render_voxel_size_step),
            "--render_batchsize",
            str(args.render_batchsize),
            "--device",
            args.render_device,
            "--output_path",
            str(render_output_dir),
        ]
        run_command(
            render_cmd,
            cwd=SCRIPT_DIR,
            quiet=not args.verbose_subprocess,
        )
    else:
        print(f"[INFO] Reusing render outputs for round {round_idx}.")

    return render_video_path, render_score_path


def select_reference_frames(
    pipe,
    points_path,
    all_exist_cam_w2cs,
    new_cam_w2c,
    intrinsic,
    all_ref_frames,
    args,
    save_dir=None,
):
    import open3d as o3d

    if len(all_exist_cam_w2cs) == 0 or len(all_ref_frames) == 0:
        return []

    fx, fy, cx, cy = intrinsic
    cam_intrinsics = torch.tensor(
        [fx * args.width, fy * args.height, cx * args.width, cy * args.height],
        dtype=torch.float32,
        device=pipe.device,
    )
    exist_cam_tensor = torch.as_tensor(all_exist_cam_w2cs, dtype=torch.float32, device=pipe.device)
    new_cam_tensor = torch.as_tensor(new_cam_w2c, dtype=torch.float32, device=pipe.device)

    pcd = o3d.io.read_point_cloud(str(points_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if points.size == 0:
        return []
    all_points = np.concatenate([points, colors], axis=-1).astype(np.float32)
    all_points = torch.from_numpy(all_points).to(pipe.device)

    hist_mask_parts = []
    new_mask_parts = []
    batch_size = args.point_retrieval_batch_size
    with torch.no_grad():
        num_points = all_points.shape[0]
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            points_chunk = all_points[start:end]
            hist_mask_parts.append(
                frustum_frontmost_mask_triton(
                    points_chunk,
                    exist_cam_tensor,
                    cam_intrinsics,
                    (args.height, args.width),
                )
            )
            new_mask_parts.append(
                frustum_frontmost_mask_triton(
                    points_chunk,
                    new_cam_tensor,
                    cam_intrinsics,
                    (args.height, args.width),
                )
            )
    hist_seeable_points = torch.cat(hist_mask_parts, dim=1)
    new_seeable_points = torch.cat(new_mask_parts, dim=1)
    matching_results = find_mask_matches(hist_seeable_points, new_seeable_points, method="hungarian")

    selected = []
    for new_idx, hist_idx in matching_results.items():
        if hist_idx < 0:
            continue
        hist_mask = hist_seeable_points[hist_idx]
        new_mask = new_seeable_points[new_idx]
        intersection = int((hist_mask & new_mask).sum().item())
        selected.append((all_ref_frames[hist_idx], intersection))
    selected = selected[: args.ref_num] if args.ref_num > 0 else []

    if save_dir is not None:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx, (frame, score) in enumerate(selected):
            frame.save(save_dir / f"{idx:02d}-{score}.jpg")

    return [frame for frame, _ in selected]


def generate_video(args):
    img_path = resolve_path(args.img_path)
    save_path = resolve_path(args.save_path)
    camera_w2c_path = resolve_path(args.camera_w2c_path)
    vace_path = resolve_path(args.vace_path or args.control_path)
    lora_path = resolve_path(args.lora_path)
    if img_path is None or not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if camera_w2c_path is None or not camera_w2c_path.exists():
        raise FileNotFoundError(f"Camera w2c file not found: {camera_w2c_path}")
    if vace_path is None or not vace_path.exists():
        raise FileNotFoundError(f"VACE checkpoint not found: {vace_path}")

    default_work_dir = save_path.with_suffix("").parent / f"{save_path.stem}_assets"
    work_dir = resolve_path(args.work_dir) if args.work_dir else default_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    trimmed_w2c_path, all_extrinsics_np = trim_camera_w2c(camera_w2c_path, args.max_frames, work_dir)
    total_frames = len(all_extrinsics_np)
    total_rounds = compute_total_rounds(
        total_frames=total_frames,
        first_round_frames=args.first_round_frames,
        round_frames=args.round_frames,
        hist_frames=args.hist_frames,
    )
    prompt_list = get_prompt_list(args, total_rounds)
    all_extrinsics = torch.from_numpy(all_extrinsics_np).to(torch.float32)

    intrinsic_path = resolve_path(args.camera_intrinsics_path)
    intrinsic = read_intrinsics_from_txt(intrinsic_path) if intrinsic_path is not None else None

    pipe, lora_state_dict = build_pipe(
        vace_path=vace_path,
        lora_path=lora_path,
        hist_frames=args.hist_frames,
        model_root=args.model_root,
    )
    first_image = Image.open(img_path).convert("RGB")
    all_hist_frames = [first_image]
    all_generated_latents = []
    all_hist_extrinsics = all_extrinsics[:1]
    for round_idx, prompt in enumerate(prompt_list):
        is_first_round = round_idx == 0
        round_start_idx, round_end_idx, target_frames = get_round_range(
            round_idx=round_idx,
            total_frames=total_frames,
            first_round_frames=args.first_round_frames,
            round_frames=args.round_frames,
            hist_frames=args.hist_frames,
        )
        print(
            f"[INFO] Round {round_idx + 1}/{total_rounds}: "
            f"frames [{round_start_idx}, {round_end_idx}) target={target_frames}"
        )

        if is_first_round:
            set_pipe_units(pipe, hist_frames=args.hist_frames, use_history=False)
            pipe.unload_lora(pipe.dit, lora_state_dict=lora_state_dict)
        else:
            set_pipe_units(pipe, hist_frames=args.hist_frames, use_history=True)
            if lora_state_dict is not None:
                pipe.load_lora(pipe.dit, lora_state_dict=lora_state_dict)

        map_frames = all_hist_frames[:: args.map_stride]
        map_extrinsics = all_hist_extrinsics[:: args.map_stride]
        map_outputs = run_mapanything(
            args=args,
            round_idx=round_idx,
            map_frames=map_frames,
            map_extrinsics=map_extrinsics,
            intrinsic_path=intrinsic_path,
            work_dir=work_dir,
        )

        if intrinsic is None:
            intrinsic = read_intrinsics_from_txt(map_outputs["map_intrinsic_path"])
            intrinsic_path = map_outputs["map_intrinsic_path"]

        round_extrinsic = all_extrinsics[round_start_idx:round_end_idx]
        all_hist_extrinsics = all_extrinsics[:round_end_idx]
        padding_num = 0
        if round_extrinsic.shape[0] < target_frames:
            padding_num = target_frames - round_extrinsic.shape[0]
            last_extrinsic = round_extrinsic[-1:].repeat(padding_num, 1, 1)
            round_extrinsic = torch.cat([round_extrinsic, last_extrinsic], dim=0)

        point_video_path, point_score_path = run_render(
            args=args,
            round_idx=round_idx,
            points_path=map_outputs["points_path"],
            map_video_path=map_outputs["map_video_path"],
            round_extrinsic=round_extrinsic,
            intrinsic=torch.from_numpy(intrinsic),
            work_dir=work_dir,
        )

        round_hist_frames = all_hist_frames[-args.hist_frames :]
        round_ref_frames = all_hist_frames[:-args.hist_frames]
        round_hist_latents_num = max(0, (len(round_hist_frames) - 1) // 4 + 1)
        input_point = load_video_frames(point_video_path, args.height, args.width)
        input_point_score = load_video_frames(point_score_path, args.height, args.width)
        input_point_score = [255 - np.array(frame) for frame in input_point_score]

        if is_first_round:
            input_video = None
            input_image = round_hist_frames[0]
            selected_exist_frames = None
            ar_hist_latents_num = 0
        else:
            input_video = round_hist_frames
            input_image = None
            all_exist_cam_w2cs = all_extrinsics[:round_start_idx].cpu().numpy()[:: args.ref_hist_stride]
            all_ref_frames = round_ref_frames[:: args.ref_hist_stride]
            if len(all_exist_cam_w2cs) != len(all_ref_frames):
                raise ValueError(
                    "Mismatch between history cameras and history frames: "
                    f"{len(all_exist_cam_w2cs)} vs {len(all_ref_frames)}"
                )
            new_cam_w2c = round_extrinsic.cpu().numpy()[:: args.ref_stride]
            selected_exist_frames = select_reference_frames(
                pipe=pipe,
                points_path=map_outputs["points_path"],
                all_exist_cam_w2cs=all_exist_cam_w2cs,
                new_cam_w2c=new_cam_w2c,
                intrinsic=intrinsic,
                all_ref_frames=all_ref_frames,
                args=args,
                save_dir=work_dir / f"round_{round_idx:02d}" / "selected_ref_frames",
            )
            if args.ref_num == 0 and selected_exist_frames is not None:
                selected_exist_frames = [round_hist_frames[0]]
            ar_hist_latents_num = round_hist_latents_num

        round_video, round_latents = pipe.call_latent_inference(
            prompt=f"{prompt} 4K, realistic, high quality, HDR.",
            negative_prompt=NEGATIVE_PROMPT,
            seed=args.seed,
            tiled=False,
            height=args.height,
            width=args.width,
            input_video=input_video,
            input_image=input_image,
            control_video=input_point,
            control_score=input_point_score,
            ref_images=selected_exist_frames,
            ar_hist_latents_num=ar_hist_latents_num,
            vace_scale=args.vace_scale,
            num_frames=target_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            sampler=args.sampler,
            return_latents=True,
        )

        round_video_save_path = work_dir / f"round_{round_idx:02d}" / "generated_round.mp4"
        save_video(round_video, str(round_video_save_path), fps=args.fps, quality=5)

        if padding_num > 0:
            all_hist_frames.extend(round_video[len(round_hist_frames) : -padding_num])
        else:
            all_hist_frames.extend(round_video[len(round_hist_frames) :])

        if input_video is not None:
            all_generated_latents.append(round_latents[:, :, round_hist_latents_num:])
        else:
            all_generated_latents.append(round_latents)

    pipe.load_models_to_device(["vae"])
    all_generated_latents = torch.cat(all_generated_latents, dim=2)
    final_video = pipe.vae.decode(
        all_generated_latents,
        device=pipe.device,
        tiled=False,
    )
    final_video = pipe.vae_output_to_video(final_video)
    pipe.load_models_to_device([])

    ensure_parent(save_path)
    save_video(final_video, str(save_path), fps=args.fps, quality=5)
    trimmed_copy_path = work_dir / "camera_w2c_used.txt"
    if trimmed_copy_path != trimmed_w2c_path:
        shutil.copy2(trimmed_w2c_path, trimmed_copy_path)
    print(f"[INFO] Saved final video to {save_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="test_cases/case_2/img.jpg")
    parser.add_argument("--camera_w2c_path", type=str, default="test_cases/case_2/w2c.txt")
    parser.add_argument("--save_path", type=str, default="test_cases/case_2/output.mp4")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--vace_path", type=str, default="/home/aiscuser/4D-Representation/wan2.2/outputs/vace_point-bs64-lr1e-5/step-8500.safetensors")
    parser.add_argument("--control_path", type=str, default="", help="Alias of --vace_path.")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--model_root", type=str, default="", help="Wan2.2-TI2V-5B checkpoint directory.")
    parser.add_argument("--camera_intrinsics_path", type=str, default="test_cases/case_2/intrinsics.txt")
    parser.add_argument("--mask_path", type=str, nargs="+", default=[])
    parser.add_argument("--per_frame_mask", action="store_true", default=True)
    parser.add_argument("--work_dir", type=str, default="test_cases/case_2/output/")

    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--max_frames", type=int, default=194)
    parser.add_argument("--first_round_frames", type=int, default=121)
    parser.add_argument("--round_frames", type=int, default=81)
    parser.add_argument("--hist_frames", type=int, default=9)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--map_fps", type=int, default=4)
    parser.add_argument("--ref_fps", type=int, default=2)
    parser.add_argument("--ref_hist_fps", type=int, default=6)
    parser.add_argument("--ref_num", type=int, default=7)

    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--sampler", type=str, default="uni_pc")
    parser.add_argument("--vace_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20917)

    parser.add_argument("--map_python", type=str, default=sys.executable)
    parser.add_argument("--map_model_path", type=str, required=True)
    parser.add_argument("--render_python", type=str, default=sys.executable)
    parser.add_argument("--map_device", type=str, default="cuda")
    parser.add_argument("--render_device", type=str, default="cuda")
    parser.add_argument("--map_conf_percentile", type=float, default=0.0)
    parser.add_argument("--map_voxel_size", type=float, default=0.01)
    parser.add_argument("--render_voxel_size", type=float, default=0.01)
    parser.add_argument("--render_voxel_size_step", type=float, default=0.005)
    parser.add_argument("--render_batchsize", type=int, default=64)
    parser.add_argument("--point_retrieval_batch_size", type=int, default=10000000)
    parser.add_argument("--force_rebuild_intermediate", action="store_true", default=False)
    parser.add_argument("--verbose_subprocess", action="store_true", default=False)

    args = parser.parse_args()
    args.mask_path = [resolve_path(path) for path in args.mask_path]
    args.map_stride = max(1, int(round(args.fps / args.map_fps)))
    args.ref_stride = max(1, int(round(args.fps / args.ref_fps)))
    args.ref_hist_stride = max(1, int(round(args.fps / args.ref_hist_fps)))
    return args


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    generate_video(args)
    end_time = time.time()
    print(f"[INFO] Total end-to-end time: {end_time - start_time:.2f} seconds")
