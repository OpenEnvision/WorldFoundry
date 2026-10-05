#!/usr/bin/env python3
"""Multi-GPU backend for the 3D reward."""

from __future__ import annotations

from _bootstrap import setup_paths

setup_paths()

import torch
import numpy as np
from PIL import Image
from io import BytesIO
import multiprocessing as mp
import os
import json
import datetime
import base64
import re
import uuid
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

from model.vlm import (
    DEFAULT_VLM_BACKEND,
    chat_vlm_call,
    clean_json_content,
    get_openrouter_api_key,
    parse_json_content,
    resolve_model_name,
    resolve_vlm_backend,
)
from reward_3d_backend import DEFAULT_RECONSTRUCTION_MODEL, Reward3DBackend

load_dotenv()

DEFAULT_SCORING_MODEL: str | None = None
DEFAULT_VLM_FRAME_COUNT = 8


def _parse_visible_gpu_env() -> list[str]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _pil_image_to_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    buffered = BytesIO()
    save_kwargs = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = 90
    image.save(buffered, **save_kwargs)
    mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _tensor_to_pil_image(image_tensor) -> Image.Image:
    if image_tensor.dtype.is_floating_point:
        image_tensor = (image_tensor.clamp(0, 1) * 255.0).to(torch.uint8)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image_np)


def _tensor_to_sampled_pil_frames(video_tensor, max_frames: int = DEFAULT_VLM_FRAME_COUNT) -> list[Image.Image]:
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected video tensor with shape (T, C, H, W), got {tuple(video_tensor.shape)}")
    frame_indices = _sample_indices(video_tensor.shape[0], max_frames)
    return [_tensor_to_pil_image(video_tensor[idx].cpu()) for idx in frame_indices]


def _save_vlm_inputs(
    *,
    prompt_text: str,
    images: list[Image.Image],
    save_dir: str | os.PathLike[str],
    prefix: str,
    image_format: str,
) -> list[str]:
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    prompt_path = save_root / f"{prefix}_prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    image_paths: list[str] = []
    normalized_format = image_format.upper()
    suffix = ".png" if normalized_format == "PNG" else ".jpg"
    for idx, image in enumerate(images):
        image_path = save_root / f"{prefix}_{idx:02d}{suffix}"
        save_kwargs: dict[str, Any] = {"format": normalized_format}
        if normalized_format == "JPEG":
            save_kwargs["quality"] = 95
        image.save(image_path, **save_kwargs)
        image_paths.append(str(image_path))

    print(f"Saved VLM prompt to {prompt_path}")
    if image_paths:
        print(f"Saved VLM inputs for {prefix}:")
        for path in image_paths:
            print(f"  {path}")
    return image_paths


def _normalize_vlm_score(raw_score: Any) -> float:
    score = float(raw_score)
    if score > 1.0:
        score = score / 10.0
    return float(np.clip(score, 0.0, 1.0))


def _response_content_to_text(response: dict) -> str:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def _extract_vlm_judgment(response: dict) -> tuple[float, str | None, str]:
    raw_text = _response_content_to_text(response)
    try:
        parsed = parse_json_content(response)
        if isinstance(parsed, dict) and "score" in parsed:
            reason = parsed.get("reason")
            reason_text = None if reason is None else str(reason)
            return _normalize_vlm_score(parsed["score"]), reason_text, raw_text
    except Exception:
        pass

    cleaned = clean_json_content(raw_text)

    patterns = [
        r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r"<Score>\s*([0-9]+(?:\.[0-9]+)?)\s*</Score>",
        r"([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned, flags=re.IGNORECASE)
            reason_text = reason_match.group(1) if reason_match else None
            return _normalize_vlm_score(match.group(1)), reason_text, raw_text

    raise ValueError(f"Could not extract score from VLM response: {cleaned[:500]}")


class VLMScorer:
    """Multimodal scorer backed by the unified VLM interface."""

    def __init__(
        self,
        model_name: str | None = DEFAULT_SCORING_MODEL,
        backend: str | None = DEFAULT_VLM_BACKEND,
        timeout: int = 240,
    ):
        self.backend = resolve_vlm_backend(backend)
        self.model_name = resolve_model_name(model_name, self.backend)
        if self.backend == "openrouter" and not get_openrouter_api_key():
            raise RuntimeError(
                "Missing OpenRouter key; set WORLDFOUNDRY_OPENROUTER_API_KEY "
                "or OPENROUTER_API_KEY"
            )
        self.timeout = timeout
        self.video_task_template = """You are a calibrated 3D consistency judge for static-scene reconstruction.
You will be given several sampled frames from a Gaussian-Splat render reconstructed from a source video after dynamic foreground regions were masked and video-inpainted.
The static-scene prompt for this 3D evaluation is:
{prompt}

Judge the 3D consistency quality of the static scene from these rendered frames. Be robust to normal Gaussian-Splat artifacts, video-inpainting artifacts, blur, rain/fog/low-light appearance, and moderate texture noise.
Dynamic foreground actors and actions (people, characters, animals, vehicles, limbs, clothing motion, manipulated objects, etc.) may be absent or incomplete because they were intentionally removed before reconstruction. Do not penalize their absence.
Focus on:
1. Whether the reconstructed static background geometry is coherent across views or across time.
2. Whether static structures, terrain, roads, walls, gates, trees, rooms, large props, and scene layout are spatially plausible.
3. Whether the render preserves a recognizable scene layout and camera-consistent spatial organization.
4. Whether the reconstruction is faithful to the static scene description and expected camera/view behavior.
5. Whether artifacts dominate the reconstruction enough to prevent understanding the static scene.
6. If the prompt says the camera is static, do not require camera motion or parallax.

Scoring calibration:
- 0.8-1.0: clear, coherent 3D scene with stable layout; minor blur/noise/inpainting artifacts are acceptable.
- 0.6-0.8: recognizable and mostly coherent static scene, but with noticeable blur, holes, noise, or incomplete geometry.
- 0.4-0.6: partially recognizable scene with significant artifacts, but some spatial layout or depth cues remain.
- 0.2-0.4: mostly failed reconstruction; scene is barely recognizable, flat/collapsed, or spatially incoherent.
- 0.0-0.2: unusable render with no interpretable static scene.

Important: Do not assign a very low score solely because the render is blurry or noisy. If the scene is visually consistent and the static layout is recognizable, the score should usually be at least 0.5.

Return strict JSON only:
{{"score": 0.0-1.0, "reason": "short"}}
"""
        self.image_task_template = """You are a calibrated 3D consistency judge for static-scene reconstruction.
You will be given a single meta-view image rendered from a 3D reconstruction of a source video after dynamic foreground regions were masked and video-inpainted.
The static-scene prompt for this 3D evaluation is:
{prompt}

Judge the 3D consistency quality of the static scene from this rendered meta-view. Be robust to normal Gaussian-Splat artifacts, video-inpainting artifacts, blur, rain/fog/low-light appearance, and moderate texture noise.
Dynamic foreground actors and actions may be absent or incomplete because they were intentionally removed before reconstruction. Do not penalize their absence.
Focus on whether the static background layout is recognizable, geometrically plausible, structurally coherent, and not catastrophically flat, floating, duplicated, or collapsed. If the prompt says the camera is static, do not require camera motion or parallax.

Scoring calibration:
- 0.8-1.0: clear, coherent static scene; minor blur/noise/inpainting artifacts are acceptable.
- 0.6-0.8: recognizable and mostly coherent static scene, but with noticeable artifacts or incomplete geometry.
- 0.4-0.6: partially recognizable scene with significant artifacts, but some spatial layout remains.
- 0.2-0.4: mostly failed reconstruction; barely recognizable or spatially incoherent.
- 0.0-0.2: unusable image with no interpretable static scene.

Important: Do not assign a very low score solely because the meta-view is blurry or noisy. If the static scene layout is recognizable, the score should usually be at least 0.5.

Return strict JSON only:
{{"score": 0.0-1.0, "reason": "short"}}
"""

    def _call(
        self,
        prompt_text: str,
        images: list[Image.Image],
        image_format: str = "JPEG",
        save_dir: str | os.PathLike[str] | None = None,
        prefix: str = "vlm_input",
    ) -> float:
        if save_dir is not None:
            _save_vlm_inputs(
                prompt_text=prompt_text,
                images=images,
                save_dir=save_dir,
                prefix=prefix,
                image_format=image_format,
            )

        content = [{"type": "text", "text": prompt_text}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _pil_image_to_data_url(image, image_format=image_format)},
                }
            )

        response = chat_vlm_call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a calibrated 3D reconstruction judge. Follow the scoring rubric carefully and return strict JSON only.",
                },
                {"role": "user", "content": content},
            ],
            model_name=self.model_name,
            backend=self.backend,
            timeout=self.timeout,
            max_retries=3,
        )
        score, reason, raw_text = _extract_vlm_judgment(response)
        print("VLM raw output:")
        print(raw_text)
        print(f"VLM parsed score: {score:.3f}")
        if reason is not None:
            print(f"VLM parsed reason: {reason}")
        return score

    def score_video_tensor(
        self,
        prompt: str,
        video_tensor,
        save_dir: str | os.PathLike[str] | None = None,
        prefix: str = "gs_vlm_input",
    ) -> float:
        pil_frames = _tensor_to_sampled_pil_frames(video_tensor)
        prompt_text = self.video_task_template.format(prompt=prompt)
        return self._call(
            prompt_text,
            pil_frames,
            image_format="JPEG",
            save_dir=save_dir,
            prefix=prefix,
        )

    def score_image_tensor(
        self,
        prompt: str,
        image_tensor,
        save_dir: str | os.PathLike[str] | None = None,
        prefix: str = "meta_vlm_input",
    ) -> float:
        pil_image = _tensor_to_pil_image(image_tensor.cpu())
        prompt_text = self.image_task_template.format(prompt=prompt)
        return self._call(
            prompt_text,
            [pil_image],
            image_format="PNG",
            save_dir=save_dir,
            prefix=prefix,
        )


def reward_3d_worker_process(gpu_id, model_name, scorer_type, scoring_model_name, task_queue, result_queue):
    """
    Worker process that runs the 3D reward stack on a specific GPU.

    Args:
        gpu_id: CUDA device ID
        model_name: Reconstruction model name
        scorer_type: Type of VLM backend
        scoring_model_name: VLM model used to score GS/meta renderings
        task_queue: Queue to receive tasks (batch_idx, video_frames, prompt, save_dir)
        result_queue: Queue to send results including per-video artifact paths
    """
    try:
        import torch as torch_local
        visible_devices = _parse_visible_gpu_env()
        physical_gpu = visible_devices[gpu_id] if gpu_id < len(visible_devices) else str(gpu_id)
        device = torch_local.device(f"cuda:{gpu_id}")
        torch_local.cuda.set_device(device)

        print(
            f"[Process {os.getpid()}] Initializing 3D reward backend on visible GPU slot {gpu_id} "
            f"(physical GPU {physical_gpu})"
        )
        reward_3d_backend = Reward3DBackend(device=device, model_name=model_name)
        scorer_backend = resolve_vlm_backend(scorer_type)
        scorer = VLMScorer(model_name=scoring_model_name, backend=scorer_backend)
        print(
            f"[Process {os.getpid()}] 3D reward backend ready on visible GPU slot {gpu_id} "
            f"(physical GPU {physical_gpu}); "
            f"VLM scoring via backend={scorer.backend} model={scorer.model_name}"
        )

        lpips_model = None

        def compute_lpips_gs_score(gs_video_tensor, input_video_tensor):
            nonlocal lpips_model
            if lpips_model is None:
                try:
                    import lpips as lpips_lib
                except Exception as e:
                    print(f"[Process {os.getpid()}] Failed to import lpips: {e}")
                    return 0.0
                lpips_model = lpips_lib.LPIPS(net="alex").to(device)
                lpips_model.eval()

            if gs_video_tensor.dim() != 4 or input_video_tensor.dim() != 4:
                print(f"[Process {os.getpid()}] Invalid LPIPS tensor shapes: gs={gs_video_tensor.shape}, input={input_video_tensor.shape}")
                return 0.0

            num_frames = min(gs_video_tensor.shape[0], input_video_tensor.shape[0])
            if num_frames <= 0:
                print(f"[Process {os.getpid()}] Empty video for LPIPS scoring")
                return 0.0

            gs_clip = gs_video_tensor[:num_frames].float()
            input_clip = input_video_tensor[:num_frames].float()

            if input_clip.shape[-2:] != gs_clip.shape[-2:]:
                input_clip = torch_local.nn.functional.interpolate(
                    input_clip,
                    size=gs_clip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            gs_norm = gs_clip * 2.0 - 1.0
            input_norm = input_clip * 2.0 - 1.0

            with torch_local.no_grad():
                lpips_values = lpips_model(gs_norm, input_norm)

            lpips_mean = lpips_values.mean().item()
            gs_score = float(np.clip(1.0 - lpips_mean, 0.0, 1.0))
            print(f"[Process {os.getpid()}] LPIPS mean: {lpips_mean:.4f}, GS score: {gs_score:.4f}")
            return gs_score

        # Signal that initialization is complete
        result_queue.put(("READY", gpu_id))

        # Process tasks from queue
        while True:
            task = task_queue.get()

            # Check for shutdown signal
            if task is None:
                print(f"[Process {os.getpid()}] Received shutdown signal")
                break

            if isinstance(task, dict) and task.get("kind") == "trajectory":
                request_id = task["request_id"]
                try:
                    from depth_anything_3.utils.geometry import affine_inverse_np, as_homogeneous

                    frame_indices = task["frame_indices"]
                    process_res = int(task.get("process_res") or 504)
                    frames = []
                    for img in task["video_frames"]:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        frames.append(np.array(img))

                    print(
                        f"[Process {os.getpid()}] Extracting DA3 camera trajectory "
                        f"for request {request_id} on GPU {gpu_id}"
                    )
                    prediction = reward_3d_backend.model.inference(
                        image=frames,
                        extrinsics=None,
                        intrinsics=None,
                        process_res=process_res,
                        align_to_input_ext_scale=True,
                        infer_gs=False,
                        export_dir=None,
                    )
                    extrinsics = as_homogeneous(np.asarray(prediction.extrinsics, dtype=np.float32))
                    poses_c2w = affine_inverse_np(extrinsics)
                    trajectory = {
                        f"frame{frame_idx}": pose.tolist()
                        for frame_idx, pose in zip(frame_indices, poses_c2w)
                    }
                    result_queue.put(("TRAJECTORY", request_id, trajectory))
                except Exception as e:
                    print(f"[Process {os.getpid()}] Error extracting trajectory: {e}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put(("TRAJECTORY_ERROR", request_id, traceback.format_exc()))
                continue

            camera_trajectory = None
            dynamic_masks = None
            if len(task) == 4:
                batch_idx, video_frames, prompt, save_dir = task
                use_lpips = False
            elif len(task) == 5:
                batch_idx, video_frames, prompt, save_dir, use_lpips = task
            elif len(task) == 6:
                batch_idx, video_frames, prompt, save_dir, use_lpips, camera_trajectory = task
            elif len(task) == 7:
                batch_idx, video_frames, prompt, save_dir, use_lpips, camera_trajectory, dynamic_masks = task
            else:
                raise ValueError(f"Unexpected task format: {task}")

            try:
                print(f"[Process {os.getpid()}] Processing batch {batch_idx} on GPU {gpu_id}")

                # Convert PIL images to tensor
                frames_tensors = []
                for img in video_frames:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert to tensor (C, H, W) in range [0, 1]
                    frame_tensor = torch_local.from_numpy(np.array(img)).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                    frames_tensors.append(frame_tensor)

                # Stack to (T, C, H, W)
                video_tensor = torch_local.stack(frames_tensors, dim=0).to(device)

                gs_video, meta_view, camera_motion_score, trajectory_comparison_image = reward_3d_backend(
                    video_tensor,
                    camera_trajectory=camera_trajectory,
                    dynamic_masks=dynamic_masks,
                )

                # Save gs_video and meta_view
                print(f"[Process {os.getpid()}] Saving GS video and meta view for batch {batch_idx}")
                import cv2 as cv2_local

                # Save GS video
                gs_video_path = os.path.join(save_dir, f"batch_{batch_idx}_gs_video.mp4")
                gs_frames_np = (gs_video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                height, width = gs_frames_np.shape[1:3]
                fourcc = cv2_local.VideoWriter_fourcc(*'mp4v')
                gs_writer = cv2_local.VideoWriter(gs_video_path, fourcc, 24, (width, height))
                for frame in gs_frames_np:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2_local.cvtColor(frame, cv2_local.COLOR_RGB2BGR)
                    gs_writer.write(frame_bgr)
                gs_writer.release()
                print(f"[Process {os.getpid()}] Saved GS video to {gs_video_path}")

                # Save meta view (single image, not video)
                meta_view_path = os.path.join(save_dir, f"batch_{batch_idx}_meta_view.png")
                meta_view_np = (meta_view.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                meta_view_bgr = cv2_local.cvtColor(meta_view_np, cv2_local.COLOR_RGB2BGR)
                cv2_local.imwrite(meta_view_path, meta_view_bgr)
                print(f"[Process {os.getpid()}] Saved meta view to {meta_view_path}")

                trajectory_comparison_path = ""
                if trajectory_comparison_image is not None:
                    trajectory_comparison_path = os.path.join(save_dir, f"batch_{batch_idx}_trajectory_comparison.png")
                    trajectory_comparison_bgr = cv2_local.cvtColor(trajectory_comparison_image, cv2_local.COLOR_RGB2BGR)
                    cv2_local.imwrite(trajectory_comparison_path, trajectory_comparison_bgr)
                    print(f"[Process {os.getpid()}] Saved trajectory comparison to {trajectory_comparison_path}")

                # Debug: Check tensor properties
                print(f"[Process {os.getpid()}] gs_video shape: {gs_video.shape}, dtype: {gs_video.dtype}, range: [{gs_video.min():.3f}, {gs_video.max():.3f}]")
                print(f"[Process {os.getpid()}] meta_view shape: {meta_view.shape}, dtype: {meta_view.dtype}, range: [{meta_view.min():.3f}, {meta_view.max():.3f}]")

                # Score GS reconstruction with LPIPS or the selected VLM backend.
                print(f"[Process {os.getpid()}] Scoring GS video for batch {batch_idx} (lpips={use_lpips})")
                if use_lpips:
                    gs_score = compute_lpips_gs_score(gs_video, video_tensor)
                else:
                    gs_score = scorer.score_video_tensor(
                        prompt,
                        gs_video,
                        save_dir=save_dir,
                        prefix=f"batch_{batch_idx}_gs_vlm_input",
                    )

                # Score meta-view with the selected multimodal model.
                print(f"[Process {os.getpid()}] Scoring meta view for batch {batch_idx}")
                meta_score = scorer.score_image_tensor(
                    prompt,
                    meta_view,
                    save_dir=save_dir,
                    prefix=f"batch_{batch_idx}_meta_vlm_input",
                )

                # Paper setting: R_3D = S_meta + S_recon + S_traj, each bounded in [0, 1].
                gs_score = float(np.clip(gs_score, 0.0, 1.0))
                meta_score = float(np.clip(meta_score, 0.0, 1.0))
                camera_motion_score = float(np.clip(camera_motion_score, 0.0, 1.0))
                final_score = gs_score + meta_score + camera_motion_score

                print(f"[Process {os.getpid()}] Batch {batch_idx} completed:")
                print(f"  GS score: {gs_score:.3f}, Meta score: {meta_score:.3f}, Camera motion: {camera_motion_score:.3f}, Final: {final_score:.3f}")

                result_queue.put((
                    batch_idx,
                    gs_score,
                    meta_score,
                    camera_motion_score,
                    final_score,
                    gs_video_path,
                    meta_view_path,
                    trajectory_comparison_path,
                ))

            except Exception as e:
                print(f"[Process {os.getpid()}] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                # Try to recover CUDA state
                try:
                    torch_local.cuda.empty_cache()
                    torch_local.cuda.synchronize()
                except:
                    pass

                result_queue.put((batch_idx, 0.0, 0.0, 0.0, 0.0, "", "", ""))

    except Exception as e:
        print(f"[Process {os.getpid()}] Fatal error in worker process: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("ERROR", gpu_id))


class MultiGPUReward3DManager:
    """Manager for multi-GPU 3D reward evaluation."""

    def __init__(
        self,
        model_name=DEFAULT_RECONSTRUCTION_MODEL,
        scorer_type=DEFAULT_VLM_BACKEND,
        scoring_model_name=DEFAULT_SCORING_MODEL,
        use_lpips=False,
        num_workers: int = 1,
    ):
        self.model_name = model_name
        self.scorer_type = resolve_vlm_backend(scorer_type)
        self.scoring_model_name = resolve_model_name(scoring_model_name, self.scorer_type)
        self.use_lpips = use_lpips
        self.num_workers = max(1, int(num_workers))
        self.num_gpus = 0
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.current_gpu_index = 0  # For round-robin assignment
        self.call_counter = 0  # Track number of calls for logging
        self.last_results = None

    def initialize(self):
        if not torch.cuda.is_available():
            print("CUDA not available. Cannot run the 3D reward backend.")
            return

        available_gpus = torch.cuda.device_count()
        self.num_gpus = min(available_gpus, self.num_workers)
        print(
            f"Initializing the 3D reward backend on {self.num_gpus} GPU(s) "
            f"(available={available_gpus}, requested_workers={self.num_workers})"
        )
        ctx = mp.get_context("spawn")

        # Create shared result queue
        self.result_queue = ctx.Queue()
        queue_by_gpu = {}
        process_by_gpu = {}

        # Create a task queue and worker process for each GPU
        for gpu_id in range(self.num_gpus):
            task_queue = ctx.Queue()
            self.task_queues.append(task_queue)
            queue_by_gpu[gpu_id] = task_queue

            # Start worker process
            process = ctx.Process(
                target=reward_3d_worker_process,
                args=(
                    gpu_id,
                    self.model_name,
                    self.scorer_type,
                    self.scoring_model_name,
                    task_queue,
                    self.result_queue,
                )
            )
            process.start()
            self.processes.append(process)
            process_by_gpu[gpu_id] = process
            print(f"Started worker process for GPU {gpu_id} (PID: {process.pid})")

        # Wait for all workers to finish initialization
        ready_count = 0
        failed_count = 0
        ready_gpu_ids = []
        while ready_count + failed_count < self.num_gpus:
            msg_type, gpu_id = self.result_queue.get()
            if msg_type == "READY":
                print(f"Worker for GPU {gpu_id} is ready")
                ready_count += 1
                ready_gpu_ids.append(gpu_id)
            elif msg_type == "ERROR":
                print(f"Worker for GPU {gpu_id} failed to initialize")
                failed_count += 1

        if ready_count == 0:
            self.shutdown()
            raise RuntimeError("All 3D reward workers failed to initialize.")

        if failed_count > 0:
            print(f"{failed_count} worker(s) failed to initialize; continuing with {ready_count} worker(s)")
            self.num_gpus = ready_count
            self.task_queues = [queue_by_gpu[gpu_id] for gpu_id in ready_gpu_ids]
            self.processes = [process_by_gpu[gpu_id] for gpu_id in ready_gpu_ids]

        print(f"3D reward backend initialized with {self.num_gpus} worker process(es)")

    def compute_batch_scores(
        self,
        batch_videos,
        batch_prompts,
        camera_trajectories=None,
        dynamic_masks=None,
        use_lpips=None,
    ):
        """
        Compute scores for a batch with load balancing across GPUs

        Args:
            batch_videos: List of videos, each video is a list of frame bytes (JPEG)
            batch_prompts: List of text prompts (length = batch_size)
            camera_trajectories: Optional list of camera trajectories (length = batch_size)
            dynamic_masks: Optional list of per-video dynamic mask frame lists
            use_lpips: Use LPIPS to score GS video instead of the selected VLM backend

        Returns:
            List of final scores (length = batch_size)
        """
        if use_lpips is None:
            use_lpips = self.use_lpips
        if not self.processes:
            print("Error: 3D reward worker processes not initialized")
            return [0.0] * len(batch_videos)

        batch_size = len(batch_videos)
        results = {
            'final_scores': [0.0] * batch_size,
            'gs_scores': [0.0] * batch_size,
            'meta_scores': [0.0] * batch_size,
            'camera_motion_scores': [0.0] * batch_size,
        }

        # Create output directory for logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = f"logs/reward_3d/call_{self.call_counter}_{timestamp}"
        os.makedirs(batch_dir, exist_ok=True)
        self.call_counter += 1

        if camera_trajectories is None:
            camera_trajectories = [None] * batch_size
        elif not isinstance(camera_trajectories, list):
            camera_trajectories = [camera_trajectories] * batch_size
        elif len(camera_trajectories) != batch_size:
            if len(camera_trajectories) < batch_size:
                camera_trajectories = camera_trajectories + [None] * (batch_size - len(camera_trajectories))
            else:
                camera_trajectories = camera_trajectories[:batch_size]

        if dynamic_masks is None:
            dynamic_masks = [None] * batch_size
        elif not isinstance(dynamic_masks, list):
            dynamic_masks = [dynamic_masks] * batch_size
        elif len(dynamic_masks) != batch_size:
            if len(dynamic_masks) < batch_size:
                dynamic_masks = dynamic_masks + [None] * (batch_size - len(dynamic_masks))
            else:
                dynamic_masks = dynamic_masks[:batch_size]

        # Prepare tasks
        tasks = []
        for batch_idx, (video_frames, prompt, camera_trajectory, video_dynamic_masks) in enumerate(
            zip(batch_videos, batch_prompts, camera_trajectories, dynamic_masks)
        ):
            try:
                # Convert frame bytes to PIL Images
                frames = [Image.open(BytesIO(frame_bytes)).convert('RGB') for frame_bytes in video_frames]
                print(f"Preparing batch {batch_idx}: {len(frames)} frames")
                tasks.append((batch_idx, frames, prompt, use_lpips, camera_trajectory, video_dynamic_masks))
            except Exception as e:
                print(f"Error preparing batch {batch_idx}: {e}")
                # Will use default score 0.0

        # Distribute tasks to workers using round-robin
        for batch_idx, frames, prompt, task_use_lpips, camera_trajectory, video_dynamic_masks in tasks:
            gpu_idx = self.current_gpu_index % self.num_gpus
            print(f"Assigning batch {batch_idx} to GPU {gpu_idx}")
            # Pass batch_dir so worker can save videos
            self.task_queues[gpu_idx].put(
                (batch_idx, frames, prompt, batch_dir, task_use_lpips, camera_trajectory, video_dynamic_masks)
            )
            self.current_gpu_index += 1

        # Collect results
        completed = 0
        per_video_results = []

        while completed < len(tasks):
            batch_idx, gs_score, meta_score, camera_motion_score, final_score, gs_video_path, meta_view_path, trajectory_comparison_path = self.result_queue.get()
            results['gs_scores'][batch_idx] = gs_score
            results['meta_scores'][batch_idx] = meta_score
            results['camera_motion_scores'][batch_idx] = camera_motion_score
            results['final_scores'][batch_idx] = final_score
            completed += 1

            # Store result for logging
            per_video_results.append({
                "video_id": batch_idx,
                "prompt": batch_prompts[batch_idx],
                "gs_score": float(gs_score),
                "meta_score": float(meta_score),
                "camera_motion_score": float(camera_motion_score),
                "final_score": float(final_score),
                "lpips": bool(use_lpips),
                "scorer_type": self.scorer_type,
                "scoring_model_name": self.scoring_model_name,
                "gs_video_path": gs_video_path,
                "meta_view_path": meta_view_path,
                "trajectory_comparison_path": trajectory_comparison_path,
            })

            print(f"Received result for batch {batch_idx}: GS={gs_score:.3f}, Meta={meta_score:.3f}, Motion={camera_motion_score:.3f}, Final={final_score:.3f}")
            if gs_video_path:
                print(f"  GS video: {gs_video_path}")
            if meta_view_path:
                print(f"  Meta view: {meta_view_path}")
            if trajectory_comparison_path:
                print(f"  Trajectory compare: {trajectory_comparison_path}")

        # Save results to JSON file
        results_path = os.path.join(batch_dir, "reward_3d_scores.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(per_video_results, f, ensure_ascii=False, indent=2)
        print(f"Saved 3D reward scores to {results_path}")

        self.last_results = {
            "batch_dir": batch_dir,
            "final_scores": list(results["final_scores"]),
            "gs_scores": list(results["gs_scores"]),
            "meta_scores": list(results["meta_scores"]),
            "camera_motion_scores": list(results["camera_motion_scores"]),
            "scorer_type": self.scorer_type,
            "scoring_model_name": self.scoring_model_name,
            "per_video_results": per_video_results,
        }

        return results['final_scores']

    def extract_camera_trajectory(
        self,
        video_frames,
        frame_indices,
        process_res: int = 504,
    ) -> dict[str, list[list[float]]]:
        """Extract a DA3 camera trajectory using the persistent worker pool."""
        if not self.processes:
            raise RuntimeError("3D reward worker processes not initialized")
        if len(video_frames) != len(frame_indices):
            raise ValueError(
                f"Frame/index count mismatch: frames={len(video_frames)}, indices={len(frame_indices)}"
            )

        pil_frames = []
        for frame_bytes in video_frames:
            pil_frames.append(Image.open(BytesIO(frame_bytes)).convert("RGB"))

        request_id = uuid.uuid4().hex
        gpu_idx = self.current_gpu_index % self.num_gpus
        print(f"Assigning trajectory request {request_id} to GPU {gpu_idx}")
        self.task_queues[gpu_idx].put(
            {
                "kind": "trajectory",
                "request_id": request_id,
                "video_frames": pil_frames,
                "frame_indices": list(frame_indices),
                "process_res": process_res,
            }
        )
        self.current_gpu_index += 1

        while True:
            message = self.result_queue.get()
            if not isinstance(message, tuple) or len(message) < 2:
                raise RuntimeError(f"Unexpected 3D worker response: {message}")
            message_type = message[0]
            if message_type == "TRAJECTORY" and message[1] == request_id:
                return message[2]
            if message_type == "TRAJECTORY_ERROR" and message[1] == request_id:
                raise RuntimeError(message[2])
            raise RuntimeError(f"Unexpected 3D worker response while waiting for trajectory: {message}")

    def shutdown(self):
        """Shutdown all worker processes"""
        if not self.processes and not self.task_queues:
            return

        print("Shutting down 3D reward worker processes...")

        # Send shutdown signal to all workers
        for task_queue in self.task_queues:
            task_queue.put(None)

        # Wait for all processes to finish
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                print(f"Force terminating process {process.pid}")
                process.terminate()
                process.join()

        print("All 3D reward worker processes shut down")
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.num_gpus = 0


def _sample_indices(total_frames: int, max_frames: Optional[int]) -> list[int]:
    if total_frames <= 0:
        return []
    if max_frames is None or max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))
    if max_frames == 1:
        return [0]
    return [
        int(round(i * (total_frames - 1) / (max_frames - 1)))
        for i in range(max_frames)
    ]


def load_video_frames_as_jpeg_with_indices(
    video_path: str | os.PathLike[str],
    max_frames: Optional[int] = 32,
) -> tuple[list[bytes], list[int]]:
    """Read a video file and return sampled JPEG-encoded frames plus source indices."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(_sample_indices(total_frames, max_frames))
    encoded_frames: list[bytes] = []
    frame_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in target_indices:
            encode_ok, buffer = cv2.imencode(".jpg", frame)
            if not encode_ok:
                raise RuntimeError(f"Failed to encode frame {frame_idx} from {video_path}")
            encoded_frames.append(buffer.tobytes())
            frame_indices.append(frame_idx)
        frame_idx += 1

    capture.release()

    if not encoded_frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return encoded_frames, frame_indices


def load_video_frames_as_jpeg(
    video_path: str | os.PathLike[str],
    max_frames: Optional[int] = 32,
) -> list[bytes]:
    """Read a video file and return sampled JPEG-encoded frames."""
    encoded_frames, _ = load_video_frames_as_jpeg_with_indices(video_path, max_frames=max_frames)
    return encoded_frames


def load_dynamic_masks(
    mask_video_path: str | os.PathLike[str],
    max_frames: Optional[int] = 32,
    threshold: int = 16,
) -> list[np.ndarray]:
    """Read sampled binary dynamic masks from a video file."""
    import cv2

    capture = cv2.VideoCapture(str(mask_video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open dynamic mask video: {mask_video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(_sample_indices(total_frames, max_frames))
    masks: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in target_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append(gray > threshold)
        frame_idx += 1

    capture.release()

    if not masks:
        raise ValueError(f"No dynamic masks extracted from video: {mask_video_path}")
    return masks


def load_camera_trajectory(
    camera_trajectory_path: Optional[str | os.PathLike[str]] = None,
    camera_trajectory_json: Optional[str] = None,
) -> Any:
    """Load camera trajectory from a JSON file or a raw JSON string."""
    if camera_trajectory_path:
        with open(camera_trajectory_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if camera_trajectory_json:
        return json.loads(camera_trajectory_json)
    return None


def score_video_file(
    video_path: str | os.PathLike[str],
    prompt: str = "",
    *,
    model_name: str = DEFAULT_RECONSTRUCTION_MODEL,
    scorer_type: str | None = DEFAULT_VLM_BACKEND,
    scoring_model_name: str | None = DEFAULT_SCORING_MODEL,
    use_lpips: bool = False,
    num_workers: int = 1,
    max_frames: Optional[int] = 32,
    camera_trajectory: Any = None,
    dynamic_mask_video_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Score a single video file and return the detailed result."""
    video_path = str(Path(video_path).resolve())
    frames = load_video_frames_as_jpeg(video_path, max_frames=max_frames)
    dynamic_masks = None
    resolved_dynamic_mask_video_path = None
    if dynamic_mask_video_path:
        resolved_dynamic_mask_video_path = str(Path(dynamic_mask_video_path).resolve())
        dynamic_masks = load_dynamic_masks(resolved_dynamic_mask_video_path, max_frames=max_frames)
        if len(dynamic_masks) != len(frames):
            raise ValueError(
                "Dynamic mask frame count does not match sampled video frames: "
                f"masks={len(dynamic_masks)}, frames={len(frames)}"
            )

    manager = MultiGPUReward3DManager(
        model_name=model_name,
        scorer_type=scorer_type,
        scoring_model_name=scoring_model_name,
        use_lpips=use_lpips,
        num_workers=num_workers,
    )

    try:
        manager.initialize()
        if not manager.processes:
            raise RuntimeError("3D reward backend failed to initialize. CUDA is required.")

        manager.compute_batch_scores(
            [frames],
            [prompt],
            camera_trajectories=[camera_trajectory],
            dynamic_masks=[dynamic_masks],
        )
        if not manager.last_results or not manager.last_results["per_video_results"]:
            raise RuntimeError("3D reward backend did not return any per-video result.")

        result = dict(manager.last_results["per_video_results"][0])
        result["video_path"] = video_path
        result["reconstruction_model_name"] = model_name
        result["scorer_type"] = manager.scorer_type
        result["scoring_model_name"] = manager.scoring_model_name
        result["num_workers"] = num_workers
        result["max_frames"] = len(frames)
        result["dynamic_mask_video_path"] = resolved_dynamic_mask_video_path
        result["batch_dir"] = manager.last_results["batch_dir"]
        return result
    finally:
        if manager.processes:
            manager.shutdown()
