# SPDX-License-Identifier: Apache-2.0
"""Native InSpatio World 1.5 orchestration over shared base models."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CHECKPOINT_REPO = "inspatio/world-1.5"
CHECKPOINT_FILENAME = "InSpatio-World-1.5-1.3B.safetensors"
CHECKPOINT_SHA256 = "44f121356f81b7d1651f7878540a53b3becc2814307497d8f7a54e8687df25f9"
DEFAULT_WAN_MODEL_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
SOURCE_REVISION = "dd3561f544053fe22d739b6b2f8461c9c97bf8cb"
CHECKPOINT_REVISION = "6a5cf5e27b8a05930911a5416fa8c68eabdda414"


def local_component(source):
    from worldfoundry.core.io.paths import resolve_local_hf_model_path

    candidate = Path(source).expanduser()
    if not candidate.exists():
        candidate = Path(resolve_local_hf_model_path(str(source)))
    # HF snapshot files are symlinks to extensionless blobs. Keep the leaf
    # name so version validation and the core safetensors loader retain it.
    return candidate.parent.resolve() / candidate.name if candidate.is_file() else candidate.resolve()


class InspatioWorldV15Runtime:
    """Local-only loading, independent request RNG and bounded VAE/video I/O."""

    def __init__(self, checkpoint_source=DEFAULT_CHECKPOINT_REPO, *,
                 wan_model_path=DEFAULT_WAN_MODEL_REPO, da3_model_path=DEFAULT_DA3_MODEL_REPO,
                 device="cuda", weight_dtype=None, compile_dit=False):
        self.checkpoint_source = str(checkpoint_source)
        self.wan_model_path = str(wan_model_path)
        self.da3_model_path = str(da3_model_path)
        self.device = str(device)
        self.weight_dtype = weight_dtype
        self.compile_dit = bool(compile_dit)
        self.rollout = self.vae = self.text_encoder = None
        self._lock = threading.RLock()

    def _checkpoint(self):
        source = local_component(self.checkpoint_source)
        if source.is_file():
            if source.name != CHECKPOINT_FILENAME:
                raise ValueError(f"InSpatio 1.5 requires {CHECKPOINT_FILENAME}; older weights are a different model")
            return source
        for candidate in (source / CHECKPOINT_FILENAME, source / "InSpatio-World-1.3B" / CHECKPOINT_FILENAME):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Missing InSpatio 1.5 checkpoint {CHECKPOINT_FILENAME} under {source}")

    def plan(self):
        checkpoint = self._checkpoint()
        wan = local_component(self.wan_model_path)
        required = ("config.json", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer_config.json",
                    "google/umt5-xxl/spiece.model")
        for name in required:
            if not (wan / name).is_file():
                raise FileNotFoundError(f"Missing Wan component: {wan / name}")
        text = next((wan / name for name in ("models_t5_umt5-xxl-enc-bf16.safetensors",
                                             "models_t5_umt5-xxl-enc-bf16.pth") if (wan / name).is_file()), None)
        if text is None:
            raise FileNotFoundError(f"Missing Wan UMT5 encoder under {wan}")
        return {"model_id": "inspatio-world-1p5", "checkpoint_path": str(checkpoint),
                "wan_model_path": str(wan), "text_encoder_path": str(text),
                "resolution": [480, 832], "frames_per_block": 3,
                "source_revision": SOURCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION}

    def _load_components(self):
        if self.rollout is not None:
            return
        import torch

        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.camera_21 import WanVAE
        from worldfoundry.base_models.diffusion_model.models.denoisers.wan_causal_world import WanCausalWorldDenoiser
        from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import T5EncoderModel
        from worldfoundry.base_models.diffusion_model.runners.inspatio_world import InspatioCausalRollout

        plan = self.plan()
        dtype = self.weight_dtype or torch.bfloat16
        if dtype != torch.bfloat16:
            raise ValueError("The released InSpatio 1.5 inference path uses torch.bfloat16")
        wan = Path(plan["wan_model_path"])
        denoiser = WanCausalWorldDenoiser.from_pretrained(
            plan["checkpoint_path"], wan, device=self.device, dtype=dtype,
        )
        text_encoder = T5EncoderModel(
            text_len=512, dtype=dtype, device="cpu",
            checkpoint_path=plan["text_encoder_path"], tokenizer_path=str(wan / "google/umt5-xxl"),
            load_with_core_loader=True, return_full_context=True,
        )
        vae = WanVAE(vae_pth=str(wan / "Wan2.1_VAE.pth"), dtype=torch.float32, device="cpu")
        vae.model.to(device=self.device, dtype=dtype)
        # Cast std BEFORE inversion, as in the released v1.5 encoder.
        vae.mean = vae.mean.to(device=self.device, dtype=dtype)
        vae.std = vae.std.to(device=self.device, dtype=dtype)
        vae.scale = [vae.mean, 1 / vae.std]

        def encode_text(text_prompts):
            text_encoder.to(self.device)
            try:
                return {"prompt_embeds": torch.stack(text_encoder(text_prompts, self.device))}
            finally:
                text_encoder.to("cpu")

        config = SimpleNamespace(denoising_step_list=[1000, 750, 500, 250],
                                 warp_denoising_step=True, num_frame_per_block=3)
        rollout = InspatioCausalRollout(config, generator=denoiser, text_encoder=encode_text).eval()
        if self.compile_dit:
            denoiser.model = torch.compile(denoiser.model, mode="max-autotune", fullgraph=False, dynamic=False)
        self.vae, self.text_encoder, self.rollout = vae, text_encoder, rollout

    def _estimate_depth(self, record):
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import DepthAnything3

        from .inspatio_world_runtime.depth.depth_utils import smooth_gaussian
        from .v15_io import encode_depth, encode_video_frame, video_writer, write_range
        from .v15_scene import RESOLUTION

        scene = Path(record["path"])
        if record["views"] > 1000:
            raise ValueError("DA3 supports up to 1000 source frames here; supply a prepared scene for longer inputs")
        if record["kind"] == "image":
            frames = [np.asarray(Image.open(path).convert("RGB")) for path in sorted((scene / "input").glob("view_*.png"))]
        else:
            from worldfoundry.core.media.codecs.video import load_video_frames
            frames = [np.asarray(frame) for frame in load_video_frames(scene / "input/video.mp4")]
        depth_model = DepthAnything3.from_pretrained(str(local_component(self.da3_model_path))).to(self.device).eval()
        try:
            prediction = depth_model.inference(frames, use_ray_pose=False, process_res=504)
        finally:
            del depth_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        depths = np.asarray(prediction.depth, dtype=np.float32)
        intrinsic = np.array(prediction.intrinsics, copy=True)
        poses = np.asarray(prediction.extrinsics)
        tcw = np.tile(np.eye(4, dtype=poses.dtype), (len(poses), 1, 1))
        tcw[:, :3, :4] = poses[:, :3, :4]
        ctw = np.linalg.inv(tcw)
        ctw = np.linalg.inv(ctw[0]) @ ctw
        smoothed = np.asarray(smooth_gaussian(ctw, sigma=2.0))
        source_tcw = np.linalg.inv(smoothed)
        height, width = RESOLUTION
        inferred_height, inferred_width = prediction.processed_images.shape[1:3]
        intrinsic[:, 0] *= width / inferred_width
        intrinsic[:, 1] *= height / inferred_height
        directory = scene / "depth"
        directory.mkdir(exist_ok=True)
        np.savetxt(directory / "source_intrinsics.txt", intrinsic.reshape(-1, 9))
        np.savetxt(directory / "source_tcw.txt", source_tcw.reshape(-1, 16))
        values = [cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST) for depth in depths]
        finite = np.concatenate([value[np.isfinite(value)] for value in values])
        if not finite.size:
            raise ValueError("DA3 returned no finite depth")
        minimum = 0.0 if record["kind"] == "image" else float(finite.min())
        maximum = max(float(finite.max()), minimum + 1e-6)
        write_range(directory / "metadata.txt", minimum, maximum)
        if record["kind"] == "image":
            from .v15_io import image_depth_paths
            for path, value in zip(image_depth_paths(directory, len(values)), values):
                value = np.where(np.isfinite(value) & (value >= 0), value, 0)
                Image.fromarray(encode_depth(value, minimum, maximum)).save(path)
        else:
            with video_writer(directory / "depth.mp4", record["fps"], width, height, depth=True) as writer:
                for value in values:
                    if not np.isfinite(value).all():
                        raise ValueError("DA3 returned non-finite video depth")
                    writer.write(encode_video_frame(encode_depth(value, minimum, maximum)).tobytes())

    def _encode_condition(self, chunks):
        import torch

        from .v15_io import normalize_rgb

        self.vae.model.clear_cache()
        try:
            latents = [self.vae.model.cached_encode(
                normalize_rgb(frames, self.device, self.vae.model.conv1.weight.dtype).permute(1, 0, 2, 3)[None],
                self.vae.scale,
            ) for frames in chunks]
            return torch.cat(latents, dim=2).permute(0, 2, 1, 3, 4)
        finally:
            chunks.close()
            self.vae.model.clear_cache()

    def _infer_record(self, record, condition_dir, *, seed):
        import torch

        from worldfoundry.base_models.diffusion_model.runners.inspatio_world import sample_noise_like

        from .v15_io import iter_mask_latents, iter_video_chunks
        from .v15_scene import padded_frame_count

        self._load_components()
        valid = int(record["valid_frames"])
        padded = padded_frame_count(valid)
        def chunks(name):
            return iter_video_chunks(Path(condition_dir) / f"{name}.mp4", valid, padded,
                                     repeat_first=name == "source" and record["kind"] == "image" and record["views"] == 1)
        # Retain the released order, including initial-noise and inter-step draws.
        render = self._encode_condition(chunks("render"))
        source = self._encode_condition(chunks("source"))
        mask_chunks = chunks("mask")
        try:
            mask = torch.cat(list(iter_mask_latents(mask_chunks, self.device, source.dtype)), dim=1)
        finally:
            mask_chunks.close()
        rng = torch.Generator(device=self.device).manual_seed(int(seed))
        noise = sample_noise_like(source, rng)
        return self.rollout.inference(noise, [record["text"]], ref_latent=source,
                                      render_latent=render, mask_latent=mask, decode=False, noise_generator=rng)

    def predict(self, *, images=None, videos=None, scene_dir=None, prompt="", traj_txt_path=None,
                output_dir=None, output_path=None, seed=0, return_frames=False, return_latents=False):
        import torch

        from .v15_io import video_writer
        from .v15_render import render_one
        from .v15_scene import load_scene, source_type_for, stage_direct_input, validate_scene

        if scene_dir is not None and (images is not None or videos is not None):
            raise ValueError("Supply a prepared scene_dir or direct images/videos")
        if output_dir is None:
            output_dir = Path(output_path).expanduser().parent if output_path is not None else tempfile.gettempdir()
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="inspatio-v15-", dir=directory))
        with self._lock, torch.no_grad():
            self.plan()
            if scene_dir is not None:
                record = load_scene(scene_dir, prompt=prompt, traj_txt_path=traj_txt_path)
            else:
                record = stage_direct_input(work / ".prepared", images=images, videos=videos,
                                            prompt=prompt, traj_txt_path=traj_txt_path)
                self._estimate_depth(record)
                validate_scene(record)
            details = render_one(record, torch.device(self.device), work)
            latents = self._infer_record(record, work / record["id"], seed=seed)
            destination = (Path(output_path).expanduser().resolve() if output_path is not None
                           else work / record["id"] / "pred.mp4")
            valid, written = int(record["valid_frames"]), 0
            self.vae.model.clear_cache()
            try:
                with video_writer(destination, record["fps"]) as writer:
                    for latent in latents.permute(0, 2, 1, 3, 4).split(1, dim=2):
                        pixels = (self.vae.model.cached_decode(latent, self.vae.scale).float().clamp(-1, 1) * .5 + .5)
                        count = min(pixels.shape[2], valid - written)
                        if count:
                            rgb = (pixels[0, :, :count].permute(1, 2, 3, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                            writer.write(rgb.tobytes())
                            written += count
                        if written == valid:
                            break
                    if written != valid:
                        raise RuntimeError(f"Generated {written} frames, expected {valid}")
            finally:
                self.vae.model.clear_cache()
            result = {"model_id": "inspatio-world-1p5", "artifact_path": str(destination),
                      "video_path": str(destination), "videos": [str(destination)],
                      "source_type": source_type_for(record), "valid_frames": valid,
                      "fps": record["fps"], "seed": int(seed), "work_dir": str(work),
                      "view_selection": details}
            if return_frames:
                from worldfoundry.core.media.codecs.video import load_video_frames
                result["video"] = load_video_frames(destination)
            if return_latents:
                result["latents"] = latents.detach().cpu()
            return result
