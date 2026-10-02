import argparse
import os
import re
import json
import sys
from pathlib import Path
from typing import List, Optional

_RUNTIME_ROOT = str(Path(__file__).resolve().parents[1])
if _RUNTIME_ROOT in sys.path:
    sys.path.remove(_RUNTIME_ROOT)
sys.path.insert(0, _RUNTIME_ROOT)

from uni3c_cam_render_api import render_from_image_and_traj

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms
from einops import rearrange
import torchvision.transforms.functional as TF
from PIL import Image

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalInferencePipeline

from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.video_x_fun import (
    CLIPModel,
)
from utils.camera_pose import process_pose_file
from videox_fun.utils.utils import get_video_to_video_render_latent
from worldfoundry.core.utils.tensors.torch import set_seed_everywhere as set_seed
from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.core.media.codecs.video import write_video_torchvision as write_video

from worldfoundry.core.vram import DynamicSwapInstaller, get_cuda_free_memory_gb


# -------------------------
# Image & Prompt helpers
# -------------------------
def apply_transform(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)


def load_prompt_index(json_path: str) -> dict:
    """
    JSON expected: list of {"name": "...png", "describe": "..."}.
    Build:
      - exact filename -> describe
      - stem -> describe
    """
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    by_name = {}
    by_stem = {}
    for it in items:
        name = str(it.get("name", "")).strip()
        desc = str(it.get("describe", "")).strip()
        if not name:
            continue
        by_name[name] = desc
        stem = Path(name).stem
        if stem and stem not in by_stem:
            by_stem[stem] = desc
    return {"by_name": by_name, "by_stem": by_stem, "items": items}


def query_prompt(prompt_index: dict, image_path: str, default_prompt: str = "") -> str:
    img_name = Path(image_path).name
    img_stem = Path(image_path).stem

    if img_name in prompt_index["by_name"]:
        return prompt_index["by_name"][img_name] or default_prompt

    if img_stem in prompt_index["by_stem"]:
        return prompt_index["by_stem"][img_stem] or default_prompt

    # loose hit: stem contained in JSON "name"
    for it in prompt_index["items"]:
        n = str(it.get("name", ""))
        if img_stem and (img_stem in n):
            d = str(it.get("describe", "")).strip()
            if d:
                return d

    return default_prompt


def safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: str, exts=(".png", ".jpg", ".jpeg", ".webp")):
        self.image_dir = Path(image_dir)
        self.paths = []
        for p in sorted(self.image_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                self.paths.append(str(p))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


# -------------------------
# History helpers
# -------------------------
def _normalize_feats(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return feats / (feats.norm(dim=-1, keepdim=True) + eps)


def _pool_latent_frames(latent: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    支持:
      - [B, C, H, W] -> [B, C]
      - [B, F, C, H, W] -> [B*F, C]
    """
    if latent.dim() == 4:
        # [B,C,H,W]
        if mode == "mean":
            return latent.mean(dim=(-1, -2))
        elif mode == "max":
            return latent.amax(dim=(-1, -2))
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")

    elif latent.dim() == 5:
        # [B,F,C,H,W]
        B, F, C, H, W = latent.shape
        x = latent.reshape(B * F, C, H, W)
        if mode == "mean":
            return x.mean(dim=(-1, -2))   # [B*F, C]
        elif mode == "max":
            return x.amax(dim=(-1, -2))
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")

    else:
        raise ValueError(f"Unsupported latent dim: {latent.dim()}")


def _append_cache_from_pred_latent(
    cache_list: List[torch.Tensor],
    pred_latent: torch.Tensor,
    detach: bool = True
) -> None:
    """
    pred_latent: [B, F, C, H, W]
    cache_list: List[[B, F_i, C, H, W]]
    """
    assert pred_latent.dim() == 5, f"expect [B,F,C,H,W], got {pred_latent.shape}"
    frames = pred_latent.detach() if detach else pred_latent
    cache_list.append(frames)


def _pack_cache_list(cache_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    List[[B,F_i,C,H,W]] -> [B,F_total,C,H,W]
    """
    if not cache_list:
        return None

    B, _, C, H, W = cache_list[0].shape
    device, dtype = cache_list[0].device, cache_list[0].dtype

    for t in cache_list:
        assert t.dim() == 5 and t.shape[0] == B and t.shape[2] == C and t.shape[3] == H and t.shape[4] == W, \
            f"Incompatible cache tensor: expect [B,*,C,H,W], got {t.shape}"
        assert t.device == device and t.dtype == dtype, "All cache tensors must share device/dtype"

    return torch.cat(cache_list, dim=1)  # [B,F_total,C,H,W]


def _trim_cache_list_to_max_frames(
    cache_list: List[torch.Tensor],
    max_frames: int
) -> List[torch.Tensor]:
    if max_frames <= 0:
        return []

    total_frames = sum(x.shape[1] for x in cache_list)
    if total_frames <= max_frames:
        return cache_list

    new_list = list(cache_list)
    while len(new_list) > 0 and total_frames > max_frames:
        oldest = new_list[0]
        oldest_frames = oldest.shape[1]
        need_remove = total_frames - max_frames

        if oldest_frames <= need_remove:
            new_list.pop(0)
            total_frames -= oldest_frames
        else:
            new_list[0] = oldest[:, need_remove:, ...].contiguous()
            total_frames -= need_remove

    return new_list


def _select_topk_from_cache_list(
    query_first_latent: torch.Tensor,   # [B,Fq,C,H,W]，只用首帧
    cache_list: List[torch.Tensor],     # List[[B,F_i,C,H,W]]
    topk: int,
    cache_pool: str = "mean"
) -> torch.Tensor:
    """
    返回 [B, C, K, H, W]
    """
    assert query_first_latent.dim() == 5
    B, _, C, H, W = query_first_latent.shape

    def _empty_like():
        return query_first_latent.new_empty((B, C, 0, H, W))

    if (not cache_list) or (topk <= 0):
        return _empty_like()

    cache_frames = _pack_cache_list(cache_list)  # [B,F_total,C,H,W]
    if cache_frames is None or cache_frames.size(1) == 0:
        return _empty_like()

    # query 取首帧
    q = query_first_latent[:, 0, ...]  # [B,C,H,W]
    q_feats = _normalize_feats(_pool_latent_frames(q, mode=cache_pool))  # [B,C]

    # cache feats
    Bc, F_total, Cc, Hc, Wc = cache_frames.shape
    assert Bc == B and Cc == C and Hc == H and Wc == W, "cache dims mismatch"

    cache_feats_flat = _pool_latent_frames(cache_frames, mode=cache_pool)  # [B*F_total,C]
    cache_feats = _normalize_feats(cache_feats_flat.view(B, F_total, C))   # [B,F_total,C]

    K = min(topk, F_total)
    sim = torch.einsum("bc,bfc->bf", q_feats, cache_feats)  # [B,F_total]
    _, idx = torch.topk(sim, k=K, dim=1)                    # [B,K]

    gather_idx = idx.view(B, K, 1, 1, 1).expand(B, K, C, H, W)
    topk_frames = torch.gather(cache_frames, dim=1, index=gather_idx)  # [B,K,C,H,W]
    topk_frames = topk_frames.permute(0, 2, 1, 3, 4).contiguous()      # [B,C,K,H,W]
    return topk_frames


# -------------------------
# AR helpers
# -------------------------
def video_last_frame_to_pil(video_btchw: torch.Tensor) -> Image.Image:
    """
    video_btchw: [B,T,C,H,W].
    Robustly map to uint8 RGB. Supports outputs in [-1,1] or [0,1].
    """
    assert video_btchw.ndim == 5
    frame = video_btchw[0, -1].detach().float().cpu()  # [C,H,W]

    if frame.min().item() < 0:
        frame = (frame * 0.5 + 0.5).clamp(0, 1)
    else:
        frame = frame.clamp(0, 1)

    frame_u8 = (frame * 255.0).round().to(torch.uint8)
    frame_u8 = frame_u8.permute(1, 2, 0).numpy()
    return Image.fromarray(frame_u8, mode="RGB")


def build_i2v_conditions_from_pil(
    start_pil: Image.Image,
    pipeline,
    clip_image_encoder,
    sampled_noise: torch.Tensor,
    render_input: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    use_render: bool = False,
):
    """
    Returns:
        y_input
        clip_context
        render_latent_bfchw
        start_latent_bfchw
    """
    # clip context
    clip_image = start_pil.convert("RGB")
    clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device=device, dtype=dtype)
    clip_context = clip_image_encoder([clip_image[:, None, :, :]])

    # start image latent
    # y_input from VAE latent of resized start image
    start_tensor = apply_transform(start_pil.convert("RGB")).squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    start_latent = pipeline.vae.encode_to_latent(start_tensor).to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)


    start_latent_bfchw = start_latent.permute(0, 2, 1, 3, 4).contiguous()  # [B,F,C,H,W]

    # noise is [B,T,16,60,104] in your code; conv_in expects [B,16,T,60,104]
    start_latents_conv_in = torch.zeros_like(sampled_noise).permute(0, 2, 1, 3, 4)
    if sampled_noise.size(1) != 1:
        start_latents_conv_in[:, :, :1] = start_latent
    y_input = start_latents_conv_in

    # render latent
    render_latent_bcfhw = pipeline.vae.encode_to_latent(render_input).to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)

    # 如果启用 render，则把 render_latent 在通道维拼到 y_input 上
    if use_render:
        y_input = torch.cat([y_input, render_latent_bcfhw], dim=1)


    return y_input, clip_context, start_latent_bfchw


def build_camera_latents_for_segment(
    control_camera_video_full: torch.Tensor,   # [L,H,W,C]
    control_camera_traj_full,                  # list[L][D]
    current_start_pil,
    sample_size: list,
    seg_id: int,
    camera_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    control_camera_video_full: output of process_pose_file(...)
    expected shape like [L, H, W, C]
    returns y_camera_input tensor ready for pipeline, and render_video.
    """
    start = seg_id * camera_length
    end = (seg_id + 1) * camera_length

    seg_cam_video = control_camera_video_full[start:end]   # [L,H,W,C]
    seg_cam_traj = control_camera_traj_full[start:end]     # list of [D]

    render_frames, mask_frames = render_from_image_and_traj(
        reference_image=current_start_pil,
        traj_params=seg_cam_traj,
        output_path=os.environ.get("WORLDFOUNDRY_MAGICWORLD_UNI3C_OUTPUT", "/tmp/worldfoundry-magicworld-uni3c"),
        traj_type="free1",
        nframe=camera_length,
    )

    render_video, _, _, _, _ = get_video_to_video_render_latent(
        render_frames,
        mask_frames,
        video_length=camera_length,
        sample_size=sample_size,
        ref_image=None
    )

    seg_cam_video = seg_cam_video.permute([3, 0, 1, 2]).unsqueeze(0)  # [B,C,L,H,W]

    control_camera_latents = torch.concat(
        [
            torch.repeat_interleave(seg_cam_video[:, :, 0:1], repeats=4, dim=2),
            seg_cam_video[:, :, 1:]
        ],
        dim=2
    ).transpose(1, 2)

    b, f, c, h, w = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)

    return control_camera_latents.to(device=device, dtype=dtype), render_video.to(device=device, dtype=dtype)


def normalize_pred_latents_to_bfchw(
    pred_latents: torch.Tensor,
    num_output_frames: int
) -> torch.Tensor:
    """
    统一把 pipeline 返回的 latent 转成 [B,F,C,H,W]
    支持:
      - [B,F,C,H,W]
      - [B,C,F,H,W]
    """
    if pred_latents.dim() != 5:
        raise ValueError(f"_latents should be 5D, got {pred_latents.shape}")

    if pred_latents.shape[1] == num_output_frames:
        return pred_latents.contiguous()
    elif pred_latents.shape[2] == num_output_frames:
        return pred_latents.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(
            f"Cannot infer latent layout from _latents shape: {pred_latents.shape}, "
            f"num_output_frames={num_output_frames}"
        )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Folder containing start images")
    parser.add_argument("--extended_prompt_path", type=str, required=True, help="JSON prompt index")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--control_camera_txt", type=str, required=True, help="One fixed camera txt used for all images")

    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--i2v", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_with_index", action="store_true")

    parser.add_argument("--use_render", action="store_true", help="Concat render_latent into y_input")
    parser.add_argument("--use_history", action="store_true", help="Enable history retrieval and history cache update")
    parser.add_argument("--history_max_frames", type=int, default=20)
    parser.add_argument("--history_topk", type=int, default=3)

    args = parser.parse_args()

    # -------- distributed init --------
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda")
        set_seed(args.seed)

    torch.set_grad_enabled(False)

    print(f"[rank{local_rank}] Free VRAM {get_cuda_free_memory_gb(device)} GB")
    low_memory = get_cuda_free_memory_gb(device) < 40

    # -------- config --------
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(os.environ["WORLDFOUNDRY_MAGICWORLD_DEFAULT_CONFIG"])
    config = OmegaConf.merge(default_config, config)

    # -------- init pipeline --------
    if not hasattr(config, "denoising_step_list"):
        raise ValueError("MagicWorld-Fast requires denoising_step_list in its inference config")
    pipeline = CausalInferencePipeline(config, device=device)

    # -------- load checkpoint --------
    state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    pipeline.generator.load_state_dict(
        state_dict["generator" if not args.use_ema else "generator_ema"]
    )
    checkpoint_step = os.path.basename(os.path.dirname(args.checkpoint_path))
    checkpoint_step = checkpoint_step.split("_")[-1]

    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)

    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # -------- clip image encoder --------
    clip_config_path = os.environ.get(
        "WORLDFOUNDRY_MAGICWORLD_CONFIG",
        str(
            resolve_data_path(
                "models", "runtime", "configs", "video_x_fun", "wan2.1", "wan_civitai.yaml"
            )
        ),
    )
    clip_name = os.environ.get(
        "WORLDFOUNDRY_MAGICWORLD_WAN_ROOT",
        "checkpoints/Wan2.1-Fun-V1.1-1.3B-InP",
    )
    clip_config = OmegaConf.load(clip_config_path)

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(
            clip_name,
            clip_config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")
        ),
    ).to(device=device, dtype=torch.bfloat16)
    clip_image_encoder = clip_image_encoder.eval()

    # -------- prompt index --------
    prompt_index = load_prompt_index(args.extended_prompt_path)

    # -------- camera (fixed) --------
    sample_size = [480, 832]
    temporal_compression_ratio = 4
    camera_length = (args.num_output_frames - 1) * temporal_compression_ratio + 1

    control_camera_video_full = process_pose_file(
        args.control_camera_txt, sample_size[1], sample_size[0]
    )  # [L,H,W,C], plucker embedding
    control_camera_traj_full = process_pose_file(
        args.control_camera_txt, sample_size[1], sample_size[0], return_poses=True
    )

    segments = 1


    need_len = segments * camera_length
    if control_camera_video_full.shape[0] < need_len:
        raise ValueError(
            f"Fixed camera pose too short: got {control_camera_video_full.shape[0]}, need >= {need_len} "
            f"(segments={segments}, camera_length={camera_length})."
        )

    # -------- output dir --------
    out_dir = os.path.join(args.output_folder, checkpoint_step)
    if local_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # -------- dataset & loader --------
    dataset = ImageFolderDataset(args.data_path)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True)

    # -------- checks --------
    if not args.i2v:
        raise ValueError("This script is written for I2V mode. Please pass --i2v.")

    if args.num_samples != 1:
        raise ValueError("This script currently supports --num_samples 1.")

    default_prompt = ""

    # -------- run --------
    for idx, img_path in enumerate(loader):
        img_path = img_path[0]
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem
        history_cache = []   # List[[B,F,C,H,W]]

        prompt = query_prompt(prompt_index, img_path, default_prompt=default_prompt)
        if not prompt:
            prompt = "A high-quality cinematic video."

        prompts = [prompt] * args.num_samples

        # 自回归交互：start 图像会被更新为上一段 last frame (RGB)
        current_start_pil = Image.open(img_path).convert("RGB")

        all_video = []

        for seg_id in range(segments):
            # -------- noise per segment --------
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16
            )

            # -------- fixed camera segment --------
            y_camera_input, render_input = build_camera_latents_for_segment(
                control_camera_video_full=control_camera_video_full,
                control_camera_traj_full=control_camera_traj_full,
                current_start_pil=current_start_pil,
                sample_size=sample_size,
                seg_id=seg_id,
                camera_length=camera_length,
                device=device,
                dtype=torch.bfloat16
            )

            # -------- build conditions from current_start_pil --------
            y_input, clip_context, start_latent_bfchw = build_i2v_conditions_from_pil(
                current_start_pil,
                pipeline=pipeline,
                clip_image_encoder=clip_image_encoder,
                sampled_noise=sampled_noise,
                render_input=render_input,
                device=device,
                dtype=torch.bfloat16,
                use_render=args.use_render,
            )

            # -------- build y_history from cache --------
            if not args.use_history:
                y_history = None
            else:
                if len(history_cache) == 0:
                    y_history = None
                else:
                    y_history = _select_topk_from_cache_list(
                        query_first_latent=start_latent_bfchw,   # [B,F,C,H,W]
                        cache_list=history_cache,
                        topk=args.history_topk,
                        cache_pool="mean",
                    )  # [B,C,K,H,W]

                    if y_history.size(2) == 0:
                        y_history = None
                    else:
                        print("y_history:", y_history.size())

            # -------- inference --------
            video, _latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=None,
                y_input=y_input,
                y_camera_input=y_camera_input,
                y_history=y_history,
                clip_context=clip_context,
                low_memory=low_memory,
            )

            # -------- append current latents to history cache --------
            if args.use_history and (_latents is not None):
                pred_latent_bfchw = normalize_pred_latents_to_bfchw(
                    pred_latents=_latents,
                    num_output_frames=args.num_output_frames
                )

                _append_cache_from_pred_latent(
                    cache_list=history_cache,
                    pred_latent=pred_latent_bfchw,
                    detach=True
                )
                history_cache = _trim_cache_list_to_max_frames(
                    history_cache,
                    max_frames=args.history_max_frames
                )

                total_hist_frames = sum(x.shape[1] for x in history_cache)
                print(f"history cache size: {len(history_cache)} chunks, total_frames={total_hist_frames}")

            # -------- update start_image for next segment --------
            current_start_pil = video_last_frame_to_pil(video)

            # -------- concat output videos --------
            if seg_id == 0:
                seg_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            else:
                seg_video = rearrange(video[:, 1:], "b t c h w -> b t h w c").cpu()

            all_video.append(seg_video)

        # -------- final output --------
        video_out = 255.0 * torch.cat(all_video, dim=1)  # [B,T_total,H,W,C]
        pipeline.vae.model.clear_cache()

        # -------- filename --------
        if args.save_with_index:
            base = f"{idx:06d}_{img_stem}"
        else:
            base = img_stem

        base = safe_filename(base)
        output_path = os.path.join(out_dir, f"{base}.mp4")

        # -------- write --------
        write_video(output_path, video_out[0], fps=16)

        if local_rank == 0:
            print(f"[OK] {img_name} -> {output_path}")

        # -------- release history cache for current img_path --------
        history_cache.clear()
        del history_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
