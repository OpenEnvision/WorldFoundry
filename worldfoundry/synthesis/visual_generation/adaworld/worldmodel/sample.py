from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2 as cv
import imageio
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm
from worldfoundry.core.io.paths import resolve_data_path

from sample_utils import *


DEFAULT_CONFIG = resolve_data_path("models", "runtime", "configs", "adaworld", "worldmodel/inference/adaworld.yaml")
DEFAULT_OUTPUT = Path("outputs") / "adaworld.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaWorld inference-only runner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("WORLDFOUNDRY_ADAWORLD_DATA_ROOT"))
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--target-video", type=Path)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--video-len", type=int, default=20)
    parser.add_argument("--context-frame", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--cfg-scale", type=float, default=1.1)
    parser.add_argument("--aug-level", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=32)
    return parser.parse_args()


def discover_samples(args: argparse.Namespace) -> list[tuple[dict[str, object], dict[str, object]]]:
    if args.source_video:
        target_video = args.target_video or args.source_video
        return [
            (
                {"file_name": str(args.source_video), "start_ind": args.start_index},
                {"file_name": str(target_video), "start_ind": args.start_index},
            )
        ]
    if not args.data_root:
        raise ValueError("AdaWorld inference requires --data-root or --source-video.")

    data_root = Path(args.data_root)
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for family in ("procgen", "retro"):
        family_root = data_root / family
        if not family_root.is_dir():
            continue
        for env_dir in sorted(path for path in family_root.iterdir() if path.is_dir()):
            candidate = env_dir / "test" / "00000.mp4"
            if candidate.is_file():
                item = {"file_name": str(candidate), "start_ind": args.start_index}
                pairs.append((item, item))
    if not pairs:
        raise FileNotFoundError(f"No AdaWorld demo videos were found under {data_root}.")
    return pairs[: max(1, args.num_samples)]


def load_video_slices(
    video_path: str,
    *,
    video_len: int,
    resolution: int,
    start_id: int = 0,
    frame_skip: int = 1,
) -> list[torch.Tensor]:
    cap = cv.VideoCapture(video_path)
    if "retro" in video_path:
        frame_skip = 4
    elif "procgen" not in video_path and "ssv2" not in video_path and "mira" not in video_path:
        frame_skip = 2
    num_frames = video_len * frame_skip

    cap.set(cv.CAP_PROP_POS_FRAMES, start_id)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        elif frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
            break
        else:
            raise RuntimeError(f"Could not read frames from {video_path}.")
    cap.release()

    video = torch.stack(frames[::frame_skip]) / 255.0
    if video.shape[1] != video.shape[2]:
        square_len = min(video.shape[1], video.shape[2])
        h_crop = (video.shape[1] - square_len) // 2
        w_crop = (video.shape[2] - square_len) // 2
        video = video[:, h_crop : h_crop + square_len, w_crop : w_crop + square_len]

    video = rearrange(video, "t h w c -> t c h w")
    if video.shape[-1] != resolution or video.shape[-2] != resolution:
        video = F.interpolate(video, resolution, mode="bicubic")
    return [frame for frame in video]


def get_sample(source_video_dict: dict[str, object], target_video_dict: dict[str, object], args: argparse.Namespace):
    source_image_seq = load_video_slices(
        str(source_video_dict["file_name"]),
        start_id=int(source_video_dict["start_ind"]),
        video_len=args.video_len,
        resolution=args.resolution,
    )
    target_image_seq = load_video_slices(
        str(target_video_dict["file_name"]),
        start_id=int(target_video_dict["start_ind"]),
        video_len=args.video_len,
        resolution=args.resolution,
    )
    lam_inputs = torch.stack(source_image_seq[:2] + [target_image_seq[0]])
    filled_seq = [torch.zeros_like(target_image_seq[0])] * (args.context_frame - 1) + target_image_seq
    gt_frames = torch.stack(filled_seq[args.context_frame - 1 :])
    next_frames = torch.Tensor(filled_seq[args.context_frame])
    prev_frames = torch.stack(filled_seq[: args.context_frame])
    next_frames = next_frames * 2.0 - 1.0
    prev_frames = prev_frames * 2.0 - 1.0
    img_seq = torch.cat([prev_frames, next_frames[None]])

    return {
        "source_video": torch.stack(source_image_seq).to("cuda"),
        "gt_frames": gt_frames.to("cuda"),
        "img_seq": img_seq.to("cuda"),
        "cond_frames_without_noise": prev_frames[-1][None].to("cuda"),
        "cond_frames": (prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]))[None].to("cuda"),
        "lam_inputs": lam_inputs[None].to("cuda"),
        "context_len": torch.Tensor([1]).to("cuda"),
        "context_aug": torch.Tensor([args.aug_level]).to("cuda"),
    }


def run_fdm(fdm_model, source_video_dict: dict[str, object], target_video_dict: dict[str, object], args: argparse.Namespace):
    value_dict = get_sample(source_video_dict, target_video_dict, args)
    sampler = init_sampling(steps=args.num_steps, cfg_scale=args.cfg_scale, n_context_frames=args.context_frame)
    return do_sample(
        fdm_model,
        sampler,
        value_dict,
        input_res=args.resolution,
        force_uc_zero_embeddings=["cond_frames_without_noise", "cond_frames", "lam_inputs"],
    )


def save_video(samples: torch.Tensor, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = 255.0 * rearrange(samples.clamp(0, 1).cpu().numpy(), "t c h w -> t h w c")
    writer = imageio.get_writer(str(output_path), fps=fps)
    for frame in frames.astype("uint8"):
        writer.append_data(frame)
    writer.close()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaWorld inference requires CUDA.")
    seed_everything(args.seed)

    samples_to_run = discover_samples(args)
    model = init_model(str(args.config), str(args.checkpoint))
    generated = []
    with torch.no_grad(), autocast("cuda"):
        for source_item, target_item in tqdm(samples_to_run, desc="AdaWorld inference"):
            samples = run_fdm(model, source_item, target_item, args)
            generated.append(samples.cpu())

    model.cpu()
    torch.cuda.empty_cache()
    save_video(torch.cat(generated, dim=0), args.output_path, args.fps)
    print(f"AdaWorld inference saved to {args.output_path}")


if __name__ == "__main__":
    main()
