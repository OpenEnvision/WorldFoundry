import os
import json
import argparse
import torch.distributed as dist
from PIL import Image
import decord
from worldfoundry.core.io import save_video
from worldfoundry.synthesis.visual_generation.longvie.worldfoundry_runtime import (
    LongVieOfficialRuntime,
)

# Target resolution (width, height)
TARGET_SIZE = (640, 352)


def load_image(path):
    return Image.open(path).convert("RGB").resize(TARGET_SIZE)


def resize_video_frames(video_np):
    return [Image.fromarray(frame).resize(TARGET_SIZE) for frame in video_np]


def main(args):
    runtime = LongVieOfficialRuntime(
        control_weight_path=args.control_weight_path,
        dit_weight_path=args.dit_weight_path or None,
        weight_dir=None,
        wan_base_dir=args.wan_base_dir,
        tokenizer_dir=args.tokenizer_dir,
        device="cuda",
        use_usp=args.use_usp,
        ring_degree=args.ring_degree,
        ulysses_degree=args.ulysses_degree,
        variant="longvie-2" if args.dit_weight_path else "longvie-1",
    )
    pipe = runtime.load()

    with open(args.json_file, "r") as f:
        samples = json.load(f)

    image = load_image(args.image_path)
    history = []
    noise = None

    for i, sample in enumerate(samples):
        dense_vr = decord.VideoReader(sample["depth"])
        sparse_vr = decord.VideoReader(sample["track"])

        dense_frames = resize_video_frames(dense_vr[:].asnumpy())
        sparse_frames = resize_video_frames(sparse_vr[:].asnumpy())

        video, noise = pipe(
            input_image=image,
            prompt=sample["text"],
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=args.seed,
            tiled=False,
            height=TARGET_SIZE[1],
            width=TARGET_SIZE[0],
            dense_video=dense_frames,
            sparse_video=sparse_frames,
            history=history,
            noise=noise,
        )

        image = video[-1]
        history = video[-8:]

        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir = f"./gen_videos/{args.video_name}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{i}.mp4")
            save_video(video, save_path, fps=16, quality=10)
            print(f"[Saved] {save_path}")

        if dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_weight_path", type=str, required=True)
    parser.add_argument("--dit_weight_path", type=str, default="")
    parser.add_argument("--wan_base_dir", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--use_usp", action="store_true", help="Enable USP (default: False; set to True if provided).")
    args = parser.parse_args()
    main(args)
