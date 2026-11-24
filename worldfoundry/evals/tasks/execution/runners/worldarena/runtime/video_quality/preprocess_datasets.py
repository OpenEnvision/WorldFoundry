import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset structure preprocessing script")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json")
    parser.add_argument(
        "--gen_video_dir", type=str, required=True, help="Path to generated videos (e.g., Genie_agi_out_sort)"
    )
    parser.add_argument("--output_base", type=str, default="your absolute path", help="Output base directory")
    return parser.parse_args()


def extract_frames(video_path, output_dir):
    """Extract a video into individual frames."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Format as frame_00000.jpg
        frame_name = f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1
    cap.release()


def process_item(item, gen_video_dir, output_base):
    # 1. Extract IDs from a path like /.../327/651177/...
    gt_video_path = Path(item["gt_path"])
    parts = gt_video_path.parts
    # Assume the fourth item from the end is ID1 (e.g., 327) and the third is ID2 (e.g., 651177)
    # id2 is like episode0.mp4; drop the extension to keep the prefix only
    id1, id2 = parts[-5], parts[-1].split(".")[0]

    # 2. Build GT path structure
    gt_root = Path(output_base) / "gt_dataset" / id1 / id2
    prompt_dir = gt_root / "prompt"
    video_dir_gt = gt_root / "video"

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(video_dir_gt, exist_ok=True)

    # Save prompt.txt
    prompt_content = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
    with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_content)

    # Save init_frame.png
    src_image = Path(item["image"])
    if src_image.exists():
        shutil.copy2(src_image, prompt_dir / "init_frame.png")

    # Unpack GT video frames
    if gt_video_path.exists():
        extract_frames(gt_video_path, video_dir_gt)

    # 3. Build Generated path structure
    gen_root_1 = Path(output_base) / "generated_dataset" / id1 / id2 / "1"
    video_dir_gen_1 = gen_root_1 / "video"
    os.makedirs(video_dir_gen_1, exist_ok=True)

    # Find generated video (format 327_651177.mp4)
    target_gen_video_1 = Path(gen_video_dir) / f"{id1}_{id2}.mp4"
    if target_gen_video_1.exists():
        extract_frames(target_gen_video_1, video_dir_gen_1)
    else:
        print(f"Warning: Generated video not found for {id1}_{id2}")


def main():
    args = parse_args()

    if not os.path.exists(args.summary_json):
        print(f"Error: summary.json not found at {args.summary_json}")
        return

    with open(args.summary_json, encoding="utf-8") as f:
        data = json.load(f)

    print(f">>> Starting preprocessing for {len(data)} items...")
    for item in tqdm(data):
        try:
            process_item(item, args.gen_video_dir, args.output_base)
        except Exception as e:
            print(f"Error processing {item.get('gt_path')}: {e}")

    print(f"\n>>> Preprocessing Complete. Structure saved in: {args.output_base}")


if __name__ == "__main__":
    main()
