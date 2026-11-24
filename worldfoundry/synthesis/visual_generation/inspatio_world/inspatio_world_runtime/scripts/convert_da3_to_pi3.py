#!/usr/bin/env python
"""Convert DA3 depth output to Pi3 format expected by the v2v pipeline.

DA3 outputs:
  - depth/0000.png: RGBA float32 PNG (4 uint8 bytes per float32 pixel)
  - extrinsic.txt: np.savetxt, 3 rows per frame (3x4 w2c matrix)
  - intrinsic.txt: np.savetxt, 3 rows per frame (3x3 matrix)
  - frames/0000.jpg: extracted frames

Pi3 format expected by pipeline:
  - depth/000000.png: uint16 PNG
  - metadata.txt: "min max" on one line
  - extrinsics.txt: one line per frame, np.array2string format (3x4)
  - intrinsics.txt: one line, np.array2string format (3x3)
  - images/000000.png: frames as PNG

Usage:
    python convert_da3_to_pi3.py --da3_dir /path/to/da3_output --output_dir /path/to/pi3_output --video_path /path/to/original.mp4
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image


def read_da3_depth(path):
    """Read DA3 RGBA float32 depth PNG."""
    img = Image.open(path)
    depth_uint8 = np.array(img)  # (H, W, 4)
    h, w = depth_uint8.shape[:2]
    depth_float32 = np.frombuffer(depth_uint8.tobytes(), dtype=np.float32).reshape(h, w)
    return depth_float32


def convert_depths(da3_dir, output_dir):
    """Convert DA3 RGBA float32 depth PNGs to Pi3 uint16 PNGs + metadata.txt."""
    da3_depth_dir = os.path.join(da3_dir, "depth")
    out_depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(out_depth_dir, exist_ok=True)

    depth_files = sorted(
        [f for f in os.listdir(da3_depth_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    # First pass: compute global min/max
    global_min = float("inf")
    global_max = float("-inf")
    all_depths = []
    for df in depth_files:
        depth = read_da3_depth(os.path.join(da3_depth_dir, df))
        all_depths.append(depth)
        global_min = min(global_min, float(depth.min()))
        global_max = max(global_max, float(depth.max()))

    # Write metadata.txt
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"{global_min} {global_max}\n")

    # Second pass: normalize to uint16 and save
    depth_range = global_max - global_min
    if depth_range < 1e-8:
        depth_range = 1.0  # avoid division by zero

    for i, depth in enumerate(all_depths):
        normalized = (depth - global_min) / depth_range  # [0, 1]
        uint16_depth = (normalized * 65535.0).clip(0, 65535).astype(np.uint16)
        out_path = os.path.join(out_depth_dir, f"{i:06d}.png")
        Image.fromarray(uint16_depth).save(out_path)

    print(f"  Converted {len(all_depths)} depth maps (min={global_min:.4f}, max={global_max:.4f})")


def convert_extrinsics(da3_dir, output_dir):
    """Convert DA3 extrinsic.txt (np.savetxt, 3 rows per frame) to Pi3 extrinsics.txt (np.array2string per line)."""
    da3_path = os.path.join(da3_dir, "extrinsic.txt")
    data = np.loadtxt(da3_path)  # (N*3, 4) for 3x4 matrices

    n_rows = data.shape[0]
    assert n_rows % 3 == 0, f"extrinsic.txt has {n_rows} rows, not divisible by 3"
    n_frames = n_rows // 3

    out_path = os.path.join(output_dir, "extrinsics.txt")
    with open(out_path, "w") as f:
        for i in range(n_frames):
            mat = data[i * 3 : (i + 1) * 3]  # (3, 4)
            line = np.array2string(mat, separator=", ", suppress_small=True)
            # Make it a single line
            line = line.replace("\n", "")
            f.write(line + "\n")

    print(f"  Converted {n_frames} extrinsic matrices")


def convert_intrinsics(da3_dir, output_dir):
    """Convert DA3 intrinsic.txt (np.savetxt, 3 rows per frame) to Pi3 intrinsics.txt (np.array2string, first frame only)."""
    da3_path = os.path.join(da3_dir, "intrinsic.txt")
    data = np.loadtxt(da3_path)  # (N*3, 3) for 3x3 matrices

    n_rows = data.shape[0]
    assert n_rows % 3 == 0, f"intrinsic.txt has {n_rows} rows, not divisible by 3"

    # Take first frame's intrinsic
    mat = data[0:3]  # (3, 3)

    out_path = os.path.join(output_dir, "intrinsics.txt")
    with open(out_path, "w") as f:
        line = np.array2string(mat, separator=", ", suppress_small=True)
        line = line.replace("\n", "")
        f.write(line + "\n")

    print("  Converted intrinsics (first frame)")


def convert_images(da3_dir, output_dir, video_path=None, target_width=832, target_height=480):
    """Convert frames to Pi3 images/*.png format.

    If video_path is provided, read from original video with center-crop + resize
    to preserve aspect ratio (avoids DA3's stretching distortion).
    Otherwise fall back to copying DA3 frames/ directly.
    """
    out_images_dir = os.path.join(output_dir, "images")
    os.makedirs(out_images_dir, exist_ok=True)

    if video_path and os.path.isfile(video_path):
        # Read from original video with crop+resize to preserve aspect ratio
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Count DA3 frames to know how many to extract
        da3_frames_dir = os.path.join(da3_dir, "frames")
        n_da3_frames = len([f for f in os.listdir(da3_frames_dir) if f.endswith((".jpg", ".png"))])

        target_ar = target_width / target_height  # 832/480 = 1.7333

        count = 0
        while count < n_da3_frames:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            src_ar = w / h

            if src_ar > target_ar:
                # Source is wider -> crop width
                crop_w = int(h * target_ar)
                x0 = (w - crop_w) // 2
                frame = frame[:, x0 : x0 + crop_w]
            elif src_ar < target_ar:
                # Source is taller -> crop height
                crop_h = int(w / target_ar)
                y0 = (h - crop_h) // 2
                frame = frame[y0 : y0 + crop_h, :]

            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            # BGR -> RGB -> PIL -> save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(os.path.join(out_images_dir, f"{count:06d}.png"))
            count += 1

        cap.release()
        print(f"  Converted {count} frames from original video (crop+resize)")
    else:
        # Fallback: copy DA3 frames directly
        da3_frames_dir = os.path.join(da3_dir, "frames")
        frame_files = sorted(
            [f for f in os.listdir(da3_frames_dir) if f.endswith((".jpg", ".png"))],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        for i, ff in enumerate(frame_files):
            src = os.path.join(da3_frames_dir, ff)
            dst = os.path.join(out_images_dir, f"{i:06d}.png")
            if ff.endswith(".png"):
                shutil.copy2(src, dst)
            else:
                img = Image.open(src)
                img.save(dst)

        print(f"  Converted {len(frame_files)} frames (DA3 fallback)")


def main():
    parser = argparse.ArgumentParser(description="Convert DA3 depth output to Pi3 format")
    parser.add_argument("--da3_dir", type=str, required=True, help="DA3 output directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Pi3-format output directory")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Original video path (for crop+resize to fix DA3 stretch distortion)",
    )
    args = parser.parse_args()

    print("Converting DA3 -> Pi3 format:")
    print(f"  DA3 dir:    {args.da3_dir}")
    print(f"  Output dir: {args.output_dir}")
    if args.video_path:
        print(f"  Video path: {args.video_path}")

    # Clean output directory to avoid stale files from previous runs
    if os.path.exists(args.output_dir):
        for sub in ["depth", "images"]:
            sub_dir = os.path.join(args.output_dir, sub)
            if os.path.exists(sub_dir):
                shutil.rmtree(sub_dir)
        for f in ["metadata.txt", "extrinsics.txt", "intrinsics.txt"]:
            fp = os.path.join(args.output_dir, f)
            if os.path.exists(fp):
                os.remove(fp)
    os.makedirs(args.output_dir, exist_ok=True)

    convert_depths(args.da3_dir, args.output_dir)
    convert_extrinsics(args.da3_dir, args.output_dir)
    convert_intrinsics(args.da3_dir, args.output_dir)
    convert_images(args.da3_dir, args.output_dir, video_path=args.video_path)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
