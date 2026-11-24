import argparse
import json
import os
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

GT_REAL_DIR = "data/Real_Raw"
OUTPUT_DIR = "data/mapanything/outputs/real"


def load_intrinsics_from_json(clip_id: str) -> np.ndarray:
    """Load 3x3 intrinsic matrix from the real clip's JSON file."""
    json_path = os.path.join(GT_REAL_DIR, clip_id, f"{clip_id}-intrinsics.json")
    with open(json_path) as f:
        data = json.load(f)
    cam = data["camera"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def extract_frames_from_mp4(mp4_path: str, sample_step: int = 1) -> list:
    """Extract frames from mp4. Returns list of BGR numpy arrays."""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def estimate_poses_mapanything(frames: list, device: str = "cuda") -> np.ndarray:
    """
    Run MapAnything on a list of BGR frames.
    Returns (N, 4, 4) c2w poses in OpenCV convention.
    """
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
    model.eval()

    # Write frames to temp dir as PNG for load_images
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, bgr in enumerate(frames):
            path = os.path.join(tmp, f"{i:05d}.png")
            cv2.imwrite(path, bgr)
            paths.append(path)

        views = load_images(paths)

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )

    poses = []
    for pred in outputs:
        p = pred["camera_poses"][0].cpu().numpy()
        poses.append(p)

    return np.stack(poses, axis=0).astype(np.float64)


def get_all_clip_ids() -> list:
    """Get all real clip IDs from the GT directory."""
    ids = []
    for name in sorted(os.listdir(GT_REAL_DIR)):
        mp4 = os.path.join(GT_REAL_DIR, name, f"{name}.mp4")
        if os.path.isfile(mp4):
            ids.append(name)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-ids", nargs="*", default=None, help="Specific clip IDs to process. Default: all.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips that already have output.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clip_ids = args.clip_ids if args.clip_ids else get_all_clip_ids()
    print(f"Processing {len(clip_ids)} real clips")

    failed = []
    for clip_id in tqdm(clip_ids, desc="MapAnything pose extraction", unit="clip"):
        out_path = os.path.join(args.output_dir, f"{clip_id}_mapanything.npz")

        if args.skip_existing and os.path.exists(out_path):
            continue

        mp4_path = os.path.join(GT_REAL_DIR, clip_id, f"{clip_id}.mp4")
        if not os.path.isfile(mp4_path):
            print(f"[SKIP] {clip_id}: no mp4 found")
            failed.append(clip_id)
            continue

        try:
            # Load intrinsics from JSON
            K = load_intrinsics_from_json(clip_id)

            # Extract all frames
            frames = extract_frames_from_mp4(mp4_path)
            print(f"  {clip_id}: {len(frames)} frames")

            # Run MapAnything
            poses = estimate_poses_mapanything(frames, device=args.device)
            print(f"  {clip_id}: got {len(poses)} poses")

            # Save in same format as MegaSAM output
            np.savez(
                out_path,
                cam_c2w=poses,
                intrinsic=K,
            )

        except Exception as e:
            print(f"[ERROR] {clip_id}: {e}")
            failed.append(clip_id)
            continue

    print(f"\nDone. {len(clip_ids) - len(failed)}/{len(clip_ids)} succeeded.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
