import os
import re
import logging
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Do not mutate the process-wide root logger. Quiet only the noisy
# third-party loggers this script actually triggers.
logging.getLogger("open_clip").setLevel(logging.WARNING)
logging.getLogger("ImageReward").setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.WARNING)

from automated.io.frames import FrameReader, VideoReader
from automated.io.metadata import (
    load_intrinsics,
    load_gt_frame_count_synthetic,
    load_intrinsics_json,
    load_gt_frame_count_real,
    extract_gt_frame_at,
    extract_gt_frames_batch,
)
from automated.metrics.reference_fidelity import gt_phase_pixel_fidelity
from automated.metrics.temporal import temporal_flow_score
from automated.metrics.visual_quality import aesthetic_score, image_quality_score, image_reward_score
from automated.metrics.geometry import (
    geometry_3d_consistency,
    object_centric_identity_consistency,
)
from automated.metrics.camera_controllability import (
    load_gt_poses_synthetic,
    load_gt_poses_real,
    load_intrinsics_synthetic,
    load_intrinsics_real,
    camera_controllability_score,
)
from automated.metrics.clip_metrics import clip_frame_similarity

GT_ROOT_SYNTHETIC  = "data/Synthetic_processed"
GEN_ROOT_SYNTHETIC = "output/Synthetic"                      # default; override with --gen_root
KEYFRAMES_SYN_XLS  = "data/Synthetic_processed/Synthetic_ExitReenter.xlsx"
GT_VIDEOS_SYNTHETIC = "data/Synthetic_videos"                # original UE5 MP4s

GT_ROOT_REAL       = "data/Real_Raw"
GEN_ROOT_REAL      = "output/Real"                           # default; override with --gen_root
KEYFRAMES_REAL_XLS = "data/Real_Raw/Real_ExitReenter.xlsx"
REAL_SELECTED_XLS  = "data/Real_Selected.xls"                # Availability filter


def _synthetic_gt_video_path(video_folder: str) -> str:
    """
    Map a synthetic clip id (e.g. 'CityPark_005', 'Middle_Age_008') to its
    UE5 GT video path (e.g. '.../CityPark5.mp4', '.../MiddleAge8.mp4').

    Convention: remove all underscores from the scene name, strip leading zeros
    from the clip number.  e.g. Japanese_Street_003 → JapaneseStreet3.mp4
    """
    import re
    m = re.match(r'^(.+)_(\d+)$', video_folder)
    if not m:
        return ""
    scene_raw = m.group(1)           # e.g. "Middle_Age"
    num       = str(int(m.group(2))) # strip leading zeros: "005" → "5"
    scene     = scene_raw.replace("_", "")  # "MiddleAge"
    return os.path.join(GT_VIDEOS_SYNTHETIC, f"{scene}{num}.mp4")

OUT_DIR = "outputs"

_RAW_NORM = {
    "TemporalScore":      (0,    1),
    "AestheticScore":     (0,   10),
    "ImageQuality":       (0,    1),
    "GeoConsistencyMean": (0,    1),
    "GeoConsistencyMin":  (0,    1),
    "ImageRewardScore":   (0,    1),   # already sigmoid-normalised
    "GTRevisitSim":       (0,    1),   # CLIP sim: generated R-phase vs GT R-phase
    "ObjIdentityMean":    (0,    1),   # object-centric patch-token sim (mean)
    "ObjIdentityMin":     (0,    1),   # object-centric patch-token sim (min)
    "CameraControllability": (0, 1),  # geodesic rotation error → exp score
    # GT_R_PSNR / GT_R_SSIM / GT_R_LPIPS are reported as raw values (no fixed range)
}

_COMPOSITE_COLS = [
    "MotionSmoothness",
    "VisualQuality",
    "ObjIdentityConsistency",   # object-centric DINOv2 patch-token identity (R phase)
    "Geo3DConsistency",
    "GTRevisitSim",             # CLIP sim: generated R-phase vs GT R-phase frames
    "ImageRewardScore",         # human preference score (NaN when no prompt → skipped in mean)
    "CameraControllability",    # geodesic rotation score in [0,1] (NaN for N/A models)
]


def _compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Build the named composite metrics (0-100) from normalized raw columns."""

    # MotionSmoothness (V+R phases only)
    if "TemporalScore_pct" in df.columns:
        df["MotionSmoothness"] = df["TemporalScore_pct"].round(2)

    # VisualQuality
    vq_cols = [c for c in ("AestheticScore_pct", "ImageQuality_pct") if c in df.columns]
    if vq_cols:
        df["VisualQuality"] = df[vq_cols].mean(axis=1).round(2)

    # Geo3DConsistency (V+R phases)
    has_geo_mean = "GeoConsistencyMean_pct" in df.columns
    has_geo_min  = "GeoConsistencyMin_pct"  in df.columns
    if has_geo_mean and has_geo_min:
        df["Geo3DConsistency"] = (
            0.7 * df["GeoConsistencyMean_pct"] + 0.3 * df["GeoConsistencyMin_pct"]
        ).round(2)
    elif has_geo_mean:
        df["Geo3DConsistency"] = df["GeoConsistencyMean_pct"].round(2)

    # Prompt-conditioned metrics — NaN for clips without prompts; skipna=True in V-D-R Score mean
    if "ImageRewardScore_pct" in df.columns:
        df["ImageRewardScore"] = df["ImageRewardScore_pct"].round(2)

    # GTRevisitSim — GT-grounded R-phase fidelity (NaN when GT video unavailable)
    if "GTRevisitSim_pct" in df.columns:
        df["GTRevisitSim"] = df["GTRevisitSim_pct"].round(2)

    # ObjIdentityConsistency — object-centric DINOv2 patch-token identity (R phase)
    has_obj_mean = "ObjIdentityMean_pct" in df.columns
    has_obj_min  = "ObjIdentityMin_pct"  in df.columns
    if has_obj_mean and has_obj_min:
        df["ObjIdentityConsistency"] = (
            0.7 * df["ObjIdentityMean_pct"] + 0.3 * df["ObjIdentityMin_pct"]
        ).round(2)
    elif has_obj_mean:
        df["ObjIdentityConsistency"] = df["ObjIdentityMean_pct"].round(2)

    # CameraControllability — geodesic rotation score (NaN for N/A models)
    if "CameraControllability_pct" in df.columns:
        df["CameraControllability"] = df["CameraControllability_pct"].round(2)

    return df


def load_keyframe_map_synthetic() -> dict:
    """
    Returns {video_folder: {"h_start": int, "r_start": int, "prompt": str}}
    Excludes rows with skip == 'yes'.
    Indices are in the original UE5 GT frame space.
    """
    if not os.path.exists(KEYFRAMES_SYN_XLS):
        return {}
    df = pd.read_excel(KEYFRAMES_SYN_XLS)
    skip_col = [c for c in df.columns if c.lower() == "skip"]
    if skip_col:
        df = df[df[skip_col[0]].astype(str).str.lower() != "yes"]
    result = {}
    for _, row in df.iterrows():
        vid = str(row["video_folder"]).strip()
        try:
            h = int(float(row["exits"]))
            r = int(float(row["re-enter"]))
        except (ValueError, TypeError, KeyError):
            continue
        result[vid] = {
            "h_start": h,
            "r_start": r,
            "prompt":  str(row.get("prompt", "")).strip(),
        }
    return result


def load_keyframe_map_real() -> dict:
    """
    Returns {"001": {"h_start": int, "r_start": int}, ...}
    Indices are in the GT timestamps.txt frame space.
    """
    if not os.path.exists(KEYFRAMES_REAL_XLS):
        return {}
    df = pd.read_excel(KEYFRAMES_REAL_XLS)
    result = {}
    for _, row in df.iterrows():
        try:
            clip_id = str(int(float(row["id"]))).zfill(3)
            h = int(float(row["exit"]))
            r = int(float(row["reenter"]))
        except (ValueError, TypeError, KeyError):
            continue
        result[clip_id] = {"h_start": h, "r_start": r}
    return result


def load_real_selected_ids() -> set | None:
    """
    Load the set of real clip IDs marked as 'yes' in Real_Selected.xls.
    Returns a set of zero-padded ID strings (e.g. {"001", "002", ...}),
    or None if the file does not exist (no filtering applied).
    """
    if not os.path.exists(REAL_SELECTED_XLS):
        return None
    df = pd.read_excel(REAL_SELECTED_XLS)
    yes_rows = df[df["Availability"].astype(str).str.strip().str.lower() == "yes"]
    ids = set()
    for val in yes_rows["ID"]:
        try:
            ids.add(str(int(float(val))).zfill(3))
        except (ValueError, TypeError):
            continue
    return ids


def load_prompt_map_synthetic(prompt_src: str) -> dict:
    """
    Returns {"Barnyard_001": "prompt text", ...} from a synthetic metadata CSV.

    Expects a CSV with 'video_folder' and 'prompt' columns
    (e.g. data/Synthetic_processed/metadata.csv).
    All models share the same synthetic prompts — pass the same CSV for
    LingBot-World, Wan2.2, Matrix-Game, and OpenSora.
    """
    if not prompt_src or not os.path.exists(prompt_src):
        return {}
    try:
        df = pd.read_csv(prompt_src)
    except Exception:
        return {}
    if "video_folder" not in df.columns or "prompt" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        vid    = str(row["video_folder"]).strip()
        prompt = str(row["prompt"]).strip()
        skip   = str(row.get("skip", "no")).strip().lower()
        if vid and prompt and prompt.lower() != "nan" and skip != "yes":
            result[vid] = prompt
    return result


def load_prompt_map_real(prompt_src: str) -> dict:
    """
    Returns {"001": "prompt text", ...} from a model-specific prompt file.

    Supported formats (auto-detected):
      CSV with 'video_folder' column  — LingBot-World, Wan2.2
        columns: video_folder, prompt, ...
      CSV with 'id' column            — OpenSora
        columns: id, prompt
      Per-clip meta.json fallback     — Matrix-Game has no prompt field,
        returns {} (prompt will be empty for that model)
    """
    if not prompt_src or not os.path.exists(prompt_src):
        return {}

    try:
        df = pd.read_csv(prompt_src)
    except Exception:
        return {}

    # Detect ID column
    if "video_folder" in df.columns:
        id_col = "video_folder"
    elif "id" in df.columns:
        id_col = "id"
    else:
        return {}

    if "prompt" not in df.columns:
        return {}

    result = {}
    for _, row in df.iterrows():
        try:
            raw_id = str(row[id_col]).strip()
            clip_id = str(int(float(raw_id))).zfill(3)
            prompt  = str(row["prompt"]).strip()
        except (ValueError, TypeError):
            continue
        if prompt and prompt.lower() != "nan":
            result[clip_id] = prompt
    return result


def map_keyframes(h_gt: int, r_gt: int, gt_total: int, gen_total: int) -> tuple:
    """
    Scale GT keyframe indices to generated video frame space.
    Ensures at least 2 frames exist in each of the V and R phases.
    """
    h = round(h_gt / gt_total * gen_total)
    r = round(r_gt / gt_total * gen_total)
    h = max(2, min(h, gen_total - 4))
    r = max(h + 2, min(r, gen_total - 1))
    return int(h), int(r)


_CLIP_NAME_RE = re.compile(r"^(.+)_(\d+)$")


def _make_reader(clip: dict, max_side: int):
    """Return a FrameReader or VideoReader depending on how the clip was discovered."""
    if clip.get("frames_dir"):
        return FrameReader(clip["frames_dir"], max_side=max_side)
    if clip.get("video_path"):
        return VideoReader(clip["video_path"], max_side=max_side)
    raise ValueError(f"Clip {clip['id']} has neither frames_dir nor video_path")


def _find_frames_dir(clip_path: str):
    """Return the PNG frames directory for a clip, or None if none found.

    Tries {clip_path}/frames/ first (standard layout),
    then {clip_path}/samples-rgb/ (Stable-Virtual-Camera layout),
    then {clip_path} itself (PNG-direct layout used by OpenSora).
    """
    frames_subdir = os.path.join(clip_path, "frames")
    if os.path.isdir(frames_subdir):
        if any(f.endswith(".png") for f in os.listdir(frames_subdir)):
            return frames_subdir
    samples_subdir = os.path.join(clip_path, "samples-rgb")
    if os.path.isdir(samples_subdir):
        if any(f.endswith(".png") for f in os.listdir(samples_subdir)):
            return samples_subdir
    if any(f.endswith(".png") for f in os.listdir(clip_path)):
        return clip_path
    return None


def _add_synthetic_clip(clips: list, clip_name: str, scene: str,
                         frames_dir: str, kf_map: dict,
                         prompt_map: dict = None,
                         video_path: str = None) -> None:
    """frames_dir: path to PNG frames dir (None for mp4-based models).
       video_path: path to generated video mp4 (None for PNG-based models).
    """
    if clip_name not in kf_map:
        return
    gt_dir        = os.path.join(GT_ROOT_SYNTHETIC, scene, clip_name)
    gt_image_path = os.path.join(gt_dir, "image.jpg")
    gt_intr_path  = os.path.join(gt_dir, "intrinsics.npy")
    if not (os.path.exists(gt_image_path) and os.path.exists(gt_intr_path)):
        return
    gt_video_path = _synthetic_gt_video_path(clip_name)
    if not os.path.exists(gt_video_path):
        return
    kf = kf_map[clip_name]
    if prompt_map is not None:
        prompt = prompt_map.get(clip_name, kf.get("prompt", ""))
    else:
        prompt = kf.get("prompt", "")
    clips.append({
        "id":            clip_name,
        "scene":         scene,
        "data_type":     "synthetic",
        "frames_dir":    frames_dir,    # None for mp4-based models
        "video_path":    video_path,    # None for PNG-based models
        "gt_image":      gt_image_path,
        "gt_intrinsics": gt_intr_path,
        "gt_video":      gt_video_path,
        "h_start_gt":    kf["h_start"],
        "r_start_gt":    kf["r_start"],
        "prompt":        prompt,
    })


def discover_clips_synthetic(gen_root: str = GEN_ROOT_SYNTHETIC,
                              prompt_map: dict = None) -> list:
    """Scan gen_root for synthetic clips. Auto-detects five layouts:

      Nested + frames/  {Scene}/{clip}/frames/*.png  — LingBot-World, Wan2.2
      Flat   + frames/  {clip}/frames/*.png           — Matrix-Game
      Nested PNG-direct {Scene}/{clip}/*.png          — OpenSora
      Flat   + video    {clip}/video.mp4              — FantasyWorld
      Flat   mp4        {clip_id}.mp4                 — CogVideo

    prompt_map: optional {video_folder: prompt} from load_prompt_map_synthetic().
    """
    kf_map = load_keyframe_map_synthetic()
    clips  = []
    for top_name in sorted(os.listdir(gen_root)):
        top_path = os.path.join(gen_root, top_name)

        # CogVideo flat mp4: {clip_id}.mp4 directly in gen_root
        if top_name.endswith(".mp4") and _CLIP_NAME_RE.match(top_name[:-4]):
            clip_name = top_name[:-4]
            scene = clip_name.rsplit("_", 1)[0]
            _add_synthetic_clip(clips, clip_name, scene, None, kf_map, prompt_map,
                                video_path=top_path)
            continue

        if not os.path.isdir(top_path):
            continue

        if _CLIP_NAME_RE.match(top_name):
            # Skip directory if a same-named flat mp4 already exists (avoids CogVideo duplicates)
            if os.path.exists(os.path.join(gen_root, top_name + ".mp4")):
                continue
            # Flat clip directory — check mp4 first (FantasyWorld), then PNG
            mp4_path = os.path.join(top_path, "video.mp4")
            if os.path.exists(mp4_path):
                scene = top_name.rsplit("_", 1)[0]
                _add_synthetic_clip(clips, top_name, scene, None, kf_map, prompt_map,
                                    video_path=mp4_path)
                continue
            frames_dir = _find_frames_dir(top_path)
            if frames_dir is None:
                continue
            scene = top_name.rsplit("_", 1)[0]
            _add_synthetic_clip(clips, top_name, scene, frames_dir, kf_map, prompt_map)
        else:
            # Nested layout: top_name is a scene dir
            for clip_name in sorted(os.listdir(top_path)):
                if not _CLIP_NAME_RE.match(clip_name):
                    continue
                clip_path = os.path.join(top_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue
                mp4_path = os.path.join(clip_path, "video.mp4")
                if os.path.exists(mp4_path):
                    scene = clip_name.rsplit("_", 1)[0]
                    _add_synthetic_clip(clips, clip_name, scene, None, kf_map, prompt_map,
                                        video_path=mp4_path)
                    continue
                frames_dir = _find_frames_dir(clip_path)
                if frames_dir is None:
                    continue
                scene = clip_name.rsplit("_", 1)[0]
                _add_synthetic_clip(clips, clip_name, scene, frames_dir, kf_map, prompt_map)

    return clips


def discover_clips_real(gen_root: str = GEN_ROOT_REAL,
                        prompt_map: dict = None) -> list:
    """Scan gen_root for real clips. Auto-detects four layouts:

      Standard  {NNN}/frames/*.png      — LingBot-World, Wan2.2, Matrix-Game
      OpenSora  real_{NNN}/*.png        — PNG-direct with real_ prefix
      FantasyWorld {NNN}/video.mp4      — mp4 in numbered subdir
      CogVideo  {NNN}.mp4              — flat mp4 at root level

    prompt_map: optional {clip_id: prompt_text} from load_prompt_map_real().
    Clips not marked 'yes' in Real_Selected.xls are excluded automatically.
    """
    kf_map     = load_keyframe_map_real()
    selected   = load_real_selected_ids()
    prompt_map = prompt_map or {}
    clips      = []

    def _add_real_clip(clip_id, frames_dir=None, video_path=None):
        gt_dir        = os.path.join(GT_ROOT_REAL, clip_id)
        gt_mp4        = os.path.join(gt_dir, f"{clip_id}.mp4")
        gt_intr_json  = os.path.join(gt_dir, f"{clip_id}-intrinsics.json")
        gt_timestamps = os.path.join(gt_dir, "timestamps.txt")
        if not all(os.path.exists(p) for p in [gt_mp4, gt_intr_json, gt_timestamps]):
            return
        if clip_id not in kf_map:
            return
        if selected is not None and clip_id not in selected:
            return
        kf = kf_map[clip_id]
        clips.append({
            "id":                 clip_id,
            "scene":              "real",
            "data_type":          "real",
            "frames_dir":         frames_dir,
            "video_path":         video_path,
            "gt_mp4":             gt_mp4,
            "gt_intrinsics_json": gt_intr_json,
            "gt_timestamps":      gt_timestamps,
            "h_start_gt":         kf["h_start"],
            "r_start_gt":         kf["r_start"],
            "prompt":             prompt_map.get(clip_id, ""),
        })

    for entry in sorted(os.listdir(gen_root)):
        entry_path = os.path.join(gen_root, entry)

        # CogVideo flat mp4: {NNN}.mp4 directly in gen_root
        if entry.endswith(".mp4") and re.match(r"^\d+\.mp4$", entry):
            clip_id = entry[:-4].zfill(3)
            _add_real_clip(clip_id, video_path=entry_path)
            continue

        if not os.path.isdir(entry_path):
            continue

        if re.match(r"^real_\d+$", entry):
            clip_id = entry[5:].zfill(3)
        elif re.match(r"^\d+$", entry):
            clip_id = entry.zfill(3)
        else:
            continue

        # Skip directory if a same-named flat mp4 already exists (avoids CogVideo duplicates)
        if os.path.exists(os.path.join(gen_root, clip_id + ".mp4")):
            continue

        # FantasyWorld: {NNN}/video.mp4
        mp4_path = os.path.join(entry_path, "video.mp4")
        if os.path.exists(mp4_path):
            _add_real_clip(clip_id, video_path=mp4_path)
            continue

        # PNG layouts
        frames_dir = _find_frames_dir(entry_path)
        if frames_dir is None:
            continue
        _add_real_clip(clip_id, frames_dir=frames_dir)

    return clips


def main():
    ap = argparse.ArgumentParser(description="MemoBench evaluation.")
    ap.add_argument("--mode", choices=["synthetic", "real", "both"],
                    default="synthetic",
                    help="Which data type to evaluate. Default: synthetic.")
    ap.add_argument("--gen_root", default=None,
                    help="Override generated clips root (single --mode only).")
    ap.add_argument("--gen_root_syn", default=None,
                    help="Synthetic gen root for --mode both.")
    ap.add_argument("--gen_root_real", default=None,
                    help="Real gen root for --mode both.")
    ap.add_argument("--prompt_src_syn", default=None,
                    help="Path to synthetic prompt CSV with video_folder+prompt columns. "
                         "All models share the same synthetic prompts; use "
                         "data/Synthetic_processed/metadata.csv "
                         "for LingBot-World, Wan2.2, Matrix-Game, and OpenSora.")
    ap.add_argument("--prompt_src_real", default=None,
                    help="Path to real prompt CSV (enables ImageReward for real clips). "
                         "Supported: Lingbot/Wan2.2 metadata.csv (video_folder col), "
                         "OpenSora prompts_real.csv (id col).")
    ap.add_argument("--clip", default=None,
                    help="Evaluate only this clip ID (e.g. Barnyard_001). Default: all.")
    ap.add_argument("--scene", default=None,
                    help="Evaluate only clips from this scene. Default: all.")
    ap.add_argument("--max_side", type=int, default=640,
                    help="Resize frames to this max side length. Default: 640.")
    ap.add_argument("--sample_step", type=int, default=4,
                    help="Frame step for temporal + quality metrics. Default: 4.")
    ap.add_argument("--device", default=None,
                    help="Device: 'cuda', 'cuda:1', 'cpu'. Default: auto-detect.")
    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path. Default: outputs/eval_{mode}_{timestamp}.csv")
    ap.add_argument("--camera_ctrl", action="store_true", default=True,
                    help="Compute CameraControllability via MapAnything pose estimation "
                         "vs GT poses. Enabled by default for all models.")
    args = ap.parse_args()

    clips = []
    if args.mode in ("synthetic", "both"):
        gen_root        = args.gen_root_syn or args.gen_root or GEN_ROOT_SYNTHETIC
        syn_prompt_map  = load_prompt_map_synthetic(args.prompt_src_syn)
        clips          += discover_clips_synthetic(gen_root, prompt_map=syn_prompt_map)
    if args.mode in ("real", "both"):
        gen_root    = args.gen_root_real or args.gen_root or GEN_ROOT_REAL
        prompt_map  = load_prompt_map_real(args.prompt_src_real)
        clips      += discover_clips_real(gen_root, prompt_map=prompt_map)

    if args.clip:
        clips = [c for c in clips if c["id"] == args.clip]
    if args.scene:
        clips = [c for c in clips if c["scene"] == args.scene]

    if not clips:
        print("No matching clips found. Check --clip / --scene or your data paths.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv   = args.out_csv or os.path.join(OUT_DIR, f"eval_{args.mode}_{timestamp}.csv")

    print(f"Evaluating {len(clips)} clip(s) [{args.mode}] → {out_csv}")

    rows = []
    for clip in tqdm(clips, desc="Evaluating"):
        try:
            row = _eval_clip(clip, args.max_side, args.sample_step, args.device,
                             camera_ctrl=args.camera_ctrl)
        except Exception as e:
            print(f"\n[WARN] {clip['id']} failed: {e}")
            row = {"id": clip["id"], "scene": clip["scene"],
                   "data_type": clip["data_type"], "error": str(e)}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Normalize raw diagnostics → 0-100
    for metric, (lo, hi) in _RAW_NORM.items():
        if metric in df.columns:
            df[f"{metric}_pct"] = (
                (df[metric] - lo) / (hi - lo) * 100
            ).clip(0, 100).round(1)

    df = _compute_composite(df)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    present = [c for c in _COMPOSITE_COLS if c in df.columns]
    print("\nComposite metrics (0-100):")
    if present:
        print(df[present].mean(skipna=True).round(2).to_string())

    print("\nDiagnostic means (raw):")
    diag_cols = [
        "AestheticScore", "ImageQuality",
        "ObjIdentityMean", "ObjIdentityMin",
        "GeoConsistencyMean", "GeoConsistencyMin",
        "CameraControllability", "MeanRotErrorDeg",
        "ImageRewardScore",
        "GTRevisitSim",
        "GT_O_PSNR", "GT_O_SSIM", "GT_O_LPIPS",
        "GT_H_PSNR", "GT_H_SSIM", "GT_H_LPIPS",
        "GT_R_PSNR", "GT_R_SSIM", "GT_R_LPIPS",
        "GT_ALL_PSNR", "GT_ALL_SSIM", "GT_ALL_LPIPS",
    ]
    present_diag = [c for c in diag_cols if c in df.columns]
    if present_diag:
        print(df[present_diag].mean().to_string())


def _eval_clip(clip: dict, max_side: int, sample_step: int, device: str,
               camera_ctrl: bool = False) -> dict:
    data_type = clip["data_type"]

    # --- Intrinsics and GT frame count ---
    if data_type == "synthetic":
        intrinsics = load_intrinsics(clip["gt_intrinsics"])
        gt_total   = load_gt_frame_count_synthetic(clip["gt_video"])
    else:  # real
        intrinsics = load_intrinsics_json(clip["gt_intrinsics_json"])
        gt_total   = load_gt_frame_count_real(clip["gt_timestamps"])

    # --- Generated frames ---
    gen = _make_reader(clip, max_side=max_side)
    N   = gen.num_frames

    # --- Scale GT keyframe indices → generated video frame space ---
    h_start, r_start = map_keyframes(
        clip["h_start_gt"], clip["r_start_gt"], gt_total, N
    )

    # 1. Motion smoothness — V + R phases only (D phase excluded)
    temp_scores = []
    if h_start >= 2:
        temp_scores.append(temporal_flow_score(gen, start=0, end=h_start,
                                               sample_step=sample_step, device=device))
    if (N - 1 - r_start) >= 2:
        temp_scores.append(temporal_flow_score(gen, start=r_start, end=N - 1,
                                               sample_step=sample_step, device=device))
    temp_score = float(np.mean(temp_scores)) if temp_scores else 0.0

    # 3. Visual quality (aesthetic + sharpness, full video)
    aes = aesthetic_score(gen, sample_step=sample_step, device=device)
    iq  = image_quality_score(gen, sample_step=sample_step, device=device)

    # 4. Object-centric identity consistency — DINOv2 patch tokens (top-40% stable patches)
    obj_id_res = object_centric_identity_consistency(gen, n_sample=9, device=device,
                                                     start=r_start, end=N - 1)

    # 5. Geo3D consistency — V + R phases (Depth Anything V2 cosine similarity)
    geo_o = geometry_3d_consistency(gen, device=device, n_sample=3,
                                    start=0, end=h_start)
    geo_r = geometry_3d_consistency(gen, device=device, n_sample=3,
                                    start=r_start, end=N - 1)
    geo_mean = round((geo_o["geo_consistency"]     + geo_r["geo_consistency"])     / 2, 4)
    geo_min  = round(min(geo_o["min_geo_consistency"], geo_r["min_geo_consistency"]), 4)

    # 6. Prompt-based metrics (both synthetic and real when prompt available)
    prompt = clip.get("prompt", "")
    ir = image_reward_score(gen, prompt, n_sample=5, device=device) \
         if prompt else None

    # 9. GT-grounded fidelity — per-frame PSNR/SSIM/LPIPS for each phase and
    #    the whole video.  GT frames are resampled to match the generated frame
    #    count within each phase.  GTRevisitSim (CLIP) unchanged (3 pairs).
    gt_r_clip_sim  = None
    gt_o_psnr = gt_o_ssim = gt_o_lpips   = None
    gt_h_psnr = gt_h_ssim = gt_h_lpips   = None
    gt_r_psnr = gt_r_ssim = gt_r_lpips   = None
    gt_all_psnr = gt_all_ssim = gt_all_lpips = None
    try:
        gt_video_path = clip.get("gt_video") or clip.get("gt_mp4")
        if gt_video_path and os.path.exists(gt_video_path):
            h_gt      = clip["h_start_gt"]
            r_gt      = clip["r_start_gt"]
            n_gt_last = gt_total - 1

            # GTRevisitSim: CLIP sim on 3 uniformly spaced R-phase pairs (unchanged)
            gt_r_idx3  = [int(r_gt     + k * (n_gt_last - r_gt)     / 2) for k in range(3)]
            gen_r_idx3 = [int(r_start  + k * (N - 1     - r_start)  / 2) for k in range(3)]
            sims = []
            for gt_idx, gen_idx in zip(gt_r_idx3, gen_r_idx3):
                gt_frame  = extract_gt_frame_at(gt_video_path, gt_idx)
                gen_frame = gen.get(gen_idx)
                sims.append(clip_frame_similarity(gt_frame, gen_frame, device=device))
            gt_r_clip_sim = round(float(np.mean(sims)), 4)

            # Per-phase and full-video pixel-level fidelity (all frames)
            def _resample_gt_idx(gt_s, gt_e, n):
                if n <= 0: return []
                if n == 1 or gt_s == gt_e: return [gt_s] * n
                return [round(gt_s + i * (gt_e - gt_s) / (n - 1)) for i in range(n)]

            phase_defs = [
                ("O",   0,        h_start,  0,    h_gt      ),
                ("H",   h_start,  r_start,  h_gt, r_gt      ),
                ("R",   r_start,  N - 1,    r_gt, n_gt_last ),
                ("ALL", 0,        N - 1,    0,    n_gt_last ),
            ]

            phase_results = {}
            for pname, gen_s, gen_e, gt_s, gt_e in phase_defs:
                gen_indices = list(range(gen_s, gen_e + 1))
                n = len(gen_indices)
                if n == 0:
                    continue
                gt_indices     = _resample_gt_idx(gt_s, gt_e, n)
                gt_frames_list = extract_gt_frames_batch(gt_video_path, gt_indices)
                gen_frames_list = [gen.get(i) for i in gen_indices]
                psnr, ssim, lpips_val = gt_phase_pixel_fidelity(
                    gt_frames_list, gen_frames_list, device=device
                )
                phase_results[pname] = (round(psnr, 4), round(ssim, 4), round(lpips_val, 4))

            if "O"   in phase_results: gt_o_psnr,   gt_o_ssim,   gt_o_lpips   = phase_results["O"]
            if "H"   in phase_results: gt_h_psnr,   gt_h_ssim,   gt_h_lpips   = phase_results["H"]
            if "R"   in phase_results: gt_r_psnr,   gt_r_ssim,   gt_r_lpips   = phase_results["R"]
            if "ALL" in phase_results: gt_all_psnr, gt_all_ssim, gt_all_lpips = phase_results["ALL"]
    except Exception:
        gt_r_clip_sim  = None
        gt_o_psnr = gt_o_ssim = gt_o_lpips   = None
        gt_h_psnr = gt_h_ssim = gt_h_lpips   = None
        gt_r_psnr = gt_r_ssim = gt_r_lpips   = None
        gt_all_psnr = gt_all_ssim = gt_all_lpips = None

    # 10. Camera Controllability — geodesic rotation error vs GT poses
    #     Only computed when --camera_ctrl is passed (lingbot-world, wan2.2).
    #     N/A for matrix-game (keyboard/mouse input) and open-sora (no camera control).
    cam_ctrl_score   = None
    mean_rot_err     = None
    ate_rot_deg      = None
    total_gt_rot_deg = None
    if camera_ctrl:
        try:
            if data_type == "synthetic":
                gt_poses_cam = load_gt_poses_synthetic(clip["scene"], clip["id"], N)
                K_cam        = load_intrinsics_synthetic(clip["scene"], clip["id"])
            else:
                gt_poses_cam = load_gt_poses_real(clip["id"], N)
                K_cam        = load_intrinsics_real(clip["id"])
            if gt_poses_cam is not None and K_cam is not None:
                cam_res          = camera_controllability_score(gen, gt_poses_cam, K_cam,
                                                                device=device)
                cam_ctrl_score   = cam_res["camera_controllability"]
                mean_rot_err     = cam_res["mean_rot_error_deg"]
                ate_rot_deg      = cam_res["ate_rot_deg"]
                total_gt_rot_deg = cam_res["total_gt_rotation_deg"]
        except Exception:
            pass

    return {
        "id":         clip["id"],
        "scene":      clip["scene"],
        "data_type":  data_type,
        "num_frames": N,
        "h_start":    h_start,
        "r_start":    r_start,
        "gt_frames":  gt_total,
        # Motion (V+R only)
        "TemporalScore": round(temp_score, 4),
        # Visual quality
        "AestheticScore": aes,
        "ImageQuality":   iq,
        # Object-centric identity (R phase) — patch tokens, top-40%
        "ObjIdentityMean": obj_id_res["obj_identity_consistency"],
        "ObjIdentityMin":  obj_id_res["obj_min_identity_consistency"],
        # Geo3D consistency (V+R)
        "GeoConsistencyMean": geo_mean,
        "GeoConsistencyMin":  geo_min,
        # Prompt-based (when prompt available)
        "ImageRewardScore":   ir,   # ImageReward (2023) human preference
        # GT-grounded fidelity (CLIP sim R-phase; PSNR/SSIM/LPIPS per-frame all phases)
        "GTRevisitSim":  gt_r_clip_sim,  # CLIP sim: generated R-phase vs GT R-phase (3 pairs)
        "GT_O_PSNR":     gt_o_psnr,      # V-phase pixel PSNR  (all gen frames vs GT resampled)
        "GT_O_SSIM":     gt_o_ssim,      # V-phase SSIM
        "GT_O_LPIPS":    gt_o_lpips,     # V-phase LPIPS (lower=better)
        "GT_H_PSNR":     gt_h_psnr,      # D-phase pixel PSNR
        "GT_H_SSIM":     gt_h_ssim,      # D-phase SSIM
        "GT_H_LPIPS":    gt_h_lpips,     # D-phase LPIPS (lower=better)
        "GT_R_PSNR":     gt_r_psnr,      # R-phase pixel PSNR
        "GT_R_SSIM":     gt_r_ssim,      # R-phase SSIM
        "GT_R_LPIPS":    gt_r_lpips,     # R-phase LPIPS (lower=better)
        "GT_ALL_PSNR":   gt_all_psnr,    # whole-video pixel PSNR
        "GT_ALL_SSIM":   gt_all_ssim,    # whole-video SSIM
        "GT_ALL_LPIPS":  gt_all_lpips,   # whole-video LPIPS (lower=better)
        # Camera Controllability (None unless --camera_ctrl passed)
        "CameraControllability": cam_ctrl_score,   # ATE-based rotation coverage [0,1]
        "ATE_RotDeg":            ate_rot_deg,       # ATE rotation RMSE in degrees
        "TotalGT_RotDeg":        total_gt_rot_deg,  # end-to-end GT rotation in degrees
        "MeanRotErrorDeg":       mean_rot_err,      # frame-to-frame diagnostic
        # Intrinsics metadata
        "fx": round(intrinsics["fx"], 2),
        "fy": round(intrinsics["fy"], 2),
    }


if __name__ == "__main__":
    main()
