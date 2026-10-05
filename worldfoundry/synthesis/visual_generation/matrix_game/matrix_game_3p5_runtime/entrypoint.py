#!/usr/bin/env python3
"""Matrix-Game-3.5 inference — anchor image + camera trajectory -> generated video.

Usage:
    python -m worldfoundry.synthesis.visual_generation.matrix_game.\
matrix_game_3p5_runtime.entrypoint --person first \
        --image samples/first_person/case_0/input.png \
        --camera samples/first_person/case_0/camera.npz \
        --caption samples/first_person/case_0/caption.json

    python -m worldfoundry.synthesis.visual_generation.matrix_game.\
matrix_game_3p5_runtime.entrypoint --person third \
        --image samples/third_person/case_3/input.png \
        --camera samples/third_person/case_3/camera.npz \
        --prompt "A long-haired female character walks along the shore." \
        --refs samples/third_person/case_3/refs        # optional

Inputs:
    --image    anchor RGB frame (png/jpg), any resolution
    --camera   .npz with:
                 extrinsics_c2w : (N,4,4) float camera-to-world matrices
                                  (use --camera-convention w2c if yours are w2c)
                 intrinsics     : (N,4)|(4,)|(3,3) [fx,fy,cx,cy] in pixels of
                                  the anchor image resolution
    --prompt / --prompt-file / --caption
               text prompt (or a segment caption json for multi-block runs)
    --refs     (third person, optional) directory of protagonist reference
               crops: either plain PNGs (masks auto-generated / *_mask.png
               honored) or a full protagonist_refs export with candidates.jsonl

Each generated block is 21 latent frames = 84 new output frames; block k
consumes camera poses [1 + 84*k, 84*(k+1)]. Poses are clamped to the last
entry when the trajectory is shorter.

Weights are resolved by the WorldFoundry runtime adapter and passed as local
paths; this entrypoint never downloads model weights implicitly.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.config_paths import (
    matrix_game_35_infer_config_path,
)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
PYTHON_BIN = os.environ.get("WORLDFOUNDRY_MATRIX_GAME_3P5_PYTHON", sys.executable)
MIN_POSES = 86  # dataset minimum: 84 generated + anchor + 1


# --------------------------------------------------------------------------- #
# weight resolution: CLI flag > env var > checkpoints/ convention
# --------------------------------------------------------------------------- #

CKPT_DIR = os.path.join(REPO_DIR, "checkpoints")

WEIGHTS = {
    # key: (env var, default candidates under checkpoints/, human description)
    "wan": ("WAN22_TI2V_5B_DIR", ["Wan2.2-TI2V-5B"], "Wan2.2-TI2V-5B base model directory"),
    "tokenizer": ("UMT5_TOKENIZER_DIR", ["Wan2.2-TI2V-5B/google/umt5-xxl", "umt5-xxl"], "umt5-xxl tokenizer directory"),
    "da3": ("DA3_MODEL_PATH", ["DA3NESTED-GIANT-LARGE-1.1"], "Depth-Anything-3 model directory"),
    "ckpt_first": ("CKPT_FIRST_PERSON", ["first-person.safetensors"], "Matrix-Game-3.5 first-person checkpoint"),
    "ckpt_third": ("CKPT_THIRD_PERSON", ["third-person.safetensors"], "Matrix-Game-3.5 third-person checkpoint"),
}


def resolve_weight(key, cli_value):
    env_var, default_names, desc = WEIGHTS[key]
    candidates = (
        ([cli_value] if cli_value else [])
        + ([os.environ[env_var]] if os.environ.get(env_var) else [])
        + [os.path.join(CKPT_DIR, n) for n in default_names]
    )
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    sys.exit(
        f"ERROR: {desc} not found (tried: {', '.join(candidates)})\n"
        f"  Provide it via CLI flag, ${env_var}, or place/symlink it at "
        f"checkpoints/{default_names[0]}"
    )


# --------------------------------------------------------------------------- #
# workspace synthesis: user inputs -> a minimal scene the pipeline accepts
# --------------------------------------------------------------------------- #


def load_camera(path, convention):
    z = np.load(path)
    if "extrinsics_c2w" in z:
        extr = np.asarray(z["extrinsics_c2w"], dtype=np.float32)
    elif "extrinsics" in z:
        extr = np.asarray(z["extrinsics"], dtype=np.float32)
    else:
        sys.exit(f"ERROR: {path} must contain 'extrinsics_c2w' (N,4,4)")
    if extr.ndim != 3 or extr.shape[1:] != (4, 4):
        sys.exit(f"ERROR: extrinsics must be (N,4,4), got {extr.shape}")
    if convention == "w2c":
        extr = np.linalg.inv(extr).astype(np.float32)

    if "intrinsics" not in z:
        sys.exit(f"ERROR: {path} must contain 'intrinsics' (N,4)|(4,)|(3,3) [fx,fy,cx,cy] in pixels")
    intr = np.asarray(z["intrinsics"], dtype=np.float32)
    if intr.ndim == 1 and intr.shape[0] == 4:
        intr = np.tile(intr[None], (extr.shape[0], 1))
    elif intr.ndim == 2 and intr.shape == (3, 3):
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        intr = np.tile(np.array([[fx, fy, cx, cy]], np.float32), (extr.shape[0], 1))
    elif intr.ndim == 3 and intr.shape[1:] == (3, 3):
        intr = np.stack([intr[:, 0, 0], intr[:, 1, 1], intr[:, 0, 2], intr[:, 1, 2]], axis=1)
    elif not (intr.ndim == 2 and intr.shape[1] == 4):
        sys.exit(f"ERROR: unsupported intrinsics shape {intr.shape}")
    if intr.shape[0] < extr.shape[0]:
        pad = np.tile(intr[-1:], (extr.shape[0] - intr.shape[0], 1))
        intr = np.concatenate([intr, pad], axis=0)
    return extr, intr[: extr.shape[0]]


def pad_poses(extr, intr, n_min):
    if extr.shape[0] >= n_min:
        return extr, intr
    print(f"[infer] camera has {extr.shape[0]} poses < {n_min}; padding by repeating the last pose")
    pad = n_min - extr.shape[0]
    extr = np.concatenate([extr, np.tile(extr[-1:], (pad, 1, 1))], axis=0)
    intr = np.concatenate([intr, np.tile(intr[-1:], (pad, 1))], axis=0)
    return extr, intr


def write_anchor_video(image_path, out_path, n_frames=86):
    import cv2

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"ERROR: cannot read image {image_path}")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return h, w


def write_caption(ws, scene_name, args):
    cap_dir = os.path.join(ws, "caption", scene_name)
    os.makedirs(cap_dir, exist_ok=True)
    out = os.path.join(cap_dir, "video.json")
    if args.caption:
        shutil.copyfile(args.caption, out)
        return
    text = args.prompt
    if args.prompt_file:
        text = open(args.prompt_file, encoding="utf-8").read().strip()
    if not text:
        sys.exit("ERROR: provide --prompt, --prompt-file or --caption")
    payload = {"detailed": {"0": {"start": "", "dynamic": text}}}
    json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def write_refs(scene_dir, refs_dir):
    """Install protagonist references for the subject-ref pipeline."""
    import cv2

    dst = os.path.join(scene_dir, "objects", "protagonist_refs")
    if os.path.exists(os.path.join(refs_dir, "candidates.jsonl")):
        shutil.copytree(refs_dir, dst)  # full export — use verbatim
        return
    pngs = sorted(
        f for f in os.listdir(refs_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_mask" not in f.lower()
    )
    if not pngs:
        sys.exit(f"ERROR: no reference images found in {refs_dir}")
    img_dir = os.path.join(dst, "ref_images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i, name in enumerate(pngs):
        stem, _ = os.path.splitext(name)
        shutil.copyfile(os.path.join(refs_dir, name), os.path.join(img_dir, name))
        mask_name = f"{stem}_mask.png"
        src_mask = os.path.join(refs_dir, mask_name)
        img = cv2.imread(os.path.join(refs_dir, name), cv2.IMREAD_COLOR)
        if os.path.exists(src_mask):
            shutil.copyfile(src_mask, os.path.join(img_dir, mask_name))
        else:  # no mask: treat full crop as subject
            cv2.imwrite(os.path.join(img_dir, mask_name), np.full(img.shape[:2], 255, np.uint8))
        h, w = img.shape[:2]
        rows.append(
            {
                "ref_id": i,
                "frame_idx": i,
                "track_id": 1,
                "class_name": "person",
                "conf": 1.0,
                "bbox_xyxy": [0, 0, w, h],
                "crop_xyxy": [0, 0, w, h],
                "box_area": float(w * h),
                "mask_area": float(w * h),
                "temporal_bin_idx": i,
                "image_path": f"ref_images/{name}",
                "mask_path": f"ref_images/{mask_name}",
            }
        )
    with open(os.path.join(dst, "candidates.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[infer] installed {len(rows)} protagonist reference(s)")


def build_workspace(args):
    person_dir = "first_person" if args.person == "first" else "third_person"
    name = args.name or time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.abspath(os.path.join(args.output, person_dir, name))
    # Intermediate synthesized scene + raw run live OUTSIDE the output dir;
    # removed after a successful run (kept on failure / --keep-workspace).
    cache_root = os.environ.get(
        "WORLDFOUNDRY_MATRIX_GAME_3P5_CACHE_DIR",
        os.path.join(REPO_DIR, ".cache"),
    )
    tmp_root = os.path.join(cache_root, "infer_runs", f"{person_dir}_{name}")
    ws = os.path.join(tmp_root, "workspace")
    scene = os.path.join(ws, "scenes", "case")
    os.makedirs(out_root, exist_ok=True)
    for sub in ("rgb", "pose", "intrinsics", "depth"):
        os.makedirs(os.path.join(scene, sub), exist_ok=True)

    h, w = write_anchor_video(args.image, os.path.join(scene, "rgb", "video.mp4"))

    extr, intr = load_camera(args.camera, args.camera_convention)
    need = max(MIN_POSES, 1 + 84 * args.num_blocks)
    if extr.shape[0] < need:
        print(
            f"[infer] note: trajectory ({extr.shape[0]} poses) is shorter than {need}; the tail will hold the last pose"
        )
    extr, intr = pad_poses(extr, intr, MIN_POSES)
    np.savez_compressed(os.path.join(scene, "pose", "video.npz"), data=extr)
    np.savez_compressed(os.path.join(scene, "intrinsics", "video.npz"), data=intr)

    # Depth file must exist and parse; its values are never used for
    # generation (DA3 estimates depth at runtime).
    np.savez_compressed(os.path.join(scene, "depth", "video.npz"), frame_00000=np.full((h, w), 5.0, np.float32))
    json.dump({"format": "npz_float"}, open(os.path.join(scene, "depth", "metadata.json"), "w"))

    write_caption(ws, "case", args)
    if args.person == "third" and args.refs:
        write_refs(scene, os.path.abspath(args.refs))

    return out_root, tmp_root, ws


# --------------------------------------------------------------------------- #
# pipeline invocation
# --------------------------------------------------------------------------- #


def build_index(ws, python_bin, env):
    import sqlite3

    index_path = os.path.join(ws, "index.sqlite")
    cmd = [
        python_bin,
        "-m",
        "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.data.build_da3_video_index",
        "--camera_params_path",
        os.path.join(ws, "scenes"),
        "--prompt_path",
        os.path.join(ws, "caption"),
        "--output_path",
        index_path,
        "--workers",
        "1",
        "--worker_backend",
        "serial",
        "--quiet",
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True, stdout=subprocess.DEVNULL)
    con = sqlite3.connect(index_path)
    rows = con.execute("SELECT clean_name, scan_error FROM records").fetchall()
    con.close()
    bad = [(n, e) for n, e in rows if e]
    if bad or not rows:
        sys.exit(f"ERROR: input scene was rejected by the index scanner: {bad or 'no records'}")
    return index_path


def resolve_wan_model_sources(wan_dir):
    """Resolve the Wan2.2 text encoder and VAE reused by Matrix recipes."""
    sources = []
    for filename in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"):
        path = os.path.join(wan_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing Wan2.2 file {filename!r} under {wan_dir}")
        sources.append(path)
    return sources


def run_generation(args, index_path, tmp_root, env):
    python_bin = PYTHON_BIN
    person = "first_person" if args.person == "first" else "third_person"
    ckpt = resolve_weight("ckpt_first" if args.person == "first" else "ckpt_third", args.ckpt)
    wan = resolve_weight("wan", args.wan_dir)
    model_sources = resolve_wan_model_sources(wan)
    cache_root = env.get(
        "WORLDFOUNDRY_MATRIX_GAME_3P5_CACHE_DIR",
        os.path.join(tmp_root, "cache"),
    )

    cmd = [
        python_bin,
        "-m",
        "worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.run_inference",
        "--config",
        str(matrix_game_35_infer_config_path(person)),
        "--model_paths",
        json.dumps(model_sources),
        "--tokenizer_path",
        resolve_weight("tokenizer", args.tokenizer_dir),
        "--trained_dit",
        ckpt,
        "--dataset_index_path",
        index_path,
        "--num_inference_batches",
        "1",
        "--num_inference_blocks",
        str(args.num_blocks),
        "--inference_sample_offset",
        "0",
        "--output_path",
        os.path.join(tmp_root, "raw"),
        "--log_dir_name",
        "run",
        "--dataset_cache_dir",
        os.path.join(cache_root, "dataset_cache"),
        "--memory_latent_cache_dir",
        os.path.join(cache_root, "memory_latents"),
    ]
    if args.steps:
        cmd += ["--num_inference_steps", str(args.steps)]
    if args.cfg_scale:
        cmd += ["--guidance_scale", str(args.cfg_scale)]
    if args.seed is not None:
        cmd += ["--inference_seed", str(args.seed), "--seed", str(args.seed)]
    cmd += args.extra

    print(f"[infer] launching generation ({args.num_blocks} block(s) x 84 frames)...")
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)
    return os.path.join(tmp_root, "raw", "run", "inference")


def collect_outputs(inference_dir, out_root):
    results = {}
    for f in sorted(os.listdir(inference_dir)):
        path = os.path.join(inference_dir, f)
        if not os.path.isfile(path):
            continue
        if f.endswith("_history.mp4"):
            # pure generated rollout — this is the user-facing result
            results["result.mp4"] = path
    for name, src in results.items():
        dst = os.path.join(out_root, name)
        shutil.copyfile(src, dst)
    return results


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True, choices=["first", "third"])
    ap.add_argument("--image", required=True, help="anchor RGB image")
    ap.add_argument("--camera", required=True, help="camera .npz (extrinsics_c2w + intrinsics)")
    ap.add_argument("--prompt", default="", help="prompt text")
    ap.add_argument("--prompt-file", default="", help="prompt text file")
    ap.add_argument("--caption", default="", help="segment caption json (multi-block runs)")
    ap.add_argument("--refs", default="", help="(third person) protagonist reference image dir")
    ap.add_argument("--num-blocks", type=int, default=1, help="blocks to generate; each = 84 new frames")
    ap.add_argument("--steps", type=int, default=0, help="denoising steps (default from config: 25)")
    ap.add_argument("--cfg-scale", type=float, default=0.0, help="CFG scale (default from config: 5.0)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default=os.path.join(REPO_DIR, "outputs"))
    ap.add_argument("--name", default="", help="run name (default: image stem + timestamp)")
    ap.add_argument("--camera-convention", choices=["c2w", "w2c"], default="c2w")
    ap.add_argument("--keep-workspace", action="store_true")
    ap.add_argument(
        "--ckpt", default="", help="Matrix-Game-3.5 DiT checkpoint (default: checkpoints/<person>-person.safetensors)"
    )
    ap.add_argument("--wan-dir", default="", help="Wan2.2-TI2V-5B dir (default: checkpoints/Wan2.2-TI2V-5B)")
    ap.add_argument("--tokenizer-dir", default="", help="umt5-xxl tokenizer dir (default: checkpoints/umt5-xxl)")
    ap.add_argument("--da3-dir", default="", help="DA3 model dir (default: checkpoints/DA3NESTED-GIANT-LARGE-1.1)")
    args, extra = ap.parse_known_args()
    args.extra = extra

    if args.refs and args.person != "third":
        sys.exit("ERROR: --refs is only supported with --person third")

    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["DA3_MODEL_PATH"] = resolve_weight("da3", args.da3_dir)
    env.setdefault("WORLDFOUNDRY_SKIP_MODEL_DOWNLOAD", "true")

    out_root, tmp_root, ws = build_workspace(args)
    print(f"[infer] output dir: {out_root}")
    index_path = build_index(ws, PYTHON_BIN, env)
    inference_dir = run_generation(args, index_path, tmp_root, env)

    results = collect_outputs(inference_dir, out_root)
    if "result.mp4" not in results:
        sys.exit(f"ERROR: generation produced no result video (inspect {tmp_root})")
    if not args.keep_workspace:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print("\n[infer] done. Outputs:")
    for name in sorted(results):
        print(f"  {os.path.join(out_root, name)}")


if __name__ == "__main__":
    main()
