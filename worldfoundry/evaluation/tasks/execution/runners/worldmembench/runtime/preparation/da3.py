# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Prepare DA3 poses and scene geometry for one video.mp4 or a folder of cases."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile


def streaming_config(weights: Path) -> dict:
    """Frozen reference settings; Python Sim3 is the reference runtime's solver."""
    model = weights / "DA3NESTED-GIANT-LARGE-1.1"
    return {
        "Weights": {
            "DA3": str(model / "model.safetensors"),
            "DA3_CONFIG": str(model / "config.json"),
            "SALAD": str(weights / "dino_salad.ckpt"),
        },
        "Model": {
            "chunk_size": 120,
            "overlap": 60,
            "loop_chunk_size": 20,
            "loop_enable": True,
            "useDBoW": False,
            "delete_temp_files": True,
            "align_lib": "torch",
            "align_method": "sim3",
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "middle",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0,
            "save_depth_conf_result": True,
            "save_debug_info": False,
            "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000},
            # Upstream evaluates the reference YAML's scientific-notation strings.
            "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
            "Pointcloud_Save": {"sample_ratio": 0.015, "conf_threshold_coef": 0.75},
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1e-6"},
        },
    }


def extract_frames(video: Path, output: Path) -> int:
    import cv2

    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

    output.mkdir()
    capture = cv2.VideoCapture(str(video))
    try:
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or expected <= 0:
            raise EvaluationError(f"cannot decode video: {video}")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(output / f"frame_{count:06d}.png"), frame):
                raise EvaluationError(f"cannot write extracted frame {count}")
            count += 1
        if count != expected:
            raise EvaluationError(f"decoded {count} frames but video reports {expected}")
        return count
    finally:
        capture.release()


def convert_outputs(native: Path, output: Path, total_frames: int) -> None:
    """Preserve full-frame poses and stride-four float16 geometry, including the final frame."""
    import numpy as np

    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

    c2w = np.loadtxt(native / "camera_poses.txt", dtype=np.float64).reshape(-1, 4, 4)
    if len(c2w) != total_frames or not np.isfinite(c2w).all():
        raise EvaluationError("DA3 camera poses must be finite and cover the entire video")
    if not np.allclose(c2w[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise EvaluationError("DA3 camera poses have invalid homogeneous rows")
    frame_dir = native / "results_output"
    files = sorted(frame_dir.glob("frame_*.npz"), key=lambda path: int(path.stem.split("_")[-1]))
    if [path.name for path in files] != [f"frame_{index}.npz" for index in range(total_frames)]:
        raise EvaluationError("DA3 must export depth/confidence for every video frame")
    selected = np.arange(total_frames, dtype=np.int64)[::4]
    if selected[-1] != total_frames - 1:
        selected = np.concatenate([selected, np.asarray([total_frames - 1], dtype=np.int64)])
    depths, confidence, intrinsics = [], [], []
    for index in selected:
        with np.load(files[int(index)], allow_pickle=False) as frame:
            depth, conf = np.asarray(frame["depth"], dtype=np.float32), np.asarray(frame["conf"], dtype=np.float32)
            intrinsic = np.asarray(frame["intrinsics"], dtype=np.float32)
        if depth.ndim != 2 or conf.shape != depth.shape or intrinsic.shape != (3, 3):
            raise EvaluationError("invalid DA3 depth/confidence/intrinsics shapes")
        if (
            not all(np.isfinite(array).all() for array in (depth, conf, intrinsic))
            or np.any(depth < 0)
            or not np.any(depth > 0)
            or np.any(conf < 0)
        ):
            raise EvaluationError("invalid DA3 depth/confidence values")
        depths.append(depth.astype(np.float16))
        confidence.append(conf.astype(np.float16))
        intrinsics.append(intrinsic)
    output.mkdir()
    np.savez(
        output / "results.npz",
        extrinsics=np.linalg.inv(c2w)[:, :3],
        frame_indices=np.arange(total_frames, dtype=np.int64),
    )
    np.savez_compressed(
        output / "scene_geometry.npz",
        schema_version=np.asarray([1], dtype=np.int64),
        source_frame_count=np.asarray([total_frames], dtype=np.int64),
        frame_indices=selected,
        depths=np.stack(depths),
        confidence=np.stack(confidence),
        intrinsics=np.stack(intrinsics),
        c2w=c2w[selected],
        source=np.asarray(["da3_streaming_depth_conf_v1"]),
        sample_stride=np.asarray([4], dtype=np.int64),
    )


def ensure_artifacts(sample, context, *, weight_metric=None, require_scene=True) -> None:
    """Reuse complete artifacts; generate into an empty standard case/da3 directory."""
    import yaml

    from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.errors import EvaluationError

    paths = sample["artifacts"]
    pose = Path(paths["da3"])
    scene = Path(paths.get("da3_scene") or pose.parent / "scene_geometry.npz")
    if pose.is_file() and (not require_scene or scene.is_file()):
        return
    # Keep outputs beside the case's video link, not beside its external target.
    video = Path(sample["video"]).absolute()
    destination = video.parent / "da3"
    if pose != destination / "results.npz" or scene != destination / "scene_geometry.npz":
        raise EvaluationError("missing manifest DA3 artifacts; automatic generation uses the standard case/da3 paths")
    if destination.exists() and any(destination.iterdir()):
        raise EvaluationError(
            f"incomplete DA3 artifacts in {destination}; move the existing directory aside before regenerating"
        )
    if not context.device.startswith("cuda"):
        raise EvaluationError("DA3 generation requires CUDA; CPU scoring can reuse existing DA3 artifacts")
    config = context.config.get("da3") or {}
    if not config.get("source") or not config.get("dinov2_source"):
        raise EvaluationError("Configure da3.source and da3.dinov2_source for DA3-Streaming, or supply DA3 artifacts")
    source = Path(config["source"]).expanduser().resolve()
    dinov2 = Path(config["dinov2_source"]).expanduser().resolve()
    runner = source / "da3_streaming/da3_streaming.py"
    if not runner.is_file() or not (dinov2 / "hubconf.py").is_file():
        raise EvaluationError(
            "Configured DA3-Streaming source must contain da3_streaming/da3_streaming.py; dinov2_source must contain hubconf.py"
        )
    weights = (
        Path(config["weights"]).expanduser().resolve()
        if config.get("weights")
        else (
            context.cache_dir / "metrics" / weight_metric if weight_metric is not None else context.cache_dir / "models"
        )
    )
    settings = streaming_config(weights)
    for path in settings["Weights"].values():
        if not Path(path).is_file():
            raise EvaluationError(f"missing DA3 weight: {path}; stage the official checkpoint or configure da3.weights")
    command = [str(Path(config.get("python") or sys.executable).expanduser())]
    index = int(context.device.split(":")[1]) if ":" in context.device else 0
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index].strip() if visible else str(index)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        DINOV2_REPO_DIR=str(dinov2),
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source / "src"), env.get("PYTHONPATH"))))
    with TemporaryDirectory(prefix=".da3-", dir=video.parent) as directory:
        work = Path(directory)
        count = extract_frames(video, work / "frames")
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        # Keep verbose native output off the terminal and out of memory.
        with TemporaryFile(dir=work) as log:
            try:
                subprocess.run(
                    [
                        *command,
                        str(runner),
                        "--image_dir",
                        str(work / "frames"),
                        "--config",
                        str(config_path),
                        "--output_dir",
                        str(work / "native"),
                    ],
                    cwd=runner.parent,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as error:
                end = log.seek(0, os.SEEK_END)
                log.seek(max(0, end - 8192))
                tail = "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-20:])
                raise EvaluationError(f"DA3 exited with code {error.returncode}\n{tail}".rstrip()) from error
        convert_outputs(work / "native", work / "artifacts", count)
        # Publish both files together. rename refuses to overwrite a non-empty directory.
        (work / "artifacts").rename(destination)
