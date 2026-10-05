# eval/task_resolver.py
"""
Task resolver (output-first).

Input:
  - outputs_json: JSON mapping { task_id: video_path_or_url }
Output:
  - list[ResolvedTask] where each task has:
      - init frame path
      - wm video path (local, in run dir)
      - extracted final frame path (final_frame.png)
      - evaluation questions from YAML
  - create per-task folders under run_root/per_task to store run
  artifacts for each task (WM video output, final frame, init_frame etc.)
      
This module also performs "ingredient extraction" (final frame extraction).

Conventions (per task folder):
  - <task_dir>/<task_dir_name>.yaml
  - <task_dir>/<task_dir_name>_init_frame.png

Video link can be:
  - local path (absolute or relative to outputs_json parent)
  - http(s) URL (downloaded)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from benchmark.runners.utils import (
    load_yaml, resolve_task_paths
)
from eval.utils import (
    _resolve_video_source, _download, occlusion_base,
)

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Question:
    id: str
    question: str
    answer_type: str = "yes_no"
    notes_for_judge: str = ""


@dataclass
class ResolvedTask:
    task_id: str
    task_level: int
    task_dir: Path
    task_yaml: Path
    init_frame: Path
    gt_final_frame: Optional[Path]  # None if not yet generated

    # Model output (normalized into run directory)
    wm_video: Path

    # Extracted ingredient
    final_frame: Path

    # Eval spec
    questions: List[Question]
    spec_prompt: Optional[str] = None  # evaluation.init2final_edit_prompt (optional)

# -----------------------------
# YAML / task discovery
# -----------------------------

def discover_tasks(tasks_root: Path) -> Dict[str, Tuple[Path, Path, Path, Path]]:
    """
    Scan for task folders under tasks_root.

    Returns mapping:
      task_id -> (task_dir, task_yaml, init_frame, gt_final_frame)

    task_id comes from YAML 'id' if present, else folder name.
    """
    tasks_root = Path(tasks_root).expanduser().resolve()
    if not tasks_root.exists():
        raise FileNotFoundError(tasks_root)

    mapping: Dict[str, Tuple[Path, Path, Path]] = {}

    for d in tasks_root.rglob("*"):
        if not d.is_dir():
            continue

        # Only consider directories that look like task dirs (must contain <dirname>.yaml)
        if not (d / f"{d.name}.yaml").exists():
            continue

        try:
            task_dir, task_yaml, folder_name, init_frame = resolve_task_paths(d)
        except FileNotFoundError:
            continue

        # GT final frame path
        gt_final_frame = (task_dir / f'{folder_name}_gt_final_frame.png').resolve()

        task = load_yaml(task_yaml)  # your existing YAML loader
        task_id = str(task.get("id") or folder_name)

        if task_id in mapping:
            raise ValueError(
                f"Duplicate task_id discovered: {task_id}\n"
                f"  - {task_dir}\n"
                f"  - {mapping[task_id][0]}"
            )

        mapping[task_id] = (task_dir, task_yaml, init_frame, gt_final_frame)

    if not mapping:
        raise RuntimeError(f"No tasks found under: {tasks_root}")

    # Add base-name aliases so camera-controlled runs (whose output JSON keys
    # use the base name, e.g. "ice_on_burner") can resolve to the correct task
    # yaml and init frame.  sorted() ensures _cardboard wins over _curtain/_lightoff.
    base_aliases: Dict[str, Tuple[Path, Path, Path, Path]] = {}
    for task_id in sorted(mapping):
        base = occlusion_base(task_id)
        if base != task_id and base not in mapping and base not in base_aliases:
            base_aliases[base] = mapping[task_id]
    mapping.update(base_aliases)

    return mapping


def _load_task_level(task_yaml: Path) -> Optional[int]:
    task = load_yaml(task_yaml)
    return task.get("level")


def _load_questions(task_yaml: Path) -> Tuple[List[Question], Optional[str]]:
    task = load_yaml(task_yaml)
    eval_block = task.get("evaluation", {}) or {}

    spec_prompt = eval_block.get("init2final_edit_prompt")
    if isinstance(spec_prompt, str):
        spec_prompt = spec_prompt.strip()
    else:
        spec_prompt = None

    q_raw = eval_block.get("questions", []) or []
    questions: List[Question] = []
    for i, q in enumerate(q_raw, start=1):
        qid = str(q.get("id") or f"q{i:02d}").strip()
        questions.append(
            Question(
                id=qid,
                question=str(q.get("question", "")).strip(),
                answer_type=str(q.get("answer_type", "yes_no")).strip(),
                notes_for_judge=str(q.get("notes_for_judge", "")).strip(),
            )
        )

    return questions, spec_prompt

# -----------------------------
# Ingredient extraction (final frame)
# -----------------------------

def extract_final_frame(video_path: Path, out_png: Path) -> Dict[str, Any]:
    """
    Extract the last frame of a video and save to out_png.
    Uses OpenCV if available; falls back to imageio.
    Returns extraction metadata (also write to extraction.json by caller).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Try OpenCV
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Seek near end
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

        ok, frame = cap.read()
        # If last-frame read failed (common for some codecs), iterate to end
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last = None
            idx = 0
            while True:
                ok2, fr2 = cap.read()
                if not ok2:
                    break
                last = fr2
                idx += 1
            if last is None:
                raise RuntimeError("No frames readable from video")
            frame = last
            selected_idx = idx - 1
            frame_count = idx
        else:
            selected_idx = max(frame_count - 1, 0)

        cap.release()

        # BGR -> RGB, write PNG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image  # type: ignore
        Image.fromarray(frame_rgb).save(out_png)

        return {
            "policy": "last_frame",
            "selected_frame_index": int(selected_idx),
            "frame_count": int(frame_count),
            "fps": float(fps) if fps else None,
            "backend": "opencv",
        }
    except Exception:
        print("Extraction failed")
        return {}

    # # Fallback: imageio
    # import imageio.v3 as iio  # type: ignore
    # meta = iio.immeta(video_path, plugin="pyav") if hasattr(iio, "immeta") else {}
    # frames = iio.imread(video_path, index=-1, plugin="pyav")  # last frame
    # from PIL import Image  # type: ignore
    # Image.fromarray(frames).save(out_png)

    # return {
    #     "policy": "last_frame",
    #     "selected_frame_index": -1,
    #     "frame_count": meta.get("nframes"),
    #     "fps": meta.get("fps"),
    #     "backend": "imageio_pyav",
    # }

# -----------------------------
# Main resolver
# -----------------------------

def resolve_one_task(
    *,
    task_id: str,
    video_link: str,
    task_map: dict,
    outputs_json_path: Path,
    run_dir: Path,
    download_urls: bool = True,
) -> ResolvedTask:
    """
    Resolve artifacts for ONE task.

    Images (init_frame, gt_final_frame) and local WM videos are referenced
    by their original paths — nothing is copied. Only derived artifacts
    (final_frame.png, extraction.json) are written into:
      run_dir/per_task/<task_id>/

    URL videos are downloaded into the per-task folder (no local original exists).

    Returns a single ResolvedTask.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    outputs_json_path = Path(outputs_json_path).expanduser().resolve()

    if task_id not in task_map:
        raise KeyError(f"Unknown task_id in outputs_json: {task_id}")

    task_dir, task_yaml, init_frame, gt_final_frame = task_map[task_id]

    if not init_frame.exists():
        raise FileNotFoundError(f"Init frame missing for task {task_id}: {init_frame}")
    questions, spec_prompt = _load_questions(task_yaml)
    task_level = _load_task_level(task_yaml)

    per_task_root = run_dir / "per_task"
    run_task_dir = per_task_root / task_id
    run_task_dir.mkdir(parents=True, exist_ok=True)

    # Resolve WM video: download URLs; use original path for local files.
    kind, src = _resolve_video_source(str(video_link), outputs_json_path)
    if kind == "url":
        if not download_urls:
            raise ValueError(f"URL provided but download_urls=False for task {task_id}: {video_link}")
        run_video = run_task_dir / "wm_video.mp4"
        if not run_video.exists():
            _download(str(src), run_video)
        wm_video = run_video
    else:
        if not src.exists():
            raise FileNotFoundError(f"Video file not found for task {task_id}: {src}")
        wm_video = src  # reference original — no copy

    # Extract final frame (derived artifact — written to run dir)
    final_frame = run_task_dir / "final_frame.png"
    if not final_frame.exists():
        extract_final_frame(wm_video, final_frame)

    return ResolvedTask(
        task_id=task_id,
        task_level=task_level,
        task_dir=task_dir,
        task_yaml=task_yaml,
        init_frame=init_frame,       # original path
        gt_final_frame=gt_final_frame if gt_final_frame.exists() else None,
        wm_video=wm_video,           # original path (or downloaded)
        final_frame=final_frame,
        questions=questions,
        spec_prompt=spec_prompt,
    )


def resolve_tasks_from_outputs_json(
    *,
    tasks_root: Path,
    outputs_json: Path,
    run_dir: Path,
    download_urls: bool = True,
    pattern: Optional[str] = None,
    exclude_baseline: bool = False,
) -> List[ResolvedTask]:
    """
    Loop over (task_id -> video_link) pairs in outputs_json and resolve each task.

    If pattern is given (fnmatch-style, e.g. "ice_on_burner*"), only task IDs
    matching that pattern are resolved.

    If exclude_baseline is True, tasks whose IDs end with "_00" are skipped.

    Returns a list of ResolvedTask.
    """
    import fnmatch

    tasks_root = Path(tasks_root).expanduser().resolve()
    outputs_json = Path(outputs_json).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()

    task_map = discover_tasks(tasks_root)

    if not outputs_json.exists():
        raise FileNotFoundError(outputs_json)

    outputs_data = json.loads(outputs_json.read_text(encoding="utf-8"))
    if not isinstance(outputs_data, dict):
        raise ValueError("outputs_json must be a JSON object: { task_id: video_link }")

    if pattern:
        outputs_data = {k: v for k, v in outputs_data.items() if fnmatch.fnmatch(str(k), pattern)}

    if exclude_baseline:
        outputs_data = {k: v for k, v in outputs_data.items() if not str(k).endswith("_00")}

    resolved: List[ResolvedTask] = []
    failed = 0
    for task_id, video_link in outputs_data.items():
        try:
            resolved.append(
                resolve_one_task(
                    task_id=str(task_id),
                    video_link=str(video_link),
                    task_map=task_map,
                    outputs_json_path=outputs_json,
                    run_dir=run_dir,
                    download_urls=download_urls,
                )
            )
        except Exception as e:
            print(f"[SKIP] {task_id}: resolution failed — {e}")
            failed += 1
    if failed:
        print(f"[WARN] {failed} tasks skipped during resolution")

    return resolved

# def resolve_tasks_from_outputs_json(
#     *,
#     tasks_root: Path,
#     outputs_json: Path,
#     run_dir: Path,
#     link_mode: str = "symlink",  # "symlink" or "copy"
#     download_urls: bool = True,
# ) -> List[ResolvedTask]:
#     """
#     Creates a normalized per-task folder under run_dir/per_task/<task_id>/ and returns ResolvedTask list.

#     - link_mode="symlink": symlink wm_video to original file (fast)
#     - link_mode="copy": copy wm_video into run dir (portable)
#     - download_urls: if True, download URL videos into run dir
#     """
#     tasks_root = Path(tasks_root).expanduser().resolve()
#     outputs_json = Path(outputs_json).expanduser().resolve()
#     run_dir = Path(run_dir).expanduser().resolve()

#     task_map = discover_tasks(tasks_root)

#     if not outputs_json.exists():
#         raise FileNotFoundError(outputs_json)

#     outputs_data = json.loads(outputs_json.read_text(encoding="utf-8"))
#     if not isinstance(outputs_data, dict):
#         raise ValueError("outputs_json must be a JSON object: { task_id: video_link }")

#     resolved: List[ResolvedTask] = []
#     per_task_root = run_dir / "per_task"

#     for task_id, video_link in outputs_data.items():
#         if task_id not in task_map:
#             raise KeyError(f"Unknown task_id in outputs_json: {task_id}")

#         task_dir, task_yaml, init_frame, gt_final_frame = task_map[task_id]
#         if not init_frame.exists():
#             raise FileNotFoundError(f"Init frame missing for task {task_id}: {init_frame}")

#         questions, spec_prompt = _load_questions(task_yaml)

#         task_level = _load_task_level(task_yaml)

#         # Prepare per-task run dir
#         run_task_dir = per_task_root / task_id

#         run_task_dir.mkdir(parents=True, exist_ok=True)

#         # Normalize init frame and 'GT final frame' into run dir (copy/symlink)
#         run_init = run_task_dir / "init_frame.png"
#         if not run_init.exists():
#             _copy_or_symlink(init_frame, run_init, mode=link_mode)

#         run_gt_final = run_task_dir / "gt_final_frame.png"
#         if not run_gt_final.exists():
#             _copy_or_symlink(gt_final_frame, run_gt_final, mode=link_mode)

#         # Normalize video into run dir
#         run_video = run_task_dir / "wm_video.mp4"
#         if not run_video.exists():
#             kind, src = _resolve_video_source(str(video_link), outputs_json)

#             if kind == "url":
#                 if not download_urls:
#                     raise ValueError(f"URL provided but download_urls=False for task {task_id}: {video_link}")
#                 _download(str(src), run_video)
#             else:
#                 if not src.exists():
#                     raise FileNotFoundError(f"Video file not found for task {task_id}: {src}")
#                 _copy_or_symlink(src, run_video, mode=link_mode)

#         # Extract final frame
#         final_frame = run_task_dir / "final_frame.png"
#         extraction_meta = run_task_dir / "extraction.json"
#         if not final_frame.exists() or not extraction_meta.exists():
#             meta = extract_final_frame(run_video, final_frame)
#             extraction_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

#         resolved.append(
#             ResolvedTask(
#                 task_id=task_id,
#                 task_level=task_level,
#                 task_dir=task_dir,
#                 task_yaml=task_yaml,
#                 init_frame=run_init,
#                 gt_final_frame=run_gt_final,
#                 wm_video=run_video,
#                 final_frame=final_frame,
#                 extraction_meta=extraction_meta,
#                 questions=questions,
#                 spec_prompt=spec_prompt,
#             )
#         )

#     return resolved