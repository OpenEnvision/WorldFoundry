import hashlib
import json
import os
import sqlite3
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np
from tqdm import tqdm

DA3_VIDEO_INDEX_SCHEMA_VERSION = 3
_SCAN_WORKER_PROMPT_INDEX = None
_SCAN_WORKER_MIN_FRAME_COUNT = 1
_SCAN_WORKER_REQUIRE_COMPLETE = True
_SCAN_WORKER_REQUIRE_DYNAMIC_MASKS = False
_SCAN_WORKER_DYNAMIC_MASK_SUBDIR = "objects"
_SCAN_WORKER_DYNAMIC_MASK_FILENAME = "dynamic_masks.jsonl"


def _debug(message, *, verbose=True):
    if verbose:
        tqdm.write(f"[DA3VideoIndex] {message}")


@dataclass(frozen=True)
class DA3VideoIndexRecord:
    clean_name: str
    camera_root: str | None
    camera_root_order: int | None
    scene_dir: str
    synthetic_path: str
    video_path: str | None
    extrinsic_path: str | None
    intrinsic_path: str | None
    depth_path: str | None
    depth_format: str | None
    depth_metadata_path: str | None
    depth_metadata_json: str | None
    prompt_path: str | None
    frame_count: int | None
    has_prompt: int
    has_depth: int
    scan_error: str | None


def _split_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, str):
        return [item for item in path_or_paths.split(",") if item]
    return [str(item) for item in path_or_paths if str(item)]


def _first_file(directory, preferred_names, suffixes):
    if not directory or not os.path.isdir(directory):
        return None
    first_match = None
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                path = os.path.join(directory, entry.name)
                if entry.name in preferred_names:
                    return path
                if entry.name.endswith(suffixes) and first_match is None:
                    first_match = path
    except OSError:
        return None
    return first_match


def _normalize_worker_backend(worker_backend):
    backend = str(worker_backend or "process").strip().lower()
    aliases = {
        "threads": "thread",
        "threadpool": "thread",
        "processes": "process",
        "processpool": "process",
        "none": "serial",
        "single": "serial",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"process", "thread", "serial"}:
        raise ValueError(f"worker_backend must be 'process', 'thread', or 'serial', got {worker_backend!r}.")
    return backend


def _normalize_max_scenes(max_scenes):
    if max_scenes is None:
        return None
    max_scenes = int(max_scenes)
    if max_scenes <= 0:
        return None
    return max_scenes


def _executor_cls(worker_backend):
    worker_backend = _normalize_worker_backend(worker_backend)
    if worker_backend == "process":
        return ProcessPoolExecutor
    if worker_backend == "thread":
        return ThreadPoolExecutor
    return None


def _empty_da3_root_result(root_order, camera_root):
    return {
        "root_order": root_order,
        "root": camera_root,
        "scene_records": [],
        "dir_count": 0,
        "duplicate_count": 0,
        "missing": False,
    }


def _make_da3_scene_record(root_order, current):
    clean_name = os.path.basename(os.path.normpath(current))
    return {
        "clean_name": clean_name,
        "camera_root_order": root_order,
        "scene_dir": current,
        "synthetic_path": os.path.join(current, f"{clean_name}.video"),
        "video_path": _first_file(
            os.path.join(current, "rgb"),
            preferred_names=("video.mp4",),
            suffixes=(".mp4",),
        ),
        "extrinsic_path": _first_file(
            os.path.join(current, "pose"),
            preferred_names=("video.npz",),
            suffixes=(".npz",),
        ),
        "intrinsic_path": _first_file(
            os.path.join(current, "intrinsics"),
            preferred_names=("video.npz",),
            suffixes=(".npz",),
        ),
        "depth_path": _first_file(
            os.path.join(current, "depth"),
            preferred_names=("video.npz",),
            suffixes=(".npz",),
        ),
        "root_order": root_order,
    }


def _standard_da3_scene_record(root_order, current):
    clean_name = os.path.basename(os.path.normpath(current))
    return {
        "clean_name": clean_name,
        "camera_root_order": root_order,
        "scene_dir": current,
        "synthetic_path": os.path.join(current, f"{clean_name}.video"),
        "video_path": os.path.join(current, "rgb", "video.mp4"),
        "extrinsic_path": os.path.join(current, "pose", "video.npz"),
        "intrinsic_path": os.path.join(current, "intrinsics", "video.npz"),
        "depth_path": os.path.join(current, "depth", "video.npz"),
        "root_order": root_order,
    }


def _done_child_scene_record(args):
    root_order, child_path = args
    done_path = os.path.join(child_path, "_DONE")
    if not os.path.isfile(done_path):
        return None
    return _standard_da3_scene_record(root_order, child_path)


def _is_hidden_metadata_dir(entry):
    return entry.name.startswith(".")


def _done_child_scene_records(root_order, children, workers, max_scenes=None):
    result = _empty_da3_root_result(root_order, "")
    local_scenes = {}
    done_dirs = set()
    sorted_children = sorted(
        (entry for entry in children if not _is_hidden_metadata_dir(entry)),
        key=lambda item: item.name,
    )
    max_scenes = _normalize_max_scenes(max_scenes)
    workers = max(1, int(workers or 1))
    if max_scenes is None:
        batch_size = len(sorted_children)
    else:
        batch_size = min(len(sorted_children), max(max_scenes, workers, 32))

    checked_count = 0
    while checked_count < len(sorted_children):
        batch = sorted_children[checked_count : checked_count + batch_size]
        if not batch:
            break
        args = [(root_order, entry.path) for entry in batch]
        if workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as pool:
                records = list(pool.map(_done_child_scene_record, args))
        else:
            records = [_done_child_scene_record(arg) for arg in args]
        for offset, record in enumerate(records):
            if record is None:
                if max_scenes is not None:
                    result["scene_records"] = [local_scenes[name] for name in sorted(local_scenes)]
                    remaining_children = sorted_children[checked_count + offset :]
                    return result, remaining_children, False
                continue
            _add_done_record(local_scenes, result, record)
            done_dirs.add(record["scene_dir"])
            if max_scenes is not None and len(local_scenes) >= max_scenes:
                result["scene_records"] = [local_scenes[name] for name in sorted(local_scenes)]
                return result, [], True
        checked_count += len(batch)

    result["scene_records"] = [local_scenes[name] for name in sorted(local_scenes)]
    remaining_children = [entry for entry in sorted_children if entry.path not in done_dirs]
    return result, remaining_children, False


def _merge_da3_subtree_results(root_order, camera_root, subtree_results):
    base_result = _empty_da3_root_result(root_order, camera_root)
    base_result["dir_count"] = 1
    return _merge_da3_child_results(root_order, camera_root, base_result, subtree_results)


def _merge_da3_child_results(root_order, camera_root, base_result, subtree_results):
    result = _empty_da3_root_result(root_order, camera_root)
    result["dir_count"] = int(base_result.get("dir_count", 0) or 0)
    result["duplicate_count"] = int(base_result.get("duplicate_count", 0) or 0)
    local_scenes = {}
    for record in base_result.get("scene_records", []):
        clean_name = record["clean_name"]
        if clean_name in local_scenes:
            result["duplicate_count"] += 1
        else:
            local_scenes[clean_name] = record
    for subtree in subtree_results:
        result["dir_count"] += int(subtree.get("dir_count", 0) or 0)
        result["duplicate_count"] += int(subtree.get("duplicate_count", 0) or 0)
        for record in subtree.get("scene_records", []):
            clean_name = record["clean_name"]
            if clean_name in local_scenes:
                result["duplicate_count"] += 1
            else:
                local_scenes[clean_name] = record
    result["scene_records"] = [local_scenes[name] for name in sorted(local_scenes)]
    return result


def _add_done_record(local_scenes, result, record):
    clean_name = record["clean_name"]
    if clean_name in local_scenes:
        result["duplicate_count"] += 1
    else:
        local_scenes[clean_name] = record


def _discover_da3_subtree(args):
    if len(args) == 4:
        root_order, camera_root, start_dir, max_scenes = args
        workers = 1
        worker_backend = "serial"
    else:
        root_order, camera_root, start_dir, max_scenes, workers, worker_backend = args
    workers = max(1, int(workers or 1))
    worker_backend = _normalize_worker_backend(worker_backend)
    result = _empty_da3_root_result(root_order, camera_root)
    if not start_dir or not os.path.isdir(start_dir):
        result["missing"] = True
        return result

    local_scenes = {}
    stack = [start_dir]
    while stack:
        result["dir_count"] += 1
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                children = [entry for entry in it if entry.is_dir(follow_symlinks=False)]
        except OSError:
            continue
        remaining_limit = None if max_scenes is None else max(0, int(max_scenes) - len(local_scenes))
        if remaining_limit == 0:
            break
        marker_workers = workers if worker_backend != "serial" else 1
        done_result, children, done_limit_reached = _done_child_scene_records(
            root_order,
            children,
            marker_workers,
            remaining_limit,
        )
        result["duplicate_count"] += int(done_result.get("duplicate_count", 0) or 0)
        for record in done_result.get("scene_records", []):
            _add_done_record(local_scenes, result, record)
        if done_limit_reached or (max_scenes is not None and len(local_scenes) >= int(max_scenes)):
            break
        child_names = {entry.name for entry in children}
        if "pose" in child_names and "intrinsics" in child_names:
            record = _make_da3_scene_record(root_order, current)
            clean_name = record["clean_name"]
            if clean_name in local_scenes:
                result["duplicate_count"] += 1
            else:
                local_scenes[clean_name] = record
                if max_scenes is not None and len(local_scenes) >= int(max_scenes):
                    break
            continue
        # Push reverse-sorted children onto the LIFO stack so discovery order is
        # ascending by directory name. This makes --max_scenes deterministic.
        stack.extend(entry.path for entry in sorted(children, key=lambda e: e.name, reverse=True))
    result["scene_records"] = [local_scenes[name] for name in sorted(local_scenes)]
    return result


def _discover_da3_root(args):
    if len(args) == 3:
        root_order, camera_root, max_scenes = args
        workers = 1
        worker_backend = "serial"
    else:
        root_order, camera_root, max_scenes, workers, worker_backend = args
    workers = max(1, int(workers or 1))
    worker_backend = _normalize_worker_backend(worker_backend)
    result = {
        "root_order": root_order,
        "root": camera_root,
        "scene_records": [],
        "dir_count": 0,
        "duplicate_count": 0,
        "missing": False,
    }
    if not camera_root or not os.path.isdir(camera_root):
        result["missing"] = True
        return result

    if workers > 1 and worker_backend != "serial":
        try:
            with os.scandir(camera_root) as it:
                children = [entry for entry in it if entry.is_dir(follow_symlinks=False)]
        except OSError:
            return result
        child_names = {entry.name for entry in children}
        if "pose" in child_names and "intrinsics" in child_names:
            return _discover_da3_subtree(
                (
                    root_order,
                    camera_root,
                    camera_root,
                    max_scenes,
                    workers,
                    worker_backend,
                )
            )
        done_result, remaining_children, done_limit_reached = _done_child_scene_records(
            root_order,
            children,
            workers,
            max_scenes,
        )
        done_result["root"] = camera_root
        done_result["dir_count"] = 1
        if done_limit_reached:
            return done_result
        child_paths = [entry.path for entry in sorted(remaining_children, key=lambda entry: entry.name)]
        if max_scenes is not None:
            remaining_limit = int(max_scenes) - len(done_result.get("scene_records", []))
            if remaining_limit <= 0:
                return done_result
            if len(child_paths) == len(children) and not done_result.get("scene_records"):
                return _discover_da3_subtree(
                    (
                        root_order,
                        camera_root,
                        camera_root,
                        remaining_limit,
                        workers,
                        worker_backend,
                    )
                )
            subtree_results = []
            for child_path in child_paths:
                if remaining_limit <= 0:
                    break
                subtree = _discover_da3_subtree(
                    (
                        root_order,
                        camera_root,
                        child_path,
                        remaining_limit,
                        workers,
                        worker_backend,
                    )
                )
                subtree_results.append(subtree)
                remaining_limit -= len(subtree.get("scene_records", []))
            return _merge_da3_child_results(
                root_order,
                camera_root,
                done_result,
                subtree_results,
            )
        if child_paths:
            executor_cls = _executor_cls(worker_backend)
            subtree_results = []
            if len(child_paths) > 1:
                with executor_cls(max_workers=min(workers, len(child_paths))) as pool:
                    futures = [
                        pool.submit(
                            _discover_da3_subtree,
                            (root_order, camera_root, child_path, None),
                        )
                        for child_path in child_paths
                    ]
                    for future in as_completed(futures):
                        subtree_results.append(future.result())
            else:
                subtree_results = [_discover_da3_subtree((root_order, camera_root, child_paths[0], None))]
            subtree_results = sorted(
                subtree_results,
                key=lambda item: (
                    item.get("scene_records", [{}])[0].get("scene_dir", "") if item.get("scene_records") else ""
                ),
            )
            if done_result.get("scene_records"):
                return _merge_da3_child_results(
                    root_order,
                    camera_root,
                    done_result,
                    subtree_results,
                )
            return _merge_da3_subtree_results(
                root_order,
                camera_root,
                subtree_results,
            )
        if done_result.get("scene_records"):
            return done_result

    return _discover_da3_subtree((root_order, camera_root, camera_root, max_scenes, workers, worker_backend))


def _merge_da3_scene_root_results(root_results, *, verbose=True):
    scenes = {}
    duplicate_count = 0
    for result in sorted(root_results, key=lambda item: item["root_order"]):
        camera_root = result["root"]
        if result.get("missing"):
            _debug(f"skip missing DA3 root: {camera_root}", verbose=verbose)
            continue
        root_scene_count = 0
        duplicate_count += int(result.get("duplicate_count", 0) or 0)
        for record in result.get("scene_records", []):
            clean_name = record["clean_name"]
            if clean_name in scenes:
                duplicate_count += 1
            else:
                record["camera_root"] = camera_root
                record["camera_root_order"] = result.get("root_order")
                scenes[clean_name] = record
                root_scene_count += 1
        _debug(
            f"DA3 root scanned: root={camera_root} "
            f"dirs={int(result.get('dir_count', 0) or 0)} "
            f"new_scenes={root_scene_count} total_scenes={len(scenes)} "
            f"duplicates={duplicate_count}",
            verbose=verbose,
        )
    return scenes, duplicate_count


def _discover_da3_scenes(
    camera_roots,
    *,
    verbose=True,
    workers=1,
    worker_backend="process",
    max_scenes=None,
    max_scenes_per_root=None,
):
    max_scenes = _normalize_max_scenes(max_scenes)
    max_scenes_per_root = _normalize_max_scenes(max_scenes_per_root)
    workers = max(1, int(workers or 1))
    # A global --max_scenes must stop discovery as soon as enough unique scenes
    # have been found. Per-root caps are also handled serially so each root can
    # receive its own deterministic budget without fanning out across every
    # NFS root.
    if max_scenes is not None or max_scenes_per_root is not None:
        root_results = []
        remaining_global = max_scenes
        for root_order, camera_root in enumerate(
            tqdm(
                camera_roots,
                desc="DA3 roots",
                unit="root",
                disable=not verbose,
            )
        ):
            if remaining_global is None:
                root_limit = max_scenes_per_root
            elif max_scenes_per_root is None:
                root_limit = remaining_global
            else:
                root_limit = min(int(remaining_global), int(max_scenes_per_root))
            if root_limit is not None and root_limit <= 0:
                break
            result = _discover_da3_root((root_order, camera_root, root_limit, workers, worker_backend))
            root_results.append(result)
            if remaining_global is not None:
                remaining_global -= len(result.get("scene_records", []))
            if remaining_global is not None and remaining_global <= 0:
                break
        return _merge_da3_scene_root_results(root_results, verbose=verbose)

    root_args = [(idx, root, None, workers, worker_backend) for idx, root in enumerate(camera_roots)]
    executor_cls = _executor_cls(worker_backend)
    if executor_cls is not None and workers > 1 and len(root_args) > 1:
        root_results = []
        with executor_cls(max_workers=min(workers, len(root_args))) as pool:
            futures = [pool.submit(_discover_da3_root, arg) for arg in root_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Discover DA3 roots",
                unit="root",
                disable=not verbose,
            ):
                root_results.append(future.result())
    else:
        root_results = [
            _discover_da3_root(arg)
            for arg in tqdm(
                root_args,
                desc="DA3 roots",
                unit="root",
                disable=not verbose,
            )
        ]
    return _merge_da3_scene_root_results(root_results, verbose=verbose)


def _discover_prompt_root(args):
    root_order, prompt_root = args
    result = {
        "root_order": root_order,
        "root": prompt_root,
        "prompts": [],
        "dir_count": 0,
        "missing": False,
    }
    if not prompt_root or not os.path.isdir(prompt_root):
        result["missing"] = True
        return result

    index = {}
    stack = [prompt_root]
    while stack:
        result["dir_count"] += 1
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            continue
        json_files = [
            entry for entry in entries if entry.name.endswith(".json") and entry.is_file(follow_symlinks=False)
        ]
        if json_files:
            dir_scene_name = os.path.basename(os.path.normpath(current))
            for entry in sorted(json_files, key=lambda e: e.name):
                if entry.name == "video.json":
                    if dir_scene_name == "caption":
                        scene_name = os.path.basename(os.path.dirname(os.path.normpath(current)))
                    else:
                        scene_name = dir_scene_name
                    index.setdefault(scene_name, os.path.join(current, entry.name))
                else:
                    scene_name = entry.name[: -len(".json")]
                    index.setdefault(scene_name, os.path.join(current, entry.name))
            continue
        stack.extend(
            entry.path
            for entry in sorted(entries, key=lambda e: e.name, reverse=True)
            if entry.is_dir(follow_symlinks=False)
        )
    result["prompts"] = [(name, index[name]) for name in sorted(index)]
    return result


def _merge_prompt_root_results(root_results, *, verbose=True):
    index = {}
    for result in sorted(root_results, key=lambda item: item["root_order"]):
        prompt_root = result["root"]
        if result.get("missing"):
            _debug(f"skip missing prompt root: {prompt_root}", verbose=verbose)
            continue
        root_prompt_count = 0
        for scene_name, prompt_path in result.get("prompts", []):
            if scene_name not in index:
                root_prompt_count += 1
            index.setdefault(scene_name, prompt_path)
        _debug(
            f"prompt root scanned: root={prompt_root} "
            f"dirs={int(result.get('dir_count', 0) or 0)} "
            f"new_prompts={root_prompt_count} total_prompts={len(index)}",
            verbose=verbose,
        )
    return index


def _build_prompt_index(
    prompt_roots,
    *,
    verbose=True,
    workers=1,
    worker_backend="process",
):
    workers = max(1, int(workers or 1))
    root_args = [(idx, root) for idx, root in enumerate(prompt_roots)]
    if not root_args:
        return {}
    executor_cls = _executor_cls(worker_backend)
    if executor_cls is not None and workers > 1 and len(root_args) > 1:
        root_results = []
        with executor_cls(max_workers=min(workers, len(root_args))) as pool:
            futures = [pool.submit(_discover_prompt_root, arg) for arg in root_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Discover prompt roots",
                unit="root",
                disable=not verbose,
            ):
                root_results.append(future.result())
    else:
        root_results = [
            _discover_prompt_root(arg)
            for arg in tqdm(
                root_args,
                desc="Prompt roots",
                unit="root",
                disable=not verbose,
            )
        ]
    return _merge_prompt_root_results(root_results, verbose=verbose)


def _prompt_has_detailed(prompt_path):
    if not prompt_path:
        return False
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except (OSError, ValueError):
        return False
    detailed = payload.get("detailed", {}) or {}
    return bool(detailed)


def _npz_member_shape(npz_path, key):
    member_name = f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path) as zf:
            with zf.open(member_name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    shape, _fortran_order, _dtype = np.lib.format._read_array_header(f, version)
                return tuple(shape)
    except Exception:
        with np.load(npz_path, allow_pickle=True) as payload:
            array = np.asarray(payload[key])
        return tuple(array.shape)


def _read_extrinsic_frame_count(extrinsic_path):
    shape = _npz_member_shape(extrinsic_path, "data")
    if len(shape) != 3 or tuple(shape[1:]) != (4, 4):
        raise ValueError(f"extrinsics shape must be (N, 4, 4), got {shape}")
    return int(shape[0])


def _existing_file(path):
    return path if path and os.path.isfile(path) else None


def _normalize_npz_depth_format(value):
    name = str(value or "").strip().lower()
    aliases = {
        "logu10": "npz_log_u10",
        "log_u10": "npz_log_u10",
        "npz_log_u10": "npz_log_u10",
        "logu16": "npz_log_u16",
        "log_u16": "npz_log_u16",
        "npz_log_u16": "npz_log_u16",
        "float": "npz_float",
        "float32": "npz_float",
        "npz_float": "npz_float",
    }
    return aliases.get(name)


def _depth_format_from_metadata(payload):
    for key in ("format", "depth_format", "depth_encoding", "quantization"):
        normalized = _normalize_npz_depth_format(payload.get(key))
        if normalized:
            return normalized
    return "npz_float"


def _read_depth_metadata(depth_path):
    if not depth_path:
        return None, None, None
    metadata_path = os.path.join(os.path.dirname(depth_path), "metadata.json")
    if not os.path.isfile(metadata_path):
        return "npz_float", None, None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except (OSError, ValueError):
        return "npz_float", metadata_path, None
    if not isinstance(payload, dict):
        return "npz_float", metadata_path, None
    depth_format = _depth_format_from_metadata(payload)
    return depth_format, metadata_path, json.dumps(payload, ensure_ascii=False)


def _dynamic_mask_relpath(dynamic_mask_subdir, dynamic_mask_filename):
    parts = []
    if dynamic_mask_subdir:
        parts.append(str(dynamic_mask_subdir).strip("/"))
    if dynamic_mask_filename:
        parts.append(str(dynamic_mask_filename).strip("/"))
    return os.path.join(*parts) if parts else ""


def _dynamic_mask_path(scene_dir, dynamic_mask_subdir, dynamic_mask_filename):
    relpath = _dynamic_mask_relpath(dynamic_mask_subdir, dynamic_mask_filename)
    if not relpath:
        return None
    return os.path.join(scene_dir, relpath)


def _dynamic_mask_missing_error(dynamic_mask_subdir, dynamic_mask_filename):
    relpath = _dynamic_mask_relpath(dynamic_mask_subdir, dynamic_mask_filename)
    return f"missing_dynamic_mask:{relpath}" if relpath else "missing_dynamic_mask"


def _apply_dynamic_mask_requirement(records, dynamic_mask_subdir, dynamic_mask_filename):
    relpath = _dynamic_mask_relpath(dynamic_mask_subdir, dynamic_mask_filename)
    if not relpath:
        return records
    missing_error = _dynamic_mask_missing_error(dynamic_mask_subdir, dynamic_mask_filename)
    filtered = []
    for record in records:
        if record.scan_error is None:
            dynamic_mask_path = os.path.join(record.scene_dir, relpath)
            if not os.path.isfile(dynamic_mask_path):
                record = replace(record, scan_error=missing_error)
        filtered.append(record)
    return filtered


def _scan_one_scene(
    scene,
    prompt_index,
    min_frame_count,
    require_complete,
    require_dynamic_masks=False,
    dynamic_mask_subdir="objects",
    dynamic_mask_filename="dynamic_masks.jsonl",
):
    clean_name = scene["clean_name"]
    prompt_path = prompt_index.get(clean_name)
    video_path = _existing_file(scene.get("video_path"))
    extrinsic_path = _existing_file(scene.get("extrinsic_path"))
    intrinsic_path = _existing_file(scene.get("intrinsic_path"))
    depth_path = _existing_file(scene.get("depth_path"))
    depth_format, depth_metadata_path, depth_metadata_json = _read_depth_metadata(depth_path)
    has_prompt = int(_prompt_has_detailed(prompt_path))
    has_depth = int(bool(depth_path))

    frame_count = None
    scan_error = None
    checks = [
        ("missing_video", video_path),
        ("missing_extrinsic", extrinsic_path),
        ("missing_intrinsic", intrinsic_path),
        ("missing_depth", depth_path),
    ]
    if require_complete:
        for reason, value in checks:
            if not value:
                scan_error = reason
                break
        if scan_error is None and not has_prompt:
            scan_error = "missing_prompt"

    if extrinsic_path:
        try:
            frame_count = _read_extrinsic_frame_count(extrinsic_path)
        except Exception as exc:
            if scan_error is None:
                scan_error = f"invalid_extrinsics:{exc}"

    if scan_error is None and frame_count is None:
        scan_error = "missing_frame_count"
    if scan_error is None and int(frame_count) < int(min_frame_count):
        scan_error = "too_short"
    if scan_error is None and require_dynamic_masks:
        dynamic_mask_path = _dynamic_mask_path(
            scene["scene_dir"],
            dynamic_mask_subdir,
            dynamic_mask_filename,
        )
        if not dynamic_mask_path or not os.path.isfile(dynamic_mask_path):
            scan_error = _dynamic_mask_missing_error(
                dynamic_mask_subdir,
                dynamic_mask_filename,
            )

    return DA3VideoIndexRecord(
        clean_name=clean_name,
        camera_root=scene.get("camera_root"),
        camera_root_order=scene.get("camera_root_order"),
        scene_dir=scene["scene_dir"],
        synthetic_path=scene["synthetic_path"],
        video_path=video_path,
        extrinsic_path=extrinsic_path,
        intrinsic_path=intrinsic_path,
        depth_path=depth_path,
        depth_format=depth_format,
        depth_metadata_path=depth_metadata_path,
        depth_metadata_json=depth_metadata_json,
        prompt_path=prompt_path,
        frame_count=frame_count,
        has_prompt=has_prompt,
        has_depth=has_depth,
        scan_error=scan_error,
    )


def _scan_worker_init(
    prompt_index,
    min_frame_count,
    require_complete,
    require_dynamic_masks,
    dynamic_mask_subdir,
    dynamic_mask_filename,
):
    global _SCAN_WORKER_PROMPT_INDEX
    global _SCAN_WORKER_MIN_FRAME_COUNT
    global _SCAN_WORKER_REQUIRE_COMPLETE
    global _SCAN_WORKER_REQUIRE_DYNAMIC_MASKS
    global _SCAN_WORKER_DYNAMIC_MASK_SUBDIR
    global _SCAN_WORKER_DYNAMIC_MASK_FILENAME
    _SCAN_WORKER_PROMPT_INDEX = prompt_index
    _SCAN_WORKER_MIN_FRAME_COUNT = int(min_frame_count)
    _SCAN_WORKER_REQUIRE_COMPLETE = bool(require_complete)
    _SCAN_WORKER_REQUIRE_DYNAMIC_MASKS = bool(require_dynamic_masks)
    _SCAN_WORKER_DYNAMIC_MASK_SUBDIR = dynamic_mask_subdir
    _SCAN_WORKER_DYNAMIC_MASK_FILENAME = dynamic_mask_filename


def _scan_one_scene_in_worker(scene):
    return _scan_one_scene(
        scene,
        _SCAN_WORKER_PROMPT_INDEX or {},
        _SCAN_WORKER_MIN_FRAME_COUNT,
        _SCAN_WORKER_REQUIRE_COMPLETE,
        _SCAN_WORKER_REQUIRE_DYNAMIC_MASKS,
        _SCAN_WORKER_DYNAMIC_MASK_SUBDIR,
        _SCAN_WORKER_DYNAMIC_MASK_FILENAME,
    )


def _scan_scene_records(
    scene_values,
    prompt_index,
    min_frame_count,
    require_complete,
    require_dynamic_masks=False,
    dynamic_mask_subdir="objects",
    dynamic_mask_filename="dynamic_masks.jsonl",
    *,
    workers=1,
    worker_backend="process",
    verbose=True,
):
    workers = max(1, min(int(workers or 1), len(scene_values) or 1))
    worker_backend = _normalize_worker_backend(worker_backend)
    if workers <= 1 or len(scene_values) <= 1 or worker_backend == "serial":
        return [
            _scan_one_scene(
                scene,
                prompt_index,
                min_frame_count,
                bool(require_complete),
                bool(require_dynamic_masks),
                dynamic_mask_subdir,
                dynamic_mask_filename,
            )
            for scene in tqdm(
                scene_values,
                desc="Building DA3 video index",
                disable=not verbose,
            )
        ]

    records = []
    if worker_backend == "process":
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_scan_worker_init,
            initargs=(
                prompt_index,
                min_frame_count,
                bool(require_complete),
                bool(require_dynamic_masks),
                dynamic_mask_subdir,
                dynamic_mask_filename,
            ),
        ) as pool:
            futures = [pool.submit(_scan_one_scene_in_worker, scene) for scene in scene_values]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Building DA3 video index ({workers} process workers)",
                disable=not verbose,
            ):
                records.append(future.result())
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _scan_one_scene,
                    scene,
                    prompt_index,
                    min_frame_count,
                    bool(require_complete),
                    bool(require_dynamic_masks),
                    dynamic_mask_subdir,
                    dynamic_mask_filename,
                )
                for scene in scene_values
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Building DA3 video index ({workers} thread workers)",
                disable=not verbose,
            ):
                records.append(future.result())
    records.sort(key=lambda item: item.clean_name)
    return records


def _content_sha1(records):
    digest = hashlib.sha1()
    for record in sorted(records, key=lambda item: item.clean_name):
        payload = json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _record_from_mapping(row):
    payload = dict(row)
    if payload.get("frame_count") is not None:
        payload["frame_count"] = int(payload["frame_count"])
    if payload.get("camera_root_order") is not None:
        payload["camera_root_order"] = int(payload["camera_root_order"])
    payload["has_prompt"] = int(payload.get("has_prompt") or 0)
    payload["has_depth"] = int(payload.get("has_depth") or 0)
    payload.setdefault("camera_root", None)
    payload.setdefault("camera_root_order", None)
    payload.setdefault("depth_format", None)
    payload.setdefault("depth_metadata_path", None)
    payload.setdefault("depth_metadata_json", None)
    return DA3VideoIndexRecord(**payload)


def _records_columns(conn):
    rows = conn.execute("PRAGMA table_info(records)").fetchall()
    return {str(row[1]) for row in rows}


def _select_records_sql(columns):
    base_columns = [
        "clean_name",
        "camera_root",
        "camera_root_order",
        "scene_dir",
        "synthetic_path",
        "video_path",
        "extrinsic_path",
        "intrinsic_path",
        "depth_path",
        "depth_format",
        "depth_metadata_path",
        "depth_metadata_json",
        "prompt_path",
        "frame_count",
        "has_prompt",
        "has_depth",
        "scan_error",
    ]
    select_columns = [name if name in columns else f"NULL AS {name}" for name in base_columns]
    return "SELECT " + ", ".join(select_columns) + " FROM records"


def _load_all_records(index_path):
    with sqlite3.connect(index_path) as conn:
        conn.row_factory = sqlite3.Row
        query = _select_records_sql(_records_columns(conn)) + " ORDER BY clean_name"
        rows = conn.execute(query).fetchall()
    return [_record_from_mapping(row) for row in rows]


def _load_json_list(value):
    if not value:
        return []
    try:
        payload = json.loads(value)
    except ValueError:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _merge_ordered(old_items, new_items):
    merged = []
    seen = set()
    for item in list(old_items) + list(new_items):
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _write_index(output_path, records, meta, overwrite):
    output_path = os.fspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        raise FileExistsError(tmp_path)

    with sqlite3.connect(tmp_path) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE records (
                clean_name TEXT PRIMARY KEY,
                camera_root TEXT,
                camera_root_order INTEGER,
                scene_dir TEXT NOT NULL,
                synthetic_path TEXT NOT NULL,
                video_path TEXT,
                extrinsic_path TEXT,
                intrinsic_path TEXT,
                depth_path TEXT,
                depth_format TEXT,
                depth_metadata_path TEXT,
                depth_metadata_json TEXT,
                prompt_path TEXT,
                frame_count INTEGER,
                has_prompt INTEGER NOT NULL,
                has_depth INTEGER NOT NULL,
                scan_error TEXT
            )
            """
        )
        conn.execute("CREATE INDEX idx_records_synthetic_path ON records(synthetic_path)")
        conn.execute("CREATE INDEX idx_records_camera_root ON records(camera_root)")
        conn.execute("CREATE INDEX idx_records_camera_root_order ON records(camera_root_order)")
        conn.execute("CREATE INDEX idx_records_has_depth ON records(has_depth)")
        conn.execute("CREATE INDEX idx_records_has_prompt ON records(has_prompt)")
        conn.execute("CREATE INDEX idx_records_frame_count ON records(frame_count)")
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [(str(key), str(value)) for key, value in sorted(meta.items())],
        )
        conn.executemany(
            """
            INSERT INTO records(
                clean_name, camera_root, camera_root_order, scene_dir,
                synthetic_path, video_path, extrinsic_path, intrinsic_path,
                depth_path, depth_format, depth_metadata_path, depth_metadata_json,
                prompt_path, frame_count, has_prompt, has_depth, scan_error
            ) VALUES (
                :clean_name, :camera_root, :camera_root_order, :scene_dir,
                :synthetic_path, :video_path, :extrinsic_path, :intrinsic_path,
                :depth_path, :depth_format, :depth_metadata_path,
                :depth_metadata_json, :prompt_path, :frame_count, :has_prompt,
                :has_depth, :scan_error
            )
            """,
            [asdict(record) for record in records],
        )
    os.replace(tmp_path, output_path)


def build_da3_video_index(
    *,
    camera_params_path,
    prompt_path,
    output_path,
    min_frame_count=1,
    require_complete=True,
    overwrite=False,
    append=False,
    workers=32,
    worker_backend="process",
    max_scenes=None,
    max_scenes_per_root=None,
    require_dynamic_masks=False,
    dynamic_mask_subdir="objects",
    dynamic_mask_filename="dynamic_masks.jsonl",
    verbose=True,
):
    camera_roots = _split_path_list(camera_params_path)
    prompt_roots = _split_path_list(prompt_path)
    min_frame_count = max(1, int(min_frame_count or 1))
    workers = max(1, int(workers or 1))
    worker_backend = _normalize_worker_backend(worker_backend)
    max_scenes = _normalize_max_scenes(max_scenes)
    max_scenes_per_root = _normalize_max_scenes(max_scenes_per_root)
    _debug(
        f"start build output={output_path} append={bool(append)} "
        f"overwrite={bool(overwrite)} workers={workers} "
        f"worker_backend={worker_backend} max_scenes={max_scenes} "
        f"max_scenes_per_root={max_scenes_per_root} "
        f"min_frame_count={min_frame_count} require_complete={bool(require_complete)} "
        f"require_dynamic_masks={bool(require_dynamic_masks)} "
        f"dynamic_mask={_dynamic_mask_relpath(dynamic_mask_subdir, dynamic_mask_filename)}",
        verbose=verbose,
    )
    old_records = []
    old_meta = {}
    output_exists = os.path.exists(os.fspath(output_path))
    if append:
        if not output_exists:
            raise FileNotFoundError(output_path)
        old_meta = load_da3_video_index_meta(output_path)
        if str(old_meta.get("dataset_kind")) != "da3_video":
            raise ValueError(f"Cannot append to non-DA3 index: {output_path}")
        old_schema_version = int(old_meta.get("schema_version", -1))
        if old_schema_version not in {2, DA3_VIDEO_INDEX_SCHEMA_VERSION}:
            raise ValueError(
                f"Cannot append to schema_version={old_meta.get('schema_version')}; "
                f"expected one of {[2, DA3_VIDEO_INDEX_SCHEMA_VERSION]}"
            )
        old_records = _load_all_records(output_path)
        _debug(
            f"append base loaded: records={len(old_records)} "
            f"kept={old_meta.get('kept_count')} rejected={old_meta.get('rejected_count')} "
            f"content_sha1={old_meta.get('content_sha1')}",
            verbose=verbose,
        )

    scenes, duplicate_count = _discover_da3_scenes(
        camera_roots,
        verbose=verbose,
        workers=workers,
        worker_backend=worker_backend,
        max_scenes=max_scenes,
        max_scenes_per_root=max_scenes_per_root,
    )
    _debug(
        f"scene discovery done: unique_scenes={len(scenes)} duplicates_in_new_roots={duplicate_count}",
        verbose=verbose,
    )
    prompt_index = _build_prompt_index(
        prompt_roots,
        verbose=verbose,
        workers=workers,
        worker_backend=worker_backend,
    )
    _debug(
        f"prompt discovery done: prompts={len(prompt_index)}",
        verbose=verbose,
    )

    scene_values = [scenes[name] for name in sorted(scenes)]
    records = _scan_scene_records(
        scene_values,
        prompt_index,
        min_frame_count,
        bool(require_complete),
        bool(require_dynamic_masks),
        dynamic_mask_subdir,
        dynamic_mask_filename,
        workers=workers,
        worker_backend=worker_backend,
        verbose=verbose,
    )
    scanned_kept_count = sum(1 for record in records if record.scan_error is None)
    _debug(
        f"new-root probe done: scanned={len(records)} kept={scanned_kept_count} "
        f"rejected={len(records) - scanned_kept_count}",
        verbose=verbose,
    )

    if append:
        merged_by_name = {record.clean_name: record for record in old_records}
        append_duplicate_count = 0
        for record in records:
            if record.clean_name in merged_by_name:
                append_duplicate_count += 1
                continue
            merged_by_name[record.clean_name] = record
        records = sorted(merged_by_name.values(), key=lambda item: item.clean_name)
        duplicate_count += append_duplicate_count + int(old_meta.get("duplicate_clean_name_count", 0) or 0)
        camera_roots = _merge_ordered(
            _load_json_list(old_meta.get("camera_params_path")),
            camera_roots,
        )
        prompt_roots = _merge_ordered(
            _load_json_list(old_meta.get("prompt_path")),
            prompt_roots,
        )
        _debug(
            f"append merge done: old={len(old_records)} new_scanned={len(scene_values)} "
            f"new_duplicates={append_duplicate_count} merged={len(records)}",
            verbose=verbose,
        )

    if require_dynamic_masks:
        records = _apply_dynamic_mask_requirement(
            records,
            dynamic_mask_subdir,
            dynamic_mask_filename,
        )

    kept_count = sum(1 for record in records if record.scan_error is None)
    rejected_count = len(records) - kept_count
    meta = {
        "schema_version": DA3_VIDEO_INDEX_SCHEMA_VERSION,
        "dataset_kind": "da3_video",
        "scanner_version": "1",
        "camera_params_path": json.dumps(camera_roots, ensure_ascii=False),
        "prompt_path": json.dumps(prompt_roots, ensure_ascii=False),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "content_sha1": _content_sha1(records),
        "min_frame_count": min_frame_count,
        "require_complete": int(bool(require_complete)),
        "require_dynamic_masks": int(bool(require_dynamic_masks)),
        "dynamic_mask_subdir": dynamic_mask_subdir,
        "dynamic_mask_filename": dynamic_mask_filename,
        "workers": workers,
        "worker_backend": worker_backend,
        "max_scenes": "" if max_scenes is None else int(max_scenes),
        "max_scenes_per_root": ("" if max_scenes_per_root is None else int(max_scenes_per_root)),
        "record_count": len(records),
        "kept_count": kept_count,
        "rejected_count": rejected_count,
        "duplicate_clean_name_count": duplicate_count,
        "append_mode": int(bool(append)),
    }
    _write_index(output_path, records, meta, overwrite=bool(overwrite or append))
    _debug(
        f"index written: path={output_path} records={len(records)} kept={kept_count} "
        f"rejected={rejected_count} duplicates={duplicate_count} "
        f"content_sha1={meta['content_sha1']}",
        verbose=verbose,
    )
    return {
        "record_count": len(records),
        "kept_count": kept_count,
        "rejected_count": rejected_count,
        "duplicate_clean_name_count": duplicate_count,
        "output_path": os.fspath(output_path),
    }


def load_da3_video_index_meta(index_path):
    with sqlite3.connect(index_path) as conn:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(key): value for key, value in rows}


def load_da3_video_index_records(index_path, *, only_complete=True, min_frame_count=None):
    with sqlite3.connect(index_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = _records_columns(conn)
        clauses = []
        params = []
        if only_complete and "scan_error" in columns:
            clauses.append("scan_error IS NULL")
        if min_frame_count is not None and "frame_count" in columns:
            clauses.append("frame_count >= ?")
            params.append(int(min_frame_count))
        query = _select_records_sql(columns)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY synthetic_path"
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
