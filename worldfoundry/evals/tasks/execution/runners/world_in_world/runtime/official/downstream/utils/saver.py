import datetime as _dt
import json
import os
import os.path as osp
import re
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


#######saving method usied in saver.py#######
class Saver:
    """
    A simple helper class containing all path-related (file/directory) methods.
    It stores 'parallel_ith' and 'parallel_total' to generate parallelized
    output paths if needed.
    """

    def __init__(self, parallel_ith=None, parallel_total=None, exp_id="", task="AR"):
        self.parallel_ith = parallel_ith
        self.parallel_total = parallel_total
        self.exp_id = exp_id
        self.task = task
        assert self.task in ["AR", "AEQA", "VLN", "ObjNav", "IGNav"]

    def get_task_path_pref(self) -> str:
        return osp.join("downstream", "states", f"{self.task}_{self.exp_id}")

    def get_datum_path_pref(self, datum) -> str:
        if self.task in ("AR", "VLN", "ObjNav", "IGNav"):
            pref = f"E{datum['episode_id']:03d}"
        elif self.task == "AEQA":
            pref = f"Q{datum['question_id']}"
        else:
            raise ValueError(f"Invalid task: {self.task}")
        return osp.join(
            self.get_task_path_pref(),
            (osp.basename(datum["scene_id"])).split(".")[0],
            pref,
        )

    def get_action_path_pref(self, datum, ith_action: int) -> str:
        return osp.join(self.get_datum_path_pref(datum), f"A{ith_action:03d}")

    def get_image_path(self, datum, ith_action, sensor_type, suffix="") -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            f"{sensor_type}{suffix}.png",
        )

    def get_meta_path(self, datum, ith_action, sensor_type, suffix="") -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            f"{sensor_type}{suffix}.json",
        )

    def get_category_path(self, datum, target_category) -> str:
        assert self.task in ["AR", "ObjNav"]
        return osp.join(self.get_datum_path_pref(datum), f"LABEL={target_category}.txt")

    def get_planner_output_path(self, datum, ith_action, action_num, postfix="") -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            f"planner_next-{action_num}{postfix}.json",
        )

    def get_chat_log_output_path(self, action_path) -> str:
        # replace the item in action_path
        # that ends with .json with .chatlog.json
        return osp.join(
            osp.dirname(action_path),
            osp.basename(action_path).replace(".json", ".chatlog.json"),
        )

    def get_highlevel_planner_output_path(self, datum, ith_action) -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            "planner_highlevel.json",
        )

    def get_highlevel_planner_imagine_output_path(self, datum, ith_action) -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            "planner_highlevel_imagine.json",
        )

    def get_visual_prompt_path(self, datum, ith_action) -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            "visual_prompt.png",
        )

    def get_visual_prompt_waypoint_path(self, datum, ith_action) -> str:
        return osp.join(
            self.get_action_path_pref(datum, ith_action),
            "visual_prompt_waypoint.png",
        )

    def get_answerer_output_path(self, datum, ith_action) -> str:
        return osp.join(self.get_action_path_pref(datum, ith_action), "answerer.json")

    def get_metric_path(self, datum=None, subset_len=None, auto_create_dir=True) -> str:
        """
        Determine the metric file path based on whether we have a datum (episode-level file)
        or a parallel_ith scenario (global worker-level file).
        We do NOT create the file here, only return the path.
        """
        if datum is not None:
            metric_path = osp.join(self.get_datum_path_pref(datum), "metrics.jsonl")
        else:
            task_path_pref = self.get_task_path_pref()
            if self.parallel_ith is not None:
                metric_path = osp.join(task_path_pref, f"metrics_{self.parallel_ith}.jsonl")
            else:
                if subset_len is not None:
                    metric_path = osp.join(task_path_pref, f"metrics_{subset_len:05d}.jsonl")
                else:
                    metric_path = osp.join(task_path_pref, "metrics.jsonl")

        # Make sure the directory for the metric file exists
        dir_path = osp.dirname(metric_path)
        if not osp.exists(dir_path) and auto_create_dir:
            os.makedirs(dir_path, exist_ok=True)

        return metric_path


# match “data:…;base64,” + the actual data, but STOP at “?” (query) or end-of-string
_BASE64_RE = re.compile(r"(data:[^;]+;base64,)[A-Za-z0-9+/=\r\n]+(?=\?|$)")


def _strip_base64(obj: Any) -> Any:
    """
    Recursively replace Base-64 payloads with a placeholder.
    """
    if isinstance(obj, str):
        # keep group 1 (prefix), drop the payload, keep ?src=… if present
        return _BASE64_RE.sub(r"\1<base64 data removed>", obj)
    if isinstance(obj, list):
        return [_strip_base64(el) for el in obj]
    if isinstance(obj, dict):
        return {k: _strip_base64(v) for k, v in obj.items()}
    return obj


def format_chat_dialog(
    messages: list[dict[str, Any]],
    assistant_reply: str | dict[str, Any],
    remove_base64: bool = True,
) -> dict[str, Any]:
    """
    Save a chat transcript in a clear, readable JSON file and return it.

    Parameters
    ----------
    messages
        The list passed to ``openai.chat.completions.create``.
        Each element must look like
        ``{"role": "system" | "user" | "assistant", "content": str, ...}``.
    assistant_reply
        Either the raw assistant text (``str``) or the full message ``dict``
        returned by the API (e.g. ``completion.choices[0].message``).
    file_path
        Destination ``.json`` file.  Missing parent directories are created.
    remove_base64
        If ``True`` (default) replace any in-line Base-64 image data with the
        placeholder string ``"<base64 data removed>"``.
    """
    transcript: dict[str, Any] = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "dialog": messages
        + (
            [{"role": "assistant", "content": assistant_reply}]
            if isinstance(assistant_reply, str)
            else [{"assistant": assistant_reply}]
        ),
    }
    if remove_base64:
        transcript = _strip_base64(transcript)  # type: ignore[assignment]

    return transcript


def get_save_dir_from_existing(obs_path):
    """
    Get the predicted image path from the existing observation path.
    """
    folder = os.path.dirname(obs_path)
    path = os.path.join(folder, "igenex")
    os.makedirs(path, exist_ok=True)
    return path


def safe_chmod(path: str, mode: int) -> None:
    """Best-effort chmod that ignores PermissionError when you are not the owner."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        warnings.warn(f"Could not chmod {path} to {oct(mode)}; skipped.", RuntimeWarning, stacklevel=1)


def get_igenex_save_dirs(base_dir, action_ids_list):
    save_dirs = []
    for _i, action_id in enumerate(action_ids_list):
        action_seq_dir = f"PredA-{action_id}"
        save_path = osp.join(base_dir, action_seq_dir)
        # append a timestamp to avoid overwriting
        timestamp = _dt.datetime.now().strftime("%m%d_%H%M%S")
        save_path = f"{save_path}_{timestamp}"
        os.makedirs(save_path)
        safe_chmod(save_path, 0o777)  # ensure final mode even if umask interfered
        save_dirs.append(save_path)

    save_dirs = [os.path.abspath(f) for f in save_dirs]
    return save_dirs


def save_action_sequence(b_action, save_dirs):
    """
    Save action sequences as JSON files in the specified directories.

    Input:
        - b_action: Int32[Tensor, "b 14"] actions corresponding to the frames
        - save_dirs: List of directories to save the action sequences
    """
    assert len(b_action) == len(save_dirs)
    for i, dir in enumerate(save_dirs):
        action_seq_path = osp.join(dir, "action_seq.json")
        action_seq = b_action[i].tolist() if hasattr(b_action, "tolist") else b_action[i]
        with open(action_seq_path, "w") as f:
            json.dump(action_seq, f, indent=2, ensure_ascii=False)


def save_video_frames(
    videos: torch.Tensor,  # (B, T, C, H, W) or list
    out_dirs: Iterable[str],
    model_type: str = "default",
) -> None:
    """
    Save batched video tensors to image folders.

    videos      : Tensor of shape (B, T, C, H, W) **or** an iterable of per-video tensors.
    out_dirs    : One output directory per video (same length as `videos`).
    model_type  : "default" (unit-range RGB) or "nwm" (-1‥1 normalised RGB).

    Each frame is written as <idx>.jpg, and both the directory and the files
    receive chmod 0o777 so that all users can read/write them.
    """
    assert len(videos) == len(out_dirs), "videos and out_dirs length mismatch"

    for clip, folder in zip(videos, out_dirs, strict=False):
        save_images(clip, Path(folder), model_type=model_type)


def save_predict(video_tensors, b_action, save_dirs, model_type="default"):
    """
    Save video frames and action sequences into the specified directories.

    Input:
        - video_tensors: Float[Tensor, "b 14 C H W"]
        - b_action: Int32[Tensor, "b 14"] actions corresponding to the frames
        - save_dirs: List of directories to save the data
        - model_type: Model type for saving images
    """
    assert len(video_tensors) == len(save_dirs) == len(b_action)
    save_video_frames(video_tensors, save_dirs, model_type)
    save_action_sequence(b_action, save_dirs)
    return save_dirs


def clean_jpgs_under_folder(folder_path, exclude_list=None):
    """
    Remove all jpg files under the folder_path
    Args:
        folder_path (str): the path to the folder
    """
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") and file not in exclude_list:
            os.remove(os.path.join(folder_path, file))


# --------------------------------------- helpers (kept minimal) ---------------------------------
def _as_thwc(arr: np.ndarray) -> np.ndarray:
    """
    Convert numpy array with possible shapes to (T,H,W,C).
    Accepts:
      5D: (B,T,C,H,W) or (B,T,H,W,C)  -> collapse to T'
      4D: (T,C,H,W) or (T,H,W,C)
      3D: (C,H,W) or (H,W,C)          -> add T=1
      2D: (H,W)                        -> add T=1,C=1
    """
    if arr.ndim == 5:
        # collapse batch and time
        b, t = arr.shape[:2]
        arr = arr.reshape(b * t, *arr.shape[2:])  # now 4D

    elif arr.ndim == 4:
        # (T,H,W,C) or (T,C,H,W)
        if arr.shape[-1] in (1, 3, 4):  # THWC
            thwc = arr
        elif arr.shape[1] in (1, 3, 4):  # TCHW
            thwc = arr.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Ambiguous 4D shape {arr.shape}: cannot infer channels.")
        return _fix_gray(thwc)

    raise ValueError(f"Unsupported array with ndim={arr.ndim}")


def _to_uint8(x: np.ndarray) -> np.ndarray:
    """Convert float/other dtypes to uint8 safely."""
    if np.issubdtype(x.dtype, np.floating):
        maxv = np.nanmax(x)
        if maxv <= 1.00001:  # assume [0,1]
            x = x * 255.0
        x = np.clip(np.rint(x), 0, 255)
        return x.astype(np.uint8)
    # integers or others: just clip-cast
    return np.clip(x, 0, 255).astype(np.uint8)


def _fix_gray(thwc: np.ndarray) -> np.ndarray:
    """Ensure last dim exists; if missing, add channel dim as 1."""
    if thwc.ndim != 4:
        raise ValueError(f"Expected THWC, got shape {thwc.shape}")
    if thwc.shape[-1] == 0:
        raise ValueError("Channel dimension has size 0.")
    return thwc


def _stack_and_fix_channels(arrs: list[np.ndarray]) -> np.ndarray:
    """
    Stack list of PIL->np arrays to (T,H,W,C). If grayscale (H,W), add C=1.
    """
    fixed = []
    for a in arrs:
        if a.ndim == 2:  # grayscale
            a = a[..., None]
        elif a.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape {a.shape}")
        fixed.append(a)
    return np.stack(fixed, axis=0)


def _to_uint8_numpy(
    frames: list[Image.Image] | torch.Tensor | np.ndarray,
    resize_to: tuple[int, int] | None = None,  # (width, height)
) -> list[np.ndarray]:
    """
    Convert input frames to a list of uint8 numpy arrays in H×W×C (RGB).
    Handles:
      - list[PIL.Image.Image]
      - torch tensors: (T,C,H,W)/(B,T,C,H,W) or (T,H,W,C)/(B,T,H,W,C)
      - numpy arrays:  (T,C,H,W) or (T,H,W,C), uint8 or float
    """
    # 1) normalize to a single numpy array in THWC
    if isinstance(frames, list):
        arrs = [np.asarray(img) for img in frames]
        thwc = _stack_and_fix_channels(arrs)  # -> (T,H,W,C), uint8-ish
    elif isinstance(frames, torch.Tensor):
        np_frames = frames.detach().cpu().numpy()
        thwc = _as_thwc(np_frames)
    elif isinstance(frames, np.ndarray):
        thwc = _as_thwc(frames)
    else:
        raise TypeError(f"Unsupported frames type: {type(frames)}")

    # 2) dtype to uint8 (accept float in [0,1] or [0,255])
    if thwc.dtype != np.uint8:
        thwc = _to_uint8(thwc)

    # 3) ensure 3 channels for mp4 writers (replicate grayscale)
    if thwc.shape[-1] == 1:
        thwc = np.repeat(thwc, 3, axis=-1)

    # 4) optional resize (PIL expects (width, height))
    if resize_to is not None:
        w, h = resize_to
        thwc = np.stack(
            [np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in thwc],
            axis=0,
        )

    # 5) return as list of HWC uint8
    return [thwc[i] for i in range(thwc.shape[0])]


def save_video(
    frames: list[Image.Image] | torch.Tensor | np.ndarray,
    save_path: str,
    fps: int = 3,
    resize_to: tuple[int, int] | None = None,  # (width, height)
) -> None:
    """
    Write *frames* to *save_path* (mp4). Accepts:
      - list[PIL.Image]
      - torch.Tensor with shape (T,C,H,W), (B,T,C,H,W), (T,H,W,C), or (B,T,H,W,C)
      - np.ndarray with shape (T,C,H,W) or (T,H,W,C), dtype uint8 or float
    resize_to: (width, height). If None, keep original size.
    """
    frames_np = _to_uint8_numpy(frames, resize_to)  # List[np.ndarray] (H,W,C), uint8

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with imageio.get_writer(save_path, fps=fps) as writer:
        for f in frames_np:
            writer.append_data(f)
    print(f"Video saved to: {save_path}")


# -----------------------------------------------------------------------------------------


def save_images(
    frames: torch.Tensor,  # (T, C, H, W)
    folder: str,
    model_type: str = "default",
) -> None:
    """
    Save a single video’s frames to `folder` as JPG.

    The folder is created if absent, then chmod’ed to 0o777.  Each saved image
    is also chmod’ed to 0o777 to guarantee write/read access for everyone.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe_chmod(folder, 0o777)

    # Choose the per-frame conversion once outside the loop
    if model_type == "default":

        def _prep(x: torch.Tensor) -> torch.Tensor:
            return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    elif model_type == "nwm":
        from downstream.api_models.nwm import misc  # local import avoids heavy deps when unused

        def _prep(x: torch.Tensor) -> torch.Tensor:
            y = misc.unnormalize(x.detach().float()).clamp(0, 1)
            return y

    # Disable grad & avoid device-CPU round-trips inside loop
    for i, frame in enumerate(frames):
        # If the image is a torch.Tensor
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        img = _prep(frame)
        file_path = folder / f"{i}.jpg"
        save_image(img, file_path)
        safe_chmod(file_path, 0o666)

    print(f"[save_images] {len(frames)} frames written to “{folder}”.")


# ============ Save RGB for nwm model ==============
def prepare_saved_imgs_nwm(batch_rgbs, save_dirs):
    from data_filtering.filter_util import save_img

    rgb_paths = []
    assert len(batch_rgbs) == len(save_dirs)

    for img, dir in zip(batch_rgbs, save_dirs, strict=False):
        # save the img into the save_dirs
        rgb_path = osp.join(dir, "cond_rgb.png")
        save_img(img, rgb_path)

        rgb_paths.append(rgb_path)

    return rgb_paths


# ============ Save RGB and depth imgs for se3ds model ==============


def prepare_saved_imgs(batch_rgbs, batch_depths, save_dirs):
    from data_filtering.filter_util import save_img

    rgb_paths, depth_paths = [], []
    assert len(batch_rgbs) == len(batch_depths) == len(save_dirs)

    for img, depth_img, dir in zip(batch_rgbs, batch_depths, save_dirs, strict=False):
        # save the img into the save_dirs
        rgb_path = osp.join(dir, "cond_pano_rgb.png")
        depth_path = osp.join(dir, "cond_pano_depth.png")
        save_img(img, rgb_path)

        # normalize depth into [0,1] by depth_scale,
        depth_scale = 20.0
        depth_norm = depth_img / depth_scale
        # create a mask of “too far” pixels, zero them out
        depth_clipped = torch.where(depth_norm > 1.0, torch.zeros_like(depth_norm), depth_norm)
        depth_clipped = torch.clamp(depth_clipped, min=0.0, max=1.0)

        # convert to uint16 range [0,65535]
        depth_uint16 = (depth_clipped * 65535).to(torch.uint16)
        save_img(depth_uint16, depth_path)

        rgb_paths.append(rgb_path)
        depth_paths.append(depth_path)

    return rgb_paths, depth_paths


def prepare_saved_imgs_from_fpaths(batch_rgbs, batch_depths, save_dirs) -> tuple[list[str], list[str]]:
    """
    Copy each (rgb, depth) image pair to its corresponding directory in `save_dirs`,
    saving them as fixed filenames:
        - cond_pano_rgb.png
        - cond_pano_depth.png

    Returns:
        (rgb_paths, depth_paths): lists of destination file paths (as strings), in order.
    """
    if not (len(batch_rgbs) == len(batch_depths) == len(save_dirs)):
        raise ValueError(
            f"Length mismatch: len(batch_rgbs)={len(batch_rgbs)}, "
            f"len(batch_depths)={len(batch_depths)}, len(save_dirs)={len(save_dirs)}"
        )

    rgb_paths: list[str] = []
    depth_paths: list[str] = []
    for rgb_fp, depth_fp, dst_dir in zip(batch_rgbs, batch_depths, save_dirs, strict=False):
        rgb_src = Path(rgb_fp)
        depth_src = Path(depth_fp)
        dst_dir = Path(dst_dir)

        # Validate sources
        if not rgb_src.is_file():
            raise FileNotFoundError(f"RGB image not found: {rgb_src}")
        if not depth_src.is_file():
            raise FileNotFoundError(f"Depth image not found: {depth_src}")

        # Ensure destination directory exists
        dst_dir.mkdir(parents=True, exist_ok=True)
        rgb_dst = dst_dir / "cond_pano_rgb.png"
        depth_dst = dst_dir / "cond_pano_depth.png"

        # Copy (overwrite if exists), preserving basic metadata
        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(depth_src, depth_dst)
        rgb_paths.append(str(rgb_dst))
        depth_paths.append(str(depth_dst))

    return rgb_paths, depth_paths


if __name__ == "__main__":
    u = "data:image/png;base64,iVBORw0KGgoAAA?src=/tmp/foo.png"
    print(_strip_base64(u))
    # → data:image/png;base64,<base64 data removed>?src=/tmp/foo.png
