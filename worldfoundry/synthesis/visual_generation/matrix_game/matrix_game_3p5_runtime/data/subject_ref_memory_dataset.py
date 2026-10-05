import json
import os
from collections import OrderedDict

import numpy as np
import torch

from .unified_dataset import (
    DA3MosaicVideoDataset,
    _derive_seed,
)

IMAGENET_MEAN_RGB = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in {path}:{line_no}: {exc}") from exc
    return rows


def _sample_reference_indices(
    candidate_count,
    pairwise_similarity,
    *,
    num_refs,
    rng,
    dissimilar_top_k,
    max_similarity,
):
    if candidate_count <= 0 or num_refs <= 0:
        return []
    k = min(int(num_refs), int(candidate_count))
    if k <= 1:
        return [int(rng.integers(0, candidate_count))]

    sim = np.asarray(pairwise_similarity, dtype=np.float32)
    if sim.shape != (candidate_count, candidate_count):
        # Fall back to random unique refs if the similarity file is stale.
        return [int(v) for v in rng.choice(candidate_count, size=k, replace=False)]
    sim = np.nan_to_num(sim, nan=1.0, posinf=1.0, neginf=-1.0)
    sim = np.clip(sim, -1.0, 1.0)

    selected = [int(rng.integers(0, candidate_count))]
    remaining = set(range(candidate_count))
    remaining.discard(selected[0])
    top_k = max(1, int(dissimilar_top_k))
    max_sim_threshold = float(max_similarity)
    while remaining and len(selected) < k:
        scored = [(max(float(sim[int(idx), int(sel)]) for sel in selected), int(idx)) for idx in sorted(remaining)]
        scored.sort(key=lambda item: item[0])
        valid = [idx for max_sim, idx in scored if max_sim <= max_sim_threshold]
        choice_pool = (
            valid[: min(top_k, len(valid))] if valid else [idx for _, idx in scored[: min(top_k, len(scored))]]
        )
        choice = int(choice_pool[int(rng.integers(0, len(choice_pool)))])
        selected.append(choice)
        remaining.discard(choice)
    return selected


def _safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class SubjectRefMemoryDA3MosaicVideoDataset(DA3MosaicVideoDataset):
    """DA3 video dataset with VAE reference-memory data preparation.

    This first-step dataset only prepares reference RGB canvases and
    protagonist patch masks. Transformer-side reference memory is added in a
    later step. Base DA3 behavior is unchanged when this dataset type is not
    selected.
    """

    def __init__(
        self,
        *args,
        subject_ref_dir_name="protagonist_refs",
        subject_num_refs_max=2,
        subject_dissimilar_top_k=8,
        subject_max_similarity=0.94,
        subject_shuffle_refs=True,
        subject_ref_canvas_slot_ratio=0.5,
        subject_ref_background="imagenet_mean",
        subject_ref_cache_max=32,
        **kwargs,
    ):
        self.subject_ref_dir_name = str(subject_ref_dir_name or "protagonist_refs")
        self.subject_num_refs_max = max(0, int(subject_num_refs_max))
        self.subject_dissimilar_top_k = max(1, int(subject_dissimilar_top_k))
        self.subject_max_similarity = float(subject_max_similarity)
        self.subject_shuffle_refs = bool(subject_shuffle_refs)
        self.subject_ref_canvas_slot_ratio = min(
            1.0,
            max(0.05, float(subject_ref_canvas_slot_ratio)),
        )
        self.subject_ref_background = str(subject_ref_background or "imagenet_mean").strip().lower()
        self.subject_ref_cache_max = max(1, int(subject_ref_cache_max))
        self._subject_ref_cache = OrderedDict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        info = data.get("info") or {}
        ref_seed_key = info.get("clean_name") or info.get("video_path") or int(index)
        rng = np.random.default_rng(_derive_seed(self.seed, "inference_subject_ref", ref_seed_key))
        data.update(self._sample_subject_ref_canvases(data, rng))
        return data

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "dataset_kind": "da3_video_subject_ref",
                "subject_ref_dir_name": self.subject_ref_dir_name,
                "subject_num_refs_max": int(self.subject_num_refs_max),
                "subject_dissimilar_top_k": int(self.subject_dissimilar_top_k),
                "subject_max_similarity": float(self.subject_max_similarity),
                "subject_shuffle_refs": bool(self.subject_shuffle_refs),
                "subject_ref_canvas_slot_ratio": float(self.subject_ref_canvas_slot_ratio),
                "subject_ref_background": self.subject_ref_background,
            }
        )
        return base

    def _scene_dir_from_info(self, info):
        scene_dir = info.get("dirname") or info.get("scene_dir")
        if scene_dir:
            return str(scene_dir)
        video_path = info.get("video_path")
        if video_path:
            return os.path.dirname(os.path.dirname(str(video_path)))
        return ""

    def _subject_ref_dir(self, info):
        scene_dir = self._scene_dir_from_info(info)
        if not scene_dir:
            return ""
        return os.path.join(
            scene_dir,
            "objects",
            self.subject_ref_dir_name,
        )

    def _load_subject_refs(self, ref_dir):
        if not ref_dir:
            return None
        cached = self._subject_ref_cache.get(ref_dir)
        if cached is not None:
            self._subject_ref_cache.move_to_end(ref_dir)
            return cached

        candidates_path = os.path.join(ref_dir, "candidates.jsonl")
        if not os.path.exists(candidates_path):
            return None
        candidates = _read_jsonl(candidates_path)
        candidates = [row for row in candidates if row.get("frame_idx") is not None]
        if not candidates:
            return None

        pairwise = None
        pairwise_path = os.path.join(ref_dir, "pairwise_similarity.npy")
        if os.path.exists(pairwise_path):
            try:
                pairwise = np.load(pairwise_path).astype(np.float32, copy=False)
            except Exception:
                pairwise = None

        selected_track_ids = []
        track_path = os.path.join(ref_dir, "track.json")
        if os.path.exists(track_path):
            with open(track_path, "r", encoding="utf-8") as f:
                track = json.load(f) or {}
            for value in track.get("selected_track_ids") or []:
                try:
                    selected_track_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            if not selected_track_ids and track.get("track_id") is not None:
                try:
                    selected_track_ids.append(int(track["track_id"]))
                except (TypeError, ValueError):
                    pass

        payload = {
            "candidates": candidates,
            "pairwise": pairwise,
            "selected_track_ids": selected_track_ids,
            "ref_dir": ref_dir,
        }
        self._subject_ref_cache[ref_dir] = payload
        self._subject_ref_cache.move_to_end(ref_dir)
        while len(self._subject_ref_cache) > self.subject_ref_cache_max:
            self._subject_ref_cache.popitem(last=False)
        return payload

    @staticmethod
    def _normal_to_uint8_canvas_value(background):
        if str(background).strip().lower() == "zero":
            return np.asarray([127.5, 127.5, 127.5], dtype=np.float32)
        if str(background).strip().lower() == "black":
            return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        return IMAGENET_MEAN_RGB

    @staticmethod
    def _pil_read_rgb(path):
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy()

    @staticmethod
    def _pil_read_mask(path):
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("L")).copy()

    def _read_ref_rgb(self, path):
        import time

        import cv2

        errors = []
        for attempt in range(3):
            try:
                return self._pil_read_rgb(path)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1} pil={type(exc).__name__}: {exc}")
            image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            errors.append(f"attempt={attempt + 1} cv2=None")
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(
            f"failed to read subject ref cutout after cv2/PIL retries: {path} size={size} errors={errors}"
        )

    def _read_ref_mask(self, path):
        import time

        import cv2

        errors = []
        for attempt in range(3):
            try:
                return self._pil_read_mask(path)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1} pil={type(exc).__name__}: {exc}")
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
            errors.append(f"attempt={attempt + 1} cv2=None")
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(f"failed to read subject ref mask after cv2/PIL retries: {path} size={size} errors={errors}")

    def _load_saved_ref_cutout_rgb(self, ref_dir, row):
        image_path = row.get("image_path")
        if not image_path:
            raise ValueError(f"subject ref candidate missing image_path: {row}")
        path = os.path.join(str(ref_dir), str(image_path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"subject ref cutout does not exist: {path}")
        image_rgb = self._read_ref_rgb(path)

        mask_path = row.get("mask_path")
        if not mask_path:
            raise ValueError(f"subject ref candidate missing mask_path: {row}")
        full_mask_path = os.path.join(str(ref_dir), str(mask_path))
        if not os.path.exists(full_mask_path):
            raise FileNotFoundError(f"subject ref mask does not exist: {full_mask_path}")
        mask = self._read_ref_mask(full_mask_path)
        if mask.shape[:2] != image_rgb.shape[:2]:
            import cv2

            mask = cv2.resize(
                mask,
                (int(image_rgb.shape[1]), int(image_rgb.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        return image_rgb, mask > 0

    def _build_subject_ref_canvas_from_cutout(self, cutout_rgb, mask_np, row):
        import cv2

        height = int(self.height)
        width = int(self.width)
        bg = self._normal_to_uint8_canvas_value(self.subject_ref_background)
        crop = np.asarray(cutout_rgb, dtype=np.float32)
        if crop.ndim != 3 or crop.shape[-1] != 3:
            return None, {"status": "bad_cutout_shape"}
        if mask_np is not None:
            crop_mask = np.asarray(mask_np, dtype=bool)
            if crop_mask.shape[:2] != crop.shape[:2]:
                crop_mask = (
                    cv2.resize(
                        crop_mask.astype(np.uint8),
                        (int(crop.shape[1]), int(crop.shape[0])),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )
        else:
            crop_mask = np.ones(crop.shape[:2], dtype=bool)

        slot_size = max(
            1,
            int(round(min(height, width) * self.subject_ref_canvas_slot_ratio)),
        )
        slot_size = min(slot_size, height, width)
        resized = cv2.resize(
            np.clip(crop, 0.0, 255.0).astype(np.uint8),
            (slot_size, slot_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        resized_mask = (
            cv2.resize(
                crop_mask.astype(np.uint8),
                (slot_size, slot_size),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

        canvas = np.empty((height, width, 3), dtype=np.float32)
        canvas[:, :] = bg.reshape(1, 1, 3)
        y0 = height - slot_size
        x0 = width - slot_size
        canvas[y0:height, x0:width] = resized
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()
        tensor = tensor.mul(2.0 / 255.0).sub(1.0).clamp(-1.0, 1.0)
        canvas_info = {
            "status": "ok",
            "source": "saved_cutout",
            "image_path": str(row.get("image_path") or ""),
            "mask_path": str(row.get("mask_path") or ""),
            "slot_xyxy": [int(x0), int(y0), int(width), int(height)],
            "slot_size": int(slot_size),
            "mask_pixel_count": int(crop_mask.sum()),
            "slot_mask_pixel_count": int(resized_mask.sum()),
            "foreground_enhance": "precomputed",
            "background": self.subject_ref_background,
        }
        return tensor, canvas_info

    def _sample_subject_ref_canvases(self, data, rng):
        info = data.get("info") or {}
        empty = {
            "subject_ref_images": torch.zeros(0, 3, int(self.height), int(self.width), dtype=torch.float32),
            "subject_ref_info": {
                "enabled": False,
                "status": "disabled" if self.subject_num_refs_max <= 0 else "missing_refs",
                "actual_ref_count": 0,
                "selected_ref_indices": [],
                "selected_frame_indices": [],
                "num_available_refs": 0,
            },
        }
        if self.subject_num_refs_max <= 0:
            return empty
        ref_dir = self._subject_ref_dir(info)
        refs = self._load_subject_refs(ref_dir)
        if refs is None:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty
        candidates = refs["candidates"]
        if not candidates:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty

        pairwise = refs.get("pairwise")
        if pairwise is None:
            selected = [
                int(v)
                for v in rng.choice(
                    len(candidates),
                    size=min(self.subject_num_refs_max, len(candidates)),
                    replace=False,
                )
            ]
        else:
            selected = _sample_reference_indices(
                len(candidates),
                pairwise,
                num_refs=min(self.subject_num_refs_max, len(candidates)),
                rng=rng,
                dissimilar_top_k=self.subject_dissimilar_top_k,
                max_similarity=self.subject_max_similarity,
            )
        if self.subject_shuffle_refs and len(selected) > 1:
            selected = [int(v) for v in rng.permutation(selected).tolist()]
        selected_rows = [candidates[int(idx)] for idx in selected]
        frame_ids = [int(row["frame_idx"]) for row in selected_rows]
        if not frame_ids:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty

        canvases = []
        ref_records = []
        for row in selected_rows:
            frame_idx = int(row["frame_idx"])
            cutout_rgb, cutout_mask = self._load_saved_ref_cutout_rgb(
                ref_dir,
                row,
            )
            canvas, canvas_info = self._build_subject_ref_canvas_from_cutout(
                cutout_rgb,
                cutout_mask,
                row,
            )
            ref_record = {
                "ref_id": _safe_int(row.get("ref_id"), len(ref_records)),
                "frame_idx": int(frame_idx),
                "track_id": _safe_int(row.get("track_id"), -1),
                "image_path": str(row.get("image_path") or ""),
                "mask_path": str(row.get("mask_path") or ""),
                "candidate_crop_xyxy": [int(v) for v in (row.get("crop_xyxy") or [])],
                "candidate_bbox_xyxy": [float(v) for v in (row.get("bbox_xyxy") or [])],
                "mask_status": "saved_cutout_mask",
                **(canvas_info or {}),
            }
            ref_records.append(ref_record)
            if canvas is not None:
                canvases.append(canvas)
        if not canvases:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty
        ref_images = torch.stack(canvases, dim=0).to(dtype=torch.float32)
        return {
            "subject_ref_images": ref_images,
            "subject_ref_info": {
                "enabled": True,
                "status": "ok",
                "ref_dir": ref_dir,
                "actual_ref_count": int(ref_images.shape[0]),
                "num_available_refs": int(len(candidates)),
                "selected_ref_indices": [int(v) for v in selected],
                "selected_frame_indices": [int(v) for v in frame_ids],
                "selected_track_ids": [int(v) for v in refs["selected_track_ids"]],
                "canvas_slot_ratio": float(self.subject_ref_canvas_slot_ratio),
                "background": self.subject_ref_background,
                "ref_records": ref_records,
            },
        }
