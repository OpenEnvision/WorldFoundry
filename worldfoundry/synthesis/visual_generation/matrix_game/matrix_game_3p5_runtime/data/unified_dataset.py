import glob
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Bumping this invalidates every on-disk dataset cache. Bump whenever the
# payload schema or any logic that affects the scan output changes in a way
# that would make older caches return wrong data.
#
# v2: legacy mosaic latent scan discovered ``depth/video.zip`` co-located
#     with ``pose/`` and ``intrinsics/`` and records it in ``info["depth_path"]``.
#     Older caches stored ``depth_path=None`` for the same scenes, which would
#     silently disable frustum retrieval and Mosaic conditioning.
# v3: video-flow datasets add a brand-new ``info`` payload shape
#     (``video_path`` / ``frame_count`` instead of ``latent_path`` /
#     ``latent``); the cache fingerprint now also folds in
#     ``height``/``width``/``memory_latent_cache_version``/``dataset_kind``
#     so a video-flow
#     scan never collides with a latent-flow scan in the same cache dir.
# v4: Subset selection and path shuffling use private deterministic RNGs
#     instead of mutating global random state.
# v5: Inference datasets scan only their selected split instead of
#     scanning the full corpus and relying on runtime ``inference_paths``
#     selection. Manifests also carry explicit split/sample keys for
#     future save/load driven eval runs.
# v6: The native inference manifest uses inference-only names and no longer
#     serializes held-out-validation fields from the upstream trainer.
DATASET_CACHE_VERSION = 6
DEFAULT_DATASET_PATH_SHUFFLE_SEED = 42


def _derive_seed(*parts) -> int:
    """Stable, well-distributed 32-bit seed derived from a tuple of parts.

    Uses BLAKE2b-32 over a ``|``-joined UTF-8 string of the parts, so any
    callable site that needs a seed for a specific *role* (per-rank, per-
    epoch, per-sample, ...) can compose a unique int without worrying
    about hash collisions or pollutting the global ``random`` state.

    Examples:
        >>> _derive_seed(0, "split")
        >>> _derive_seed(0, "inference_sample", 4, 17)
    """
    blob = b"|".join(str(p).encode() for p in parts)
    h = hashlib.blake2b(blob, digest_size=4)
    return int.from_bytes(h.digest(), "little")


def _sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _video_frames_uint8_to_normalized_tensor(
    frames_np,
    *,
    height,
    width,
    normalize=True,
):
    """Convert decord ``uint8`` NHWC frames to ``(C,T,H,W)``.

    Cropping happens before float conversion so unused pixels never pay the
    uint8->float32 conversion cost. When ``normalize`` is false, the returned
    tensor stays uint8 unless resize is required; resized tensors remain float
    in pixel range [0, 255] and are normalized on GPU before VAE encode.
    """
    import torch.nn.functional as F

    frames_np = np.asarray(frames_np)
    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(f"Expected video frames as uint8 NHWC array with 3 channels, got shape {frames_np.shape}.")

    height = int(height)
    width = int(width)
    src_h, src_w = int(frames_np.shape[1]), int(frames_np.shape[2])
    crop_h = min(src_h, height)
    crop_w = min(src_w, width)
    top = max((src_h - crop_h) // 2, 0)
    left = max((src_w - crop_w) // 2, 0)
    frames_np = frames_np[:, top : top + crop_h, left : left + crop_w, :]

    frames = torch.as_tensor(frames_np).permute(0, 3, 1, 2)
    if normalize:
        frames = frames.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        frames.mul_(2.0 / 255.0).sub_(1.0)
    else:
        frames = frames.contiguous(memory_format=torch.contiguous_format)

    if frames.shape[-2:] != (height, width):
        if not normalize:
            frames = frames.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        frames = F.interpolate(
            frames,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    frames = frames.permute(1, 0, 2, 3)
    return frames


def _split_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, str):
        return path_or_paths.split(",") if "," in path_or_paths else [path_or_paths]
    return list(path_or_paths)


# The DiT's 3D RoPE temporal frequency table is precomputed with a fixed
# 1024-entry timeline (``precompute_freqs_cis_3d(head_dim)`` in
# ``diffsynth/models/wan_video_dit.py``). Flows that index RoPE with the
# REAL latent timeline used by original-index history retrieval can reach
# indices up to about
# ``frame_count // 4 + latent_window_size``; past the table length the
# embedding lookup raises a CUDA indexing error mid-run, which is near
# impossible to trace back to one over-long scene. Such scenes are
# filtered at scan time with the dedicated ``rope_time_overflow`` label.
ROPE_FREQS_TABLE_END = 1024


def _rope_time_overflow(frame_count, latent_window_size, table_end=ROPE_FREQS_TABLE_END):
    """Return the (conservative) max real-timeline RoPE index for a scene
    when it would overflow the precomputed freqs table, else ``None``.

    Conservative bound: the largest generating start (``frame_count`` minus
    one noisy window) contributes ``frame_count // 4`` and the noisy window
    itself adds up to ``latent_window_size`` latent steps.
    """
    max_rope_time = int(frame_count) // 4 + int(latent_window_size)
    if max_rope_time >= int(table_end):
        return max_rope_time
    return None


def _normalize_vipe_prompt_type(value):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("vipe_prompt_type must be 1 or 2.") from exc
    if value not in (1, 2):
        raise ValueError(f"vipe_prompt_type must be 1 or 2, got {value!r}.")
    return value


class _MosaicDatasetBase(torch.utils.data.Dataset):
    VAE_HW_SCALING = 16
    VAE_T_SCALING = 4

    def __init__(
        self,
        base_path=None,
        camera_params_path=None,
        depth_path=None,
        latent_window_size=None,
        inference_assign_yaml=None,
        inference_ratio=0.05,
        inference_paths=None,
        filter_yaml=None,
        include_yaml=None,
        random_start_latent_prob=0.0,
        seed=None,
        rank=0,
        max_data_items=None,
        max_scan_items=None,
        frustum_handler_cls=None,
        dataset_cache_dir=None,
        max_frames_per_scene=None,
        force_override_extrinsic=None,
        mosaic_intrinsics_mode="per_frame",
        mosaic_query_reference_frame=2,
        mosaic_view_change_prope=False,
        vipe_prompt_type=1,
        dataset_index_path=None,
        min_frame_count=None,
        dataset_compact_mode=False,
        dataset_path_shuffle_seed=DEFAULT_DATASET_PATH_SHUFFLE_SEED,
        **kwargs,
    ):
        if base_path is None:
            raise ValueError("base_path is required for _MosaicDatasetBase.")
        if camera_params_path is None:
            camera_params_path = kwargs.get("camera_params_path")
        if camera_params_path is None:
            raise ValueError("camera_params_path is required for _MosaicDatasetBase.")
        if depth_path is None:
            depth_path = kwargs.get("depth_path")
        if depth_path is None:
            raise ValueError("depth_path is required for _MosaicDatasetBase.")
        if latent_window_size is None:
            latent_window_size = kwargs.get("latent_window_size")
        if latent_window_size is None:
            raise ValueError("latent_window_size is required for _MosaicDatasetBase.")
        if force_override_extrinsic is None:
            force_override_extrinsic = kwargs.get("force_override_extrinsic")

        self.base_path = _split_path_list(base_path)
        self.camera_params_path = _split_path_list(camera_params_path)
        self.depth_path = _split_path_list(depth_path)
        self.force_override_extrinsic = str(force_override_extrinsic) if force_override_extrinsic else None
        self._force_override_extrinsic_npz_paths = self._discover_force_override_extrinsic_npz_paths(
            self.force_override_extrinsic
        )
        self.metadata_path = None
        self.repeat = 1
        self.data_file_keys = tuple()
        self.main_data_operator = lambda x: x
        self.special_operator_map = {}
        self.max_data_items = max_data_items
        self.load_from_cache = False
        self.data = []
        self.cached_data = []

        self.latent_window_size = int(latent_window_size)
        self.mosaic_intrinsics_mode = self._normalize_mosaic_intrinsics_mode(mosaic_intrinsics_mode)
        self.mosaic_query_reference_frame = self._normalize_mosaic_query_reference_frame(mosaic_query_reference_frame)
        self.mosaic_view_change_prope = bool(mosaic_view_change_prope)
        self.vipe_prompt_type = _normalize_vipe_prompt_type(vipe_prompt_type)
        self.random_start_latent_prob = float(random_start_latent_prob)
        self.seed = seed
        self.rank = int(rank)
        # Optional inference cap: restrict each scene to its first N frames. ``None``
        # (or any non-positive value) disables the cap. Applies to:
        #   * scan: ``too_short`` is evaluated against the capped count and
        #     ``info["frame_count"]`` is stored already capped.
        #   * runtime: ``__getitem__`` derives ``G`` / latent windows
        #     from the capped ``info[...]`` values, so generating /
        #     memory frames are guaranteed to fall inside [0, N).
        # Use to overfit on a small temporal slice, or to run end-to-end
        # smoke tests on long clips without paying for the full footage.
        if max_frames_per_scene is None or int(max_frames_per_scene) <= 0:
            self.max_frames_per_scene = None
        else:
            self.max_frames_per_scene = int(max_frames_per_scene)
        self._frustum_handler_cls = frustum_handler_cls
        self.handler = None

        # Cache / manifest plumbing. The cache lives under ``dataset_cache_dir``
        # (e.g. ``./.cache/dataset_info``) and is a JSON snapshot of the full
        # scan output. ``save_manifest`` writes the exact same payload to an
        # arbitrary path, so the log directory's copy and the on-disk cache
        # stay byte-identical and are always in sync.
        self.dataset_cache_dir = dataset_cache_dir
        self._filter_yaml_path = filter_yaml
        self._include_yaml_path = include_yaml
        self._inference_assign_yaml_path = inference_assign_yaml
        self._inference_ratio = float(inference_ratio) if inference_ratio is not None else None
        self._max_scan_items = max_scan_items
        self.dataset_index_path = str(dataset_index_path) if dataset_index_path else None
        self.dataset_compact_mode = bool(dataset_compact_mode)
        if dataset_path_shuffle_seed is None:
            dataset_path_shuffle_seed = DEFAULT_DATASET_PATH_SHUFFLE_SEED
        self.dataset_path_shuffle_seed = int(dataset_path_shuffle_seed)
        if min_frame_count is None or int(min_frame_count) <= 0:
            self.min_frame_count = None
        else:
            self.min_frame_count = int(min_frame_count)
        self._last_dataset_cache_payload = None
        self._last_dataset_cache_path = None
        self._last_dataset_cache_hit = False

        rawpath = self._collect_latent_paths(filter_yaml)
        rawpath = self._apply_include_yaml(rawpath, include_yaml)
        self.inference_paths = []
        if inference_paths is not None:
            self.inference_paths = list(dict.fromkeys(str(path) for path in inference_paths))
            self.rawpath = self._filter_raw_paths_by_keys(rawpath, self.inference_paths)
            inference_assigns = []
        else:
            self.rawpath, inference_assigns = self._select_inference_paths(
                rawpath,
                inference_assign_yaml=inference_assign_yaml,
                inference_ratio=inference_ratio,
            )

        self.reason = {}
        scan_rawpath = self.rawpath[:max_scan_items] if max_scan_items is not None else self.rawpath
        self.dataset_info = self._build_or_load_dataset_info(
            scan_rawpath,
            inference_assigns,
        )
        self.path = sorted(self.dataset_info.keys())
        self.inference_paths = list(dict.fromkeys(path for path in self.inference_paths if path in self.dataset_info))

        # A private seed keeps inference ordering deterministic without
        # changing Python's global random state.
        shuffle_rng = random.Random(self.dataset_path_shuffle_seed)
        shuffle_rng.shuffle(self.path)

        # Runtime inference sampling goes through ``_select_path`` which
        # prefers ``self.inference_paths`` when it is non-empty. Keep that
        # actual sampling list aligned with the fixed inference ``self.path``
        # order; otherwise the fixed ``self.path`` shuffle is bypassed.
        self.inference_paths = list(self.path)

        if self.rank is not None:
            self._rank = int(self.rank)
        else:
            self._rank = 0
        self._inference_sample_offset = 0

    @staticmethod
    def _load_yaml_tokens(yaml_path):
        if yaml_path is None:
            return []
        import yaml

        with open(yaml_path, "r") as f:
            payload = yaml.safe_load(f) or []
        if isinstance(payload, dict):
            for key in ("include", "include_hashes", "hashes", "items"):
                if key in payload:
                    payload = payload[key] or []
                    break
            else:
                payload = list(payload.values())
        if isinstance(payload, (str, int, float)):
            payload = [payload]
        return [str(item).strip() for item in payload if str(item).strip()]

    def _apply_include_yaml(self, rawpath, include_yaml):
        include_tokens = self._load_yaml_tokens(include_yaml)
        if not include_tokens:
            return rawpath
        return [path for path in rawpath if any(include_token in path for include_token in include_tokens)]

    @staticmethod
    def _raw_path_to_dataset_key(path):
        clean_name = os.path.basename(path).split(".")[0]
        return f"{os.path.dirname(path)}/{clean_name}"

    def _filter_raw_paths_by_keys(self, rawpath, keys):
        key_set = set(keys or [])
        if not key_set:
            return []
        return [path for path in rawpath if self._raw_path_to_dataset_key(path) in key_set]

    def _select_inference_paths(self, rawpath, inference_assign_yaml=None, inference_ratio=0.05):
        if inference_assign_yaml is not None:
            import yaml

            with open(inference_assign_yaml, "r") as f:
                inference_assigns = yaml.safe_load(f) or []
            matched = []
            for path in rawpath:
                if any(assign in path for assign in inference_assigns):
                    matched.append(path)
                    self.inference_paths.append(self._raw_path_to_dataset_key(path))
            return matched, inference_assigns

        inference_assigns = []
        shuffled = list(rawpath)
        # Stable per-seed subset selection for inference sampling.
        split_rng = random.Random(_derive_seed(self.seed, "split"))
        split_rng.shuffle(shuffled)
        split = int(len(shuffled) * inference_ratio)
        matched = shuffled[:split]
        self.inference_paths.extend(self._raw_path_to_dataset_key(path) for path in matched)
        return matched, inference_assigns

    # ------------------------------------------------------------------
    # Dataset-info cache & manifest
    # ------------------------------------------------------------------

    def _dataset_cache_inputs_dict(self):
        """Serializable snapshot of constructor inputs for this scan.

        Subclasses can override to add backend-specific knobs (e.g. prompt
        roots or sqlite index paths). The dict ends up inside both the cache
        file and the log-dir manifest so the
        operator can inspect what produced a given inference run.
        """
        inputs = {
            "base_path": list(self.base_path),
            "camera_params_path": list(self.camera_params_path),
            "depth_path": list(self.depth_path),
            "filter_yaml": self._filter_yaml_path,
            "include_yaml": self._include_yaml_path,
            "inference_assign_yaml": self._inference_assign_yaml_path,
            "inference_ratio": self._inference_ratio,
            "seed": self.seed,
            "latent_window_size": int(self.latent_window_size),
            "max_scan_items": self._max_scan_items,
            "dataset_index_path": self.dataset_index_path,
            "min_frame_count": self.min_frame_count,
            "dataset_compact_mode": bool(self.dataset_compact_mode),
            # Caps each scene's recorded length, so different cap values
            # need different manifest files (otherwise switching the cap
            # would silently reuse a stale ``info["frame_count"]`` /
            # ``info["latent"]``).
            "max_frames_per_scene": self.max_frames_per_scene,
            "mosaic_intrinsics_mode": self.mosaic_intrinsics_mode,
            "mosaic_query_reference_frame": int(self.mosaic_query_reference_frame),
            "mosaic_view_change_prope": bool(self.mosaic_view_change_prope),
            "vipe_prompt_type": int(self.vipe_prompt_type),
        }
        inputs["dataset_path_shuffle_seed"] = int(self.dataset_path_shuffle_seed)
        if self.force_override_extrinsic:
            inputs["force_override_extrinsic"] = self.force_override_extrinsic
        return inputs

    @staticmethod
    def _discover_force_override_extrinsic_npz_paths(path):
        if not path or not os.path.isdir(path):
            return None
        npz_paths = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not npz_paths:
            raise ValueError(f"force_override_extrinsic directory has no .npz files: {path}")
        return npz_paths

    @staticmethod
    def _canonical_view_change_like_latents(latents):
        if latents.ndim == 5:
            t, h, w = latents.shape[2], latents.shape[-2], latents.shape[-1]
        elif latents.ndim == 4:
            t, h, w = latents.shape[1], latents.shape[-2], latents.shape[-1]
        else:
            raise ValueError(f"Expected latent with 4 or 5 dims, got shape {tuple(latents.shape)}.")
        vc = torch.zeros(int(t), int(h), int(w), 3, dtype=latents.dtype, device=latents.device)
        vc[..., 0] = 1.0
        return vc

    @staticmethod
    def _normalize_mosaic_intrinsics_mode(mode):
        mode = str(mode or "per_frame").strip().lower()
        aliases = {
            "per-frame": "per_frame",
            "perframe": "per_frame",
            "frame": "per_frame",
            "mean": "episode_mean",
            "episode-mean": "episode_mean",
            "episode_kmean": "episode_mean",
            "kmean": "episode_mean",
            "first": "first_frame",
            "first-frame": "first_frame",
            "firstframe": "first_frame",
            "frame0": "first_frame",
            "first_frame_only": "first_frame",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"per_frame", "episode_mean", "first_frame"}:
            raise ValueError(
                f"mosaic_intrinsics_mode must be 'per_frame', 'episode_mean', or 'first_frame', got {mode!r}."
            )
        return mode

    @staticmethod
    def _normalize_mosaic_query_reference_frame(frame):
        frame = int(frame)
        if frame < 1 or frame > 4:
            raise ValueError(f"mosaic_query_reference_frame must be in [1, 4] (1-based), got {frame}.")
        return frame

    def _intrinsics_temporal_mean_enabled(self):
        return self.mosaic_intrinsics_mode == "episode_mean"

    def _dataset_cache_fingerprint(self, rawpath, inference_assigns):
        """Stable hex fingerprint of everything that affects the scan.

        Includes the *exact* list of latent files we're about to scan
        (so adding / removing a sample invalidates the cache) plus the
        camera / depth / prompt roots, the val assignment, the latent
        window threshold (used to drop too-short scenes) and the class
        name. Two datasets with the same fingerprint are guaranteed to
        produce byte-identical ``dataset_info`` from a re-scan, so the
        cache can be safely reused.
        """
        payload = {
            "version": DATASET_CACHE_VERSION,
            "class": type(self).__name__,
            "rawpath_sha": _sha1_text("\n".join(sorted(rawpath))),
            "rawpath_count": len(rawpath),
            "inference_assigns": sorted(list(inference_assigns or [])),
            "inputs": self._dataset_cache_inputs_dict(),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]

    def _build_or_load_dataset_info(self, rawpath, inference_assigns):
        """Cache-wrapped scan.

        On a cache hit the scan is skipped entirely and ``self.reason`` /
        ``self.inference_paths`` are repopulated from the cached payload so
        the rest of ``__init__`` is oblivious to whether we scanned or
        loaded.
        """
        fingerprint = self._dataset_cache_fingerprint(rawpath, inference_assigns)
        cache_dir = self.dataset_cache_dir
        cache_path = os.path.join(cache_dir, f"{fingerprint}.json") if cache_dir else None

        # Snapshot what inference_paths held *before* the scan: the cache
        # records only the delta the scan itself appended (see below).
        pre_inference_keys_len = len(self.inference_paths)

        if cache_path and os.path.exists(cache_path):
            payload = self._dataset_cache_try_load(cache_path, fingerprint)
            if payload is not None:
                # _build_dataset_info would have populated self.reason
                # and appended to self.inference_paths; do the same here.
                self.reason.update(payload.get("reason", {}))
                self.inference_paths.extend(payload.get("inference_keys_appended", []))
                self._last_dataset_cache_payload = payload
                self._last_dataset_cache_path = cache_path
                self._last_dataset_cache_hit = True
                if self.rank == 0:
                    print(
                        f"[{type(self).__name__}] cache HIT  fp={fingerprint}  "
                        f"kept={len(payload.get('dataset_info', {}))}/"
                        f"{payload.get('rawpath_count', len(rawpath))}  "
                        f"path={cache_path}"
                    )
                return payload["dataset_info"]

        dataset_info = self._build_dataset_info(rawpath, inference_assigns)
        inference_keys_appended = list(self.inference_paths[pre_inference_keys_len:])
        payload = self._dataset_cache_make_payload(
            dataset_info=dataset_info,
            fingerprint=fingerprint,
            inference_keys_appended=inference_keys_appended,
            rawpath=rawpath,
            inference_assigns=inference_assigns,
        )
        self._last_dataset_cache_payload = payload
        self._last_dataset_cache_path = cache_path
        self._last_dataset_cache_hit = False

        if cache_path and self.rank == 0:
            try:
                self._dataset_cache_write(cache_path, payload)
                print(
                    f"[{type(self).__name__}] cache MISS fp={fingerprint}  "
                    f"kept={len(dataset_info)}/{len(rawpath)}  "
                    f"saved={cache_path}"
                )
            except OSError as exc:
                # Cache failure must never break inference; just warn.
                print(f"[{type(self).__name__}] WARNING: cache write to {cache_path} failed: {exc}")
        return dataset_info

    def _dataset_cache_make_payload(
        self,
        *,
        dataset_info,
        fingerprint,
        inference_keys_appended,
        rawpath,
        inference_assigns,
    ):
        sample_keys = sorted(dataset_info.keys())
        inference_keys = sorted(dict.fromkeys(getattr(self, "_split_inference_keys", None) or self.inference_paths))
        return {
            "version": DATASET_CACHE_VERSION,
            "fingerprint": fingerprint,
            "class": type(self).__name__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "is_inference_dataset": True,
            "split_role": "inference",
            "sample_keys": sample_keys,
            "inference_keys": inference_keys,
            "split_seed": getattr(self, "_split_seed", None),
            "split_source_count": getattr(self, "_split_source_count", None),
            "split": {
                "role": "inference",
                "sample_keys": sample_keys,
                "inference_keys": inference_keys,
                "split_seed": getattr(self, "_split_seed", None),
                "split_source_count": getattr(self, "_split_source_count", None),
            },
            "inputs": self._dataset_cache_inputs_dict(),
            "inference_assigns": sorted(list(inference_assigns or [])),
            "rawpath_count": len(rawpath),
            "kept_count": len(dataset_info),
            "dataset_info": dataset_info,
            "inference_keys_appended": list(inference_keys_appended),
            "reason": dict(self.reason),
            "kept_paths": sorted(info.get("latent_path", "") for info in dataset_info.values()),
            "kept_clean_names": sorted(info.get("clean_name", "") for info in dataset_info.values()),
        }

    def _dataset_cache_try_load(self, cache_path, expected_fingerprint):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != DATASET_CACHE_VERSION:
            return None
        if payload.get("fingerprint") != expected_fingerprint:
            return None
        if not isinstance(payload.get("dataset_info"), dict):
            return None
        return payload

    @staticmethod
    def _dataset_cache_write(cache_path, payload):
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Atomic write so concurrent ranks / repeated runs never observe
        # a half-written JSON.
        tmp_path = cache_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)

    def save_manifest(self, path):
        """Write the dataset manifest (== cache payload) to ``path``.

        Same content as the on-disk cache so an operator can grep through
        the log directory to see *exactly* which scenes were kept /
        excluded for any given run, without having to rummage in the
        ``.cache`` directory. Returns ``path`` on success, ``None`` if
        the dataset never produced a cache payload (e.g. when
        ``dataset_cache_dir`` was unset *and* nothing was scanned).
        """
        payload = self._last_dataset_cache_payload
        if payload is None:
            return None
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return path

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def get_inference_paths(self):
        return self.inference_paths

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        return len(self.path)

    # ------------------------------------------------------------------
    # ``max_frames_per_scene`` cap helpers. Subclasses call these from
    # their ``_build_dataset_info`` so the manifest stores
    # ALREADY-CAPPED frame / latent counts -- the runtime
    # ``__getitem__`` then doesn't need any extra cap logic.
    # ------------------------------------------------------------------

    def _cap_frame_count(self, actual: int) -> int:
        """Return ``min(actual, max_frames_per_scene)`` (or ``actual``
        when no cap is configured). Used by video-flow datasets whose
        ``info["frame_count"]`` is the canonical scene length.
        """
        if self.max_frames_per_scene is None:
            return int(actual)
        return min(int(actual), int(self.max_frames_per_scene))

    # ------------------------------------------------------------------
    # Per-sample reproducible RNG. ``__getitem__`` stashes one of these
    # on ``self`` so any helper down the call chain can pull a draw
    # without having to thread an rng arg through every layer.
    # ------------------------------------------------------------------

    def _make_sample_rng(self, index):
        """Build & stash a per-sample numpy ``Generator``.

        The seed is derived from ``(seed, rank, index)`` so the same
        inference request is reproducible while ranks remain decorrelated.

        The stash also avoids worker-init races: PyTorch DataLoader
        spawns workers with their own python/numpy random states, but
        we deliberately want our randomness to be *uncoupled* from
        those (so two runs with the same args and seed but
        different DataLoader worker counts still produce the same
        sample stream).
        """
        s = _derive_seed(
            self.seed,
            "inference_sample",
            self._rank,
            int(index),
        )
        rng = np.random.default_rng(s)
        self._sample_rng = rng
        return rng

    def set_inference_sample_offset(self, offset):
        self._inference_sample_offset = int(offset)

    def _select_path(self, index):
        if len(self.path) == 0:
            raise RuntimeError(
                "_MosaicDatasetBase is empty. Check base_path, camera_params_path, "
                f"depth_path, and filter reasons: {self.reason}"
            )
        if hasattr(self, "current_video_idx"):
            return self.path[self.current_video_idx]

        candidates = self.inference_paths or self.path
        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
            world_size = int(torch.distributed.get_world_size())
        resolved = self._inference_sample_offset + world_size * int(index) + rank
        return candidates[resolved % len(candidates)]

    def normalize_and_scale_intrinsics(self, intrinsics, H_img, W_img, temporal_mean=True):
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError(f"intrinsics shape must be (N, 3, 3), but got {intrinsics.shape}")

        processed_intrinsics = intrinsics.astype(np.float32, copy=True)
        if getattr(self, "mosaic_intrinsics_mode", "per_frame") == "first_frame":
            processed_intrinsics = np.repeat(
                processed_intrinsics[:1],
                repeats=processed_intrinsics.shape[0],
                axis=0,
            )
        cx = processed_intrinsics[:, 0, 2]
        cy = processed_intrinsics[:, 1, 2]
        target_cx = float(W_img) * 0.5
        target_cy = float(H_img) * 0.5

        valid_mask = (np.abs(cx) > 1e-6) & (np.abs(cy) > 1e-6)
        if np.any(valid_mask):
            fx_norm = processed_intrinsics[valid_mask, 0, 0] / cx[valid_mask]
            fy_norm = processed_intrinsics[valid_mask, 1, 1] / cy[valid_mask]
            skew_norm = processed_intrinsics[valid_mask, 0, 1] / cx[valid_mask]
            processed_intrinsics[valid_mask, 0, 0] = fx_norm * target_cx
            processed_intrinsics[valid_mask, 1, 1] = fy_norm * target_cy
            processed_intrinsics[valid_mask, 0, 1] = skew_norm * target_cx
            processed_intrinsics[valid_mask, 0, 2] = target_cx
            processed_intrinsics[valid_mask, 1, 2] = target_cy

        if temporal_mean:
            mean_intrinsics = processed_intrinsics.mean(axis=0, keepdims=True)
            processed_intrinsics = np.repeat(mean_intrinsics, repeats=processed_intrinsics.shape[0], axis=0)
        return processed_intrinsics

    def _get_extrinsic_path(self, info):
        fixed_path = info.get("force_override_extrinsic_path")
        if fixed_path:
            return fixed_path
        override_path = getattr(self, "force_override_extrinsic", None)
        if not override_path:
            return info["extrinsic_path"]
        override_npz_paths = getattr(self, "_force_override_extrinsic_npz_paths", None)
        if override_npz_paths:
            sample_rng = getattr(self, "_sample_rng", None)
            selected_idx = int(sample_rng.integers(0, len(override_npz_paths))) if sample_rng is not None else 0
            selected_path = override_npz_paths[selected_idx]
            info["force_override_extrinsic_path"] = selected_path
            return selected_path
        if os.path.isdir(override_path):
            raise ValueError(f"force_override_extrinsic directory has no .npz files: {override_path}")
        if override_path:
            return override_path
        return info["extrinsic_path"]

    def _get_frustum_handler(self):
        if self.handler is not None:
            return self.handler
        handler_cls = self._frustum_handler_cls
        if handler_cls is None:
            from ..frustum.frustum_handler import FrustumHandler

            handler_cls = FrustumHandler
        # latent_stride=16 matches the WAN VAE this dataset feeds. Override
        # if you wire in a VAE with a different spatial downsample ratio.
        self.handler = handler_cls(np.eye(3), latent_stride=16)
        return self.handler


class _MosaicVideoSharedDataset(_MosaicDatasetBase):
    def __init__(
        self,
        *args,
        prompt="",
        prompt_path=None,
        vipe_prompt_type=1,
        vipe_prompt_segment_format=False,
        align_generating_to_prompt_segments=False,
        require_depth=False,
        allow_no_prompt=False,
        loose_prompt_match=False,
        mosaic_fuse_mode="fill_stop",
        candidates_per_query_group=20,
        mosaic_selection_mode="projection_iou",
        mosaic_candidate_nms_mode=None,
        mosaic_candidate_nms_projection_iou_threshold=0.7,
        mosaic_candidate_nms_min_temporal_gap=0,
        mosaic_candidate_nms_pose_distance_threshold=0.0,
        mosaic_candidate_nms_pool_multiplier=2.0,
        mosaic_coverage_grid_downsample=4,
        mosaic_coverage_pool_stride=2,
        **kwargs,
    ):
        self.default_prompt = prompt
        # Lazy-built scene_hash -> (extrinsic_path, intrinsic_path) lookup.
        # DA3 strict mode populates this directly from sqlite index records.
        self._vipe_camera_index = None
        # Companion index: scene_hash -> path to the indexed depth npz.
        self._vipe_depth_zip_index = None
        # Optional prompt source. ``prompt_path`` mirrors ``camera_params_path``:
        # one or more roots holding ``<group>/<scene>/video.json`` (the DL3DV
        # caption layout). When empty, every sample falls back to
        # ``default_prompt`` for backwards compatibility.
        self._vipe_prompt_roots = _split_path_list(prompt_path) if prompt_path else []
        self._vipe_prompt_index = None
        self._vipe_prompt_path_index = None
        self._vipe_prompt_segment_cache = {}
        self._vipe_prompt_payload_cache = {}
        self.vipe_prompt_type = _normalize_vipe_prompt_type(vipe_prompt_type)
        # Retained only as an informational attribute: prompt JSON forms
        # (list segment entries vs legacy per-frame dicts) are auto-detected
        # in _load_prompt_segments, and this flag NO LONGER implicitly
        # enables align_generating_to_prompt_segments -- the two switches
        # are orthogonal now.
        self.vipe_prompt_segment_format = bool(vipe_prompt_segment_format)
        self.align_generating_to_prompt_segments = bool(align_generating_to_prompt_segments)
        # Strict DA3 inference drops scenes missing depth during indexing.
        self.require_depth = bool(require_depth)
        # When True, scenes whose caption JSON is missing / unreadable /
        # has an empty ``detailed`` block are STILL kept at scan time and
        # silently fall back to ``default_prompt`` at __getitem__ time
        # (via ``_resolve_prompt_for_scene`` which already returns the
        # default for missing index entries). Useful for partially
        # captioned datasets where visual conditioning should still run
        # signal with an empty / generic prompt than drop the scene
        # entirely. Default ``False`` preserves the legacy behaviour
        # where missing-caption scenes get reason="no_prompt" and are excluded.
        self.allow_no_prompt = bool(allow_no_prompt)
        # When True, ``_lookup_prompt_path`` falls back from a strict
        # ``scene_hash == key`` lookup to a longest-prefix match: any
        # indexed key that is a prefix of ``scene_hash`` ending on a
        # token boundary (``_``, ``-``, ``.``) is accepted, with the
        # longest such key winning. The intended use case is scenes
        # whose hash carries an extra trailing suffix that the caption
        # JSON does not, e.g. scene
        # ``8d4iysZ0pOU_0003750_0005550_f301-376`` matched against
        # prompt ``8d4iysZ0pOU_0003750_0005550.json``. Token-boundary
        # anchoring is what keeps unrelated scenes that happen to share
        # a leading character run from colliding ("foo" never matches
        # "foobar"). Default ``False`` preserves strict equality.
        self.loose_prompt_match = bool(loose_prompt_match)
        self.mosaic_fuse_mode = str(mosaic_fuse_mode or "fill_stop")
        if self.mosaic_fuse_mode == "random":
            raise ValueError("Matrix-Game inference requires a deterministic fuse mode")
        # Candidate budget per query group passed to ``select_candidates``.
        self.candidates_per_query_group = max(1, int(candidates_per_query_group))
        self.mosaic_selection_mode = str(mosaic_selection_mode or "projection_iou")
        self.mosaic_candidate_nms_mode = (
            None if mosaic_candidate_nms_mode in (None, "", "none", "None") else str(mosaic_candidate_nms_mode)
        )
        self.mosaic_candidate_nms_projection_iou_threshold = float(mosaic_candidate_nms_projection_iou_threshold)
        self.mosaic_candidate_nms_min_temporal_gap = int(mosaic_candidate_nms_min_temporal_gap)
        self.mosaic_candidate_nms_pose_distance_threshold = float(mosaic_candidate_nms_pose_distance_threshold)
        self.mosaic_candidate_nms_pool_multiplier = float(mosaic_candidate_nms_pool_multiplier)
        self.mosaic_coverage_grid_downsample = max(1, int(mosaic_coverage_grid_downsample))
        self.mosaic_coverage_pool_stride = max(1, int(mosaic_coverage_pool_stride))
        super().__init__(*args, vipe_prompt_type=vipe_prompt_type, **kwargs)

    def _dataset_cache_inputs_dict(self):
        # Inherit base inputs and add VIPE-specific knobs so the cache /
        # manifest can distinguish runs that differ only in their prompt
        # / default-prompt / depth-requirement configuration.
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "prompt_path": list(self._vipe_prompt_roots),
                "default_prompt": self.default_prompt,
                "vipe_prompt_type": int(self.vipe_prompt_type),
                "vipe_prompt_segment_format": bool(self.vipe_prompt_segment_format),
                "align_generating_to_prompt_segments": bool(self.align_generating_to_prompt_segments),
                "require_depth": bool(self.require_depth),
                # Folded into the fingerprint because flipping this flag
                # changes which scenes survive the scan, which would
                # otherwise be invisible to the manifest cache.
                "allow_no_prompt": bool(self.allow_no_prompt),
                # Same reasoning as ``allow_no_prompt``: enabling loose
                # matching can rescue scenes that strict lookup would
                # have dropped (and bind different scenes to different
                # JSONs), so the cached manifest must distinguish runs
                # that flipped this flag.
                "loose_prompt_match": bool(self.loose_prompt_match),
                "mosaic_fuse_mode": self.mosaic_fuse_mode,
                "candidates_per_query_group": int(self.candidates_per_query_group),
                "mosaic_selection_mode": self.mosaic_selection_mode,
                "mosaic_candidate_nms_mode": self.mosaic_candidate_nms_mode,
                "mosaic_candidate_nms_projection_iou_threshold": float(
                    self.mosaic_candidate_nms_projection_iou_threshold
                ),
                "mosaic_candidate_nms_min_temporal_gap": int(self.mosaic_candidate_nms_min_temporal_gap),
                "mosaic_candidate_nms_pose_distance_threshold": float(
                    self.mosaic_candidate_nms_pose_distance_threshold
                ),
                "mosaic_candidate_nms_pool_multiplier": float(self.mosaic_candidate_nms_pool_multiplier),
                "mosaic_coverage_grid_downsample": int(self.mosaic_coverage_grid_downsample),
                "mosaic_coverage_pool_stride": int(self.mosaic_coverage_pool_stride),
            }
        )
        return base

    def _compose_vipe_prompt(self, start, dynamic):
        dynamic = str(dynamic or "").strip()
        if self.vipe_prompt_type == 1:
            return dynamic
        start = str(start or "").strip()
        return "".join(part for part in (start, dynamic) if part).strip()

    def _build_vipe_prompt_index(self):
        """BFS index of ``scene_hash -> *.json`` under each prompt root.

        Two layouts are supported:

        * **DL3DV / per-scene-dir** (legacy):
          ``<root>/<group>/<scene_hash>/video.json``. The scene's
          directory name is the key.
        * **citywalk / flat-file** (added 2026-05): each scene is a single
          JSON file directly under some grouping directory, e.g.
          ``<root>/citywalk/<scene_hash>.json``. The basename minus the
          ``.json`` extension is the key. Useful when an external pipeline
          drops one prompt JSON per scene without wrapping it in a folder.

        BFS rule: while descending with ``os.scandir``, if the current
        directory contains any ``.json`` file children we treat it as a
        scene-container leaf -- index those JSONs (per the two layouts
        above) and stop descending. Otherwise keep descending into
        subdirectories.
        """
        index = {}
        path_index = {}
        for prompt_root in self._vipe_prompt_roots:
            if not prompt_root or not os.path.isdir(prompt_root):
                continue
            stack = [prompt_root]
            while stack:
                current = stack.pop()
                try:
                    with os.scandir(current) as it:
                        entries = list(it)
                except OSError:
                    continue
                json_files = [e for e in entries if e.name.endswith(".json") and e.is_file(follow_symlinks=False)]
                if json_files:
                    dir_scene_name = os.path.basename(os.path.normpath(current))
                    for je in json_files:
                        if je.name == "video.json":
                            # Layout A: directory name == scene_hash.
                            path = os.path.join(current, je.name)
                            index.setdefault(dir_scene_name, path)
                            path_index.setdefault(dir_scene_name, []).append(path)
                        else:
                            # Layout B: file basename (sans .json) == scene_hash.
                            scene_name = je.name[: -len(".json")]
                            path = os.path.join(current, je.name)
                            index.setdefault(scene_name, path)
                            path_index.setdefault(scene_name, []).append(path)
                    # Either layout: this directory is a scene-container,
                    # don't descend further.
                    continue
                stack.extend(e.path for e in entries if e.is_dir(follow_symlinks=False))
        self._vipe_prompt_path_index = path_index
        return index

    # Token-boundary chars accepted as the cut point between an indexed
    # prompt key and the trailing suffix-only-on-the-scene-side. ``/``
    # is included for completeness even though scene hashes practically
    # never contain it. ``_PROMPT_PREFIX_BOUNDARY_CHARS`` is a class
    # attr so subclasses can extend it without touching the lookup.
    _PROMPT_PREFIX_BOUNDARY_CHARS = ("_", "-", ".", "/")

    def _lookup_prompt_path(self, scene_hash):
        """Resolve ``scene_hash`` to a prompt JSON path via the index.

        Strict mode (``loose_prompt_match=False``, the default) is a
        plain ``dict.get(scene_hash)`` against ``_vipe_prompt_index``.

        Loose mode (``loose_prompt_match=True``) additionally accepts
        any indexed key that is a *prefix* of ``scene_hash`` ending on
        a token boundary character (``_``, ``-``, ``.``, ``/``). The
        longest such key wins. Strict equality is always preferred.

        Implementation note: rather than scanning all indexed keys
        (which is O(N) per query and would dominate the scan on large
        datasets), we walk ``scene_hash`` from longest to shortest
        prefix and ask the dict directly. That's at most
        ``len(scene_hash)`` O(1) hash lookups (~50 chars in practice),
        so the total scan cost stays linear in the number of scenes.
        """
        if not self._vipe_prompt_index:
            return None
        path = self._vipe_prompt_index.get(scene_hash)
        if path is not None:
            return path
        if not self.loose_prompt_match:
            return None
        boundary = self._PROMPT_PREFIX_BOUNDARY_CHARS
        # Walk shrinking prefixes; require the cut character (the one
        # *immediately after* the candidate prefix) to be a token
        # boundary so e.g. "foo" never matches "foobar". We start at
        # ``len-1`` because ``len(scene_hash)`` would just repeat the
        # strict lookup above.
        for L in range(len(scene_hash) - 1, 0, -1):
            if scene_hash[L] not in boundary:
                continue
            cand = scene_hash[:L]
            path = self._vipe_prompt_index.get(cand)
            if path is not None:
                return path
        return None

    def _lookup_prompt_paths(self, scene_hash):
        """Resolve all prompt JSON candidates for ``scene_hash`` in root order."""
        if not self._vipe_prompt_index:
            return []
        if self._vipe_prompt_path_index is None:
            self._vipe_prompt_index = self._build_vipe_prompt_index()
        candidates = list((self._vipe_prompt_path_index or {}).get(scene_hash) or [])
        if candidates:
            return candidates
        if self.loose_prompt_match:
            boundary = self._PROMPT_PREFIX_BOUNDARY_CHARS
            for L in range(len(scene_hash) - 1, 0, -1):
                if scene_hash[L] not in boundary:
                    continue
                cand = scene_hash[:L]
                candidates = list((self._vipe_prompt_path_index or {}).get(cand) or [])
                if candidates:
                    return candidates
        path = self._lookup_prompt_path(scene_hash)
        return [] if path is None else [path]

    def _load_prompt_segments(self, json_path):
        """Parse prompt JSON into either segments or per-frame cache entries."""
        cached = self._vipe_prompt_payload_cache.get(json_path)
        if cached is not None:
            return cached
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        detailed = payload.get("detailed", {}) or {}
        if not detailed:
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        if isinstance(detailed, list):
            # List-form segment entries are auto-detected; the old
            # vipe_prompt_segment_format gate is gone (it used to silently
            # blank BOTH the segments and the prompt when off).
            segments = []
            for item in detailed:
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                if not isinstance(prompt, dict):
                    continue
                try:
                    start_frame = int(item["start"])
                except (KeyError, TypeError, ValueError):
                    continue
                start = (prompt.get("start") or "").strip()
                dyn = (prompt.get("dynamic") or "").strip()
                if start or dyn:
                    segments.append((start_frame, start, dyn))
            segments.sort(key=lambda item: item[0])
            result = {
                "format": "segments",
                "segments": segments,
                "source_format": "segment_entries_v1",
            }
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = segments
            return result
        if not isinstance(detailed, dict):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        try:
            keys = sorted(detailed.keys(), key=lambda k: int(k))
        except (TypeError, ValueError):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        if payload.get("prompt_cache_format") == "frame_entries_v1":
            frames = {}
            for k in keys:
                entry = detailed[k]
                if not isinstance(entry, dict):
                    continue
                text = self._compose_vipe_prompt(entry.get("start"), entry.get("dynamic"))
                if text:
                    frames[int(k)] = text
            result = {"format": "frame_entries_v1", "frames": frames}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        segments = []
        prev_start = prev_dyn = None
        for k in keys:
            entry = detailed[k]
            if not isinstance(entry, dict):
                continue
            start = (entry.get("start") or "").strip()
            dyn = (entry.get("dynamic") or "").strip()
            if (start, dyn) != (prev_start, prev_dyn):
                segments.append((int(k), start, dyn))
                prev_start, prev_dyn = start, dyn
        result = {"format": "segments", "segments": segments}
        self._vipe_prompt_payload_cache[json_path] = result
        self._vipe_prompt_segment_cache[json_path] = segments
        return result

    def _resolve_prompt_from_path(self, json_path, frame_min, frame_max):
        prompt_payload = self._load_prompt_segments(json_path)
        if prompt_payload.get("format") == "frame_entries_v1":
            frames = prompt_payload.get("frames") or {}
            lo = min(int(frame_min), int(frame_max))
            hi = max(int(frame_min), int(frame_max))
            prompt_hashes = [frames.get(frame) for frame in range(lo, hi + 1) if frames.get(frame)]
            if not prompt_hashes:
                return ""
            counter = Counter(prompt_hashes)
            text = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
            return text.strip()

        segments = prompt_payload.get("segments") or []
        if not segments:
            return ""

        def _segment_index_for(frame):
            # Largest segment whose start_frame <= frame. ``segments`` is
            # already sorted ascending and starts at frame 0 in practice;
            # if frame predates segment 0 we still return 0.
            idx = 0
            for i, (s, _, _) in enumerate(segments):
                if s <= frame:
                    idx = i
                else:
                    break
            return idx

        lo = min(int(frame_min), int(frame_max))
        hi = max(int(frame_min), int(frame_max))
        first_idx = _segment_index_for(lo)
        last_idx = _segment_index_for(hi)
        best_idx = first_idx
        best_overlap = -1
        for idx in range(first_idx, last_idx + 1):
            start_frame, _, _ = segments[idx]
            next_start = segments[idx + 1][0] if idx + 1 < len(segments) else hi + 1
            overlap_lo = max(lo, start_frame)
            overlap_hi = min(hi + 1, next_start)
            overlap = max(0, overlap_hi - overlap_lo)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        _start_frame, start, dyn = segments[best_idx]
        return self._compose_vipe_prompt(start, dyn)

    def _resolve_prompt_for_scene(self, scene_hash, frame_min, frame_max):
        """Resolve the prompt for a video-frame window ``[frame_min, frame_max]``.

        Rules (matching the user spec):
          * Frame keys in ``detailed`` use *video* frame indices, not latent.
          * Pick the prompt segment that covers the most frames in the query
            window and return only its ``dynamic`` text.
          * If frame indices exceed the largest documented key, the
            algorithm naturally pins them to the last segment, which is
            equivalent to padding with the final prompt.

        Falls back to ``default_prompt`` when no prompt source is configured
        or when no caption is available for this scene.
        """
        if not self._vipe_prompt_roots and not self._vipe_prompt_index:
            return self.default_prompt
        if self._vipe_prompt_index is None:
            self._vipe_prompt_index = self._build_vipe_prompt_index()
        for json_path in self._lookup_prompt_paths(scene_hash):
            text = self._resolve_prompt_from_path(json_path, frame_min, frame_max)
            if text:
                return text
        return self.default_prompt

    def _load_extrinsics(self, path):
        """Load VIPE-style camera poses and return them in W2C convention.

        VIPE writes the per-frame camera-to-world transforms under the
        `"data"` key of pose/video.npz with shape (N, 4, 4). We invert
        each matrix to obtain W2C, which is what the rest of the mosaic
        pipeline (frustum handler, PRoPE camera unit, etc.) consumes.
        """
        with np.load(path, allow_pickle=True) as f:
            array = np.asarray(f["data"])
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise ValueError(f"{path} extrinsics shape must be (N, 4, 4), but got {array.shape}")
        w2c = np.linalg.inv(array.astype(np.float64)).astype(np.float32)
        return w2c

    def _load_intrinsics(self, path, num_frames):
        """Load VIPE-style intrinsics (shape (N, 4): [fx, fy, cx, cy] in pixels)
        and expand to a (N, 3, 3) K-matrix stack, broadcasting if needed.
        """
        with np.load(path, allow_pickle=True) as f:
            array = np.asarray(f["data"])
        array = self._intrinsics_packed_to_matrix(array)
        if array.shape == (3, 3):
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise ValueError(f"intrinsics shape must be (N, 3, 3), but got {array.shape}")
        if array.shape[0] == 1 and num_frames > 1:
            array = np.repeat(array, repeats=num_frames, axis=0)
        return array.astype(np.float32)

    @staticmethod
    def _intrinsics_packed_to_matrix(array):
        """Accept VIPE's packed (N, 4) [fx, fy, cx, cy] OR an already-built
        (3, 3) / (N, 3, 3) K matrix and always return a K-matrix tensor."""
        if array.ndim == 2 and array.shape[-1] == 4:
            n = array.shape[0]
            ks = np.zeros((n, 3, 3), dtype=np.float32)
            ks[:, 0, 0] = array[:, 0]  # fx
            ks[:, 1, 1] = array[:, 1]  # fy
            ks[:, 0, 2] = array[:, 2]  # cx
            ks[:, 1, 2] = array[:, 3]  # cy
            ks[:, 2, 2] = 1.0
            return ks
        return array


def _load_depth_npz_lazy(
    depth_path_or_npz,
    frame_count,
    *,
    depth_format=None,
    depth_metadata=None,
    require_depth_format=False,
):
    """Load the first ``frame_count`` per-frame depths from a depth ``.npz``.

    Supports three on-disk layouts so older caches keep working:

    1. **Per-frame keys** (preferred, producer side now writes this):
       ``frame_{i:05d}`` -- one (H, W) array per frame. We only fetch the
       first ``frame_count`` keys so the rest of the archive is never
       decompressed.
    2. ``depth_{i:05d}`` -- legacy per-frame key naming used by older
       mosaic caches.
    3. Bulk ``data`` or ``depth`` key -- single (N, H, W) array; we
       still slice ``[:frame_count]``.

    Args:
        depth_path_or_npz: path to ``video.npz``, or an already-opened
            ``NpzFile``-like context manager (used by unit tests to spy on
            which keys are actually accessed).
        frame_count: number of frames to load from the start of the clip.

    Returns:
        ``np.ndarray`` of shape ``(frame_count, H, W)``, dtype ``float32``.
    """
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame_count must be >= 0, got {frame_count}")

    if hasattr(depth_path_or_npz, "files"):
        _validate_depth_format_available(
            depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
        return _load_depth_npz_from_handle(
            depth_path_or_npz,
            frame_count,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )

    depth_format, depth_metadata = _resolve_depth_metadata_for_npz(depth_path_or_npz, depth_format, depth_metadata)
    _validate_depth_format_available(
        depth_path_or_npz,
        depth_format=depth_format,
        depth_metadata=depth_metadata,
        require_depth_format=require_depth_format,
    )
    with np.load(depth_path_or_npz, allow_pickle=True) as f:
        return _load_depth_npz_from_handle(
            f,
            frame_count,
            source=depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )


def _load_depth_npz_sparse(
    depth_path_or_npz,
    frame_count,
    frame_ids,
    *,
    depth_format=None,
    depth_metadata=None,
    require_depth_format=False,
):
    """Load only selected frame depths while preserving raw frame-id indexing.

    Returned arrays have first dimension ``frame_count`` so existing frustum
    code can keep indexing ``depths[raw_frame_id]``. Unselected frames are
    zero-filled and should not be consumed by the caller.
    """
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame_count must be >= 0, got {frame_count}")
    frame_ids_iter = [] if frame_ids is None else frame_ids
    selected_ids = sorted({int(fid) for fid in frame_ids_iter if 0 <= int(fid) < frame_count})
    if not selected_ids:
        return np.zeros((0,), dtype=np.float32)

    if hasattr(depth_path_or_npz, "files"):
        _validate_depth_format_available(
            depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
        return _load_depth_npz_sparse_from_handle(
            depth_path_or_npz,
            frame_count,
            selected_ids,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )

    depth_format, depth_metadata = _resolve_depth_metadata_for_npz(depth_path_or_npz, depth_format, depth_metadata)
    _validate_depth_format_available(
        depth_path_or_npz,
        depth_format=depth_format,
        depth_metadata=depth_metadata,
        require_depth_format=require_depth_format,
    )
    with np.load(depth_path_or_npz, allow_pickle=True) as f:
        return _load_depth_npz_sparse_from_handle(
            f,
            frame_count,
            selected_ids,
            source=depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )


def _load_depth_npz_from_handle(f, frame_count, *, source=None, depth_format=None, depth_metadata=None):
    keys = list(getattr(f, "files", []))
    source_str = "" if source is None else f"{source} "

    # Per-frame keys take priority because they let us read only the
    # frames we need without decompressing the rest of the archive.
    has_frame_keys = any(k.startswith("frame_") for k in keys)
    has_depth_xxxxx_keys = any(k.startswith("depth_") and k[len("depth_") :].isdigit() for k in keys)

    if has_frame_keys:
        prefix = "frame_"
    elif has_depth_xxxxx_keys:
        prefix = "depth_"
    elif "data" in keys:
        depths = _decode_depth_array(f["data"], depth_format, depth_metadata)
        return _validate_and_slice_bulk_depth(depths, frame_count, source_str, "data")
    elif "depth" in keys:
        depths = _decode_depth_array(f["depth"], depth_format, depth_metadata)
        return _validate_and_slice_bulk_depth(depths, frame_count, source_str, "depth")
    else:
        raise KeyError(
            f"{source_str}depth npz has no recognised schema; expected one of "
            "{'frame_xxxxx', 'depth_xxxxx', 'data', 'depth'} but got "
            f"{sorted(keys)[:8]}..."
        )

    frames = []
    for i in range(frame_count):
        key = f"{prefix}{i:05d}"
        if key not in keys:
            raise KeyError(
                f"{source_str}depth npz missing key {key!r} (requested "
                f"frame_count={frame_count}, available "
                f"{prefix}* count={sum(1 for k in keys if k.startswith(prefix))})"
            )
        frames.append(_decode_depth_array(f[key], depth_format, depth_metadata))

    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(frames, axis=0)


def _load_depth_npz_sparse_from_handle(
    f,
    frame_count,
    frame_ids,
    *,
    source=None,
    depth_format=None,
    depth_metadata=None,
):
    keys = list(getattr(f, "files", []))
    source_str = "" if source is None else f"{source} "
    has_frame_keys = any(k.startswith("frame_") for k in keys)
    has_depth_xxxxx_keys = any(k.startswith("depth_") and k[len("depth_") :].isdigit() for k in keys)

    if has_frame_keys:
        prefix = "frame_"
    elif has_depth_xxxxx_keys:
        prefix = "depth_"
    elif "data" in keys:
        depths = _decode_depth_array(f["data"], depth_format, depth_metadata)
        full = _validate_and_slice_bulk_depth(depths, frame_count, source_str, "data")
        return _zero_unselected_depth_frames(full, frame_ids)
    elif "depth" in keys:
        depths = _decode_depth_array(f["depth"], depth_format, depth_metadata)
        full = _validate_and_slice_bulk_depth(depths, frame_count, source_str, "depth")
        return _zero_unselected_depth_frames(full, frame_ids)
    else:
        raise KeyError(
            f"{source_str}depth npz has no recognised schema; expected one of "
            "{'frame_xxxxx', 'depth_xxxxx', 'data', 'depth'} but got "
            f"{sorted(keys)[:8]}..."
        )

    loaded = []
    for fid in frame_ids:
        key = f"{prefix}{int(fid):05d}"
        if key not in keys:
            raise KeyError(
                f"{source_str}depth npz missing key {key!r} (requested sparse "
                f"frame ids up to frame_count={frame_count}, available "
                f"{prefix}* count={sum(1 for k in keys if k.startswith(prefix))})"
            )
        loaded.append((int(fid), _decode_depth_array(f[key], depth_format, depth_metadata)))
    if not loaded:
        return np.zeros((0,), dtype=np.float32)
    first = loaded[0][1]
    out = np.zeros((frame_count, *first.shape), dtype=np.float32)
    for fid, frame in loaded:
        if frame.shape != first.shape:
            raise ValueError(
                f"{source_str}{prefix}{fid:05d} depth has shape {frame.shape} "
                f"but first selected frame has shape {first.shape}; all frames "
                "must share the same resolution."
            )
        out[fid] = frame
    return out


def _zero_unselected_depth_frames(depths, frame_ids):
    out = np.zeros_like(depths, dtype=np.float32)
    valid_ids = [int(fid) for fid in frame_ids if 0 <= int(fid) < int(depths.shape[0])]
    if valid_ids:
        out[valid_ids] = depths[valid_ids]
    return out


def _resolve_depth_metadata_for_npz(depth_path, depth_format=None, depth_metadata=None):
    if depth_format or depth_metadata:
        return depth_format, depth_metadata
    try:
        metadata_path = os.path.join(os.path.dirname(os.fspath(depth_path)), "metadata.json")
    except TypeError:
        return depth_format, depth_metadata
    if not os.path.isfile(metadata_path):
        return depth_format, depth_metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except (OSError, ValueError):
        return depth_format, depth_metadata
    if not isinstance(payload, dict):
        return depth_format, depth_metadata
    return _depth_encoding_name(depth_metadata=payload)[0], payload


def _validate_depth_format_available(depth_path, *, depth_format=None, depth_metadata=None, require_depth_format=False):
    if not require_depth_format:
        return
    encoding, _metadata = _depth_encoding_name(depth_format, depth_metadata)
    if encoding:
        return
    raise ValueError(
        "depth_format/depth_metadata is required for this depth npz. "
        f"Got none for {depth_path!r}. Rebuild the DA3 index with current "
        "build_da3_video_index.py, keep depth/metadata.json next to video.npz, "
        "or pass --dataset_compact_mode only for old float16/float NPZ datasets."
    )


def _normalize_depth_metadata(depth_metadata):
    if isinstance(depth_metadata, dict):
        return depth_metadata
    if isinstance(depth_metadata, str) and depth_metadata:
        try:
            payload = json.loads(depth_metadata)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _depth_encoding_name(depth_format=None, depth_metadata=None):
    metadata = _normalize_depth_metadata(depth_metadata)
    candidates = (
        depth_format,
        metadata.get("depth_format"),
        metadata.get("format"),
        metadata.get("depth_encoding"),
        metadata.get("quantization"),
    )
    for value in candidates:
        name = str(value or "").strip().lower()
        if name in {"npz_log_u10", "log_u10", "logu10"}:
            return "npz_log_u10", metadata
        if name in {"npz_log_u16", "log_u16", "logu16"}:
            return "npz_log_u16", metadata
        if name in {"npz_float", "float", "float32"}:
            return "npz_float", metadata
    return "", metadata


def _decode_depth_array(array, depth_format=None, depth_metadata=None):
    encoding, metadata = _depth_encoding_name(depth_format, depth_metadata)
    if encoding == "npz_log_u10":
        return _decode_log_uint_depth(array, metadata, default_bit_depth=10)
    if encoding == "npz_log_u16":
        return _decode_log_uint_depth(array, metadata, default_bit_depth=16)
    return np.asarray(array, dtype=np.float32)


def _decode_log_uint_depth(array, metadata, *, default_bit_depth):
    bit_depth = int(metadata.get("bit_depth") or default_bit_depth)
    max_code = float((1 << bit_depth) - 1)
    depth_min = float(metadata.get("depth_min", metadata.get("configured_depth_min", 0.1)))
    depth_max = float(metadata.get("depth_max", metadata.get("configured_depth_max", 1000.0)))
    values = np.asarray(array, dtype=np.float32)
    values = np.clip(values, 0.0, max_code) / max_code
    log_min = np.log(np.float32(depth_min))
    log_max = np.log(np.float32(depth_max))
    return np.exp(values * (log_max - log_min) + log_min).astype(np.float32)


def _validate_and_slice_bulk_depth(depths, frame_count, source_str, key_name):
    if depths.ndim != 3:
        raise ValueError(f"{source_str}{key_name!r} depth data must have shape (N, H, W), got {depths.shape}")
    return depths[:frame_count]


# ---------------------------------------------------------------------------
# DA3 inference dataset. It returns raw RGB frames; the runtime materializes
# clean, noisy, and memory latents on the active GPU rank.
# ---------------------------------------------------------------------------


class _MosaicVideoDatasetBase(_MosaicVideoSharedDataset):
    """Shared DA3 inference dataset runtime.

    DA3 strict mode drives each scene from sqlite-indexed RGB, camera, prompt,
    and depth records. The runtime encodes clean/noisy/memory frames on GPU.
    Key runtime properties:
        * Memory uses **single-frame** latents (1 source frame == 1 latent
          slot), so frustum candidate ``frame_id`` indexes the memory buffer
          directly.
        * Rollout start points are selected at the **frame** level.
        * A single clean anchor precedes each generated window.
        * Optional per-frame memory-latent cache via sha256 fanout
          (``--memory_latent_cache_dir`` / ``--memory_latent_cache_version``).
    """

    DEPTH_PREFERRED_FILENAMES = ("video.zip",)
    DEPTH_FILE_SUFFIXES = (".zip",)

    def __init__(
        self,
        *args,
        height,
        width,
        memory_latent_cache_dir="",
        memory_latent_cache_version="wan2.2_ti2v_5b_vae",
        memory_vae_encode_input_frames=1,
        min_anchor_frame_idx=0,
        **kwargs,
    ):
        rs_prob = float(kwargs.get("random_start_latent_prob", 0.0) or 0.0)
        if rs_prob != 0.0:
            raise ValueError("Matrix-Game inference requires random_start_latent_prob=0")

        self.height = int(height)
        self.width = int(width)
        # Wan 2.2 TI2V-5B: VAE upsampling_factor=16 AND DIT patch_size=2,
        # so the pipeline's height_division_factor is 16*2 = 32. Heights
        # that are 16-aligned but not 32-aligned (e.g. 720) end up with
        # an odd latent row count (45) which the pipeline silently
        # rounds up via ``check_resize_height_width`` (-> 736 -> 46
        # latent rows), producing a noise tensor that no longer matches
        # our mosaic_latent. Enforce 32-alignment up front so the
        # operator picks a compatible resolution rather than getting a
        # cryptic "(1,48,20,45,80) vs (1,48,20,46,80)" error from
        # WanVideoUnit_MosaicLatent.
        if self.height % 32 != 0 or self.width % 32 != 0:

            def _suggest(v):
                return ((v + 31) // 32 * 32, max(32, v // 32 * 32))

            h_up, h_down = _suggest(self.height)
            w_up, w_down = _suggest(self.width)
            raise ValueError(
                f"height/width must be divisible by 32 (VAE upsampling x DIT patch_size); "
                f"got {self.height}x{self.width}. "
                f"Closest valid options: {h_down}x{w_down}, {h_up}x{w_up} "
                f"(720p flows usually use 704x1280 or 736x1280)."
            )
        self.memory_latent_cache_dir = str(memory_latent_cache_dir or "")
        self.memory_latent_cache_version = str(memory_latent_cache_version)
        # Encode style for memory latents; folded into the
        # cache key when != 1 (see _memory_cache_hash).
        self.memory_vae_encode_input_frames = max(1, int(memory_vae_encode_input_frames or 1))

        self.min_anchor_frame_idx = max(0, int(min_anchor_frame_idx or 0))

        # DA3 strict mode populates this directly from sqlite index records.
        self._vipe_video_index = None

        super().__init__(*args, **kwargs)

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "dataset_kind": "video",
                "height": self.height,
                "width": self.width,
                "memory_latent_cache_version": self.memory_latent_cache_version,
            }
        )
        return base

    # ------------------------------------------------------------------
    # Memory latent cache helpers shared with GPU materialization.
    # ------------------------------------------------------------------

    def _memory_cache_hash(self, clean_name, frame_id):
        """Stable per-frame hash used for atomic cache writes.

        ``memory_vae_encode_input_frames`` is part of the namespace when it is
        non-default because replicated input encoding is numerically distinct:
        the runtime's replicate-N encode (keep-last-latent) is numerically
        different from the single-frame encode, so non-default values get
        their own namespace while the default 1 keeps existing hashes.
        """
        text = f"{clean_name}|{int(frame_id)}|{self.height}|{self.width}|{self.memory_latent_cache_version}"
        encode_input_frames = int(getattr(self, "memory_vae_encode_input_frames", 1) or 1)
        if encode_input_frames != 1:
            text = f"{text}|mvif{encode_input_frames}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _memory_cache_path(self, clean_name, frame_id):
        """Two-level fanout dir layout to keep any single subdir <100k files."""
        if not self.memory_latent_cache_dir:
            return None
        h = self._memory_cache_hash(clean_name, frame_id)
        return os.path.join(
            self.memory_latent_cache_dir,
            h[:2],
            h[2:4],
            f"{clean_name}_{int(frame_id):06d}_{h}.pt",
        )

    def _read_video_frames(
        self,
        video_path,
        frame_indices,
        *,
        normalize=True,
    ):
        """Read ``frame_indices`` from ``video_path`` (decord) and return
        a ``(C, T, H, W)`` tensor.

        ``frame_indices`` must be a sorted list of unique ints. Replicates
        ``batch_encode_videos.VideoLatentDataset.{load_video,center_crop_resize}``
        so memory and rollout encodes see byte-identical inputs.
        ``normalize=False`` keeps uint8 pixels for GPU-side normalization in the
        runtime materialization path.
        """
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu())
        frames = reader.get_batch(list(frame_indices)).asnumpy()
        out = _video_frames_uint8_to_normalized_tensor(
            frames,
            height=self.height,
            width=self.width,
            normalize=normalize,
        )
        return out

    def _prompt_aligned_generating_start_candidates(
        self,
        info,
        *,
        lo,
        hi,
    ):
        if not getattr(self, "align_generating_to_prompt_segments", False):
            return []
        prompt_path = info.get("prompt_path")
        if not prompt_path:
            return []
        prompt_payload = self._load_prompt_segments(prompt_path)
        if prompt_payload.get("format") != "segments":
            return []
        candidates = []
        segments = list(prompt_payload.get("segments") or [])
        for index, segment in enumerate(segments):
            try:
                segment_start = int(segment[0])
            except (TypeError, ValueError, IndexError):
                continue
            # The first segment starts at frame 0, but G needs at least one
            # frame of clean context, so start the first prompt at start+1.
            candidate = segment_start + 1 if index == 0 else segment_start
            next_start = int(segments[index + 1][0]) if index + 1 < len(segments) else int(hi)
            if candidate < int(lo) or candidate >= int(hi) or candidate >= next_start:
                continue
            candidates.append(int(candidate))
        return sorted(dict.fromkeys(candidates))

    def _sample_generating_first_frame_idx(
        self,
        info,
        *,
        lo,
        hi,
    ):
        candidates = self._prompt_aligned_generating_start_candidates(
            info,
            lo=lo,
            hi=hi,
        )
        if candidates:
            return int(candidates[0])
        return int(lo)

    def __getitem__(self, index):
        # Stash deterministic per-sample RNG for subject-reference selection
        # and optional extrinsic overrides.
        self._make_sample_rng(index)
        path = self._select_path(index)
        info = dict(self.dataset_info[path])

        frame_count = int(info["frame_count"])
        num_generating_frames = 4 * int(self.latent_window_size)
        max_clean_context = 1

        # The lower bound reserves enough clean history for retrieval.
        lo = max_clean_context
        lo = max(lo, int(self.min_anchor_frame_idx) + 1)
        hi = frame_count - num_generating_frames
        if hi <= lo:
            # ``too_short`` filter should have dropped this scene; fail loud.
            raise RuntimeError(
                f"_MosaicVideoDatasetBase: scene {info['clean_name']} too short "
                f"(frame_count={frame_count}, num_generating_frames="
                f"{num_generating_frames}, max_clean_context="
                f"{max_clean_context})"
            )
        G = self._sample_generating_first_frame_idx(
            info,
            lo=lo,
            hi=hi,
        )

        latent_window_size = int(self.latent_window_size)
        H_lat = self.height // self.VAE_HW_SCALING
        W_lat = self.width // self.VAE_HW_SCALING

        # ----- Camera / depth (re-implemented from read_camera_params,
        # but keyed off frame indices instead of clean_latents.shape) -----
        extrinsics = self._load_extrinsics(self._get_extrinsic_path(info))
        intrinsics = self._load_intrinsics(info["intrinsic_path"], num_frames=extrinsics.shape[0])
        # The dataset's mosaic_intrinsics_mode controls whether K remains
        # per-frame, collapses to a per-episode mean, or reuses frame 0.
        scaled_intrinsics = self.normalize_and_scale_intrinsics(
            intrinsics,
            H_img=self.height,
            W_img=self.width,
            temporal_mean=self._intrinsics_temporal_mean_enabled(),
        )

        clean_frame_indices = [G - 1] * self.VAE_T_SCALING
        noisy_frame_indices = list(range(G, G + num_generating_frames))

        clean_frame_extrinsics = extrinsics[clean_frame_indices]
        clean_frame_intrinsics = scaled_intrinsics[clean_frame_indices]
        noisy_frame_extrinsics = extrinsics[noisy_frame_indices]
        noisy_frame_intrinsics = scaled_intrinsics[noisy_frame_indices]

        # Memory frustum needs depths and w2c up to frame G (exclusive).
        # DA3 strict mode requires a depth file during dataset indexing.
        depth_path = info.get("depth_path")
        if not depth_path:
            raise RuntimeError(
                f"DA3 sample {info.get('clean_name')} has no depth_path; dataset indexing should have filtered it out."
            )
        sparse_depth_after_selection = self.mosaic_selection_mode in {
            "pose_nearest",
            "pose_pool_temporal_earliest",
        }
        if sparse_depth_after_selection:
            depths = np.zeros((0,), dtype=np.float32)
        else:
            depths = self._load_depths_for_memory(
                depth_path,
                frame_count=G,
                depth_format=info.get("depth_format"),
                depth_metadata=info.get("depth_metadata"),
            )
        w2c_memory = extrinsics[:G]

        scene_hash = info.get("clean_name") or os.path.basename(info.get("video_path", "")).split(".")[0]
        prompt_text = self._resolve_prompt_for_scene(
            scene_hash,
            frame_min=noisy_frame_indices[0],
            frame_max=noisy_frame_indices[-1],
        )

        # ----- Phase 1: select frustum candidates (CPU-only) -----
        # Per-frame K split:
        #   * memory_K: K for frames [0..G) that the frustum can pick from.
        #   * query_K : K for the noisy/query frames (== noisy_frame_intrinsics).
        # Inference paths keep using the single-anchor
        # K shape, so the handler accepts both modes.
        memory_K = scaled_intrinsics[:G].astype(np.float32, copy=False)
        query_K_arr = scaled_intrinsics[noisy_frame_indices].astype(np.float32, copy=False)
        handler = self._get_frustum_handler()
        depth_downsample_pool = {}
        try:
            # See note in _query_frustum: distance_threshold is now in METERS.
            cand_frame_ids = handler.select_candidates(
                query_extrinsics=noisy_frame_extrinsics,
                depths=depths,
                w2c=w2c_memory,
                memory_K=memory_K,
                query_K=query_K_arr,
                H_lat=H_lat,
                W_lat=W_lat,
                total_latents=G,
                latent_merge_4frames=False,
                candidates_per_query_group=self.candidates_per_query_group,
                angle_threshold=None,
                distance_threshold=None,
                temporal_threshold=None,
                geometry_rep_mode="second_frame",
                selection_mode=self.mosaic_selection_mode,
                query_reference_frame=self.mosaic_query_reference_frame,
                candidate_nms_mode=self.mosaic_candidate_nms_mode,
                candidate_nms_projection_iou_threshold=(self.mosaic_candidate_nms_projection_iou_threshold),
                candidate_nms_min_temporal_gap=(self.mosaic_candidate_nms_min_temporal_gap),
                candidate_nms_pose_distance_threshold=(self.mosaic_candidate_nms_pose_distance_threshold),
                candidate_nms_pool_multiplier=(self.mosaic_candidate_nms_pool_multiplier),
                coverage_grid_downsample=self.mosaic_coverage_grid_downsample,
                coverage_pool_stride=self.mosaic_coverage_pool_stride,
                depth_downsample_pool=depth_downsample_pool,
            )
        finally:
            # The pool can hold one latent-sized depth map per candidate
            # group. Keep it scoped to this sample only.
            depth_downsample_pool.clear()

        # ----- Build memory_set + cache lookup -----
        memory_set = set()
        for group in cand_frame_ids:
            for fid in group:
                fid = int(fid)
                if 0 <= fid < G:
                    memory_set.add(fid)

        if sparse_depth_after_selection:
            depths = self._load_depths_for_memory(
                depth_path,
                frame_count=G,
                frame_ids=sorted(memory_set),
                depth_format=info.get("depth_format"),
                depth_metadata=info.get("depth_metadata"),
            )

        cached_paths = {}
        miss_set = set()
        target_paths = {}
        for fid in memory_set:
            cache_path = self._memory_cache_path(info["clean_name"], fid)
            if cache_path is not None:
                target_paths[fid] = cache_path
                if os.path.exists(cache_path):
                    cached_paths[fid] = cache_path
                    continue
            miss_set.add(fid)
        # Read the clean anchor, generated window, and uncached memory frames
        # in one video decode.
        cn_start = G - 1
        cn_end = G + num_generating_frames
        cn_frame_indices = list(range(cn_start, cn_end))
        memory_read_ids = set(miss_set)
        all_indices_to_read = sorted(memory_read_ids | set(cn_frame_indices))
        if all_indices_to_read:
            all_frames = self._read_video_frames(
                info["video_path"],
                all_indices_to_read,
                normalize=False,
            )
            # all_frames: (C, T_all, H, W), T_all == len(all_indices_to_read)
            idx_to_pos = {fid: pos for pos, fid in enumerate(all_indices_to_read)}
        else:
            all_frames = None
            idx_to_pos = {}

        memory_frames_by_id = {}
        for fid in sorted(miss_set):
            pos = idx_to_pos[fid]
            # Slice as (C, 1, H, W) so materialization can stack and call
            # pipe.vae.encode without an extra unsqueeze.
            memory_frames_by_id[fid] = all_frames[:, pos : pos + 1].contiguous()

        # cn_frames must come back in frame-order (matches encode expectations).
        cn_positions = [idx_to_pos[fid] for fid in cn_frame_indices]
        cn_frames = all_frames[:, cn_positions].contiguous()
        assert cn_frames.shape[1] == 1 + num_generating_frames

        # ----- Index tensors for the DIT (positional, not frame-level) -----
        clean_latent_indices_start = torch.tensor([0], dtype=torch.long)
        clean_latent_indices = torch.tensor([1], dtype=torch.long)
        noisy_latent_indices = torch.arange(2, 2 + latent_window_size, dtype=torch.long)[None, :]

        per_call_info = {
            **info,
            "generating_first_frame_idx": int(G),
            "cn_drop_count": 0,
        }

        data = {
            "clean_latent_indices_start": clean_latent_indices_start,
            "clean_latent_indices": clean_latent_indices,
            "noisy_latent_indices": noisy_latent_indices,
            "clean_latent_indices_prope_intrinsic": torch.from_numpy(clean_frame_intrinsics).float(),
            "clean_latent_indices_prope_extrinsic": torch.from_numpy(clean_frame_extrinsics).float(),
            "noisy_latent_indices_prope_intrinsic": torch.from_numpy(noisy_frame_intrinsics).float(),
            "noisy_latent_indices_prope_extrinsic": torch.from_numpy(noisy_frame_extrinsics).float(),
            "memory_frames_by_id": memory_frames_by_id,
            "cached_memory_paths_by_id": cached_paths,
            "memory_cache_hash_by_id": target_paths,
            "frame_pixel_range": "uint8_or_255",
            "cn_frames": cn_frames,
            "cn_drop_count": 0,
            "cn_anchor_warmup_frames": 0,
            "fused_query_inputs": {
                "query_extrinsics": np.asarray(noisy_frame_extrinsics, dtype=np.float32),
                "candidate_frame_ids": cand_frame_ids,
                "w2c": w2c_memory,
                "depths": depths,
                # Single-K compatibility alias; inference uses the per-frame
                # ``memory_K`` and ``query_K`` fields below.
                "K": np.asarray(query_K_arr[0], dtype=np.float32),
                "memory_K": memory_K,
                "query_K": query_K_arr,
                "query_frame_indices": noisy_frame_indices,
                "H_lat": int(H_lat),
                "W_lat": int(W_lat),
                "memory_total_frames": int(G),
                "query_reference_frame": int(self.mosaic_query_reference_frame),
            },
            "needs_vae_materialization": True,
            "prompt": prompt_text,
            "info": per_call_info,
            "lookup": {},
            "is_starting": False,
            "updatetqdm": ".",
        }
        return data

    def read_camera_params(self, data, info, lookup=None):
        """Per-section PRoPE camera recompute used by ``run_mosaic_inference``.

        The inference loop drifts ``clean_latent_indices`` /
        ``noisy_latent_indices`` by ``latent_stride`` per section and
        calls this method on each section to refresh PRoPE camera fields
        and ``prompt`` for the new window. Mirrors the parent's API but
        works in frame-level coordinates instead of latent-level
        ``_latent_indices_to_frame_indices`` math (which assumed 4-frame
        latent compression -- not applicable to single-frame memory).

        Frame indices that walk past the source video's last frame get
        clamped (the predicted-only sections genuinely have no GT
        camera; clamping to the final pose is the same fallback the
        DA3 inference uses).
        """
        extrinsics = self._load_extrinsics(self._get_extrinsic_path(info))
        intrinsics = self._load_intrinsics(info["intrinsic_path"], num_frames=extrinsics.shape[0])
        # Inference uses the configured mosaic intrinsics mode.
        scaled_intrinsics = self.normalize_and_scale_intrinsics(
            intrinsics,
            H_img=self.height,
            W_img=self.width,
            temporal_mean=self._intrinsics_temporal_mean_enabled(),
        )

        # Original G this sample was drawn at; inference drifts by
        # full-window strides so re-derive the drifted G from the
        # currently-running clean_latent_indices.
        g_orig = int(info.get("generating_first_frame_idx", 0))
        # Section-0 clean_latent_indices == [1]; per-section drift of
        # ``latent_stride`` (== latent_window_size) maps to a frame drift
        # of ``stride * 4`` since 1 latent step covers 4 noisy frames.
        clean_pos = int(data["clean_latent_indices"][0].item())
        drift_frames = (clean_pos - 1) * 4
        G = g_orig + drift_frames

        is_starting = bool(data.get("is_starting", True))
        if is_starting:
            # Section-0 bootstrap anchor is a genuine single-frame
            # (image-mode) latent, so replicate its one pose.
            clean_frame_indices = [G - 1] * self.VAE_T_SCALING
        else:
            # Sections >= 1 feed back the last GENERATED latent -- a dense
            # block covering frames [G-4..G-1]. PRoPE applies one camera
            # per sub-frame slot, so give it the four distinct poses. Replicating the
            # last pose here would assert zero ego-motion through a moving
            # anchor.
            clean_frame_indices = [G - 4, G - 3, G - 2, G - 1]
        num_generating_frames = 4 * int(self.latent_window_size)
        noisy_frame_indices = list(range(G, G + num_generating_frames))

        num_camera_frames = int(extrinsics.shape[0])
        max_idx = num_camera_frames - 1
        clean_frame_indices = [min(max(0, fid), max_idx) for fid in clean_frame_indices]
        noisy_frame_indices = [min(max(0, fid), max_idx) for fid in noisy_frame_indices]

        scene_hash = info.get("clean_name") or os.path.splitext(os.path.basename(info.get("video_path", "")))[0]
        data["prompt"] = self._resolve_prompt_for_scene(
            scene_hash,
            frame_min=min(noisy_frame_indices),
            frame_max=max(noisy_frame_indices),
        )
        # ``w2c`` / ``depths`` are not consumed by inference's downstream
        # flow (it runs its own DA3 + register_source_sequence), but a
        # stable shape keeps progress reporting meaningful.
        data["w2c"] = extrinsics[: noisy_frame_indices[0]]
        data["depths"] = np.zeros((0,), dtype=np.float32)

        data["clean_latent_indices_prope_intrinsic"] = torch.from_numpy(scaled_intrinsics[clean_frame_indices]).float()
        data["clean_latent_indices_prope_extrinsic"] = torch.from_numpy(extrinsics[clean_frame_indices]).float()
        data["noisy_latent_indices_prope_intrinsic"] = torch.from_numpy(scaled_intrinsics[noisy_frame_indices]).float()
        data["noisy_latent_indices_prope_extrinsic"] = torch.from_numpy(extrinsics[noisy_frame_indices]).float()
        return data


class DA3MosaicVideoDataset(_MosaicVideoDatasetBase):
    """Video-flow dataset for DA3-batched outputs.

    Layout uses sqlite-indexed ``rgb/``, ``pose/`` and ``intrinsics/`` records.
    Depth lives at ``depth/video.npz``; the loader auto-
    detects the on-disk schema (see ``_load_depth_npz_lazy``):

    - **Preferred (current producer)**: per-frame keys ``frame_{i:05d}``,
      one (H, W) array each. Only the first ``frame_count`` keys are
      read so the rest of the archive stays compressed on disk.
    - Legacy bulk ``data`` (or ``depth``) key with a single (N, H, W)
      array; still supported for older caches.
    """

    DEPTH_PREFERRED_FILENAMES = ("video.npz",)
    DEPTH_FILE_SUFFIXES = (".npz",)

    def __init__(self, *args, dataset_compact_mode=False, **kwargs):
        self.dataset_compact_mode = bool(dataset_compact_mode)
        if not kwargs.get("dataset_index_path"):
            raise ValueError(
                "DA3MosaicVideoDataset strict mode requires --dataset_index_path. "
                "DA3 video data is read only from the sqlite index."
            )
        # Parent classes still carry legacy constructor requirements for
        # directory scans. In strict DA3 mode these roots are inert: sqlite
        # records provide every video/camera/depth/prompt path.
        args = list(args)
        for idx in range(min(3, len(args))):
            args[idx] = ()
        args = tuple(args)
        for legacy_key in ("base_path", "camera_params_path", "depth_path"):
            kwargs[legacy_key] = ()
        super().__init__(*args, dataset_compact_mode=dataset_compact_mode, **kwargs)

    def _da3_index_debug(self, message):
        if getattr(self, "rank", 0) == 0:
            print(f"[DA3MosaicVideoDataset:index] {message}")

    def _da3_index_meta(self):
        if not getattr(self, "dataset_index_path", None):
            raise ValueError("DA3MosaicVideoDataset strict mode requires --dataset_index_path.")
        cached = getattr(self, "_da3_video_index_meta", None)
        if cached is not None:
            return cached
        try:
            from .da3_video_index import (
                DA3_VIDEO_INDEX_SCHEMA_VERSION,
                load_da3_video_index_meta,
            )

            meta = load_da3_video_index_meta(self.dataset_index_path)
            if str(meta.get("dataset_kind")) != "da3_video":
                raise ValueError("dataset_kind is not da3_video")
            schema_version = int(meta.get("schema_version", -1))
            supported_schema_versions = {2, DA3_VIDEO_INDEX_SCHEMA_VERSION}
            if schema_version not in supported_schema_versions and not bool(
                getattr(self, "dataset_compact_mode", False)
            ):
                raise ValueError(f"schema_version={schema_version} expected one of {sorted(supported_schema_versions)}")
            if schema_version != DA3_VIDEO_INDEX_SCHEMA_VERSION:
                self._da3_index_debug(
                    f"accepting legacy schema_version={schema_version}; current={DA3_VIDEO_INDEX_SCHEMA_VERSION}"
                )
        except Exception as exc:
            raise ValueError(
                f"DA3MosaicVideoDataset strict mode cannot use dataset_index_path={self.dataset_index_path!r}: {exc}"
            ) from exc
        self._da3_video_index_meta = meta
        self._da3_index_debug(
            f"loaded meta path={self.dataset_index_path} "
            f"records={meta.get('record_count')} kept={meta.get('kept_count')} "
            f"rejected={meta.get('rejected_count')} "
            f"content_sha1={meta.get('content_sha1')}"
        )
        return meta

    def _using_da3_video_index(self):
        self._da3_index_meta()
        return True

    def _load_da3_video_index_records(self):
        cached = getattr(self, "_da3_video_index_records", None)
        if cached is not None:
            return cached
        self._using_da3_video_index()
        from .da3_video_index import load_da3_video_index_records

        records = load_da3_video_index_records(
            self.dataset_index_path,
            only_complete=False,
            min_frame_count=None,
        )
        meta = self._da3_index_meta() or {}
        self._da3_index_debug(
            f"loaded records total={len(records)} "
            f"min_frame_count={self.min_frame_count} "
            f"sqlite_kept={meta.get('kept_count')}"
        )
        self._da3_video_index_records = records
        self._da3_video_index_records_by_path = {record["synthetic_path"]: record for record in records}
        self._vipe_camera_index = {
            record["clean_name"]: (
                record["extrinsic_path"],
                record["intrinsic_path"],
            )
            for record in records
        }
        self._vipe_video_index = {record["clean_name"]: record["video_path"] for record in records}
        self._vipe_depth_zip_index = {
            record["clean_name"]: record["depth_path"] for record in records if record.get("depth_path")
        }
        self._vipe_prompt_index = {
            record["clean_name"]: record["prompt_path"] for record in records if record.get("prompt_path")
        }
        self._vipe_prompt_path_index = {
            record["clean_name"]: [record["prompt_path"]] for record in records if record.get("prompt_path")
        }
        return records

    def _collect_latent_paths(self, filter_yaml):
        self._using_da3_video_index()
        records = self._load_da3_video_index_records()
        rawpath = sorted(record["synthetic_path"] for record in records)
        before_filter = len(rawpath)
        if filter_yaml is None:
            self._da3_index_debug(f"collected raw paths from index: {before_filter}")
            return rawpath

        import yaml

        with open(filter_yaml, "r") as f:
            filter_criteria = yaml.safe_load(f) or []
        filtered = [path for path in rawpath if not any(filter_token in path for filter_token in filter_criteria)]
        self._da3_index_debug(f"filter_yaml applied: before={before_filter} after={len(filtered)} path={filter_yaml}")
        return filtered

    def _apply_include_yaml(self, rawpath, include_yaml):
        before = len(rawpath)
        result = super()._apply_include_yaml(rawpath, include_yaml)
        if self._using_da3_video_index() and include_yaml:
            self._da3_index_debug(f"include_yaml applied: before={before} after={len(result)} path={include_yaml}")
        return result

    def _select_inference_paths(self, rawpath, inference_assign_yaml=None, inference_ratio=0.05):
        self._using_da3_video_index()
        if inference_assign_yaml is not None:
            return super()._select_inference_paths(
                rawpath,
                inference_assign_yaml=inference_assign_yaml,
                inference_ratio=inference_ratio,
            )
        inference_ratio = 0.0 if inference_ratio is None else float(inference_ratio)
        inference_ratio = max(0.0, min(1.0, inference_ratio))
        split_seed = _derive_seed(self.seed, "split")
        inference_paths_set = set()
        for path in sorted(rawpath):
            key = self._raw_path_to_dataset_key(path)
            token = f"{self.seed}|split|{key}"
            score = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:16], 16)
            score = score / float(16**16)
            if score < inference_ratio:
                inference_paths_set.add(path)
        matched = [path for path in sorted(rawpath) if path in inference_paths_set]
        inference_keys = [self._raw_path_to_dataset_key(path) for path in matched]
        self._split_seed = int(split_seed)
        self._split_source_count = len(rawpath)
        self._split_inference_keys = inference_keys
        self.inference_paths.extend(inference_keys)
        self._da3_index_debug(
            f"stable inference selection: source={len(rawpath)} "
            f"selected={len(matched)} "
            f"inference_ratio={inference_ratio} seed={self.seed}"
        )
        return matched, []

    def _da3_dynamic_min_frame_count(self):
        num_generating_frames = 4 * int(self.latent_window_size)
        dynamic_min = num_generating_frames + 2
        if self.min_frame_count is None:
            return int(dynamic_min)
        return max(int(self.min_frame_count), int(dynamic_min))

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base["dataset_kind"] = "da3_video"
        base["dataset_compact_mode"] = bool(getattr(self, "dataset_compact_mode", False))
        # Folding the RoPE table bound into the cache key forces a rescan
        # of manifests built before the ``rope_time_overflow`` filter
        # existed, so the filter applies to cached datasets too.
        base["rope_freqs_table_end"] = ROPE_FREQS_TABLE_END
        meta = self._da3_index_meta()
        if meta:
            base["dataset_index"] = {
                "path": self.dataset_index_path,
                "schema_version": meta.get("schema_version"),
                "content_sha1": meta.get("content_sha1"),
            }
        return base

    def _build_dataset_info(self, rawpath, inference_assigns):
        self._using_da3_video_index()
        records_by_path = getattr(self, "_da3_video_index_records_by_path", None)
        if records_by_path is None:
            self._load_da3_video_index_records()
            records_by_path = self._da3_video_index_records_by_path

        dataset_info = {}
        min_frames = self._da3_dynamic_min_frame_count()
        self._da3_index_debug(
            f"building dataset_info from index: raw_paths={len(rawpath)} dynamic_min_frame_count={min_frames}"
        )
        for synthetic_path in tqdm(
            rawpath,
            desc="Build DA3 dataset_info from index",
            disable=getattr(self, "rank", 0) != 0,
        ):
            record = records_by_path.get(synthetic_path)
            if record is None:
                continue
            clean_name = record["clean_name"]
            scan_error = record.get("scan_error")
            if scan_error:
                self.reason[clean_name] = scan_error
                continue
            if not record.get("prompt_path") and not self.allow_no_prompt:
                self.reason[clean_name] = "no_prompt"
                continue
            if not record.get("video_path"):
                self.reason[clean_name] = "no_video"
                continue
            if not record.get("extrinsic_path") or not record.get("intrinsic_path"):
                self.reason[clean_name] = "no_camera_params"
                continue
            if record.get("frame_count") is None:
                self.reason[clean_name] = "missing_frame_count"
                continue
            try:
                frame_count = self._cap_frame_count(int(record["frame_count"]))
            except (TypeError, ValueError) as exc:
                self.reason[clean_name] = f"invalid_frame_count:{exc}"
                continue
            if frame_count < min_frames:
                self.reason[clean_name] = "too_short"
                continue
            overflow_time = _rope_time_overflow(frame_count, self.latent_window_size)
            if overflow_time is not None:
                # Stable label (no per-scene suffix) so the reason counter
                # aggregates; the scan summary below reports the count.
                self.reason[clean_name] = "rope_time_overflow"
                continue
            if self.require_depth and not record.get("depth_path"):
                self.reason[clean_name] = "no_depth"
                continue
            depth_format = record.get("depth_format")
            depth_metadata = record.get("depth_metadata_json")
            if (
                getattr(self, "dataset_compact_mode", False)
                and record.get("depth_path")
                and not (depth_format or depth_metadata)
            ):
                depth_format = "npz_float"
            else:
                depth_format, depth_metadata = _resolve_depth_metadata_for_npz(
                    record.get("depth_path"),
                    depth_format,
                    depth_metadata,
                )
            if record.get("depth_path") and not (depth_format or depth_metadata):
                if getattr(self, "dataset_compact_mode", False):
                    depth_format = "npz_float"
                else:
                    self.reason[clean_name] = "missing_depth_format"
                    continue
            info = {
                "frame_count": int(frame_count),
                "clean_name": clean_name,
                "camera_root": record.get("camera_root"),
                "camera_root_order": record.get("camera_root_order"),
                "video_path": record["video_path"],
                "depth_path": record["depth_path"],
                "depth_format": depth_format,
                "depth_metadata_path": record.get("depth_metadata_path"),
                "depth_metadata": depth_metadata,
                "dirname": record["scene_dir"],
                "extrinsic_path": record["extrinsic_path"],
                "intrinsic_path": record["intrinsic_path"],
                "prompt_path": record.get("prompt_path"),
            }
            key = f"{info['dirname']}/{clean_name}"
            dataset_info[key] = info
            if record.get("prompt_path"):
                reason = "success" if record.get("depth_path") else "success_no_depth"
            else:
                reason = "success_no_prompt" if record.get("depth_path") else "success_no_prompt_no_depth"
            self.reason[clean_name] = reason

        info_keys = set(dataset_info)
        self.inference_paths = [key for key in dict.fromkeys(self.inference_paths) if key in info_keys]
        self._split_inference_keys = sorted(info_keys)
        reason_counts = Counter(self.reason.values())
        self._da3_index_debug(f"dataset_info ready: kept={len(dataset_info)} reasons={dict(reason_counts)}")
        return dataset_info

    def _load_depths_for_memory(
        self,
        depth_path,
        frame_count,
        frame_ids=None,
        *,
        depth_format=None,
        depth_metadata=None,
        require_depth_format=None,
    ):
        if require_depth_format is None:
            require_depth_format = not bool(getattr(self, "dataset_compact_mode", False))
        if frame_ids is not None:
            return _load_depth_npz_sparse(
                depth_path,
                frame_count=frame_count,
                frame_ids=frame_ids,
                depth_format=depth_format,
                depth_metadata=depth_metadata,
                require_depth_format=require_depth_format,
            )
        return _load_depth_npz_lazy(
            depth_path,
            frame_count=frame_count,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
