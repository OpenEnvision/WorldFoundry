"""ViGeo depth estimation for the cloud-warp inference backend.

This matches DA3's *output contract*, not its method. DA3 is pose-conditioned -- the GT extrinsics and
intrinsics are network inputs, so its depth is solved inside the GT camera model -- while ViGeo accepts
images only, predicting its own poses and focal length with depth up-to-scale. The bridge is that DA3's
own alignment step effectively replaces intrinsics/extrinsics with the GT ones and uses the Umeyama scale
purely to rescale depth, so ViGeo can take the same route: rescale by the Umeyama scale between the GT
camera centres and ViGeo's predicted ones, then unproject with the GT K.

Because `FrameBank.ingest` only calls `depth_window`, the FrameBank, point cloud, renderers and every
hygiene / recall gate are reused untouched -- an A/B isolates the geometry model alone.

Two things do not transfer from DA3:

- **conf units.** ViGeo conf is raw logits (can be negative); DA3 conf is positive. `conf_transform="exp"`
  maps it positive and, being monotone, leaves percentile gates such as `conf_percentile` unaffected.
  Absolute thresholds (`hygiene_conf_abs`, `consist_conf_ref`, `self_reanchor_anchor_min_conf`) are
  calibrated in DA3 units; measured equivalents for DA3's 2.5 are ~0.17 (real footage) / ~0.084 (game).
- **focal length.** ViGeo self-estimates focal, so unprojecting with the GT K is a pinhole mismatch that
  varies per window -- the observed GT/ViGeo ratio spans ~0.78-1.13 over a rollout and crosses 1.0, so it
  is not a fixed bias. `EVOKE_VIGEO_DEBUG=1` logs it per window.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from worldfoundry import base_models

from .da3_cloud import _umeyama_scale_collinear_safe

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Model source has one canonical owner; checkpoint locations remain configurable.
_VIGEO_SRC = Path(base_models.__file__).resolve().parent / "three_dimensions/depth/vigeo"
_VIGEO_WEIGHTS = Path(os.environ.get("EVOKE_VIGEO_WEIGHTS", str(_REPO_ROOT / "models" / "ViGeo1.1")))

_PATCH = 14  # ViGeo.patch_size
# Tokens prepended to every frame's patch grid: 1 cls + num_register_tokens (4) + 1 camera, i.e.
# dinov2 patch_start_idx = 6. Getting this wrong under-sizes the kv-cache budget below.
_EXTRA_TOKENS = 6
# ViGeo does not split the cache budget evenly across global blocks: dinov2 weights it by
# softmax((1 - last_scores) / 0.5), so the weakest block receives a fraction of the even share. Measured
# worst case (share x num_global_blocks) once last_scores has diverged; used to size the budget against
# the weakest block rather than the average one.
_BUDGET_BLOCK_SKEW = 0.35
# Below this the weakest block's budget drops to about one frame, at which point eviction silently
# becomes a no-op (see _resolve_budget).
_MIN_KEEP_FRAMES = 4

# Exceptions that mean "the environment or the call is wrong", never "this window is geometrically
# degenerate". These must not be turned into a skipped window: doing so produces a run that completes
# with an all-black warp and a zero exit code.
_FATAL_EXC = (MemoryError, KeyError, AttributeError, ImportError, TypeError, ValueError)


def check_assets(src=None, weights=None) -> None:
    """Raise if the ViGeo source tree or weights are missing.

    Called from the config validator and the inference pre-flight check. Without it the first failure
    surfaces inside a per-window `except` and is reported as geometric degeneracy, so a run with a bad
    path produces an all-black warp and still exits successfully.
    """
    src = Path(src).expanduser().resolve() if src else _VIGEO_SRC
    weights = Path(weights) if weights else _VIGEO_WEIGHTS
    if src != _VIGEO_SRC:
        raise ValueError("Evoke uses WorldFoundry's shared ViGeo implementation; omit the ViGeo source override.")
    if not (src / "vigeo.py").is_file():
        raise FileNotFoundError(
            f"ViGeo source is missing from the shared base model at {src}.")
    if not (weights / "vigeo.pt").is_file():
        raise FileNotFoundError(
            f"ViGeo weights not found: {weights / 'vigeo.pt'} does not exist. Fetch the "
            f"pkqbajng/ViGeo1.1 snapshot or point EVOKE_VIGEO_WEIGHTS at it.")


def _fit_patch_res(h: int, w: int, long_side: int, patch: int = _PATCH):
    """Scale (h, w) to a ~long_side long edge with both sides multiples of `patch`. Mirrors DA3's
    `upper_bound_resize`, so ViGeo does not resize a second time internally."""
    s = float(long_side) / float(max(h, w))
    nh = max(patch, int(round(h * s / patch)) * patch)
    nw = max(patch, int(round(w * s / patch)) * patch)
    return nh, nw


class ViGeoDepthEstimator:
    """ViGeo-backed streaming depth estimator.

    Per-knob semantics live on `VigeoBackendConfig` in evoke/utils/train_config.py. Defaults here
    reproduce the validated recipe, so constructing with no arguments gives a measured configuration
    rather than an untested one.

    The instance is **stateful** while streaming (kv-cache, locked scale). Every stream boundary -- a new
    sample, a new rollout, a new video -- must call `reset_stream()`; the inference pipeline does this
    for their callers.
    """

    def __init__(self, device="cuda", process_res: int = 644,
                 src: Path = _VIGEO_SRC, weights: Path = _VIGEO_WEIGHTS,
                 num_tokens: Optional[int] = None, mode: str = "chunk",
                 chunk_size: int = 16, intr_source: str = "gt",
                 conf_transform: str = "exp", scale_mode: str = "anchor",
                 anchor_windows: int = 4, total_budget: int = 0,
                 cache_keep_frames: int = 6, scale_value: float = 0.0,
                 depth_median_target: float = 1.0):
        # Validated here too, not only in the config/CLI layers: a typo would otherwise fall through to a
        # silently different recipe (conf_transform="expo" leaves conf as raw logits, inverting every
        # absolute conf gate).
        if str(mode) not in ("offline", "chunk", "online"):
            raise ValueError(f"ViGeo mode must be offline/chunk/online, got {mode!r}")
        if str(scale_mode) not in ("per_window", "anchor", "depth_median", "fixed"):
            raise ValueError("ViGeo scale_mode must be per_window/anchor/depth_median/fixed, "
                             f"got {scale_mode!r}")
        # Caught here rather than at the first window: scale_mode=fixed with the 0.0 sentinel would divide
        # depth by zero and surface as inf points in the renderer, several seconds and one forward later.
        if str(scale_mode) == "fixed" and not (float(scale_value) > 0):
            raise ValueError(f"ViGeo scale_mode=fixed needs scale_value > 0, got {scale_value!r}")
        if not (float(depth_median_target) > 0):
            raise ValueError(f"ViGeo depth_median_target must be > 0, got {depth_median_target!r}")
        if str(intr_source) not in ("gt", "vigeo"):
            raise ValueError(f"ViGeo intr_source must be gt/vigeo, got {intr_source!r}")
        if str(conf_transform) not in ("exp", "none"):
            raise ValueError(f"ViGeo conf_transform must be exp/none, got {conf_transform!r}")
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.src = Path(src)
        self.weights = Path(weights)
        # 0 is the project-wide "auto" sentinel for this knob; passing it straight through would ask ViGeo
        # for a 1x1-patch encoder, whose 1-pixel depth map is then upsampled into flat garbage silently.
        self.num_tokens = int(num_tokens) if num_tokens else None
        self.mode = str(mode)
        self.chunk_size = int(chunk_size)
        self.intr_source = str(intr_source)
        self.conf_transform = str(conf_transform)
        self.scale_mode = str(scale_mode)
        self.scale_value = float(scale_value)
        self.depth_median_target = float(depth_median_target)
        self._skip_reason = None
        self.anchor_windows = int(anchor_windows)
        self.total_budget = int(total_budget)
        self.cache_keep_frames = int(cache_keep_frames)
        self._model = None
        self._dbg = bool(os.environ.get("EVOKE_VIGEO_DEBUG"))
        self._focal_ratios: list[float] = []   # diagnostic: GT focal / ViGeo focal
        self._kv = None
        self._win_seen = 0
        self._anchor_scales: list[float] = []
        self._scale_locked: Optional[float] = None
        self._parallax_warned = False        # the zero-parallax diagnostic fires once per stream

    @property
    def streaming(self) -> bool:
        return self.mode in ("chunk", "online")

    def _resolve_budget(self, h: int, w: int) -> int:
        """Convert "frames of context to keep" into ViGeo's `total_budget` (a token count).

        Trap: in `attention.eviction()` `num_to_keep = cache_budget - num_anchor_tokens` with
        `num_anchor_tokens = tokens_per_frame`. If the budget reaching a *single global block* is at most
        one frame's tokens, `num_to_keep <= 0` and the function returns `k, v` unchanged -- eviction is
        silently skipped and the cache grows without bound.

        Sizing therefore has to target the **weakest** block, not the average one: dinov2 distributes
        total_budget by `softmax((1 - last_scores) / 0.5)`, so once last_scores diverges (from the second
        cached forward onwards) the minimum block share is roughly `_BUDGET_BLOCK_SKEW` of the even
        split. `cache_keep_frames` is floored at `_MIN_KEEP_FRAMES` for the same reason -- lower values
        put the weakest block inside the no-eviction region even though the even-split arithmetic looks
        fine.
        """
        if self.total_budget > 0:
            return self.total_budget
        tpf = (h // _PATCH) * (w // _PATCH) + _EXTRA_TOKENS
        nblk = int(getattr(self._model.pretrained, "num_global_blocks", 0))
        if nblk <= 0:
            # A zero budget makes dinov2 disable eviction outright, so refuse rather than run unbounded.
            raise RuntimeError(
                f"ViGeo reports num_global_blocks={nblk}; cannot size the streaming kv-cache budget. "
                f"Use mode='offline' or set total_budget explicitly.")
        keep = max(_MIN_KEEP_FRAMES, self.cache_keep_frames)
        budget = nblk * keep * tpf
        if self._dbg:
            worst = int(budget * _BUDGET_BLOCK_SKEW / nblk)
            print(f"[vigeo] cache budget: {nblk} global blocks x {keep} frames x {tpf} tokens/frame "
                  f"= {budget}; weakest block ~{worst} tokens = {worst / tpf:.2f} frames "
                  f"(must exceed 1.0 or eviction silently stops)", flush=True)
        return budget

    def _invalidate_geometry(self, why: str):
        """Drop the kv-cache together with the depth scale derived from it.

        `_kv`, `_scale_locked` and `_anchor_scales` are one unit: the locked scale is only meaningful
        inside the coordinate frame of the cache it was solved in. Dropping the cache alone makes the next
        window start a fresh frame with a fresh arbitrary up-to-scale factor, which then gets divided by
        the stale lock -- a permanent depth bias with no further warning. Dropping all three forces a
        clean re-anchor instead. Holds for the baseline-free modes too: "depth_median" reads the median of
        a specific window's depth map, which is in that window's arbitrary units.
        """
        had = self._kv is not None or self._scale_locked is not None or bool(self._anchor_scales)
        self._kv = None
        self._scale_locked = None
        self._anchor_scales = []
        if had:
            if self.scale_mode == "fixed":
                how = "the fixed scale is unchanged (it does not depend on the cache)"
            elif self.scale_mode == "depth_median":
                how = "the depth-median scale will be re-derived on the next window"
            else:
                how = f"the depth scale will re-anchor over the next {max(1, self.anchor_windows)} window(s)"
            print(f"[vigeo] stream geometry invalidated ({why}); {how}", flush=True)

    def reset_stream(self):
        """Drop all per-stream state. Must be called at every stream boundary (new video / rollout)."""
        if self._dbg and self._focal_ratios:
            r = np.asarray(self._focal_ratios, dtype=np.float64)
            print(f"[vigeo] stream focal ratio (GT/ViGeo) over {r.size} windows: "
                  f"median={np.median(r):.3f} min={r.min():.3f} max={r.max():.3f}", flush=True)
        self._kv = None
        self._win_seen = 0
        self._anchor_scales = []
        self._scale_locked = None
        self._focal_ratios = []
        self._parallax_warned = False
        if self._model is not None:
            self._model.reset_cache_state()

    def _lazy(self):
        if self._model is not None:
            return
        check_assets(self.src, self.weights)
        from worldfoundry.base_models.three_dimensions.depth.vigeo import ViGeo
        self._model = ViGeo.from_pretrained(str(self.weights)).to(self.device).eval()
        print(f"[vigeo] loaded {self.weights} (mask_head={self._model.mask_head is not None}) "
              f"process_res={self.process_res} num_tokens={self.num_tokens} mode={self.mode} "
              f"scale_mode={self.scale_mode} intr_source={self.intr_source}", flush=True)

    def _infer_window(self, frames_rgb: np.ndarray):
        """One window forward. Assumes `_lazy()` has already run (the caller loads outside its `try`).

        frames_rgb [N,H,W,3] in [0,1] -> (depth [N,h,w] t, conf [N,h,w] t|None, c2w_pred [N,4,4] np,
        rgb [N,h,w,3] np, (h, w), f_pix [N]). Depth is still up-to-scale.
        """
        x = torch.as_tensor(np.ascontiguousarray(frames_rgb), dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).clamp(0, 1)                       # [N,3,H,W]
        N, _, H, W = x.shape
        if self.num_tokens is None:
            h, w = _fit_patch_res(H, W, self.process_res)
            if (h, w) != (H, W):
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.to(self.device)

        if self.streaming:
            # Holding kv_caches across windows makes ViGeo treat the rollout as one video, so all windows
            # share one coordinate frame and one scale. With mode=offline errors compound window by window.
            out = self._model.infer(
                x, mode=self.mode, chunk_size=self.chunk_size,
                num_tokens=self.num_tokens, resize_output=True,
                kv_caches=self._kv, reset_cache=(self._kv is None),
                total_budget=self._resolve_budget(int(x.shape[-2]), int(x.shape[-1])))
            # Indexed, not .get(): if ViGeo ever renames this the stream must fail loudly rather than
            # silently restart its coordinate frame every window.
            self._kv = out["kv_caches"]
        else:
            # chunk_size is ignored in offline/online (ViGeo derives step_size from the sequence length).
            out = self._model.infer(
                x, mode=self.mode, num_tokens=self.num_tokens, resize_output=True)

        depth = out["depth_pred"][:, 0].float()                     # [N,h,w]
        conf = out["conf_pred"]
        conf = None if conf is None else conf[:, 0].float()
        pose = out["pose_pred"].float().cpu().numpy()                # [N,3,4] c2w
        c2w_pred = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
        c2w_pred[:, :3, :4] = pose
        h, w = int(depth.shape[-2]), int(depth.shape[-1])
        rgb = x.permute(0, 2, 3, 1).float().cpu().numpy()            # [N,h,w,3], aligned with depth
        f_pix = self._vigeo_focal_pix(out["points_pred"], depth)
        return depth, conf, c2w_pred, rgb, (h, w), f_pix

    @staticmethod
    def _scale_K(K_gt: np.ndarray, src_hw, dst_hw) -> np.ndarray:
        """GT K [N,3,3] at the source resolution -> rescaled to (h, w), matching DA3's returned `ix`."""
        (H, W), (h, w) = src_hw, dst_hw
        sx, sy = float(w) / float(W), float(h) / float(H)
        K = np.asarray(K_gt, np.float32).copy()
        if K.ndim == 2:
            K = K[None]
        K[:, 0, 0] *= sx; K[:, 0, 2] *= sx
        K[:, 1, 1] *= sy; K[:, 1, 2] *= sy
        return K.astype(np.float32)

    @staticmethod
    def _vigeo_focal_pix(points: torch.Tensor, depth: torch.Tensor) -> np.ndarray:
        """Recover the pixel focal length [N] from ViGeo's points_pred [N,h,w,3] and depth [N,h,w].

        ViGeo's camera model (vigeo/utils.py) is x = (u / focal_norm) * z with u normalised by the
        half-diagonal, so f_pix = focal_norm * sqrt(h^2 + w^2) / 2 with fx = fy and a centred
        principal point. Estimated in least-squares form sum(u*z * x) / sum(x^2) to avoid x ~= 0.
        """
        N, h, w = int(depth.shape[0]), int(depth.shape[-2]), int(depth.shape[-1])
        aspect = w / h
        span_x = aspect / (1 + aspect ** 2) ** 0.5
        span_y = 1.0 / (1 + aspect ** 2) ** 0.5
        u = torch.linspace(-span_x * (w - 1) / w, span_x * (w - 1) / w, w, dtype=torch.float32)
        v = torch.linspace(-span_y * (h - 1) / h, span_y * (h - 1) / h, h, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")                 # [h,w]
        xy = points[..., :2].float()                                 # [N,h,w,2]
        z = depth.float().unsqueeze(-1)                              # [N,h,w,1]
        uz = torch.stack([uu, vv], -1).unsqueeze(0) * z              # [N,h,w,2] == f * xy in theory
        num = (uz * xy).flatten(1).sum(1)
        den = (xy * xy).flatten(1).sum(1).clamp_min(1e-8)
        focal_norm = (num / den)                                     # [N]
        diag = float((h ** 2 + w ** 2) ** 0.5)
        return (focal_norm * (diag / 2.0)).cpu().numpy().astype(np.float32)

    def _resolve_scale(self, c_raw: Optional[float]) -> Optional[float]:
        """Turn this window's raw Umeyama scale into the scale actually applied.

        A zero-baseline window (`c_raw is None`) is rejected first, whatever the lock state: it carries no
        solvable geometry and DA3 always skips it, so returning the locked scale would let ViGeo ingest
        windows DA3 rejects.

        Under "anchor", later windows are fed self-generated frames whose apparent camera motion already
        contains generation drift; re-solving the scale there writes that drift into depth, so the median
        of the first `anchor_windows` scales is locked and reused instead.
        """
        if c_raw is None:
            return None
        if self.scale_mode != "anchor":
            return c_raw
        if self._scale_locked is not None:
            return self._scale_locked
        self._anchor_scales.append(float(c_raw))
        if len(self._anchor_scales) >= max(1, self.anchor_windows):
            self._scale_locked = float(np.median(self._anchor_scales))
            print(f"[vigeo] scale anchored: scale={self._scale_locked:.4g} (median of the first "
                  f"{len(self._anchor_scales)} windows {np.round(self._anchor_scales, 4).tolist()}); "
                  f"reused for all later windows", flush=True)
            return self._scale_locked
        return c_raw

    def _warn_if_no_parallax(self, gt_centres, depth_scaled) -> None:
        """Say so, once, when the resolved scale has put the cloud so far away that the warp is a still image.

        Parallax is set by ONE ratio: how far the camera is commanded to travel across the window versus how
        deep the cloud sits. At baseline/depth ~ 1e-3 a whole window of commanded motion moves the
        reprojection by well under a pixel, so every frame renders identically -- the warp is a still image
        at coverage 1.000, the model faithfully continues it, and because later windows then measure a
        static clip the condition never recovers.

        This is worth its own line because the failure is otherwise SILENT and self-consistent: nothing
        raises, every window succeeds, and `anchor` announces the scale that causes it in the same cheerful
        tone as a good one ("scale anchored: scale=0.00607"). It cost a full debugging round
        to find out that a healthy-looking log meant a dead camera.
        Diagnostic only -- the run continues, because a legitimately near-static window looks the same and
        must not be turned into a crash.
        """
        if self._parallax_warned:
            return
        gt = np.asarray(gt_centres, np.float64)
        if gt.ndim != 2 or gt.shape[0] < 2:
            return
        baseline = float(np.linalg.norm(gt.max(0) - gt.min(0)))     # commanded travel across the window
        med = float(np.median(depth_scaled))
        if not np.isfinite(med) or med <= 0 or baseline <= 0:
            return
        ratio = baseline / med
        if ratio >= 1e-3:
            return
        self._parallax_warned = True
        print(f"[vigeo] WARN zero-parallax geometry: the window's commanded camera travel is {baseline:.4g} "
              f"but the scaled cloud's median depth is {med:.4g} (ratio {ratio:.2e} < 1e-3), so the warp "
              f"will render as a STILL IMAGE (coverage ~1.000) and the model will have no camera signal. "
              f"scale_mode={self.scale_mode}"
              + (" -- anchor/per_window derive the scale as a ratio against the camera motion ViGeo reads "
                 "out of the frames, which collapses when those frames are near-static (i2v chunk 0 has no "
                 "real frames at all); use scale_mode=depth_median, which cannot collapse."
                 if self.scale_mode in ("anchor", "per_window")
                 else " -- check depth_median_target / scale_value against the pose track's units."),
              flush=True)

    def _resolve_scale_baseline_free(self, depth_t) -> float:
        """The scale for the two modes that never look at the camera baseline.

        Why these exist. `per_window` / `anchor` both solve `c = umeyama(GT centres, ViGeo centres)` and
        divide depth by it, which ties the cloud's size to *how much camera motion ViGeo reads out of the
        frames it was given*. Two trajectory families break that:

          * zero commanded translation (a pure pan/orbit-in-place): no baseline at all, `c_raw is None`,
            every window is skipped and the cloud stays empty -- the warp is black for the whole rollout.
          * a first chunk that barely moves: chunk 0 has no warp yet (nothing to reproject from), so its
            content is driven by image+text alone. If it comes out near-static, ViGeo correctly reports
            ~no camera motion, `c` collapses, depth is divided by ~1e-4, the cloud lands at infinity and
            the warp becomes a still image -- which the model then faithfully continues. Self-locking.

        Both disappear once the scale stops being a *ratio* -- i.e. unproject the depth map as-is and
        never derive a scale from camera centres. ViGeo's depth is up-to-scale (`_remap`:
        z = logits.clamp(-10,10).exp()), not metric like DA3's, so instead we fix the *one* free
        scalar directly:

          "depth_median"  c = median(depth) / depth_median_target, solved on the first window and locked.
                          After the division the cloud's median depth is `depth_median_target`, so the
                          commanded trajectory is read in units of scene depth: translating by 1.0 moves
                          the camera one median-scene-depth. This is the mode that makes a benchmark's
                          per-model translation knob (WorldScore's `camera_speed`) mean something --
                          under `anchor` it is algebraically a no-op, because scaling the commanded
                          motion by k scales `c` by k and the resolved depth by k, leaving parallax
                          unchanged.
          "fixed"         c = scale_value verbatim. For reproducing one specific run, or when the caller
                          already knows the conversion factor.

        Locked on the first window in both cases: one rollout shares one coordinate frame, so it needs
        exactly one scalar, and re-solving per window would reintroduce the drift `anchor` exists to avoid.
        """
        if self._scale_locked is not None:
            return self._scale_locked
        if self.scale_mode == "fixed":
            self._scale_locked = float(self.scale_value)
            print(f"[vigeo] scale fixed: scale={self._scale_locked:.4g} (baseline-free; "
                  f"no Umeyama solve)", flush=True)
            return self._scale_locked
        med = float(torch.median(depth_t.detach().float()).item())
        if not np.isfinite(med) or med <= 0:
            raise ValueError(f"ViGeo depth median is {med!r}; cannot set a depth_median scale")
        self._scale_locked = med / float(self.depth_median_target)
        print(f"[vigeo] scale from depth median: scale={self._scale_locked:.4g} "
              f"(raw median {med:.4g} -> target {self.depth_median_target:g}; baseline-free, "
              f"locked for the rollout)", flush=True)
        return self._scale_locked

    @torch.no_grad()
    def depth_single(self, frame_rgb, K_gt):
        """Single-frame depth for the i2v chunk-0 warp (no camera baseline available).

        frame_rgb [H,W,3] in [0,1] + K_gt [3,3] (source resolution)
        -> (depth [h,w] np at GT scale, intr [3,3] np at depth res, rgb [h,w,3] np in [0,1]).

        Only the baseline-free scale modes can do this: depth_median / fixed set the scale from the depth
        map itself, so one frame is enough. per_window / anchor solve a Umeyama scale from >=3 posed frames
        and cannot -> NotImplementedError, and build_single_source_warp_mono falls back to a blank warp.

        ISOLATION: the i2v chunk-0 warp is rendered before any rollout ingest, so the stream is clean when
        this runs; it still reset_stream()s before AND after so this one reference-frame forward never
        leaks its kv-cache or its locked depth-scale into the rollout's own ingest, which must own them
        (the rollout re-anchors depth_median on its first generated-frame window, exactly as in training).
        """
        if self.scale_mode not in ("depth_median", "fixed"):
            raise NotImplementedError(
                f"depth_single needs scale_mode depth_median/fixed (baseline-free), got {self.scale_mode!r}")
        fr = np.asarray(frame_rgb, np.float32)
        if fr.ndim != 3 or fr.shape[2] != 3:
            raise ValueError(f"depth_single expects [H,W,3], got {fr.shape}")
        H, W = int(fr.shape[0]), int(fr.shape[1])
        self._lazy()
        self.reset_stream()                                  # clean start (defensive; stream is already clean here)
        try:
            depth_t, _conf_t, _c2w_pred, rgb, (h, w), f_pix = self._infer_window(fr[None])
            c = self._resolve_scale_baseline_free(depth_t)   # depth_median -> median/target; fixed -> scale_value
            depth = (depth_t / float(c))[0].cpu().numpy().astype(np.float32)     # [h,w] at GT scale
            intr = self._scale_K(K_gt, (H, W), (h, w))[0]                        # [3,3] at depth res
            if self.intr_source == "vigeo":                  # ablation parity with depth_windows_batched
                intr = np.eye(3, dtype=np.float32)
                intr[0, 0] = float(f_pix[0]); intr[1, 1] = float(f_pix[0])
                intr[0, 2] = (w - 1) / 2.0;   intr[1, 2] = (h - 1) / 2.0
            return depth, intr.astype(np.float32), np.asarray(rgb[0], np.float32)
        finally:
            self.reset_stream()                              # drop this forward's kv/scale; rollout ingest re-anchors

    @torch.no_grad()
    def depth_window(self, frames_rgb, c2w_gt, K_gt):
        """frames_rgb [K,H,W,3] in [0,1] + c2w_gt [K,4,4] + K_gt [K,3,3] (source resolution)
        -> (depth [K,h,w] np, intr [K,3,3] np, conf [K,h,w] np|None, rgb [K,h,w,3] np).

        Same constraints as DA3: K >= 3 (a baseline is needed to solve the scale); a zero baseline
        (truly static) raises RuntimeError, which `FrameBank.ingest` catches and turns into a skipped
        chunk, preserving the existing blank-warp behaviour.
        """
        frames_rgb = np.asarray(frames_rgb)
        K = int(frames_rgb.shape[0])
        if K < 3:
            raise ValueError(f"ViGeo GT-pose needs >=3 frames per call (Umeyama scale), got {K}")
        self._skip_reason = None
        deps, intrs, confs, rgbs = self.depth_windows_batched(
            [frames_rgb], [np.asarray(c2w_gt)], [np.asarray(K_gt)])
        if deps[0] is None:
            # Report the actual reason: claiming "zero baseline" for every skip hid real failures behind
            # a geometric-sounding message.
            raise RuntimeError(f"ViGeo depth_window produced no usable window "
                               f"({self._skip_reason or 'reason not recorded'})")
        return (deps[0], intrs[0], confs[0], rgbs[0])

    @torch.no_grad()
    def depth_windows_batched(self, windows_rgb, windows_c2w, windows_K):
        """B windows -> (depths, intrs, confs, rgbs), each a list[B]; entries are None for skipped windows.

        Windows run one at a time: each needs its own scale solve (as with DA3), and while streaming
        consecutive windows must continue the same kv-cache.
        """
        B = len(windows_rgb)
        if B == 0:
            return [], [], [], []
        # Load outside the per-window try, matching DA3: a missing source tree, missing weights or a
        # CUDA OOM must keep its own exception type instead of being reported as a degenerate window.
        self._lazy()
        depths, intrs, confs, rgbs = [], [], [], []
        for b in range(B):
            fr = np.asarray(windows_rgb[b])
            N, H, W = int(fr.shape[0]), int(fr.shape[1]), int(fr.shape[2])
            if N < 3:
                raise ValueError(f"ViGeo GT-pose needs >=3 frames per window, window {b} got {N}")
            try:
                depth_t, conf_t, c2w_pred, rgb, (h, w), f_pix = self._infer_window(fr)
                # Scale alignment: src = GT centres, dst = ViGeo predicted centres, dst ~= c*R*src + t
                # so depth /= c -- the same convention and direction as DA3's _align_pred_robust.
                gt_c = np.asarray(windows_c2w[b], np.float64)[:, :3, 3]
                pr_c = c2w_pred[:, :3, 3].astype(np.float64)
                # Kept even in the baseline-free modes: it costs one SVD on N 3-vectors and it is the
                # single most useful diagnostic here (it is the ratio that collapses), so EVOKE_VIGEO_DEBUG
                # can still report what the solve *would* have said.
                c_raw = _umeyama_scale_collinear_safe(gt_c, pr_c)
                if self.scale_mode in ("depth_median", "fixed"):
                    c = self._resolve_scale_baseline_free(depth_t)
                else:
                    c = self._resolve_scale(c_raw)
                if c is None:
                    # Truly static / zero baseline: no solvable scale, so nothing can enter the cloud.
                    # The forward itself succeeded, so the cache stays valid (ViGeo legitimately saw these
                    # frames) -- but say so, otherwise a static clip yields a black warp in total silence.
                    self._skip_reason = "zero baseline (static or degenerate camera motion)"
                    print(f"[vigeo] window {self._win_seen + 1} skipped: {self._skip_reason}; "
                          f"no points ingested", flush=True)
                    depths.append(None); intrs.append(None); confs.append(None); rgbs.append(None)
                    continue
                depth = (depth_t / float(c)).cpu().numpy().astype(np.float32)   # [N,h,w] at GT scale
                self._warn_if_no_parallax(gt_c, depth)
                intr = self._scale_K(windows_K[b], (H, W), (h, w))
                if intr.shape[0] == 1 and N > 1:
                    intr = np.repeat(intr, N, axis=0)   # callers may pass one shared K for the window
                f_gt = float(np.mean(intr[:, 0, 0]))
                f_vg = float(np.mean(f_pix))
                self._focal_ratios.append(f_gt / max(f_vg, 1e-6))
                if self.intr_source == "vigeo":     # ablation: ViGeo focal, fx=fy, centred principal point
                    intr = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
                    intr[:, 0, 0] = f_pix; intr[:, 1, 1] = f_pix
                    intr[:, 0, 2] = (w - 1) / 2.0; intr[:, 1, 2] = (h - 1) / 2.0
                if conf_t is None:
                    conf = None
                elif self.conf_transform == "exp":
                    # clamp guards against fp32 overflow in exp, not a semantic confidence ceiling
                    conf = conf_t.clamp(-30, 30).exp().cpu().numpy().astype(np.float32)
                else:
                    conf = conf_t.cpu().numpy().astype(np.float32)
                self._win_seen += 1
                if self._dbg:
                    print(f"[vigeo] win#{self._win_seen} N={N} {H}x{W}->{h}x{w} "
                          f"scale_raw={('%.4g' % c_raw) if c_raw is not None else 'None'} scale_used={c:.4g}"
                          f"{'(locked)' if self._scale_locked is not None else ''} "
                          f"stream={'on' if self.streaming else 'off'} "
                          f"depth_med={float(np.median(depth)):.3f} f_gt={f_gt:.1f} "
                          f"f_vigeo={f_vg:.1f} ratio={f_gt / max(f_vg, 1e-6):.3f}", flush=True)
                depths.append(depth); intrs.append(intr); confs.append(conf); rgbs.append(rgb)
            except _FATAL_EXC:
                raise
            except torch.cuda.OutOfMemoryError:
                raise                       # the cache is the likely cause, so every later window fails too
            except Exception as _e:
                # Anything left is treated as this window failing. The cache no longer matches the frames
                # actually ingested, so the whole geometry unit goes with it.
                self._skip_reason = f"{type(_e).__name__}: {_e}"
                self._invalidate_geometry(f"window {self._win_seen + 1} failed")
                print(f"[vigeo] WARN stream window {self._win_seen + 1} (batch item {b}) failed "
                      f"({self._skip_reason}); skipping", flush=True)
                depths.append(None); intrs.append(None); confs.append(None); rgbs.append(None)
        return depths, intrs, confs, rgbs
