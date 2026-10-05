"""Point-cloud memory and warp rendering for Evoke's ViGeo inference backend.

External poses are camera-to-world. Unprojection uses depth-resolution intrinsics;
rendering uses intrinsics scaled to the output resolution. Rendered RGB is in
[-1, 1], with a separate visibility mask.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def unproject_depth_torch(depth: torch.Tensor, intr: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """depth [H,W] + intr [3,3] (at depth resolution) + c2w [4,4] -> world points [H,W,3]. OpenCV pinhole, GPU."""
    h, w = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ys, xs = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32),
                            torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
    z = depth.float()
    cam = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=-1)   # [H,W,3] camera frame
    R = c2w[:3, :3].float(); t = c2w[:3, 3].float()
    return cam @ R.T + t                                                      # [H,W,3] world frame


# --------------------- scale fallback for collinear / straight-line degenerate windows ---------------------
#   background: DA3 solves the depth scale with evo Umeyama Sim(3) (align_poses_umeyama), which rank-checks [collinear
#   camera centres] (= pure straight-line/forward motion, no lateral shift, no turning) and raises GeometryException -> no depth for the whole window -> all-black warp.
#   but the scale component is well defined for collinear motion (the ratio of trajectory lengths along the motion direction); the throw is only caused by the rank-deficient rotation SVD.
#   fallback: on a throw, solve the Umeyama [scale] directly with SVD (no rank check -> no throw), with the same convention and direction as evo
#   (src=GT trajectory -> dst=DA3 predicted trajectory, scale c such that dst ~= c*R*src+t, then depth/=c).
#   truly static (zero baseline, var_src ~= 0) -> return None -> keep the original "blank warp" behaviour (a static camera is a near-identity warp anyway).


def _umeyama_scale_collinear_safe(src_centers, dst_centers):
    """the [scale component] of Umeyama Sim(3), solved directly by SVD (no rank check -> collinear / straight-line motion does not throw).
    same convention as evo align_poses_umeyama(ext_ref=pred, ext_est=GT): src=GT camera centres -> dst=pred camera centres,
    c = sum(sigma_i(Sigma_dst,src)) / var_src. returns the scalar c; zero baseline (truly static) -> None."""
    src = np.asarray(src_centers, np.float64); dst = np.asarray(dst_centers, np.float64)
    n = int(src.shape[0])
    if n < 2:
        return None
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s; dc = dst - mu_d
    var_s = float((sc ** 2).sum() / n)
    if not np.isfinite(var_s) or var_s < 1e-12:                # zero baseline -> no scale (static)
        return None
    Sigma = (dc.T @ sc) / n                                    # 3x3 cross-cov (dst,src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:               # keep a right-handed frame (consistent with Umeyama)
        S[2, 2] = -1.0
    c = float((D * np.diag(S)).sum() / var_s)
    if not np.isfinite(c) or c <= 1e-9:
        return None
    return c


# ------------------------- DA3 depth estimation (GT poses) -------------------------


# ------------------------- resident GPU persistent cloud -------------------------


# ------------------------- batched GPU rasterisation render -------------------------
@torch.no_grad()
def render_cloud_batched(xyz, rgb, c2w_targets, K_render, height: int, width: int, *,
                         device="cuda", splat_radius: int = 2, invisible_fill: str = "black"):
    """point cloud -> warp for F views. project all views at once + scatter z-buffer (int-packed depth keeps the nearest point).
    xyz [P,3], rgb [P,3] in [0,1] (world frame); c2w_targets [F,4,4]; K_render [F,3,3] (render resolution, F1).
    returns warp [1,3,F,H,W] in [-1,1] (F2) + vis [1,1,F,H,W] in {0,1}."""
    device = torch.device(device)
    P = torch.as_tensor(np.asarray(xyz) if not torch.is_tensor(xyz) else xyz,
                        dtype=torch.float32, device=device).reshape(-1, 3)
    C = torch.as_tensor(np.asarray(rgb) if not torch.is_tensor(rgb) else rgb,
                        dtype=torch.float32, device=device).reshape(-1, 3)
    npt = P.shape[0]
    # the int packing stores the point index in 24 bits -> the point count must be < 2^24 (otherwise idx overflows into the depth high bits and renders silently wrong).
    # exceeding it means the caller must bound the cloud point count before rendering.
    assert npt < (1 << 24), (
        f"cloud has {npt} points >= 2^24, over the render int-pack limit -> bound the point count with voxel bound (see R6)")
    c2w = torch.as_tensor(np.asarray(c2w_targets) if not torch.is_tensor(c2w_targets) else c2w_targets,
                          dtype=torch.float32, device=device)
    Kt = torch.as_tensor(np.asarray(K_render) if not torch.is_tensor(K_render) else K_render,
                         dtype=torch.float32, device=device)
    V = c2w.shape[0]; HW = height * width
    # invisible fill colour (F2)
    if invisible_fill == "mean" and npt > 0:
        fill = C.mean(0)
    else:
        fill = torch.zeros(3, device=device)                                     # black -> -1
    if npt == 0:
        warp = (fill[None, :, None, None, None].expand(1, 3, V, height, width) * 2 - 1)
        return warp.contiguous(), torch.zeros((1, 1, V, height, width), device=device)
    R = c2w[:, :3, :3]; t = c2w[:, :3, 3]
    cam = torch.einsum("vij,vpj->vpi", R.transpose(1, 2), P[None] - t[:, None])   # x_cam=R^T(X-t) [V,P,3]
    z = cam[..., 2]
    fx = Kt[:, 0, 0][:, None]; fy = Kt[:, 1, 1][:, None]
    cx = Kt[:, 0, 2][:, None]; cy = Kt[:, 1, 2][:, None]
    px = torch.round((cam[..., 0] / z) * fx + cx).long()
    py = torch.round((cam[..., 1] / z) * fy + cy).long()
    front = z > 1e-4
    zq = (z.clamp(0, 1e6) * 1000.0).long().clamp(0, (1 << 38) - 1)                # quantised depth (small=near)
    idx = torch.arange(npt, device=device)[None].expand(V, npt)
    vidx = torch.arange(V, device=device)[:, None].expand(V, npt)
    INF = torch.full((V * HW,), (1 << 62), dtype=torch.long, device=device)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            qx = px + dx; qy = py + dy
            ok = front & (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            key = (vidx * HW + qy * width + qx)[ok]
            packed = (zq[ok] << 24) | idx[ok]                                    # amin keeps the nearest point
            INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
    valid = INF < (1 << 62)
    win = (INF & ((1 << 24) - 1)).clamp(max=npt - 1)
    img01 = fill[None].expand(V * HW, 3).clone()
    img01[valid] = C[win[valid]]
    img = img01.reshape(V, height, width, 3).clamp(0, 1)
    warp = (img.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)                     # [1,3,V,H,W] in [-1,1](F2)
    vis = valid.reshape(V, height, width).float()[None, None]                    # [1,1,V,H,W]
    return warp.contiguous(), vis.contiguous()


# --------------------- FrameBank recall-style warp (design) ---------------------
# replaces dump-all-history fusion (cross-chunk scale drift turns into salt-and-pepper): each chunk estimates depth and is ingested [independently], and when generating chunk k a few frames are recalled by viewpoint and rendered.


@torch.no_grad()
def _cloud_depth_map(xyz, c2w, K, height, width, *, device):
    """world points xyz[P,3] -> the [per-pixel nearest-point camera-z depth map] [H,W] of camera (c2w[4,4], K[3,3])
    (+inf where there is no point). serves as the reference for the "ingest consistency gate": the geometric depth the cloud predicts for this view. pure geometry, no model."""
    xyz = xyz if torch.is_tensor(xyz) else torch.as_tensor(xyz, dtype=torch.float32, device=device)
    out = torch.full((height * width,), float("inf"), device=device, dtype=torch.float32)
    if xyz.shape[0] == 0:
        return out.view(height, width)
    R = c2w[:3, :3]; t = c2w[:3, 3]
    cam = (xyz - t[None]) @ R                                    # R^T (X - t) [P,3]
    z = cam[:, 2]
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    px = torch.round(cam[:, 0] / z.clamp(min=1e-6) * fx + cx).long()
    py = torch.round(cam[:, 1] / z.clamp(min=1e-6) * fy + cy).long()
    ok = (z > 1e-4) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if ok.any():
        out.scatter_reduce_(0, py[ok] * width + px[ok], z[ok], reduce="amin", include_self=True)
    return out.view(height, width)


class DA3FrameBank:
    """per-frame world point-cloud bank (GT-pose + DA3 depth). each chunk is ingested independently (NAIVE); on recall only the hit frames are taken -> render.
    different from Pi3X's `frame_bank.FrameBank` (hence the DA3* name); the two can coexist."""

    def __init__(self, device="cuda", conf_percentile=30.0,
                 cloud_hygiene=False, hygiene_sat_max=1.0,
                 hygiene_flat_std=0.0, hygiene_flat_win=7,
                 hygiene_conf_pct=0.0, hygiene_conf_abs=0.0,
                 consist_gate=False, consist_tau=0.15,
                 consist_ref_frames=24, consist_min_ref=2000,
                 consist_adaptive=False, consist_tau_lo=0.20, consist_tau_hi=0.30, consist_conf_ref=12.0,
                 consist_probation=0, consist_probation_frac=0.0, consist_probation_win=25,
                 consist_scale_align=False,
                 color_anchor=False, color_anchor_alpha=0.5, color_anchor_ref_windows=4,
                 self_reanchor=False, self_reanchor_scale_thr=1.5, self_reanchor_rej_thr=0.8,
                 self_reanchor_min_gap=8, self_reanchor_keep_windows=3, self_reanchor_lookback=20,
                 self_reanchor_anchor_min_conf=0.0,
                 self_reanchor_pin_prime=False, self_reanchor_pin_windows=4,
                 hist_max_frames=0, reanchor_every=0, reanchor_keep_frames=0,
                 carve=False, carve_margin=0.10, carve_ref_frames=24, carve_min_views=1,
                 carve_strike_windows=1):
        self.device = torch.device(device)
        self.conf_pct = float(conf_percentile)
        # [cloud hygiene] master switch + params (off by default -> the whole block inside ingest is skipped, behaviour byte-identical)
        self.hygiene = bool(cloud_hygiene)
        # NOTE primary gate: absolute confidence. real content has high DA3 conf (~11-14, including vividly coloured objects) -> kept; degenerate flood collapses (~1-3) -> dropped.
        #   differs from conf_percentile (per-frame relative): when a whole frame collapses the percentile threshold collapses with it and cannot filter flood; only an absolute threshold can. <=0 disables.
        self.hyg_conf_abs = float(hygiene_conf_abs)
        self.hyg_sat_max = float(hygiene_sat_max)       # (optional, off by default) HSV saturation ceiling, <=0 or >=1 disables
        self.hyg_flat_std = float(hygiene_flat_std)     # (optional, off by default, use with care: also hits genuinely flat regions DA3 handles well) drop points whose local grey std < this, <=0 disables
        self.hyg_flat_win = int(hygiene_flat_win)       # local texture window (odd)
        self.hyg_conf_pct = float(hygiene_conf_pct)     # (weak, per-frame relative, useless when the whole frame collapses) confidence percentile gate, <=0 disables
        # === [method 1] ingest consistency gate (independent switch, off by default) -- the principled fix, independent of colour / shape / scale ===
        #   reproject the [established cloud] into the new frame view to get a "cloud-predicted depth" and compare it with the new frame DA3 depth: pixels whose relative
        #   deviation > tau are judged "hallucination / poisoning conflicting with known geometry" (a wall, a blue flood, any colour) -> not ingested. holes (not covered by the cloud) cannot be checked -> passed through
        #   (moving forward necessarily has to fill in new foreground). too few cloud points early on (<min_ref) -> not enabled (cold-start guard). independent of hygiene.
        self.consist_gate = bool(consist_gate)
        self.consist_tau = float(consist_tau)              # relative depth tolerance |d-d_cloud|/d_cloud; smaller is stricter, typically 0.1-0.2
        self.consist_ref_frames = int(consist_ref_frames)  # how many of the most recent frames provide the reference geometry (0=all)
        self.consist_min_ref = int(consist_min_ref)        # fewer reference points than this -> not enabled this time
        # - [variant 1] confidence-adaptive tolerance (removes the magic fixed tau; independent switch, off by default = keep the fixed consist_tau) -
        #   WARNING: the direction is the [opposite] of standard Mahalanobis: in this domain the poisoning (blue flood / render surfaces seen through walls) has collapsed DA3 conf (~1-3, flat / over-saturated) and is a [systematic error]
        #   rather than unbiased noise -> low conf must [tighten] (->tau_lo) to block it; high conf is trustworthy new real geometry (a new building round the corner) -> [loosen] (->tau_hi) to avoid salt-and-pepper.
        #   per-pixel tau = tau_lo + (tau_hi-tau_lo)*clamp(conf/conf_ref, 0, 1); large deviations (>tau_hi*d_cloud) are always deleted (wall / flood).
        #   WARNING: tau_lo=0.20 = the validated safe floor (lo=0.10 pushes late low-conf whole frames down to tau_eff~0.12 = the over-deletion zone -> coverage collapses -> blur);
        # adaptive only loosens [above] that floor (high conf -> tau_hi=0.30 to avoid salt-and-pepper), never below.
        self.consist_adaptive = bool(consist_adaptive)
        self.consist_tau_lo = float(consist_tau_lo)
        self.consist_tau_hi = float(consist_tau_hi)
        self.consist_conf_ref = float(consist_conf_ref)
        # - (independent switch, default 0=off=legacy behaviour "holes pass straight through") -- closes the "hallucination fast lane" -
        #   structural hole in M1: hole pixels with no cloud reference (_have=False) cannot be checked -> the legacy behaviour admits them unconditionally, so the hallucination the model
        # generates inside a hole is ingested and stays forever (the monotonic-degradation entry point). once on: those pixels
        #   are held in _probation for one round (not ingested now), and at the next ingest they are geometrically voted on against the new window raw DA3 depth:
        #   consistent (>=1 vote and >= conflicting votes) -> admitted and merged back; more conflicts -> dropped permanently; not visible at all -> admitted (benefit of the doubt, so they cannot starve).
        #   only delays by 1 chunk, genuinely new content still gets in (one round later). only held when consist_gate is on and the reference geometry is valid (cold start passes through).
        self.consist_probation = int(consist_probation)
        # - with frac>0 only pixels in the [interior of large holes] are held: held only when the local hole fraction over a win x win window > frac,
        # small holes / hole boundaries (new content when moving forward is mostly narrow strips) pass straight through -> greatly reduces the v1 coverage tax (v1 warp black holes x2-5).
        #   hallucination lives in the interior of large unknown regions (wall / flood); narrow strips of new content are almost never hallucination. frac=0 = v1 behaviour (hold everything).
        self.consist_probation_frac = float(consist_probation_frac)
        self.consist_probation_win = max(3, int(consist_probation_win) | 1)
        self._probation = {}   # gid:int -> dict(mask=bool[h,w], depth=[h,w] (values only where held), rgb=[h,w,3], it, cw); only the most recent round is stored
        # === [method 2] bounded / periodically re-anchored history (independent switch, off by default) -- lets old poisoning age out instead of staying forever ===
        #   hist_max_frames: sliding window, keep only this many recent frames (in gid units, 0=off);
        #   reanchor_every>0: hard re-anchor every N ingests, keeping only the most recent reanchor_keep_frames frames (0=off).
        #   WARNING: the window must be > lag*stride, otherwise frames render needs get evicted -> empty warp.
        self.hist_max_frames = int(hist_max_frames)
        self.reanchor_every = int(reanchor_every)
        self.reanchor_keep_frames = int(reanchor_keep_frames)
        self._ingest_calls = 0
        # === [method 3 free-space carving] (independent switch, off by default) -- the real "correction" ===
        #   after each ingest of a batch of new frames, use the (already gated) depth of the new frames to "see through" the old-frame cloud: if an old point falls in
        #   the free space between "the camera and the newly observed surface" (the old point is closer than the new surface by > carve_margin) -> that point hangs where space should be empty
        #   -> delete it (set the old frame dense depth to 0 + recompute the sparse points). clears floating poison (the fake near surface of a blue flood) + ghosts of objects that moved away.
        #   complementary to M1: M1 uses "depth inconsistency" to block [new points entering]; carve uses "the line of sight sees through = free space" to delete [old points].
        self.carve = bool(carve)
        self.carve_margin = float(carve_margin)       # how much closer (relatively) an old point must be than the new observed surface to count as free space -> delete; larger is more conservative
        self.carve_ref_frames = int(carve_ref_frames) # how many recent old frames at most to carve back through per step (cost control, 0=all)
        # - [variant 2] multi-view voting (cures single-frame false deletion / see-through-foliage teleport; default 1 = the old "delete if any single frame sees through" OR behaviour) -
        #   an old point must be judged "seen through (free space)" by >=carve_min_views new frames before deletion; 3-4 recommended. on-policy frame DA3 depths jitter
        #   independently -> a single-frame false trigger cannot gather enough votes -> only genuinely free space confirmed by several frames is deleted, dropping deleted% from 20-50% to single digits.
        self.carve_min_views = max(1, int(carve_min_views))
        # - (independent switch, default 1=off=legacy "delete within the window" behaviour) -- cures window-level systematic over-deletion -
        #   the 12 frames of one ingest window share a single DA3 forward pass + a single Umeyama scale -> carve votes are correlated within the window,
        # and min_views cannot suppress the systematic error of "the whole window seeing through itself" (55-85% over-deletion late in the rollout). once on (>1):
        #   a pixel satisfying the deletion criterion (votes>=min_views) only records one strike and must be confirmed in [carve_strike_windows consecutive windows]
        #   before it is really deleted; any unconfirmed window -> the strike count resets. DA3 forward passes / scales are independent across windows -> the votes are independent.
        self.carve_strike_windows = max(1, int(carve_strike_windows))
        self._carve_strike = {}  # gid:int -> int16[h,w] consecutive-confirmation count (only used when strike_windows>1)
        # === (independent switch, off by default) -- cures the compounding drift of "one Umeyama scale per window" ===
        #   measured ([consist-dbg] scale_ratio_med): the new window depth is systematically 1.05-1.18 relative to the cloud-predicted depth and always >1 = compounding inflation,
        # eating most of M1's tau budget. once on: before each window ingest, solve the scale s of this window relative to the cloud from median(d/d_cloud) over the
        #   overlap region, then depth/=s for the whole window before gating / ingesting -> the new window is anchored back to the cloud scale (the cloud is anchored to the GT prime),
        #   and from then on tau measures real geometric conflict rather than scale drift. guards: too little overlap (<2000px) or an absurd s (not in [0.5,2]) -> 1.0 (no-op);
        #   s is clipped to [0.75,1.33] (measured drift 1.05-1.18, leaving margin).
        self.consist_scale_align = bool(consist_scale_align)
        # === (independent switch, off by default; does not depend on consist_gate) -- cures "over-saturated / over-sharpened seeds" ===
        # the distilled student pred drifts in per-channel statistics from chunk0 on (saturation ratio 1.3-1.5x / sharpness 1.3-2.4x vs GT),
        #   and DA3 conf collapses on that "fake texture" input (1.7-5 vs 11-14 for real content) -> wrong depth -> the seed of cloud poisoning (still present under oracle).
        #   this feature linearly re-anchors the per-channel statistics of the decoded frames at the [DA3 ingest entry] to the "conditioning prefix (GT prime) statistics":
        #   the first color_anchor_ref_windows ingests = the reference period (for v2v the cold start is fed GT prime frames) -> accumulate channel mean/std
        #   as the reference, with [no correction] during the reference period; afterwards each window uses whole-window statistics x'=(x-mu_win)/max(sigma_win,eps)*sigma_ref+mu_ref, then
        #   x_out=alpha*x'+(1-alpha)*x, clip[0,1]. window statistics are computed over the whole window (not per frame) to prevent inter-frame flicker. only affects the DA3 input + cloud
        #   RGB (which inherits the rgb DA3 returns), never the model output; only uses past information (legal at inference, unlike using GT for the same time span, which does not exist at inference).
        self.color_anchor = bool(color_anchor)
        self.color_anchor_alpha = float(color_anchor_alpha)
        self.color_anchor_ref_windows = max(0, int(color_anchor_ref_windows))
        self._ca_ref = None          # (mu[3], sigma[3]) reference channel statistics (frozen once the reference period has accumulated)
        self._ca_seen = 0            # number of ingest windows processed (< ref_windows = the reference period)
        self._ca_ref_acc = []        # per-window (mu[3], sigma[3]) cache during the reference period (averaged into the reference)
        # === [self re-anchor] (independent switch, off by default) -- the legitimate oracle: anchor to [the best-quality window in our own history] ===
        # the oracle proves "a clean cloud -> pred keeps its quality all the way"; while the window scale ratio blows up to 2-3.1 in the mid / late rollout
        #   = the cloud and pred/DA3 have diverged at content level (the cloud is a write-off), and entry filtering (M1 / probation) cannot rescue the existing stock. once divergence is
        #   detected (current window scale_ratio>thr or rejection rate>rej_thr), this feature hard re-anchors (evicts) the cloud to the [best quality] (high conf, scale~1, low rejection)
        #   keep_windows consecutive windows within the last lookback ingest windows -> the warp is re-rendered from "our own best past" and the model is pulled back to that state to move on again.
        #   only uses past self-produced information, so it is legal at inference (unlike the oracle, which uses GT).
        self.self_reanchor = bool(self_reanchor)
        self.sr_scale_thr = float(self_reanchor_scale_thr)     # trigger when the current window median(d/d_cloud) falls outside [1/thr, thr]
        self.sr_rej_thr = float(self_reanchor_rej_thr)         # or trigger when the rejection rate over the checkable region > this
        self.sr_min_gap = int(self_reanchor_min_gap)           # minimum gap between two re-anchors (in ingests; anti-oscillation)
        self.sr_keep_windows = max(1, int(self_reanchor_keep_windows))  # keep the anchor window plus a few windows before it (geometric continuity)
        self.sr_lookback = max(2, int(self_reanchor_lookback)) # lookback range for anchor candidates (in ingest windows)
        # [v3] minimum anchor quality: a candidate window must have conf >= this (0=off=v1 behaviour), otherwise skip the re-anchor (keep the current cloud = m1 behaviour)
        #   rather than anchor to garbage. lesson from gen3-A6mW: the 3rd trigger anchored to a conf=1.53 junk window -> swapped in a ghost-polluted cloud -> the ghosts came back.
        self.sr_anchor_min_conf = float(self_reanchor_anchor_min_conf)
        # [v4] pin the prime anchor: the first pin_windows ingests (the v2v GT prefix) are kept forever and used as the fallback anchor
        self.sr_pin_prime = bool(self_reanchor_pin_prime)
        self.sr_pin_windows = max(1, int(self_reanchor_pin_windows))
        self._pinned_wins = []
        self._win_hist = []          # one entry per ingest {gids, conf, scale, rej} (recorded when the feature is on, capped at 64)
        self._sr_ingests = 0         # ingest counter (independent of the _ingest_calls used by reanchor_every)
        self._sr_last = -10**9       # ingest count at the last re-anchor
        self.pts = {}        # gid:int -> (xyz[P,3], rgb[P,3]) on device
        self.c2ws = {}       # gid:int -> c2w[4,4] tensor on device
        self.frames = {}     # gid:int -> (depth[h,w], intr[3,3], c2w[4,4], rgb[3,h,w]@depth-res in[0,1]) for multi-source priority rendering
        # A1 fix: remember the sparse point mask each frame had at ingest time (including conf_percentile and every other gate). when carve recomputes pts it must intersect with it,
        # otherwise recomputing from the dense depth would "resurrect" the low-confidence points that conf_percentile only filtered out of pts without zeroing the dense d (secondary pollution).
        self._pt_mask = {}   # gid:int -> bool[h,w]

    @property
    def num_frames(self):
        return len(self.pts)

    @torch.no_grad()
    def ingest(self, estimator, frames_rgb, c2w, K_pix, frame_ids):
        """frames_rgb [N,H,W,3] in [0,1] (np); c2w [N,4,4] (np); K_pix [3,3]; frame_ids list[int].
        one DA3 forward pass (independent scale) -> per-frame unproject + conf filtering -> store points. degenerate motion (umeyama failed) -> skip this chunk (not ingested)."""
        frames_rgb = np.asarray(frames_rgb, np.float32); c2w = np.asarray(c2w, np.float32)
        N = frames_rgb.shape[0]
        if N < 3:
            return                                              # DA3 GT-pose needs >=3 frames
        # === gated (color_anchor, independent of consist_gate); off by default = whole block skipped ===
        #   [before] estimator.depth_window (the DA3 forward), re-anchor the per-channel statistics of frames_rgb to the reference (GT prime) statistics;
        #   the corrected frames_rgb is used downstream (fed to DA3 + the cloud RGB inherits the rgb DA3 returns). window statistics are whole-window to prevent flicker.
        if self.color_anchor:
            _ca_flat = frames_rgb.reshape(-1, 3)
            _ca_mu = _ca_flat.mean(0).astype(np.float32)        # [3] whole-window per-channel mean
            _ca_sig = _ca_flat.std(0).astype(np.float32)        # [3] whole-window per-channel std
            _ca_dbg = bool(os.environ.get("EVOKE_CONSIST_DEBUG"))
            if self._ca_seen < self.color_anchor_ref_windows:   # reference period: accumulate statistics, no correction
                self._ca_ref_acc.append((_ca_mu.copy(), _ca_sig.copy()))
                _mus = np.stack([a for a, _ in self._ca_ref_acc], 0)
                _sigs = np.stack([b for _, b in self._ca_ref_acc], 0)
                self._ca_ref = (_mus.mean(0).astype(np.float32), _sigs.mean(0).astype(np.float32))
                if _ca_dbg:
                    print(f"[color-anchor] win_mu={np.round(_ca_mu, 3).tolist()} "
                          f"ref_mu={np.round(self._ca_ref[0], 3).tolist()} alpha=0.000(reference period, no correction "
                          f"{self._ca_seen + 1}/{self.color_anchor_ref_windows})", flush=True)
            elif self._ca_ref is not None:                      # after the reference period: per-channel linear match -> alpha blend
                _r_mu, _r_sig = self._ca_ref
                _eps = 1e-6
                _xp = (_ca_flat - _ca_mu[None]) / np.maximum(_ca_sig[None], _eps) * _r_sig[None] + _r_mu[None]
                _a = self.color_anchor_alpha
                _xo = _a * _xp + (1.0 - _a) * _ca_flat
                frames_rgb = np.clip(_xo, 0.0, 1.0).reshape(frames_rgb.shape).astype(np.float32)
                if _ca_dbg:
                    print(f"[color-anchor] win_mu={np.round(_ca_mu, 3).tolist()} "
                          f"ref_mu={np.round(_r_mu, 3).tolist()} alpha={_a:.3f}", flush=True)
            self._ca_seen += 1
        Ks = np.stack([np.asarray(K_pix, np.float32)] * N)
        try:
            depth, intr, conf, rgb = estimator.depth_window(frames_rgb, c2w, Ks)
        except Exception as _e:                                 # degenerate motion GeometryException etc -> skip, do not crash
            print(f"[da3-framebank] WARN ingest failed ({type(_e).__name__}: {_e}); skipping this chunk", flush=True)
            return
        # === re-check the pixels held last round (gated: consist_probation>0 and something held; off by default = whole block skipped) ===
        #   the "no cloud reference" pixels held at the last ingest are now geometrically voted on against the raw DA3 depth of this round's N frames:
        #   project the held world point into the new frames -> visible and |z_held-d_new|<=tau*d_new counts as a "consistent" vote / visible but deviating >tau counts as a "conflicting" vote;
        #   consistent>=1 and >=conflicting -> admitted (merged back into that gid's dense depth + pts recomputed, with the A1-style mask intersection so other filtered points are not resurrected);
        #   conflicting>consistent -> dropped permanently; not visible at all (0 votes) -> admitted (benefit of the doubt, so they cannot starve). probation lasts one round only and is cleared once processed.
        if self.consist_probation > 0 and self._probation:
            _pb_held = 0; _pb_adm = 0; _pb_drop = 0; _pb_unver = 0
            _pb_new = [(torch.as_tensor(depth[i], device=self.device),
                        torch.as_tensor(intr[i], device=self.device),
                        torch.as_tensor(c2w[i], device=self.device)) for i in range(N)]
            for _g, _rec in list(self._probation.items()):
                if _g not in self.frames:                       # already evicted (evict/reanchor) -> give up
                    continue
                _hm = _rec["mask"]                              # bool[h,w] held pixels
                _hd = _rec["depth"]                             # [h,w] held depth (values only where mask is set)
                if not bool(_hm.any()):
                    continue
                _Xh = unproject_depth_torch(_hd, _rec["it"], _rec["cw"])[_hm]   # [P,3] held world points
                _ok = torch.zeros(_Xh.shape[0], dtype=torch.int16, device=self.device)
                _bad = torch.zeros_like(_ok)
                for (_dn, _itn, _cwn) in _pb_new:               # project into each new frame of this round (same projection as _carve_recent)
                    _Rn = _cwn[:3, :3]; _tn = _cwn[:3, 3]
                    _cam = (_Xh - _tn[None]) @ _Rn              # R^T(X-t) -> new frame camera space [P,3]
                    _z = _cam[:, 2]
                    _fx = _itn[0, 0]; _fy = _itn[1, 1]; _cx = _itn[0, 2]; _cy = _itn[1, 2]
                    _hn, _wn = int(_dn.shape[0]), int(_dn.shape[1])
                    _px = torch.round(_cam[:, 0] / _z.clamp(min=1e-6) * _fx + _cx).long()
                    _py = torch.round(_cam[:, 1] / _z.clamp(min=1e-6) * _fy + _cy).long()
                    _inb = (_z > 1e-4) & (_px >= 0) & (_px < _wn) & (_py >= 0) & (_py < _hn)
                    _dat = _dn.reshape(-1)[(_py.clamp(0, _hn - 1) * _wn + _px.clamp(0, _wn - 1))]
                    _seen = _inb & (_dat > 1e-4)                # visible in this new frame (in bounds and with a valid observation)
                    _cons = _seen & ((_z - _dat).abs() <= self.consist_tau * _dat)
                    _ok += _cons.to(torch.int16)
                    _bad += (_seen & ~_cons).to(torch.int16)
                _unver = (_ok == 0) & (_bad == 0)               # not visible at all -> unverifiable -> admitted (benefit of the doubt)
                _admv = (_ok >= 1) & (_ok >= _bad)              # consistent votes >=1 and >= conflicting votes -> admitted
                _adm = _admv | _unver
                _pb_held += int(_Xh.shape[0]); _pb_adm += int(_admv.sum())
                _pb_unver += int(_unver.sum()); _pb_drop += int((~_adm).sum())
                if bool(_adm.any()):                            # admitted: merge back into the dense depth + recompute pts/_pt_mask
                    _adm_mask = torch.zeros_like(_hm)
                    _adm_mask[_hm] = _adm
                    _dg, _itg, _cwg, _rgbg = self.frames[_g]
                    _dg2 = torch.where(_adm_mask, _hd, _dg)
                    _pm = self._pt_mask.get(_g)
                    _pm2 = (_pm | _adm_mask) if _pm is not None else _adm_mask  # admitted mask = original mask | admitted pixels
                    # A1-style intersection: keep is decided only by (valid depth & ingest mask) -> does not resurrect points that conf / other gates filtered out
                    _keep = torch.isfinite(_dg2) & (_dg2 > 1e-4) & _pm2
                    _Xg2 = unproject_depth_torch(_dg2, _itg, _cwg)
                    self.frames[_g] = (_dg2, _itg, _cwg, _rgbg)
                    self.pts[_g] = (_Xg2[_keep], _rgbg.permute(1, 2, 0)[_keep])
                    self._pt_mask[_g] = _keep
            self._probation = {}                                # probation lasts one round only
            if bool(os.environ.get("EVOKE_CONSIST_DEBUG")):
                print(f"[probation-dbg] held={_pb_held} admitted={_pb_adm} dropped={_pb_drop} "
                      f"unverifiable={_pb_unver}", flush=True)
        # [method 1] reference geometry snapshot: take the world points of the most recent frames of the [established cloud] (its state before this ingest) for the per-frame consistency check.
        #   built once outside the loop -> every frame of this chunk is checked against "the existing geometry of the previous block" (not polluted by this chunk itself).
        _ref_xyz = None
        if self.consist_gate and self.pts:
            _gs = sorted(self.pts.keys())
            _ref_gids = _gs[-self.consist_ref_frames:] if self.consist_ref_frames > 0 else _gs
            _xs = [self.pts[g][0] for g in _ref_gids if self.pts[g][0].shape[0] > 0]
            if _xs:
                _ref = torch.cat(_xs, 0)
                if _ref.shape[0] >= self.consist_min_ref:
                    if _ref.shape[0] > 300000:                  # rendering the reference depth map is O(P) -> cap it with a random downsample to control cost
                        _ref = _ref[torch.randint(0, _ref.shape[0], (300000,), device=self.device)]
                    _ref_xyz = _ref
        # [consist-dbg] M1 rejection-rate instrumentation (EVOKE_CONSIST_DEBUG=1, off by default): one line per ingest
        #   covered (checkable) pixel count / conflict-culled count / median conf -- filling the observation blind spot of "the rejection rate was never actually measured".
        _cg_dbg = bool(os.environ.get("EVOKE_CONSIST_DEBUG"))
        _cg_track = _cg_dbg or self.self_reanchor              # [self re-anchor] the trigger criteria reuse the same window statistics
        _cg_have = 0; _cg_rej = 0; _cg_tot = 0; _cg_confs = []; _cg_scales = []
        # === gated (consist_scale_align and reference geometry present); off by default = whole block skipped ===
        #   project the reference cloud per frame to get the predicted depth _cd (cached for reuse by the gate below, semantics unchanged); the per-frame median of median(d/_cd) over the
        #   overlap region, then the median over the window = the scale s of this window relative to the cloud -> depth/=s for the whole window. guards: see the __init__ comment.
        #   note: the probation re-check block runs before this and uses uncorrected depth -- its tau=0.2 criterion absorbs a 5-18% scale difference, which is acceptable.
        _cd_cache = {}
        if self.consist_scale_align and _ref_xyz is not None:
            _sa_meds = []; _sa_nov = 0
            for i in range(N):
                _d_i = torch.as_tensor(depth[i], device=self.device)
                _it_i = torch.as_tensor(intr[i], device=self.device)
                _cw_i = torch.as_tensor(c2w[i], device=self.device)
                _cd_i = _cloud_depth_map(_ref_xyz, _cw_i, _it_i,
                                         int(_d_i.shape[0]), int(_d_i.shape[1]), device=self.device)
                _cd_cache[i] = _cd_i
                _ov_i = torch.isfinite(_cd_i) & (_cd_i > 1e-4) & (_d_i > 1e-4)
                _n_i = int(_ov_i.sum()); _sa_nov += _n_i
                if _n_i > 200:
                    _sa_meds.append(float((_d_i[_ov_i] / _cd_i[_ov_i]).median()))
            if _sa_meds and _sa_nov >= 2000:
                _s_raw = float(np.median(_sa_meds))
                if 0.5 < _s_raw < 2.0:
                    _s = float(np.clip(_s_raw, 0.75, 1.33))
                    if abs(_s - 1.0) > 1e-3:
                        depth = depth / _s                        # whole-window correction (numpy); everything downstream uses the corrected depth
                    if _cg_dbg:
                        print(f"[scale-align] window_scale={_s_raw:.3f}"
                              f"{'(clip->%.3f)' % _s if abs(_s - _s_raw) > 1e-3 else ''} "
                              f"applied n_overlap={_sa_nov}", flush=True)
                elif _cg_dbg:
                    print(f"[scale-align] window_scale={_s_raw:.3f} absurd, not in (0.5,2) -> skip", flush=True)
            elif _cg_dbg:
                print(f"[scale-align] overlap too small ({_sa_nov}<2000) -> skip", flush=True)
        for i, gid in enumerate(frame_ids):
            d = torch.as_tensor(depth[i], device=self.device)
            it = torch.as_tensor(intr[i], device=self.device)
            cw = torch.as_tensor(c2w[i], device=self.device)
            wp = unproject_depth_torch(d, it, cw)               # [h,w,3]
            m = torch.isfinite(wp).all(-1) & (d > 1e-4)
            if conf is not None and self.conf_pct > 0:
                thr = float(np.percentile(conf[i].reshape(-1), self.conf_pct))
                m &= torch.as_tensor(conf[i], device=self.device) >= thr
            if rgb is not None:
                r = torch.as_tensor(np.asarray(rgb[i], np.float32), device=self.device)   # [h,w] in[0,1] (DA3 processed, aligned with depth)
            else:
                r = torch.zeros_like(wp)
            # === [cloud hygiene] gated; off by default (self.hygiene=False) -> whole block skipped, behaviour unchanged ===
            if self.hygiene and rgb is not None:
                # (1) saturation clipping: pixels with HSV saturation > sat_max are shrunk toward their own brightness (stops over-saturated colours being baked into the cloud and then amplified on-policy into a "colour flood")
                if 0.0 < self.hyg_sat_max < 1.0:
                    # hard HSV saturation ceiling: keep V (=max channel) and hue, only compress S=chroma/V down to sat_max
                    # new = V - (V - rgb) * f, f = min(1, sat_max*V/chroma) -> V channel unchanged, min channel raised -> S=sat_max
                    _mx = r.max(-1).values                                                       # V [h,w]
                    _chroma = (_mx - r.min(-1).values).clamp(min=1e-6)                            # chroma
                    _s = _chroma / _mx.clamp(min=1e-6)
                    _f = torch.where(_s > self.hyg_sat_max,
                                     (self.hyg_sat_max * _mx) / _chroma,
                                     torch.ones_like(_mx)).clamp(max=1.0)
                    r = (_mx.unsqueeze(-1) - (_mx.unsqueeze(-1) - r) * _f.unsqueeze(-1)).clamp(0.0, 1.0)
                # (2) flat (textureless) region gate: DA3 monocular depth is unreliable on flat regions -> drop pixels whose local grey std < threshold (depth set to 0 -> not rendered in the warp, leaving a hole for the model)
                _drop = torch.zeros_like(m)
                if self.hyg_flat_std > 0.0:
                    _g = (0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2]).unsqueeze(0).unsqueeze(0)
                    _w = max(3, int(self.hyg_flat_win) | 1); _pad = _w // 2
                    _mean = torch.nn.functional.avg_pool2d(_g, _w, 1, _pad)
                    _msq = torch.nn.functional.avg_pool2d(_g * _g, _w, 1, _pad)
                    _std = (_msq - _mean * _mean).clamp(min=0.0).sqrt()[0, 0]                   # [h,w]
                    _drop = _drop | (_std < self.hyg_flat_std)
                # (3a) NOTE primary gate: absolute confidence threshold -- degenerate flood collapses DA3 conf to the floor (~1-3) while real content is high (~11-14);
                #      an absolute threshold culls the former and keeps the latter. (a per-frame percentile threshold collapses together with the whole frame -> useless against flood, see (3b))
                if conf is not None and self.hyg_conf_abs > 0.0:
                    _drop = _drop | (torch.as_tensor(conf[i], device=self.device) < self.hyg_conf_abs)
                # (3b) weak: per-frame relative percentile (when the whole frame collapses the threshold collapses too -> cannot filter flood; kept for compatibility only)
                if conf is not None and self.hyg_conf_pct > 0.0:
                    _thr2 = float(np.percentile(conf[i].reshape(-1), self.hyg_conf_pct))
                    _drop = _drop | (torch.as_tensor(conf[i], device=self.device) < _thr2)
                _keep = ~_drop
                d = torch.where(_keep, d, torch.zeros_like(d))   # dropped pixels get depth 0 -> naturally excluded by the d>1e-4 test at render time
                m = m & _keep
            # === [cloud hygiene] end ===
            # === [method 1] ingest consistency gate, gated (consist_gate and enough reference geometry); independent of hygiene ===
            if _ref_xyz is not None:
                # project the established cloud into this view -> per-pixel nearest-point depth (the cloud prediction); +inf where there is no point (a hole, passed through)
                # when scale_align is on the same projection was already computed -> reuse the cache (byte-identical semantics)
                _cd = _cd_cache.get(i)
                if _cd is None:
                    _cd = _cloud_depth_map(_ref_xyz, cw, it, int(d.shape[0]), int(d.shape[1]), device=self.device)
                _have = torch.isfinite(_cd) & (_cd > 1e-4)
                # relative depth tolerance _tau (a fixed scalar threshold or, with [variant 1], a confidence-adaptive per-pixel threshold):
                if self.consist_adaptive and conf is not None:
                    # per-pixel tau = tau_lo + (tau_hi-tau_lo)*clamp(conf/conf_ref, 0, 1) -> [rises] with DA3 confidence.
                    # low conf (poisoning: flat / over-saturated render surfaces where conf collapses) -> tau_lo (strict, blocks poisoning); high conf (trustworthy new real geometry) -> tau_hi (loose, avoids salt-and-pepper).
                    # note: the direction is opposite to standard Mahalanobis -- in this domain low conf means a systematic error rather than unbiased noise.
                    _cf = torch.as_tensor(conf[i], device=self.device).to(torch.float32)
                    _u = (_cf / max(self.consist_conf_ref, 1e-6)).clamp_(0.0, 1.0)
                    _tau = self.consist_tau_lo + (self.consist_tau_hi - self.consist_tau_lo) * _u   # [h,w]
                else:
                    _tau = self.consist_tau                                    # fixed threshold (default / fallback when there is no conf)
                # relative depth conflict: the existing geometry covers this pixel but the new frame DA3 depth deviates > tau*d_cloud -> hallucination / poisoning (wall / flood / any colour) -> cull
                _conflict = _have & (d > 1e-4) & ((d - _cd).abs() > _tau * _cd)
                _keep2 = ~_conflict
                if _cg_track:
                    # WARNING: the statistics are computed [before] culling: a scale ratio over surviving pixels only would be truncated by tau -> severe underestimate of the drift
                    _ov = _have & (d > 1e-4)                    # checkable region (including what is about to be culled)
                    _cg_have += int(_ov.sum())
                    _cg_rej += int(_conflict.sum()); _cg_tot += int(d.numel())
                    if conf is not None:
                        _cg_confs.append(float(np.median(conf[i])))
                    # direct measurement of window-level scale drift: the median ratio of new frame depth to cloud-predicted depth over the overlap (~1 = scale consistent;
                    # a systematic deviation = this ingest window's Umeyama scale has drifted relative to the existing cloud, and it eats M1's tau)
                    if int(_ov.sum()) > 500:
                        _cg_scales.append(float((d[_ov] / _cd[_ov]).median()))
                d = torch.where(_keep2, d, torch.zeros_like(d))
                m = m & _keep2
                # - gated (consist_probation>0): pixels with no cloud reference (_have=False) no longer pass straight
                #   into the cloud (previously the hallucination fast lane); they are deducted from d/m this round into _probation and admitted / dropped after the next ingest re-check.
                #   only held on the path where consist_gate is on and _ref_xyz is valid (cold start / insufficient reference admits everything = legacy behaviour).
                if self.consist_probation > 0:
                    _hole = (~_have) & (d > 1e-4) & m           # pixels that passed the other gates but have no cloud reference
                    # frac>0: only hold "large-hole interiors" (local hole fraction over win x win > frac); narrow strips / small holes pass straight through.
                    #   hallucination lives inside large unknown regions, while the new real content moving forward brings in is mostly narrow boundary strips -> admitting it lowers the coverage tax.
                    if self.consist_probation_frac > 0.0 and bool(_hole.any()):
                        _w = self.consist_probation_win
                        _hf = torch.nn.functional.avg_pool2d(
                            _hole.float()[None, None], _w, stride=1, padding=_w // 2)[0, 0]
                        _hole = _hole & (_hf > self.consist_probation_frac)
                    if bool(_hole.any()):
                        self._probation[int(gid)] = {
                            "mask": _hole.clone(),
                            "depth": torch.where(_hole, d, torch.zeros_like(d)),
                            "rgb": r.clone(), "it": it.clone(), "cw": cw.clone()}
                        d = torch.where(_hole, torch.zeros_like(d), d)   # same treatment as conflict: depth set to 0
                        m = m & ~_hole
            # === [method 1] end ===
            self.pts[int(gid)] = (wp[m], r[m])
            self._pt_mask[int(gid)] = m                         # A1: lets carve keep the original gating when recomputing pts
            self.c2ws[int(gid)] = cw
            # for multi-source priority rendering: store dense depth/intr/c2w/rgb (rgb uses the DA3 processed image, letterbox-aligned with depth)
            self.frames[int(gid)] = (d, it, cw, r.permute(2, 0, 1).contiguous())
        if _cg_dbg and self.consist_gate:
            _newest = max(int(g) for g in frame_ids)
            _cstr = f" conf_med={np.median(_cg_confs):.2f}" if _cg_confs else ""
            _sstr = f" scale_ratio_med={np.median(_cg_scales):.3f}" if _cg_scales else ""
            print(f"[consist-dbg] newest_gid={_newest}(~{_newest/24.0:.1f}s) "
                  f"checkable={_cg_have}/{_cg_tot} ({100.0*_cg_have/max(_cg_tot,1):.1f}%) "
                  f"rejected={_cg_rej} ({100.0*_cg_rej/max(_cg_have,1):.1f}% of checkable){_cstr}{_sstr}", flush=True)
        # === [method 3 free-space carving] gated; use the batch of new frames just ingested to "see through" the old frames and delete floating old points in free space ===
        if self.carve:
            self._carve_recent([int(g) for g in frame_ids])
        # === [method 2] bounded / periodically re-anchored history, gated; lets old poisoning age out (evicted in one go after the whole chunk is ingested) ===
        if self.pts and (self.hist_max_frames > 0 or self.reanchor_every > 0):
            _newest = max(self.pts.keys())
            if self.hist_max_frames > 0:                          # sliding window: keep only the most recent hist_max_frames frames
                self.evict_before(_newest - self.hist_max_frames + 1)
            if self.reanchor_every > 0:                           # periodic hard re-anchor: every N ingests shrink to the most recent keep frames
                self._ingest_calls += 1
                if self._ingest_calls % self.reanchor_every == 0 and self.reanchor_keep_frames > 0:
                    self.evict_before(_newest - self.reanchor_keep_frames + 1)
        # === [self re-anchor] gated; divergence triggers a hard re-anchor of the cloud to the best contiguous window in its own history ===
        if self.self_reanchor:
            self._sr_ingests += 1
            _scale = float(np.median(_cg_scales)) if _cg_scales else None
            _rej = _cg_rej / max(_cg_have, 1)
            _conf = float(np.median(_cg_confs)) if _cg_confs else 0.0
            self._win_hist.append({"gids": [int(g) for g in frame_ids],
                                   "conf": _conf, "scale": _scale, "rej": _rej})
            # [v4] pin the prime anchor: the first sr_pin_windows ingests = the v2v cold-start GT prefix (a forever-legal clean anchor),
            #   recorded in _pinned_wins, and their gids are never evicted by a self re-anchor. lesson from the 2min long rollout: the healthy anchors inside the lookback window age out
            #   (all 31 triggers skipped), and with nowhere to anchor the cloud just rots in place; falling back to prime = resetting to "near-free generation" and re-accumulating
            #   (user observation: pure generation without warp barely drifts), which beats hugging a poisoned cloud.
            if self.sr_pin_prime and self._sr_ingests <= self.sr_pin_windows:
                self._pinned_wins.append({"gids": [int(g) for g in frame_ids],
                                          "conf": _conf, "scale": _scale, "rej": _rej})
            if len(self._win_hist) > 64:
                self._win_hist = self._win_hist[-64:]
            _diverged = ((_scale is not None and (_scale > self.sr_scale_thr or _scale < 1.0 / self.sr_scale_thr))
                         or (_cg_have > 0 and _rej > self.sr_rej_thr))
            if (_diverged and len(self._win_hist) >= 6
                    and self._sr_ingests - self._sr_last >= self.sr_min_gap):
                # candidates = the "healthy" windows in the lookback (scale~1, low rejection, gids still in the bank); the highest conf becomes the anchor
                _cands = []
                for _wi in range(max(0, len(self._win_hist) - self.sr_lookback), len(self._win_hist) - 1):
                    _w = self._win_hist[_wi]
                    if _w["scale"] is None or not (0.8 <= _w["scale"] <= 1.25) or _w["rej"] >= 0.3:
                        continue
                    if _w["conf"] < self.sr_anchor_min_conf:    # [v3] minimum anchor quality; junk windows are not candidates
                        continue
                    if not all(g in self.pts for g in _w["gids"]):
                        continue
                    _cands.append((_w["conf"], _wi))
                _pin_keep = set()
                if self.sr_pin_prime:
                    for _w in self._pinned_wins:                # pinned anchors are always in the keep set (no matter where we anchor)
                        if all(g in self.pts for g in _w["gids"]):
                            _pin_keep.update(_w["gids"])
                if _cands:
                    _best_wi = max(_cands)[1]
                    _keep = set(_pin_keep)
                    for _wi in range(max(0, _best_wi - self.sr_keep_windows + 1), _best_wi + 1):
                        _keep.update(self._win_hist[_wi]["gids"])   # the anchor window + the contiguous windows before it (geometric continuity)
                    _n_before = len(self.pts)
                    self._evict_keep(_keep)
                    self._sr_last = self._sr_ingests
                    print(f"[self-reanchor] divergence triggered (scale={_scale if _scale is not None else float('nan'):.2f} "
                          f"rej={_rej:.2f}) -> re-anchoring to the best window in history gid[{min(_keep)}..{max(_keep)}] "
                          f"(conf={self._win_hist[_best_wi]['conf']:.2f}) cloud {_n_before}->{len(self.pts)} frames", flush=True)
                elif self.sr_pin_prime and _pin_keep:
                    # [v4] no healthy anchor in the lookback -> fall back to the pinned prime anchor (resetting to "near-free generation" and re-accumulating the cloud)
                    _n_before = len(self.pts)
                    self._evict_keep(_pin_keep)
                    self._sr_last = self._sr_ingests
                    print(f"[self-reanchor] divergence triggered (scale={_scale if _scale is not None else float('nan'):.2f} "
                          f"rej={_rej:.2f}) no healthy recent anchor -> [v4] falling back to the prime anchor gid[{min(_pin_keep)}..{max(_pin_keep)}] "
                          f"cloud {_n_before}->{len(self.pts)} frames", flush=True)
                else:
                    print(f"[self-reanchor] divergence triggered (scale={_scale} rej={_rej:.2f}) but no healthy anchor in the lookback -> skipping", flush=True)

    @torch.no_grad()
    def _carve_recent(self, new_gids):
        """[free-space carving] use the dense depth of the frames just ingested (new_gids) to delete points of earlier old frames that fall in free space.
        free-space criterion: project an old point into a new frame; if its camera-z depth < that pixel's newly observed surface depth*(1-margin) -> the old point hangs
        in "the gap between the camera and the real surface" -> delete it (set the old frame dense depth to 0 + recompute the sparse points). clears floating poison / ghosts of objects that moved away."""
        new_set = set(int(g) for g in new_gids)
        # new observations = the (already gated) dense (depth, intr, c2w) of the batch just ingested
        newobs = [self.frames[g][:3] for g in new_gids if g in self.frames]
        if not newobs:
            return
        gmin = min(new_set)
        targets = sorted([g for g in self.frames.keys() if g < gmin])   # only carve strictly older frames
        if self.carve_ref_frames > 0:
            targets = targets[-self.carve_ref_frames:]
        _dbg = bool(os.environ.get("EVOKE_CARVE_DEBUG"))   # provenance diagnostics: print the carve deletion count per chunk
        _del_px = 0; _val_px = 0; _ftouch = 0; _strike_pend = 0
        for g in targets:
            d_g, it_g, cw_g, rgb_g = self.frames[g]
            valid = torch.isfinite(d_g) & (d_g > 1e-4)
            if not bool(valid.any()):
                continue
            if _dbg:
                _val_px += int(valid.sum())
            Xg = unproject_depth_torch(d_g, it_g, cw_g)                 # [h,w,3] old frame world points
            votes = torch.zeros_like(d_g, dtype=torch.int16)            # [variant 2] how many new frames judge each pixel as "seen through (free space)"
            for (d_n, it_n, cw_n) in newobs:
                Rn = cw_n[:3, :3]; tn = cw_n[:3, 3]
                cam = (Xg - tn) @ Rn                                     # R^T(X-t) -> new frame camera space [h,w,3]
                z = cam[..., 2]
                fx = it_n[0, 0]; fy = it_n[1, 1]; cx = it_n[0, 2]; cy = it_n[1, 2]
                hn, wn = int(d_n.shape[0]), int(d_n.shape[1])
                px = torch.round(cam[..., 0] / z.clamp(min=1e-6) * fx + cx).long()
                py = torch.round(cam[..., 1] / z.clamp(min=1e-6) * fy + cy).long()
                inb = (z > 1e-4) & (px >= 0) & (px < wn) & (py >= 0) & (py < hn)
                idx = (py.clamp(0, hn - 1) * wn + px.clamp(0, wn - 1)).reshape(-1)
                d_at = d_n.reshape(-1)[idx].reshape(z.shape)             # the surface depth the new frame observes at that pixel
                # free space: the new frame has a valid surface at that pixel and the old point is clearly closer -> the old point hangs in the gap (one vote from this frame)
                freespace = inb & (d_at > 1e-4) & (z < d_at * (1.0 - self.carve_margin))
                votes += (valid & freespace).to(torch.int16)
            # gated (carve_strike_windows>1): the 12 frames of one window share a DA3 forward pass + scale -> votes are
            #   correlated within the window and cannot suppress a window-level systematic error. a pixel satisfying the deletion criterion only records a strike and must be confirmed in several consecutive windows before real deletion;
            #   an unconfirmed window -> the strike resets ([consecutive] is required). the default 1 takes the else path and is byte-equivalent to the legacy behaviour.
            if self.carve_strike_windows > 1:
                _confirmed = valid & (votes >= self.carve_min_views)    # satisfies the existing deletion criterion in this window
                _st = self._carve_strike.get(g)
                if _st is None:
                    _st = torch.zeros_like(votes)
                _st = torch.where(_confirmed, _st + 1, torch.zeros_like(_st))   # confirmed +1 / unconfirmed reset to 0
                carve = valid & (_st >= self.carve_strike_windows)
                if _dbg:
                    _strike_pend += int((_confirmed & ~carve).sum())    # a strike was recorded but nothing was deleted in this window
                _st = torch.where(carve, torch.zeros_like(_st), _st)    # strike reset to 0 for already-deleted pixels
                self._carve_strike[g] = _st
            else:
                # [variant 2 multi-view voting] at least carve_min_views new frames must see through before deletion (=1 -> reproduces the legacy OR behaviour); single-frame depth jitter cannot gather enough votes
                carve = valid & (votes >= self.carve_min_views)
            if bool(carve.any()):
                d_g2 = torch.where(carve, torch.zeros_like(d_g), d_g)
                keep = torch.isfinite(d_g2) & (d_g2 > 1e-4)
                # A1 fix: intersect with the sparse point mask from ingest time so points that conf_percentile filtered out of pts without zeroing the dense d are not resurrected
                if g in self._pt_mask:
                    keep = keep & self._pt_mask[g]
                Xg2 = unproject_depth_torch(d_g2, it_g, cw_g)
                self.frames[g] = (d_g2, it_g, cw_g, rgb_g)
                self.pts[g] = (Xg2[keep], rgb_g.permute(1, 2, 0)[keep])
                self._pt_mask[g] = keep
                if _dbg:
                    _del_px += int(carve.sum()); _ftouch += 1
        if _dbg:
            _newest = max(new_set)
            _spstr = f" strike_pending={_strike_pend}" if self.carve_strike_windows > 1 else ""
            print(f"[carve-dbg] newest_gid={_newest}(~{_newest/24.0:.1f}s) targets={len(targets)} "
                  f"frames_touched={_ftouch} deleted_px={_del_px}/{_val_px} "
                  f"({100.0*_del_px/max(_val_px,1):.1f}%){_spstr}", flush=True)

    def evict_before(self, min_gid):
        """memory eviction: drop stored points with frame_id < min_gid (the recall pool starts at min_gid)."""
        for gid in [g for g in self.pts if g < int(min_gid)]:
            self.pts.pop(gid, None); self.c2ws.pop(gid, None); self.frames.pop(gid, None)
            self._pt_mask.pop(gid, None)
        for gid in [g for g in self._probation if g < int(min_gid)]:      # clear the probation area in sync
            self._probation.pop(gid, None)
        for gid in [g for g in self._carve_strike if g < int(min_gid)]:   # clear the strike counters in sync
            self._carve_strike.pop(gid, None)

    def _evict_keep(self, keep_gids):
        """[self re-anchor] selective eviction: keep only keep_gids and pop everything else (including the probation / strike side state)."""
        keep = set(int(g) for g in keep_gids)
        for gid in [g for g in self.pts if g not in keep]:
            self.pts.pop(gid, None); self.c2ws.pop(gid, None); self.frames.pop(gid, None)
            self._pt_mask.pop(gid, None); self._probation.pop(gid, None)
            self._carve_strike.pop(gid, None)


@torch.no_grad()
def _render_multisrc(store, ids_all, target_c2ws, K_pix, height, width, *,
                     nsrc=8, nearby=16, splat_radius=1, dens_thresh=0.45, dens_win=7,
                     recall_min_cov=0.5, recall_margin=0.15, device="cuda"):
    """multi-source priority fusion render core (shared by training / inference). store: gid -> (depth[h,w], intr[3,3], c2w[4,4], rgb[3,h,w]@depth-res in[0,1]).
    per target frame: the recent frames sorted by covis provide the primary source (used in full) + the others only fill holes (priority); recalled old frames only fill holes and get a local scale alignment; a density gate turns sparse points into black holes.
    target_c2ws [F,4,4] -> warp [1,3,F,H,W] in[-1,1], vis [1,1,F,H,W]. empty store -> all holes (cold start)."""
    import torch.nn.functional as _F
    H, W = int(height), int(width); HW = H * W; F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)

    # subsampled points + P_all (vectorised covis) + the reference for scale estimation
    M = 2000; FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}
    P_all = torch.full((len(ids_all), M, 3), FAR, device=device); subpts = {}
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]   # A2: zeroed-depth pixels unproject to a fake point at the camera centre and must be excluded (they pollute the covis ordering)
        if wp.shape[0] > 0:
            sp = wp[torch.randint(0, wp.shape[0], (M,), device=device)]
            subpts[g] = sp; P_all[id2row[g]] = sp
        else:
            subpts[g] = wp

    def covis_vec(rows, tpose):
        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]   # fix: subtract the camera centre (the c2w translation), not the w2c translation; otherwise R@(P-t) is not world->camera
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def est_scale(g, ref_pts, tposepts_n=200):
        d, it, cwi, _ = store[g]; h, w = d.shape
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0
        D = torch.full((h * w,), float("inf"), device=device); D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        if int(v.sum()) < tposepts_n:
            return 1.0
        s = float((D[v] / df[v]).median())
        return s if 0.2 < s < 5.0 else 1.0

    def render_one(g, R, t, scale=1.0):
        d, it, cwi, rr = store[g]
        if scale != 1.0:
            d = d * scale
        wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3); rgb = rr.permute(1, 2, 0).reshape(-1, 3)
        npt = wp.shape[0]; cam = (R @ wp.T).T + t; zc = cam[:, 2]
        px = torch.round(cam[:, 0] / zc.clamp(min=1e-6) * fx + cx).long(); py = torch.round(cam[:, 1] / zc.clamp(min=1e-6) * fy + cy).long()
        zq = (zc.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1); idx = torch.arange(npt, device=device)
        INF = torch.full((HW,), (1 << 62), dtype=torch.long, device=device)
        front = (zc > 1e-4) & (d.reshape(-1) > 1e-4)
        # real point hits (before splat, radius=0): whether a real projected point landed on each pixel -> feeds the "real sampling density" of the density gate
        hit0 = torch.zeros((HW,), dtype=torch.bool, device=device)
        ok0 = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        hit0[(py * W + px)[ok0]] = True
        for dy in range(-splat_radius, splat_radius + 1):
            for dx in range(-splat_radius, splat_radius + 1):
                qx = px + dx; qy = py + dy; ok = front & (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
                INF.scatter_reduce_(0, (qy * W + qx)[ok], (zq[ok] << 24) | idx[ok], reduce="amin", include_self=True)
        valid = INF < (1 << 62); win = (INF & ((1 << 24) - 1)).clamp(max=npt - 1)
        col = torch.zeros((HW, 3), device=device); col[valid] = rgb[win[valid]]
        return col, valid, hit0

    tc = target_c2ws.to(device).float()
    nearby_ids = ids_all[-int(nearby):]; old_ids = [g for g in ids_all if g not in nearby_ids]
    nearby_rows = [id2row[g] for g in nearby_ids]; old_rows = [id2row[g] for g in old_ids]
    nref = torch.cat([subpts[g] for g in nearby_ids if subpts[g].shape[0] > 0]) if nearby_ids else None
    # camera poses (for the recall pose-proximity gate): centre + view axis
    nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]) if nearby_ids else None
    old_cen = torch.stack([store[g][2][:3, 3] for g in old_ids]) if old_ids else None
    old_fwd = torch.stack([store[g][2][:3, 2] for g in old_ids]) if old_ids else None
    warps = []; viss = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(nearby_rows, tpose)
        order = torch.argsort(cv, descending=True).tolist()
        srcs = [nearby_ids[i] for i in order[:int(nsrc)]]
        scales = {}
        if old_ids:
            cvo = covis_vec(old_rows, tpose); oi = int(cvo.argmax())
            # pose-proximity gate: a real revisit = the old frame camera came back to a pose close to the target (distance <= 1.5x that of the nearest neighbour frame, view-axis cos > 0.5);
            # otherwise it is just a false hit where points happen to enter the frustum (moving forward into a new region) -> do not recall, leave it black for the DiT to generate.
            tcen = tpose[:3, 3]; tfwd = tpose[:3, 2]
            d_old = float((old_cen[oi] - tcen).norm())
            d_nb_min = float((nb_cen - tcen).norm(dim=1).min()) if nb_cen is not None else 1e9
            cos_fwd = float((old_fwd[oi] @ tfwd) / (old_fwd[oi].norm() * tfwd.norm() + 1e-9))
            pose_ok = (d_old <= max(d_nb_min * 1.5, 1e-6)) and (cos_fwd > 0.5)
            if float(cvo[oi]) >= recall_min_cov and float(cvo[oi]) > float(cv.max()) + recall_margin and pose_ok:
                bo = old_ids[oi]; srcs = srcs[:int(nsrc) - 1] + [bo]
                if nref is not None:
                    scales[bo] = est_scale(bo, nref)
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = w2c[:3, 3]
        fused = torch.zeros((HW, 3), device=device); filled = torch.zeros((HW,), dtype=torch.bool, device=device)
        filled_true = torch.zeros((HW,), dtype=torch.bool, device=device)   # union of real point hits (before splat) -> used by the density gate
        large_holes = torch.zeros((HW,), dtype=torch.bool, device=device)
        _fg = 2     # primary-source coverage dilation guard (px): seals the micro-seams of the primary splat so fill cannot enter primary territory
        _kl = 6     # "large hole" radius (px): only connected holes wider than ~2*_kl may be filled (= a real memory / coverage gap);
                    # thin pinholes / micro-seams of the primary source (gone after the opening) are never filled -- a secondary source filling small holes only stuffs in points at a different depth, forming overlapping scattered specks.
        for si, g in enumerate(srcs):
            col, valid, hit0 = render_one(g, R, t, scale=scales.get(g, 1.0))
            if si == 0:
                wmask = valid                                   # the primary source is used in full
                prim_guard = _F.max_pool2d(valid.reshape(1, 1, H, W).float(), 2 * _fg + 1, 1, _fg)[0, 0].reshape(-1) > 0
                # holes outside the primary territory -> morphological opening (erode then dilate) keeps only large connected holes and removes pinholes / hairline seams
                _holes = (~prim_guard).reshape(1, 1, H, W).float()
                _er = (_F.avg_pool2d(_holes, 2 * _kl + 1, 1, _kl) >= 0.999).float()
                large_holes = (_F.max_pool2d(_er, 2 * _kl + 1, 1, _kl)[0, 0].reshape(-1) > 0)
            else:
                wmask = valid & (~filled) & large_holes         # fill only patches "the large holes the primary source does not have"
            fused[wmask] = col[wmask]; filled |= wmask          # only the pixels actually written count as filled (micro-seams are not visible)
            filled_true |= (hit0 & wmask)                       # real point hits (only the ones actually used)
        vis = filled.reshape(H, W).float()
        # the density gate uses the [real point hit density] rather than the splat-dilated coverage: splat inflates coverage but the colour is blocky specks,
        # only the real hit density reflects "whether this area is really densely sampled" -> speckled areas have low real density -> keep=0 excludes them.
        dens = _F.avg_pool2d(filled_true.reshape(1, 1, H, W).float(), int(dens_win), 1, int(dens_win) // 2)[0, 0]
        keep = (vis > 0) & (dens >= dens_thresh)
        fused = fused * keep.reshape(-1, 1)
        warps.append(fused.reshape(H, W, 3).clamp(0, 1).permute(2, 0, 1) * 2 - 1)
        viss.append(keep.float())
    return torch.stack(warps, 1)[None].contiguous(), torch.stack(viss)[None, None].contiguous()


def _render_backward(store, ids_all, target_c2ws, K_pix, height, width, *,
                     nearby=16, fill_iters=12, recall_min_cov=0.5, recall_margin=0.15, device="cuda"):
    """backward-warp render core (shared by training / inference): per target frame pick the nearest source and backward-warp it (grid_sample, full resolution over the visible area),
    pose-gated recalled old frames only fill holes; where sampling is insufficient a clean black hole is left. store: gid->(depth[h,w], intr[3,3], c2w[4,4], rgb[3,h,w]@depth-res in[0,1]).
    K_pix [3,3] @(H,W) target render resolution; target_c2ws [F,4,4] -> warp [1,3,F,H,W] in[-1,1], vis [1,1,F,H,W]. empty store -> all black (cold start)."""
    import torch.nn.functional as _F
    H, W = int(height), int(width); F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)
    M = 2000; FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}
    P_all = torch.full((len(ids_all), M, 3), FAR, device=device); subpts = {}
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]   # A2: zeroed-depth pixels unproject to a fake point at the camera centre and must be excluded (they pollute the covis ordering)
        if wp.shape[0] > 0:
            sp = wp[torch.randint(0, wp.shape[0], (M,), device=device)]
            subpts[g] = sp; P_all[id2row[g]] = sp
        else:
            subpts[g] = wp

    def covis_vec(rows, tpose):
        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]   # fix: subtract the camera centre (the c2w translation), not the w2c translation; otherwise R@(P-t) is not world->camera
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def est_scale(g, ref_pts, tposepts_n=200):
        """align the DA3 depth scale of source g to the reference cloud ref_pts (the nearby primary-source cloud): project ref_pts into camera g and take the median ratio against g's depth.
        too little overlap or an absurd value -> return 1.0 (no-op). prevents cross-chunk DA3 scale drift from putting recalled hole-fill content at the wrong depth (one of the real reasons history camera control never got learned)."""
        d, it, cwi, _ = store[g]; h, w = d.shape
        if ref_pts is None or ref_pts.shape[0] == 0:
            return 1.0
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0
        D = torch.full((h * w,), float("inf"), device=device)
        D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        if int(v.sum()) < tposepts_n:
            return 1.0
        s = float((D[v] / df[v]).median())
        return s if 0.2 < s < 5.0 else 1.0

    def bwarp_one(g, tpose, scale=1.0):
        d, it, cwi, rr = store[g]; h, w = d.shape
        if scale != 1.0:
            d = d * scale
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        z = d
        Xc = (xs - it[0, 2]) / it[0, 0] * z; Yc = (ys - it[1, 2]) / it[1, 1] * z
        cam = torch.stack([Xc, Yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (cwi @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(tpose); ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]; zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        src_flat = torch.arange(h * w, device=device)
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        key = (yt * W + xt)[ok]; zq = (zt.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)[ok]
        packed = (zq << 24) | src_flat[ok]
        INF = torch.full((H * W,), (1 << 62), dtype=torch.long, device=device)
        INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = INF < (1 << 62); owner = (INF & ((1 << 24) - 1)).clamp(max=h * w - 1)
        us = (owner % w).float(); vs = (owner // w).float()
        uv = torch.stack([us, vs], 0).reshape(2, H, W); vmask = valid.reshape(H, W)
        kern = torch.ones(1, 1, 3, 3, device=device); cur_uv = uv * vmask[None]; cur_v = vmask.float()[None, None]
        for _ in range(int(fill_iters)):
            if cur_v.min() > 0:
                break
            num = _F.conv2d((cur_uv * vmask[None])[None].reshape(2, 1, H, W), kern, padding=1).reshape(2, H, W)
            cnt = _F.conv2d(cur_v, kern, padding=1)[0, 0]
            newly = (cnt > 0) & (~vmask); filled_uv = num / cnt.clamp(min=1)[None]
            cur_uv = torch.where(vmask[None], cur_uv, filled_uv); vmask = vmask | newly; cur_v = vmask.float()[None, None]
        gx = cur_uv[0] / max(w - 1, 1) * 2 - 1; gy = cur_uv[1] / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        samp = _F.grid_sample(rr[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        col = torch.where(vmask[None].expand(3, H, W), samp, torch.zeros_like(samp))
        return col, vmask

    tc = target_c2ws.to(device).float()
    nearby_ids = ids_all[-int(nearby):]; old_ids = [g for g in ids_all if g not in nearby_ids]
    nearby_rows = [id2row[g] for g in nearby_ids]; old_rows = [id2row[g] for g in old_ids]
    nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]) if nearby_ids else None
    nref = torch.cat([subpts[g] for g in nearby_ids if subpts[g].shape[0] > 0]) if nearby_ids else None  # reference for the recall scale alignment
    warps = []; viss = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(nearby_rows, tpose)
        primary = nearby_ids[int(cv.argmax())]
        fused, filled = bwarp_one(primary, tpose)
        if old_ids:
            cvo = covis_vec(old_rows, tpose); oi = int(cvo.argmax()); go = old_ids[oi]
            tcen = tpose[:3, 3]; tfwd = tpose[:3, 2]
            d_old = float((store[go][2][:3, 3] - tcen).norm())
            d_nb_min = float((nb_cen - tcen).norm(dim=1).min()) if nb_cen is not None else 1e9
            of = store[go][2][:3, 2]; cos_fwd = float((of @ tfwd) / (of.norm() * tfwd.norm() + 1e-9))
            pose_ok = (d_old <= max(d_nb_min * 1.5, 1e-6)) and (cos_fwd > 0.5)
            if pose_ok and float(cvo[oi]) >= recall_min_cov and float(cvo[oi]) > float(cv.max()) + recall_margin:
                cr, vr = bwarp_one(go, tpose, scale=est_scale(go, nref)); m = vr & (~filled)
                fused = torch.where(m[None].expand(3, H, W), cr, fused); filled = filled | vr
        warps.append((fused.clamp(0, 1) * 2 - 1)); viss.append(filled.float())
    return torch.stack(warps, 1)[None].contiguous(), torch.stack(viss)[None, None].contiguous()


def _render_backward_multisrc_zbuf(store, ids_all, target_c2ws, K_pix, height, width, *,
                                   nearby=16, fill_iters=12, recall_min_cov=0.5, recall_margin=0.15,
                                   depth_thresh=0.02, topk=8,
                                   fg_covis=0.3, fg_factor=1.5,        # far-source gate (ON by default): drop the sources with "covis<fg_covis and distance>fg_factor x the nearest near source"
                                   fg_scale_exempt=1.0,                # ON (experimentally validated): geometrically self-consistent far sources (est_scale aligned with n>=200 and scale in [0.8,1.25]) are exempt and not cut -> rescues oblique distant legitimate revisits (memory 22->23/24) without letting straight-line poison sources back in (no overlap -> no exemption, the mall reset stays fixed)
                                   zbuf_despeckle=False, zbuf_despeckle_ksize=3, zbuf_despeckle_fill_iters=4,
                                   device="cuda"):
    """backward-warp render core (optional): per-pixel top-K multi-source z-buffer fusion.

    differences from `_render_backward` -- switched by cloud_warp.render_mode (backward / backward_zbuf):
      - render_mode=backward (_render_backward): each target frame takes only 1 primary source (argmax covis) + a single recalled frame for hole filling
        + fill_iters 3x3 dilation fills the holes and marks the dilated pixels visible -> prone to visible seams where the primary source switches at chunk boundaries or the depth scale drifts.
      - render_mode=backward_zbuf (this function): z-buffer fusion over the top-K candidate sources **per pixel** (the nearest target depth wins,
        within the depth_thresh band), **no dilation**, uncovered pixels stay holes and are marked invisible. multi-source fusion no longer hard-switches a single primary source,
        so content is more stable with no chunk-boundary seams (measured in).

    signature / returns are identical to `_render_backward` (directly interchangeable):
      store: gid->(depth[h,w], intr[3,3], c2w[4,4], rgb[3,h,w]@depth-res in[0,1]);
      K_pix [3,3] @(H,W) target render resolution; target_c2ws [F,4,4]
      -> warp [1,3,F,H,W] in[-1,1], vis [1,1,F,H,W]. empty store -> all black (cold start).
    `fill_iters` is accepted but ignored (no dilation on this path); `recall_*` is accepted only for signature alignment (per-pixel fusion already covers all sources, so no separate recall is needed).
    depth_thresh/topk default to 0.02/8 and can be overridden by the env vars WARP_ZBUF_DEPTH_THRESH / WARP_ZBUF_TOPK (for tuning).
    zbuf_despeckle: per-frame "hybrid despeckle" post-processing (off by default; validated in). the zbuf single-pixel splat does no hole filling and
      leaves ~17% salt-and-pepper (scattered invalid pixels interleaved with valid content), which the VAE 8x downsampling averages into grey mush -> a poisoned training label. once on, the valid mask
      goes through a morphological open->close (ksize kernel): open clears boundary specks marked valid (salt -> a clean hole handed to the model to generate), close fills fully enclosed interior pixels (pepper = splat
      misses, content known, safe), genuinely large holes are untouched; newly filled interior pixels are filled iteratively with the neighbourhood mean (zbuf_despeckle_fill_iters times) and cleared specks are set to black.
      when off (zbuf_despeckle=False) the whole block is bypassed and the behaviour is byte-identical to the original."""
    import torch.nn.functional as _F
    H, W = int(height), int(width); F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:                                          # cold start: same as single_primary, all black / all invisible
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)
    # depth_thresh: the target-depth "proximity band" (only a closer candidate may overwrite); DA3 depth is roughly normalised -> a relatively small band by default.
    depth_thresh = float(os.environ.get("WARP_ZBUF_DEPTH_THRESH", str(depth_thresh)))
    topk = int(os.environ.get("WARP_ZBUF_TOPK", str(topk)))   # number of source frames fused per pixel for each target frame
    M = int(os.environ.get("WARP_ZBUF_COVIS_M", "2000"))      # covis subsample point count (larger = a more stable covis estimate; for diagnostics / ablations)
    _covis_min = float(os.environ.get("WARP_ZBUF_COVIS_MIN", "0"))  # >0: keep only the "near" sources with covis>=this threshold and drop low-scoring far sources (leaving the area ahead black for the DiT)
    FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}
    # an independent RNG for the covis subsampling (isolated from the global / DiT RNG): set EVOKE_WARP_SEED to reproduce / sweep the source-selection randomness
    _wseed = os.environ.get("EVOKE_WARP_SEED")
    _cgen = torch.Generator(device=device).manual_seed(int(_wseed)) if _wseed is not None else None

    # -- per-source subsampled world points (for the covis ordering; the same set as single_primary) --
    P_all = torch.full((len(ids_all), M, 3), FAR, device=device)
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)
        # A2 fix: pixels whose depth was zeroed by a gate / carve unproject to the [camera centre] (finite, so they pass the isfinite filter) -> for heavily gated frames
        # most of P_all is the same fake point -> the covis ordering / far-source gate / nref (the est_scale reference) all get polluted. must be excluded via d>1e-4.
        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]
        if wp.shape[0] > 0:
            _idx = (torch.randint(0, wp.shape[0], (M,), device=device, generator=_cgen)
                    if _cgen is not None else torch.randint(0, wp.shape[0], (M,), device=device))
            P_all[id2row[g]] = wp[_idx]

    def covis_vec(rows, tpose):
        """the fraction of source g's subsampled points that fall inside the target frustum (same implementation as single_primary)."""
        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]   # fix: subtract the camera centre (the c2w translation), not the w2c translation; otherwise R@(P-t) is not world->camera
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def splat_one(g, tpose, scale=1.0):
        """forward-splat a single source into the target. returns col[3,H,W] / zbuf[H*W] (the winner target depth, inf=no hit) / vmask[H,W].
        z-buffer: each target pixel keeps the source pixel with the **nearest target depth** (the same packed-amin trick as single_primary), no dilation.
        scale: the cross-chunk DA3 scale alignment factor for recalled old frames (nearby primary source=1.0; old frames are aligned to the nearby reference by est_scale)."""
        d, it, cwi, rr = store[g]; h, w = d.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        z = d.float()
        if scale != 1.0:                                       # cross-chunk recall scale alignment (same convention as _render_backward)
            z = z * scale
        Xc = (xs - it[0, 2]) / it[0, 0] * z; Yc = (ys - it[1, 2]) / it[1, 1] * z
        cam = torch.stack([Xc, Yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (cwi @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(tpose); ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]; zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        src_flat = torch.arange(h * w, device=device)
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        col = torch.zeros(3, H, W, device=device)
        zbuf = torch.full((H * W,), float("inf"), device=device)
        if not bool(ok.any()):
            return col, zbuf, torch.zeros(H, W, dtype=torch.bool, device=device)
        key = (yt * W + xt)[ok]; zt_ok = zt[ok]; src_ok = src_flat[ok]
        # winner = smallest target depth: pack (quantised depth<<24)|source index, then amin (same as single_primary)
        zq = (zt_ok.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)
        packed = (zq << 24) | src_ok
        INF = torch.full((H * W,), (1 << 62), dtype=torch.long, device=device)
        INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = INF < (1 << 62); owner = (INF & ((1 << 24) - 1)).clamp(max=h * w - 1)
        zbuf[valid] = ((INF[valid] >> 24).float()) / 1000.0    # the winner target depth (before quantisation)
        us = (owner % w).float(); vs = (owner // w).float()
        gx = us / max(w - 1, 1) * 2 - 1; gy = vs / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx.reshape(H, W), gy.reshape(H, W)], -1)[None]
        samp = _F.grid_sample(rr[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        vmask = valid.reshape(H, W)
        col = torch.where(vmask[None].expand(3, H, W), samp, torch.zeros_like(samp))
        return col, zbuf, vmask

    def est_scale(g, ref_pts, tposepts_n=200):
        """align the DA3 depth scale of recalled source g to the reference cloud ref_pts (the nearby primary-source cloud): project ref_pts into camera g and take the median ratio against g's depth.
        same convention as _render_backward.est_scale; too little overlap / absurd value -> 1.0 (no-op). prevents cross-chunk DA3 scale drift from putting recalled old frames at the wrong depth
        -> kicked out by the z-buffer comparison / projected to the wrong place -> recall stops working (worst on straight segments, where the new field of view ahead relies entirely on recalled history).
        returns (scale, n_overlap): a large n_overlap = real geometric overlap with the nearby scene (trustworthy); =0 -> a source we walked past with no overlap."""
        d, it, cwi, _ = store[g]; h, w = d.shape
        if ref_pts is None or ref_pts.shape[0] == 0:
            return 1.0, 0
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0, int(ok.sum())
        D = torch.full((h * w,), float("inf"), device=device)
        D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        nov = int(v.sum())
        if nov < tposepts_n:
            return 1.0, nov
        s = float((D[v] / df[v]).median())
        return (s if 0.2 < s < 5.0 else 1.0), nov

    # -- recall scale alignment: nearby (the most recent nearby frames) = the reference scale; the rest = recalled old frames, aligned to the nearby cloud before entering the z-buffer --
    nearby_ids = ids_all[-int(nearby):]; nearby_set = set(nearby_ids)
    # far-source gate (enabled when WARP_ZBUF_FG_COVIS>0): drop only the sources with "covis<tau and distance>factor x the distance of the nearest near source" (low score and far = poison sources we walked past);
    # near sources (kept by the distance term) stay regardless of covis -> the anchor can never be cut away (avoids a blunt covis_min cutting near sources and collapsing pred). memory safety: revisited frames have high covis / short distance and are always kept.
    _fg_covis = float(os.environ.get("WARP_ZBUF_FG_COVIS", fg_covis))     # default 0.3 (experimentally validated: fixes straight-line false recall + keeps memory on 92% of rotate-and-return cases); env can override / disable (=0)
    _fg_factor = float(os.environ.get("WARP_ZBUF_FG_FACTOR", fg_factor))
    _nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]).to(device).float() if (nearby_ids and _fg_covis > 0) else None
    _all_cen = {g: store[g][2][:3, 3].to(device).float() for g in ids_all} if _fg_covis > 0 else None
    _nref_l = [P_all[id2row[g]][P_all[id2row[g]][:, 0] < FAR * 0.5] for g in nearby_ids]
    _nref_l = [p for p in _nref_l if p.shape[0] > 0]
    nref = torch.cat(_nref_l) if _nref_l else None
    _es = {g: ((1.0, 0) if g in nearby_set else est_scale(g, nref)) for g in ids_all}
    src_scale = {g: _es[g][0] for g in ids_all}                                # per-source alignment scale for old frames (constant, unchanged across frames)
    _fg_scale_exempt = float(os.environ.get("WARP_ZBUF_FG_SCALE_EXEMPT", str(fg_scale_exempt)))
    # far-source gate exemption: far + low covis but "geometrically self-consistent" (real overlap with the nearby scene, n>=200, and an alignment scale in [0.8,1.25]) legitimate recalls -> not cut (rescues oblique distant revisits)
    src_aligned = ({g: (g not in nearby_set and _es[g][1] >= 200 and 0.8 <= _es[g][0] <= 1.25) for g in ids_all}
                   if _fg_scale_exempt > 0 else {})

    tc = target_c2ws.to(device).float()
    all_rows = [id2row[g] for g in ids_all]
    warps = []; viss = []
    _age_dbg = bool(os.environ.get("EVOKE_WARP_AGE_DEBUG"))   # per-chunk attribution of the winner source frame age (off by default, no side effects)
    _age_rows = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(all_rows, tpose)                       # take the top-K candidate sources by covis (the original took only argmax=top-1)
        k = min(int(topk), len(ids_all)); order = torch.topk(cv, k=k).indices.tolist()
        if _covis_min > 0:                                    # blunt cut: keep only covis>=threshold (may cut near sources by mistake -> blur, use with care)
            order = [i for i in order if float(cv[i]) >= _covis_min]
        if _fg_covis > 0 and _nb_cen is not None:             # far-source gate: drop poison sources that are "low scoring (covis<tau) and far (dist>factor x the nearest near source)"; near sources are kept as a fallback
            _tcen = tpose[:3, 3]; _dnb = float((_nb_cen - _tcen).norm(dim=1).min())
            order = [i for i in order if float(cv[i]) >= _fg_covis
                     or float((_all_cen[ids_all[i]] - _tcen).norm()) <= _fg_factor * _dnb
                     or src_aligned.get(ids_all[i], False)]      # exemption for geometrically self-consistent far sources (enabled with WARP_ZBUF_FG_SCALE_EXEMPT=1)
        cand = [ids_all[i] for i in order]
        fused = torch.zeros(3, H, W, device=device)
        fused_depth = torch.full((H * W,), float("inf"), device=device)
        covered = torch.zeros(H, W, dtype=torch.bool, device=device)
        winner_row = torch.full((H * W,), -1, dtype=torch.long, device=device) if _age_dbg else None
        for g in cand:                                        # per-pixel z-buffer fusion (candidate_depth < fused_depth)
            col, zbuf, _vm = splat_one(g, tpose, scale=src_scale[g])   # recalled old frames use the aligned scale
            update = torch.isfinite(zbuf) & (zbuf < fused_depth - depth_thresh)   # only overwrite when closer (beyond the proximity band)
            if not bool(update.any()):
                update = torch.isfinite(zbuf) & (zbuf < fused_depth)             # but empty pixels (inf) are still filled by the first hit
                if not bool(update.any()):
                    continue
            um = update.reshape(H, W)
            fused = torch.where(um[None].expand(3, H, W), col, fused)
            fused_depth = torch.where(update, zbuf, fused_depth); covered = covered | um
            if _age_dbg:
                winner_row[update] = int(id2row[g])           # record the z-buffer winning source for this pixel (for frame-age attribution)
        if _age_dbg:
            _age_rows.append(winner_row[covered.reshape(-1)])  # winning source row indices of every covered pixel in this frame
        # uncovered pixels stay 0 (mapped to black below) and are marked invisible -- the same convention as single_primary's torch.where(vmask,...); no dilation.
        # -- (optional) hybrid despeckle: per-frame morphological open->close to clear boundary specks / fill fully enclosed interior pixels (validated in) --
        #   when off the whole block is bypassed -> fused/covered unchanged -> byte-identical to the original. when on it rewrites the (color, covered) appended for this frame.
        if zbuf_despeckle:
            k = int(zbuf_despeckle_ksize)
            erosion = lambda x: -_F.max_pool2d(-x, k, 1, k // 2)
            dilation = lambda x: _F.max_pool2d(x, k, 1, k // 2)
            V = covered.float()[None, None]                            # (1,1,H,W)
            opened = dilation(erosion(V))                              # open: clear the specks marked valid (salt) -> the boundary band becomes a clean hole
            hv = erosion(dilation(opened))                             # close: fill the fully enclosed invalid pixels (pepper) = splat misses
            hv_b = hv[0, 0] > 0.5                                      # the valid mask after the hybrid step
            fill_mask = hv_b & (~covered)                             # newly filled interior pixels (black before, valid now) -> coloured with the neighbourhood mean
            removed = covered & (~hv_b)                               # cleared boundary specks (valid before, a hole now) -> set to black
            col = fused.clone()
            m = covered.float()[None, None]                           # the current valid mask (grows as colours are filled in)
            for _ in range(int(zbuf_despeckle_fill_iters)):           # iteratively fill fill_mask with the neighbourhood mean starting from the known valid pixels
                num = _F.avg_pool2d(col[None] * m, 3, 1, 1); den = _F.avg_pool2d(m, 3, 1, 1)
                upd = fill_mask & (den[0, 0] > 1e-6) & (m[0, 0] < 0.5)
                col = torch.where(upd[None].expand(3, -1, -1), (num / den.clamp(min=1e-6))[0], col)
                m[0, 0] = torch.where(upd, torch.ones_like(m[0, 0]), m[0, 0])
            col[:, removed] = 0.0                                     # specks cleared into holes -> black (handed to the model to generate)
            warps.append((col.clamp(0, 1) * 2 - 1)); viss.append(hv_b.float())
            continue
        warps.append((fused.clamp(0, 1) * 2 - 1)); viss.append(covered.float())
    vis_out = torch.stack(viss)[None, None].contiguous()
    # instrumentation (only printed on low coverage to avoid spam): distinguishes "empty store / cold start" (A) from "recall sources exist but got kicked out / drifted in scale" (B).
    _mean_cov = float(vis_out.mean())
    if _mean_cov < 0.25:
        _n_old = int(sum(1 for g in ids_all if g not in nearby_set))
        _n_aligned = int(sum(1 for g in ids_all if abs(float(src_scale[g]) - 1.0) > 1e-6))
        print(f"[zbuf-render] LOW-COV mean_cov={_mean_cov*100:.1f}% src={len(ids_all)} "
              f"nearby={len(nearby_ids)} old(recall)={_n_old} aligned={_n_aligned} "
              f"nref={'0' if nref is None else nref.shape[0]} F={F_t}", flush=True)
    if _age_dbg and _age_rows:                              # per chunk: the "frame age" distribution of the z-buffer winning sources over the covered pixels (fps=24 -> gid/24=seconds)
        _wr = torch.cat(_age_rows)
        if _wr.numel() > 0:
            _ids_t = torch.as_tensor(ids_all, device=_wr.device)
            _gid = _ids_t[_wr]; _age = (int(ids_all[-1]) - _gid).float() / 24.0   # seconds
            _n = float(_wr.numel())
            _b = lambda lo, hi: 100.0 * float(((_age >= lo) & (_age < hi)).sum()) / _n
            _oldest = int(_gid.min().item()); _newest_win = int(_gid.max().item())
            _n_old_pool = int(sum(1 for g in ids_all if (int(ids_all[-1]) - g) > 120))   # number of "old frames" (>5s) in the pool
            print(f"[warp-age] winners={int(_n)} bank_gid=[{ids_all[0]}..{ids_all[-1]}] pool={len(ids_all)} old_in_pool={_n_old_pool} | "
                  f"age<2s={_b(0,2):.1f}% 2-5s={_b(2,5):.1f}% 5-10s={_b(5,10):.1f}% "
                  f"10-20s={_b(10,20):.1f}% 20s+={_b(20,1e9):.1f}% | "
                  f"winner_gid=[{_oldest}..{_newest_win}] frac_from_ge5s={_b(5,1e9):.1f}%", flush=True)
    return torch.stack(warps, 1)[None].contiguous(), vis_out


# --------------------- i2v chunk-0 target-disparity scale ---------------------
# Default budget for the chunk-0 rescale below, in warp-render pixels. Calibrated, not guessed
# (, summarised here because that path is not in the repo):
#
# 12 i2v reference frames spanning macro / room-scale / aerial, all driven by ONE pose track so the only
# variable is the reference frame's depth histogram, swept over off / 45 / 90 / 135 / 180 px. Scored
# against chunks 1-3, which re-anchor depth_median on a multi-frame window of generated content and are
# therefore the amplitude chunk 0 should match:
#
#   arm  | chunk-0 warp cov | median flow(c0)/flow(c1..3) | cross-scene spread of that ratio
#   off  |      0.445       |            5.95x            |  7.68
#   45   |      0.998       |            0.67x            |  4.39
#   90   |      0.994       |            0.80x            |  3.22
#   135  |      0.991       |            0.88x            |  2.76   <- interior minimum
#   180  |      0.988       |            0.87x            |  3.55
#
# The coverage cliff is between `off` and any enabled value, not among the enabled ones, so coverage does
# not argue against 135; the spread column turning back up at 180 is what makes 135 a real optimum rather
# than the best endpoint of the swept range.
CHUNK0_TARGET_DISPARITY_PX_DEFAULT = 135.0


@torch.no_grad()
def solve_chunk0_disparity_scale(depth, intr_depth, ref_c2w, target_c2ws, K_out, target_px,
                                 *, quantile=0.9, k_limits=(1.0 / 64.0, 64.0)):
    """Solve the one depth-scale factor `k` that puts chunk-0's commanded parallax at `target_px`.

    WHY this exists. chunk 0's cloud scale comes from `depth_single` on the reference image alone:
    c = median(ViGeo depth) / depth_median_target, i.e. the median depth is pinned to `target` and the
    commanded translation is then read in units of scene depth. Median-pinning says nothing about the
    NEAR field, which is what parallax is actually made of (displacement ~ 1/d), and one frame gives no
    cross-frame check on it. So a close-up reference (a macro shot, an arm filling the frame) and a
    landscape reference get wildly different chunk-0 amplitudes from the identical pose track --
    measured 0.29x to 3.14x of the following chunks across six demo_hours cases, with absolute per-chunk
    flow spanning 0.19 to 4.44 px/frame. Later chunks do not share the problem: they re-anchor on a
    multi-frame window of generated content.

    WHAT it solves. For a pure-translation step (rotation is deliberately excluded -- rotational
    displacement does not scale with depth, so letting it into the solve would shrink real parallax to
    pay for a pan), a pixel with ray m = K_depth^-1 [u,v,1] (m_z = 1) at depth d lands, after scaling
    depth by k, at a distance

        |dx| = C / (k*d + dz),   C = |( fx_out*(dx_ - m_x*dz), fy_out*(dy_ - m_y*dz) )|

    from where it started, with (dx_, dy_, dz) = R_ref^T (t_ref - t_i) the translation in the reference
    camera frame. Both image components share the denominator, so |dx| is exact and strictly decreasing
    in k -- no small-angle approximation, and `dz` matters here precisely because these tracks translate
    by a large fraction of the scene depth per chunk.

    Inverting per pixel gives the k above which that pixel sits below target:

        k_p = (C/target_px - dz) / d

    and the fraction of pixels above target at a given k is the fraction with k_p > k. So the q-quantile
    of the displacement equals target_px exactly when k = q-quantile of {k_p} -- a closed form, no
    iteration. Taken per target frame and maxed over frames, so `target_px` reads as "at chunk 0's
    largest commanded translation, 90 % of pixels have moved no further than this".

    depth [h,w] (already at GT scale) + intr_depth [3,3] @depth-res + ref_c2w [4,4] +
    target_c2ws [F,4,4] + K_out [3,3] @render-res -> (k, info dict). k is clamped to k_limits: an
    unusable reference frame must not be able to fling the cloud to infinity, which is the failure mode
    this whole function exists to bound.
    """
    d = depth.float()
    h, w = int(d.shape[0]), int(d.shape[1])
    tgt = float(target_px)
    if not (tgt > 0):
        return 1.0, {"reason": "target_px<=0"}

    fx_d, fy_d = float(intr_depth[0, 0]), float(intr_depth[1, 1])
    cx_d, cy_d = float(intr_depth[0, 2]), float(intr_depth[1, 2])
    ys, xs = torch.meshgrid(torch.arange(h, device=d.device, dtype=torch.float32),
                            torch.arange(w, device=d.device, dtype=torch.float32), indexing="ij")
    m_x = (xs - cx_d) / fx_d
    m_y = (ys - cy_d) / fy_d
    fx_o, fy_o = float(K_out[0, 0]), float(K_out[1, 1])

    R0 = ref_c2w[:3, :3].float()
    t0 = ref_c2w[:3, 3].float()
    valid = torch.isfinite(d) & (d > 0)
    if not bool(valid.any()):
        return 1.0, {"reason": "no valid depth"}
    d_safe = torch.where(valid, d, torch.ones_like(d))

    best = (0.0, -1, 0.0)                       # (k, frame, |delta| of that frame)
    for i in range(int(target_c2ws.shape[0])):
        delta = R0.transpose(0, 1) @ (t0 - target_c2ws[i, :3, 3].float())    # ref-camera frame
        dx_, dy_, dz = float(delta[0]), float(delta[1]), float(delta[2])
        C = torch.sqrt((fx_o * (dx_ - m_x * dz)) ** 2 + (fy_o * (dy_ - m_y * dz)) ** 2)
        k_p = ((C / tgt) - dz) / d_safe
        # k_p <= 0 means no positive k can push this pixel past target (it is already inside it);
        # such pixels must not drag the quantile negative, so they are floored, not dropped -- dropping
        # them would change the population the quantile is taken over.
        k_p = torch.where(valid, k_p, torch.zeros_like(k_p)).clamp_min(0.0)
        k_i = float(torch.quantile(k_p.reshape(-1), float(quantile)))
        if k_i > best[0]:
            best = (k_i, i, float(np.linalg.norm([dx_, dy_, dz])))

    k, frame, tnorm = best
    info = {"frame": frame, "trans_norm": tnorm, "k_raw": k, "clamped": False}
    if not np.isfinite(k) or k <= 0:
        return 1.0, {**info, "reason": "degenerate solve (no commanded translation?)"}
    lo, hi = float(k_limits[0]), float(k_limits[1])
    if k < lo or k > hi:
        info["clamped"] = True
        k = min(max(k, lo), hi)
    # What the parallax WOULD have been without the rescale, at the frame that set k -- the number that
    # makes the correction auditable in the log.
    #
    # `k*d + dz` is the point's depth after the step, and dz is NEGATIVE for a forward walk (delta points
    # backwards from the target back to the reference), so at k=1 with a median depth of 1.0 pose unit and
    # ~1.9 units of commanded translation, most of the cloud ends up BEHIND the camera: the commanded
    # motion walks clean through the scene. A displacement is then not merely large, it is undefined, and
    # an earlier version of this print clamped the denominator and reported a meaningless 5.7e8 px. Report
    # the two numbers honestly instead: how much of the cloud the camera would have passed, and the p90
    # over what remains in front (inf once more than (1-quantile) of it is behind, because then the
    # quantile itself lies in the passed-through part).
    i = frame
    delta = R0.transpose(0, 1) @ (t0 - target_c2ws[i, :3, 3].float())
    dx_, dy_, dz = float(delta[0]), float(delta[1]), float(delta[2])
    C = torch.sqrt((fx_o * (dx_ - m_x * dz)) ** 2 + (fy_o * (dy_ - m_y * dz)) ** 2)
    z_after = d_safe + dz
    front = valid & (z_after > 1e-6)
    n_valid = int(valid.sum())
    behind_frac = 1.0 - (float(front.sum()) / max(n_valid, 1))
    info["behind_frac_before"] = behind_frac
    if behind_frac > (1.0 - float(quantile)) or not bool(front.any()):
        info["disp_p90_before"] = float("inf")
    else:
        disp0 = torch.where(front, C / z_after.clamp_min(1e-6), torch.zeros_like(C))
        # quantile over the valid population, re-mapped so the passed-through pixels still count as worse
        # than anything in front: they sit at the top of the order, which is where they belong.
        q_adj = (float(quantile) - behind_frac) / max(1.0 - behind_frac, 1e-6)
        vals = disp0[front].reshape(-1)
        info["disp_p90_before"] = float(torch.quantile(vals, min(max(q_adj, 0.0), 1.0)))
    return float(k), info


@torch.no_grad()
def build_single_source_warp_mono(estimator, ref_frame_pix, ref_pose_c2w, K_pix, target_c2ws,
                                  *, height, width, splat_radius=2, device="cuda",
                                  chunk0_target_disparity_px=CHUNK0_TARGET_DISPARITY_PX_DEFAULT):
    """i2v chunk-0 warp from the single reference image (inference-only).

    The training i2v path (online_materialize -> build_single_source_warp) gives EVERY chunk, the first
    included, a single-source warp from that chunk's own first frame, with the depth scale solved from
    >=3 posed frames of the chunk. At inference chunk 0 those frames do not exist yet, so that function
    cannot run and chunk 0 gets a blank warp (pool=0 cov=0) -> no camera signal -> the model just
    continues the still reference image. This builds the equivalent from the one frame we DO have:
    monocular depth of the reference image (estimator.depth_single), scale pinned by depth_median -- the
    same convention the rollout's later chunks use, so chunk-0 parallax is at the same scale as chunk 1 --
    unprojected at the reference pose, rendered at chunk-0's target poses. This is the ONLY thing that
    gives i2v chunk 0 a camera condition, because warp is this model's only camera channel.

    ref_frame_pix [C,H,W] or [1,C,H,W] in [-1,1]; ref_pose_c2w [4,4]; K_pix [3,3] @(H,W);
    target_c2ws [F,4,4]. -> warp [1,3,F,H,W] in [-1,1], vis [1,1,F,H,W].

    Any failure -- a backend with no single-frame path (DA3), an unsupported scale mode (per_window /
    anchor), or degenerate depth -- returns a blank warp, exactly the pre-existing cold-start behaviour;
    it never raises, so a bad reference frame can only cost the chunk-0 warp, never the whole rollout.

    `chunk0_target_disparity_px` > 0 rescales that monocular depth so chunk 0's commanded parallax lands
    on a fixed pixel budget instead of wherever the reference frame's depth histogram happened to put it
    (see solve_chunk0_disparity_scale). It touches chunk 0 only -- every later chunk re-anchors on its
    own multi-frame window and never reaches this function. 0 disables it, restoring the pure
    depth_median behaviour byte for byte.
    """
    H, W = int(height), int(width)
    target_c2ws = target_c2ws.to(device=device, dtype=torch.float32)
    F_t = int(target_c2ws.shape[0])

    def _blank():
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))

    depth_single = getattr(estimator, "depth_single", None)
    if not callable(depth_single):
        return _blank()                                      # DA3 backend has no single-frame path yet
    ref = ref_frame_pix
    if ref.ndim == 4:
        ref = ref[0]
    frame = (ref.detach().permute(1, 2, 0).float() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()   # [H,W,3] in [0,1]
    K_np = np.asarray(K_pix, np.float32)
    try:
        dep, intr, rgb = depth_single(frame, K_np)           # depth [h,w] GT scale, intr [3,3] depth-res, rgb [h,w,3]
    except NotImplementedError as _e:
        # An unsupported scale_mode, NOT a bad reference frame. Kept distinct because the generic message
        # below reads as a data problem and sent a debugging round down the wrong path: the real meaning is
        # "this recipe cannot give chunk 0 a warp", i.e. the chunk-0 flag is silently doing nothing.
        # infer_single rejects this combination up front; reaching it means another caller (training
        # validation) built the estimator directly.
        print(f"[da3-single-src-mono] WARN chunk-0 reference warp DISABLED by the depth recipe: {_e}. "
              f"Chunk 0 gets a BLANK warp, so it has no camera signal and will sit static -- the same as "
              f"chunk0_ref_warp=off. Use scale_mode=depth_median to enable it.", flush=True)
        return _blank()
    except Exception as _e:
        print(f"[da3-single-src-mono] WARN reference-frame depth failed "
              f"({type(_e).__name__}: {_e}); blank chunk-0 warp", flush=True)
        return _blank()
    d0 = torch.as_tensor(dep, device=device)
    it0 = torch.as_tensor(intr, device=device)
    # ref_pose_c2w may arrive as a CUDA tensor (target poses live on device) or a numpy array; np.asarray
    # cannot convert a CUDA tensor, so branch on type.
    if isinstance(ref_pose_c2w, torch.Tensor):
        cw0 = ref_pose_c2w.detach().to(device=device, dtype=torch.float32)
    else:
        cw0 = torch.as_tensor(np.asarray(ref_pose_c2w, np.float32), device=device)
    if float(chunk0_target_disparity_px) > 0:
        # Rescale BEFORE unprojecting: the cloud, the render and the reported coverage then all see one
        # consistent geometry, and nothing downstream needs to know this happened.
        _k, _info = solve_chunk0_disparity_scale(
            d0, it0, cw0, target_c2ws, torch.as_tensor(K_np, device=device),
            float(chunk0_target_disparity_px))
        if _info.get("reason"):
            print(f"[da3-single-src-mono] chunk0 disparity rescale SKIPPED ({_info['reason']}); "
                  f"depth left at depth_median scale", flush=True)
        else:
            d0 = d0 * float(_k)
            print(f"[da3-single-src-mono] chunk0 disparity rescale: k={_k:.4g}"
                  f"{' (CLAMPED)' if _info.get('clamped') else ''} -> p90 parallax "
                  f"{_info['disp_p90_before']:.1f} px => {float(chunk0_target_disparity_px):.1f} px "
                  f"(driving frame {_info['frame']}, |t|={_info['trans_norm']:.3f} pose units, "
                  f"{_info['behind_frac_before'] * 100:.0f}% of the cloud was behind the camera before)",
                  flush=True)
    xyz = unproject_depth_torch(d0, it0, cw0).reshape(-1, 3)                     # [P,3] world (GT-scale)
    rgb0 = torch.as_tensor(np.asarray(rgb, np.float32), device=device).reshape(-1, 3)   # [P,3] in [0,1]
    K_render = torch.as_tensor(K_np, device=device)[None].expand(F_t, 3, 3)     # GT K @ pixel-res
    return render_cloud_batched(xyz, rgb0, target_c2ws, K_render, H, W,
                                device=device, splat_radius=int(splat_radius), invisible_fill="black")
