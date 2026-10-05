"""V2V / camera-viz helpers: video/pose loading, joystick overlay, GT|pred side-by-side."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


def load_ref_video_for_v2v(
    video_path: str,
    height: int,
    width: int,
    seconds: float,
    target_fps: int = 24,
    source_fps: int = 30,
    start_seconds: float = 0.0,
) -> torch.Tensor:
    """Load a clip from source video, resample source_fps to target_fps. Returns [T, 3, H, W] in [-1, 1] fp32 CPU."""
    num_target_frames = max(1, int(round(seconds * target_fps)))
    video = _load_test_clip(
        video_path,
        num_target_frames=num_target_frames,
        source_fps=source_fps,
        target_fps=target_fps,
        height=height,
        width=width,
        start_seconds=start_seconds,
    )                                          # [3, T, H, W]
    return video.permute(1, 0, 2, 3).contiguous()    # [T, 3, H, W]


def _clamp_c2w_rotation(cam_c2w, max_deg, ref_idx: int = 0):
    """Clamp each frame's rotation so its angular deviation from the reference frame
    (default frame 0) does not exceed ``max_deg`` degrees. Translation is left untouched.

    Used by validation to constrain the camera trajectory to a high-overlap (small-turn)
    regime, so warp coverage stays dense and we can test whether the model follows warp.
    cam_c2w: np.ndarray [N,4,4]; returns a clamped copy with the same dtype.
    """
    import numpy as np
    if max_deg is None or float(max_deg) <= 0:
        return cam_c2w
    src_dtype = cam_c2w.dtype if hasattr(cam_c2w, "dtype") else np.float32
    arr = np.asarray(cam_c2w, dtype=np.float64).copy()
    max_rad = np.deg2rad(float(max_deg))
    R_ref = arr[ref_idx, :3, :3]
    for i in range(arr.shape[0]):
        R_rel = R_ref.T @ arr[i, :3, :3]
        cos = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
        ang = float(np.arccos(cos))
        if ang <= max_rad or ang < 1e-6:
            continue
        axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                         R_rel[0, 2] - R_rel[2, 0],
                         R_rel[1, 0] - R_rel[0, 1]])
        nrm = np.linalg.norm(axis)
        if nrm < 1e-8:
            continue
        axis /= nrm
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        # Rodrigues: rotate by exactly max_rad about the same axis as R_rel.
        R_clamped = np.eye(3) + np.sin(max_rad) * K + (1.0 - np.cos(max_rad)) * (K @ K)
        arr[i, :3, :3] = R_ref @ R_clamped
    return arr.astype(src_dtype)


def _extend_c2w_relative_replay(abs_c2w, total_frames):
    """Extend an absolute c2w trajectory to ``total_frames`` by *replaying the same
    per-frame local motion* (camera-frame deltas) composed onto the last real pose.

    Camera control is relative: abs[k] = abs[k-1] @ L[k], L[k] = inv(abs[k-1]) @ abs[k]
    is the motion expressed in the previous camera's local frame. To continue past the
    end of the real trajectory we cycle L[1..M-1] forward, anchored at the current pose —
    the camera keeps driving the same route pattern, appended continuously (no teleport,
    unlike a hard index-loop; no freeze, unlike clamping to the last frame).

    abs_c2w: np.ndarray [M,4,4]; returns [total_frames,4,4] (dtype preserved).
    """
    import numpy as np
    src_dtype = abs_c2w.dtype if hasattr(abs_c2w, "dtype") else np.float32
    a = np.asarray(abs_c2w, dtype=np.float64)
    M = a.shape[0]
    if M >= total_frames:
        return a[:total_frames].astype(src_dtype)
    out = np.zeros((total_frames, 4, 4), dtype=np.float64)
    out[:M] = a
    if M < 2:
        # not enough frames to derive motion → fall back to freeze (repeat last pose)
        out[M:] = a[-1]
        return out.astype(src_dtype)
    # local (previous-camera-frame) deltas of the real trajectory
    L = [np.linalg.inv(a[k - 1]) @ a[k] for k in range(1, M)]   # length M-1
    for j in range(M, total_frames):
        out[j] = out[j - 1] @ L[(j - M) % len(L)]
    return out.astype(src_dtype)


def load_pose_for_v2v(
    pose_path: str,
    target_height: int,
    target_width: int,
    source_resolution=(1080, 1920),
    pose_type: str = "vipe",
    num_target_frames: int | None = None,
    target_fps: int = 24,
    source_fps: int = 30,
    start_seconds: float = 0.0,
    max_rotation_deg: float = 0.0,
    fallback_default_intrinsic: bool = False,
    pose_extend_mode: str = "clamp",
):
    """Load pose npz, transform intrinsics to target resolution, resample fps. Returns (lingbot_Ks [4], lingbot_c2ws [N,4,4]).

    fallback_default_intrinsic: when True and the npz carries no intrinsics/intrinsic/K key
    (some store only `data`/`inds`), use a default normalized intrinsic
    [[1,0,0.5],[0,1,0.5],[0,0,1]]. transform_intrinsic_for_crop_resize auto-detects the
    normalized form (all values <= 2) and rescales to source_resolution. Default False keeps
    the sekai/vipe path byte-for-byte unchanged (raises KeyError if no intrinsics, as before).
    Also accepts `data` as an extrinsic key alias for `cam_c2w`.
    """
    import numpy as np
    from evoke.utils.inference_geometry import (
        resolve_intrinsic_source_resolution,
        transform_intrinsic_for_crop_resize,
    )

    data = np.load(pose_path, allow_pickle=True)
    if "intrinsic" in data:
        intrinsics_all = data["intrinsic"]                                             # [N,3,3] or [3,3]
    elif "intrinsics" in data:
        intrinsics_all = data["intrinsics"]
    elif "K" in data:
        intrinsics_all = data["K"]
    elif fallback_default_intrinsic:
        intrinsics_all = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        raise KeyError(f"pose npz has no intrinsics/intrinsic/K key: {pose_path}")
    if "cam_c2w" in data:
        cam_c2w_all = data["cam_c2w"]                                                   # [N,4,4]
    elif "data" in data:
        cam_c2w_all = data["data"]
    else:
        raise KeyError(f"pose npz has no cam_c2w/data key: {pose_path}")
    n_src = cam_c2w_all.shape[0]

    # resample to target_fps starting at start_seconds
    if num_target_frames is None:
        num_target_frames = max(1, int(n_src * target_fps / source_fps))
    start_src = int(round(start_seconds * source_fps))
    src_indices = [start_src + round(i * source_fps / target_fps) for i in range(num_target_frames)]
    # count how many resampled frames fall inside the real trajectory
    n_in_range = sum(1 for idx in src_indices if idx < n_src)
    if pose_extend_mode == "relative_replay" and 0 < n_in_range < num_target_frames:
        # take the real in-range poses, then replay the same local motion to fill the tail
        real_c2w = cam_c2w_all[[idx for idx in src_indices[:n_in_range]]]                # [n_in_range,4,4]
        cam_c2w = _extend_c2w_relative_replay(real_c2w, num_target_frames)               # [N_target,4,4]
        print(f"[pose] relative_replay: {n_in_range} real → {num_target_frames} frames "
              f"(tail {num_target_frames - n_in_range} replayed, no freeze/teleport)", flush=True)
    else:
        # default: clamp out-of-range indices to last frame (freezes camera past end of source)
        src_indices = [min(idx, n_src - 1) for idx in src_indices]
        cam_c2w = cam_c2w_all[src_indices]                                              # [N_target, 4, 4]
    # Optional: clamp rotation deviation from frame 0 to keep the clip high-overlap (warp-coverage test).
    if max_rotation_deg and float(max_rotation_deg) > 0:
        cam_c2w = _clamp_c2w_rotation(cam_c2w, max_rotation_deg)
    if intrinsics_all.ndim == 3:
        intrinsic = intrinsics_all[src_indices[0]]                                      # [3, 3]
    else:
        intrinsic = intrinsics_all                                                       # [3, 3]

    # transform intrinsics to target resolution using same crop-resize rules as training, including the
    # declared-vs-inferred source resolution decision (shared helper -> val/infer cannot drift from train)
    K_t = torch.from_numpy(intrinsic).float()
    h_src, w_src = resolve_intrinsic_source_resolution(
        K_t, int(source_resolution[0]), int(source_resolution[1]), tag=pose_path,
    )
    lingbot_Ks = transform_intrinsic_for_crop_resize(
        K_t, h_src, w_src, target_height, target_width,
    )                                                                                    # [4]

    lingbot_c2ws = torch.from_numpy(cam_c2w).float()                                     # [N_target, 4, 4]

    return lingbot_Ks, lingbot_c2ws


def _float_rgb_to_uint8(x) -> "np.ndarray":
    """Convert float or uint8 array to uint8 RGB; auto-detects [-1,1], [0,1], or [0,255] range."""
    import numpy as np

    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    if float(np.nanmin(arr_f)) < -0.05:
        # [-1, 1] range
        return np.clip((arr_f + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if float(np.nanmax(arr_f)) <= 1.5:
        # [0, 1] range
        return np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
    # already [0, 255] but float
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def _to_np_uint8_frames(video) -> "list[np.ndarray]":
    """Normalize any video type (tensor/ndarray/list of PIL/np/tensor) to list[H,W,3] uint8 RGB."""
    import numpy as np

    if isinstance(video, torch.Tensor):
        # normalize to [T, H, W, 3]
        if video.dim() == 4 and video.shape[0] == 3 and video.shape[1] != 3:
            video = video.permute(1, 2, 3, 0)        # [3, T, H, W] -> [T, H, W, 3]
        elif video.dim() == 4 and video.shape[1] == 3:
            video = video.permute(0, 2, 3, 1)        # [T, 3, H, W] -> [T, H, W, 3]
        elif video.dim() == 4 and video.shape[-1] == 3:
            pass                                       # [T, H, W, 3] — no-op
        else:
            raise ValueError(f"unrecognized tensor shape {tuple(video.shape)}")
        arr = _float_rgb_to_uint8(video.cpu().numpy())
        return [arr[i] for i in range(arr.shape[0])]

    if isinstance(video, np.ndarray):
        arr = _float_rgb_to_uint8(video)
        return [arr[i] for i in range(arr.shape[0])]

    if isinstance(video, list):
        out = []
        for f in video:
            if hasattr(f, "size") and not isinstance(f, np.ndarray):  # PIL.Image
                out.append(np.asarray(f.convert("RGB")))
            elif isinstance(f, torch.Tensor):
                t = f.cpu().numpy() if isinstance(f, torch.Tensor) else f
                out.append(_float_rgb_to_uint8(t))
            elif isinstance(f, np.ndarray):
                out.append(_float_rgb_to_uint8(f))
            else:
                raise ValueError(f"unrecognized list element type {type(f)}")
        return out

    raise ValueError(f"unrecognized video type {type(video)}")


def _ensure_framewise_relative(c2ws_arr):
    """Convert absolute c2w sequence to framewise relative poses via SE3 inverse."""
    import numpy as np
    F = c2ws_arr.shape[0]
    out = np.zeros_like(c2ws_arr)
    out[0] = np.eye(4)
    # rel[k] = inv(abs[k-1]) @ abs[k]  (SE3 inverse)
    for k in range(1, F):
        prev = c2ws_arr[k - 1]
        R_inv = prev[:3, :3].T
        t_inv = -R_inv @ prev[:3, 3]
        prev_inv = np.eye(4, dtype=c2ws_arr.dtype)
        prev_inv[:3, :3] = R_inv
        prev_inv[:3, 3] = t_inv
        out[k] = prev_inv @ c2ws_arr[k]
    return out


# --------------------- joystick HUD: analytic fields, drawn supersampled ---------------------
# What this replaced, and why the HUD looked low-resolution: a glow used to be N filled circles at radius
# r, r-2, r-4 ... drawn straight onto the frame buffer, so at the shipped radius of ~31 px on a 640x384
# frame a gradient was only ~15 rings -- visible concentric banding, with every ring edge aliasing. Worse,
# those circles wrote the alpha CHANNEL rather than compositing, so each one overwrote the last: the base
# plate never actually reached the 0.75 alpha it asked for (it barely rendered at all), and the knob came
# out as a bright annulus with a hollow middle. The look that shipped was therefore an artefact of the
# bug, not the design. These helpers instead evaluate each shape as a continuous function of distance on a
# numpy grid, composite properly, and draw the widget at _JOY_SS x before an INTER_AREA downscale, so
# edges land on real subpixel coverage.
_JOY_SS = 4


def _smoothstep(edge0, edge1, x):
    import numpy as np
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _field_disc(dist, radius, feather):
    """Solid disc of `radius`, alpha falling to 0 over `feather` px at the rim."""
    return 1.0 - _smoothstep(radius - feather, radius, dist)


def _field_glow(dist, r_outer, power):
    """Radial falloff: 1 at the centre, 0 at r_outer, shaped by `power`."""
    import numpy as np
    t = np.clip(1.0 - dist / max(r_outer, 1e-6), 0.0, 1.0)
    return t ** power


def _field_ring(dist, radius, width, feather):
    """Annulus centred on `radius`, `width` px thick, soft by `feather` px on both sides."""
    return (_smoothstep(radius - width * 0.5 - feather, radius - width * 0.5, dist)
            * (1.0 - _smoothstep(radius + width * 0.5, radius + width * 0.5 + feather, dist)))


def _blend(dst_rgb, dst_a, field, bgr, alpha):
    """Composite one coloured field over the accumulating BGRA layer (premultiplied, in place)."""
    a = field * float(alpha)
    for c in range(3):
        dst_rgb[:, :, c] = dst_rgb[:, :, c] * (1.0 - a) + float(bgr[c]) * a
    dst_a[:] = dst_a + a * (1.0 - dst_a)


def draw_joystick(frame_bgr, center, radius: int, vec2, label: str = None):
    """Draw a blue-ring joystick HUD onto a BGR uint8 frame in-place."""
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    x = float(np.clip(vec2[0], -1.0, 1.0))
    y = float(np.clip(vec2[1], -1.0, 1.0))
    # Knob position stays sub-pixel: rounding it to int (as the old version did) made the knob jump in
    # whole pixels, which on a ~31 px widget reads as the deflection stuttering.
    knob_x = cx + x * radius * 0.85
    knob_y = cy + y * radius * 0.85

    blue      = (255, 170, 50)
    blue_hot  = (255, 210, 90)
    base_dark = (18, 22, 30)
    base_mid  = (35, 45, 65)
    knob_core = (235, 245, 255)
    knob_edge = (165, 190, 215)

    # --- work in a local ROI at SS x resolution: the widget covers ~1.45 r, so the whole-frame float32
    #     BGRA buffer the old version allocated per joystick per frame was ~99 % wasted too.
    S = _JOY_SS
    pad = radius * 1.45 + 3
    x0, y0 = int(np.floor(cx - pad)), int(np.floor(cy - pad))
    x1, y1 = int(np.ceil(cx + pad)), int(np.ceil(cy + pad))
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return
    Hs, Ws = (y1c - y0c) * S, (x1c - x0c) * S
    # pixel centres of the supersampled grid, in original frame coordinates
    yy, xx = np.meshgrid(np.arange(Hs, dtype=np.float32), np.arange(Ws, dtype=np.float32), indexing="ij")
    fx = x0c + (xx + 0.5) / S
    fy = y0c + (yy + 0.5) / S
    d_c = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)                    # distance to widget centre
    d_k = np.sqrt((fx - knob_x) ** 2 + (fy - knob_y) ** 2)            # distance to knob centre
    fe = 1.0 / S                                                      # one output pixel of feather

    rgb = np.zeros((Hs, Ws, 3), dtype=np.float32)
    a = np.zeros((Hs, Ws), dtype=np.float32)

    # base plate (glow fields, then two inset grooves)
    # NO filled base plate. The stepped version nominally asked for one at 0.75/0.40 alpha but those were
    # per-ring OVERWRITES into the alpha channel, so almost nothing of it survived -- the shipped look was
    # effectively ring + knob over untouched video. Reinstating the plate as real coverage (which is what a
    # correct compositor does with those numbers) put a grey disc over the footage, worse than the bug it
    # fixed. So the plate is gone on purpose; contrast comes from a dark halo hugging each stroke instead,
    # which costs ~1 px and works on both bright and dark scenes.
    _blend(rgb, a, _field_ring(d_c, radius * 1.00, 5.0, fe), (12, 16, 22), 0.34)   # halo under the ring
    _blend(rgb, a, _field_ring(d_c, radius * 1.00, 2.6, fe), blue, 0.90)
    _blend(rgb, a, _field_ring(d_c, radius * 1.00, 1.0, fe), blue_hot, 0.75)
    # knob drop shadow
    d_s = np.sqrt((fx - knob_x - radius * 0.07) ** 2 + (fy - knob_y - radius * 0.09) ** 2)
    _blend(rgb, a, _field_glow(d_s, radius * 0.46, 2.0) * _field_disc(d_s, radius * 0.46, fe), (0, 0, 0), 0.34)
    # Knob body: a SOLID shaded ball, not a glow. The old version stacked filled circles whose alpha fell
    # toward the centre and overwrote as it went, which left a bright annulus with a hollow middle -- it
    # read as detail but it was an ordering artefact, and reproducing its alpha curve here made the knob
    # vanish into the plate. A disc at high alpha with a light-to-dark gradient across it reads as a ball
    # at any size, which is the point of the widget: you must be able to see where it is deflected to.
    r_k = radius * 0.40
    body = _field_disc(d_k, r_k, fe)
    lit = np.clip(0.5 - ((fx - knob_x) + (fy - knob_y)) / (2.6 * r_k), 0.0, 1.0)   # top-left lit
    for c in range(3):
        shade = knob_edge[c] + (knob_core[c] - knob_edge[c]) * lit
        aa = body * 0.94
        rgb[:, :, c] = rgb[:, :, c] * (1.0 - aa) + shade * aa
    a[:] = a + body * 0.94 * (1.0 - a)
    _blend(rgb, a, _field_ring(d_k, r_k, 1.6, fe), (250, 252, 255), 0.55)          # bright rim
    _blend(rgb, a, _field_ring(d_k, r_k + 1.4, 1.6, fe), (10, 14, 20), 0.40)       # dark outline
    # specular highlight
    d_h = np.sqrt((fx - knob_x + r_k * 0.34) ** 2 + (fy - knob_y + r_k * 0.34) ** 2)
    _blend(rgb, a, _field_disc(d_h, r_k * 0.30, fe) * 0.9, (255, 255, 255), 0.5)

    # downscale premultiplied, so partially covered pixels average colour and coverage together
    inter = cv2.INTER_AREA
    rgb_s = cv2.resize(rgb * a[:, :, None], (x1c - x0c, y1c - y0c), interpolation=inter)
    a_s = cv2.resize(a, (x1c - x0c, y1c - y0c), interpolation=inter)[:, :, None]
    roi = frame_bgr[y0c:y1c, x0c:x1c].astype(np.float32)
    frame_bgr[y0c:y1c, x0c:x1c] = np.clip(roi * (1.0 - a_s) + rgb_s, 0, 255).astype(np.uint8)

    # label: drawn supersampled into its own tile so the glyphs get the same subpixel treatment as the
    # widget, instead of cv2's 1 px-thick Hershey strokes straight onto the frame
    if label:
        fs, th = 0.5 * S, max(1, int(round(S * 0.9)))
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        tile = np.zeros((tht + base + 2 * S, tw + 2 * S, 4), dtype=np.uint8)
        cv2.putText(tile, label, (S, tht + S), cv2.FONT_HERSHEY_DUPLEX, fs,
                    (255, 255, 255, 255), th, cv2.LINE_AA)
        tile_s = cv2.resize(tile, (tile.shape[1] // S, tile.shape[0] // S), interpolation=inter)
        tx = int(round(cx - tile_s.shape[1] * 0.5))
        ty = int(round(cy - radius * 1.15 - tile_s.shape[0]))
        tx0, ty0 = max(tx, 0), max(ty, 0)
        tx1, ty1 = min(tx + tile_s.shape[1], w), min(ty + tile_s.shape[0], h)
        if tx1 > tx0 and ty1 > ty0:
            sub = tile_s[ty0 - ty:ty1 - ty, tx0 - tx:tx1 - tx].astype(np.float32)
            ta = (sub[:, :, 3:4] / 255.0) * 0.95
            dst = frame_bgr[ty0:ty1, tx0:tx1].astype(np.float32)
            frame_bgr[ty0:ty1, tx0:tx1] = np.clip(dst * (1.0 - ta) + sub[:, :, :3] * ta, 0, 255).astype(np.uint8)


def _extract_move_rot_from_c2ws(
    c2ws_absolute,
    move_scale: float = None,
    rot_scale: float = None,
    percentile: float = 90.0,
):
    """Extract per-frame (move_xy, rot_xy) 2D joystick vectors in [-1,1] from absolute c2w sequence.

    Move uses framewise-relative translation (vx=+dx, vy=-dz); Rotate uses scipy xyz Euler yaw/pitch.
    Scale is auto-adapted to the clip's motion percentile unless overridden.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as _R

    if hasattr(c2ws_absolute, "cpu"):
        c2ws_absolute = c2ws_absolute.cpu().numpy()
    c2ws_absolute = np.asarray(c2ws_absolute, dtype=np.float32)
    # convert absolute to framewise relative for joystick mapping
    c2ws_framewise = _ensure_framewise_relative(c2ws_absolute)
    F = c2ws_framewise.shape[0]

    # raw per-frame extraction
    moves_raw = np.zeros((F, 2), dtype=np.float32)
    rots_raw = np.zeros((F, 2), dtype=np.float32)
    for k in range(F):
        R_k = c2ws_framewise[k, :3, :3]
        t_k = c2ws_framewise[k, :3, 3]
        moves_raw[k, 0] =  float(t_k[0])
        moves_raw[k, 1] = -float(t_k[2])
        # scipy 'xyz' returns (pitch, yaw, roll) in camera frame
        try:
            pitch, yaw, _roll = _R.from_matrix(R_k).as_euler("xyz", degrees=False)
        except Exception:
            pitch, yaw = 0.0, 0.0
        rots_raw[k, 0] =  float(yaw)
        rots_raw[k, 1] = -float(pitch)

    # adaptive normalization scale based on clip-level percentile
    def _auto_scale(vecs2d, override):
        if override is not None and float(override) > 0:
            return float(override)
        mag = np.linalg.norm(vecs2d, axis=1)
        nz = mag[mag > 1e-6]
        if nz.size == 0:
            return 1.0
        s = float(np.percentile(nz, percentile))
        return max(s, 1e-4)

    move_s = _auto_scale(moves_raw, move_scale)
    rot_s  = _auto_scale(rots_raw,  rot_scale)

    moves = np.clip(moves_raw / move_s, -1.0, 1.0).astype(np.float32)
    rots  = np.clip(rots_raw  / rot_s,  -1.0, 1.0).astype(np.float32)
    return moves, rots


def compute_global_move_rot_scale(c2ws_absolute, percentile: float = 90.0):
    """Compute a single (move_scale, rot_scale) from the full c2w sequence for consistent chunk-by-chunk overlay."""
    import numpy as np
    moves_raw, rots_raw = _extract_move_rot_raw_only(c2ws_absolute)

    def _scale(vecs2d):
        mag = np.linalg.norm(vecs2d, axis=1)
        nz = mag[mag > 1e-6]
        if nz.size == 0:
            return 1.0
        return max(float(np.percentile(nz, percentile)), 1e-4)

    return _scale(moves_raw), _scale(rots_raw)


def resolve_joystick_scale_with_cap(
    c2ws_absolute,
    cap_max_deg: float = 0.0,
    cap_max_trans: float = 0.0,
    frames_per_chunk: int = 33,
    percentile: float = 90.0,
):
    """Joystick HUD scale derived from the trajectory cap.

    cap_max_deg / cap_max_trans are per-chunk (start->end) limits over a 33-frame chunk; clamp/resample spread
    them evenly over (frames_per_chunk - 1) framewise intervals, so a frame at the cap rate equals
    cap / (frames_per_chunk - 1). Using that as the joystick scale makes ±1 correspond to "cap maxed out".
    Capped dims use the cap; uncapped dims fall back to the p90 auto scale.
    """
    import math
    n_intervals = max(1, int(frames_per_chunk) - 1)
    auto_move, auto_rot = compute_global_move_rot_scale(c2ws_absolute, percentile=percentile)
    move_s = float(cap_max_trans) / n_intervals if float(cap_max_trans) > 0 else auto_move
    rot_s = (float(cap_max_deg) * math.pi / 180.0) / n_intervals if float(cap_max_deg) > 0 else auto_rot
    return max(move_s, 1e-4), max(rot_s, 1e-4)


def _extract_move_rot_raw_only(c2ws_absolute):
    """Extract raw (moves_raw, rots_raw) from absolute c2ws without normalizing; caller handles scaling."""
    import numpy as np
    from scipy.spatial.transform import Rotation as _R

    if hasattr(c2ws_absolute, "cpu"):
        c2ws_absolute = c2ws_absolute.cpu().numpy()
    c2ws_absolute = np.asarray(c2ws_absolute, dtype=np.float32)
    c2ws_framewise = _ensure_framewise_relative(c2ws_absolute)
    F = c2ws_framewise.shape[0]

    moves_raw = np.zeros((F, 2), dtype=np.float32)
    rots_raw = np.zeros((F, 2), dtype=np.float32)
    for k in range(F):
        R_k = c2ws_framewise[k, :3, :3]
        t_k = c2ws_framewise[k, :3, 3]
        moves_raw[k, 0] =  float(t_k[0])
        moves_raw[k, 1] = -float(t_k[2])
        try:
            pitch, yaw, _roll = _R.from_matrix(R_k).as_euler("xyz", degrees=False)
        except Exception:
            pitch, yaw = 0.0, 0.0
        rots_raw[k, 0] =  float(yaw)
        rots_raw[k, 1] = -float(pitch)
    return moves_raw, rots_raw


def add_joystick_overlay_from_c2ws(
    video_frames,
    c2ws_absolute,
    smooth_alpha: float = 0.3,
    move_scale: float = 1.0,
    rot_scale: float = None,
    label_left: str = "Move",
    label_right: str = "Rotate",
):
    """Overlay dual joystick HUD (Move bottom-left, Rotate bottom-right) driven by absolute c2ws. Returns list[H,W,3 uint8 RGB]."""
    import numpy as np

    try:
        import cv2  # noqa: F401 -- check the optional overlay backend before drawing
    except ImportError:
        print("[CamViz] cv2 not available, joystick overlay skipped")
        return list(video_frames) if not isinstance(video_frames, list) else video_frames

    if isinstance(video_frames, np.ndarray):
        video_frames = [video_frames[i] for i in range(video_frames.shape[0])]
    n_frames = len(video_frames)
    if n_frames == 0:
        return video_frames

    moves, rots = _extract_move_rot_from_c2ws(c2ws_absolute, move_scale=move_scale, rot_scale=rot_scale)
    F_pix = moves.shape[0]

    if F_pix == n_frames:
        idx_map = list(range(F_pix))
    else:
        idx_map = [min(int(round(i * (F_pix - 1) / max(n_frames - 1, 1))), F_pix - 1)
                   for i in range(n_frames)]

    h, w = video_frames[0].shape[:2]
    short = min(w, h)
    radius = int(np.clip(short * 0.08, 24, 120))
    margin = int(np.clip(short * 0.04, 12, 80))
    left_center = (margin + radius, h - margin - radius)
    right_center = (w - margin - radius, h - margin - radius)

    ts = np.zeros(2, dtype=np.float32)
    rs = np.zeros(2, dtype=np.float32)
    result = []
    for pf in range(n_frames):
        ci = idx_map[pf]
        frame_bgr = video_frames[pf][:, :, ::-1].copy()
        tv = moves[ci]
        rv = rots[ci]
        ts = (1.0 - smooth_alpha) * ts + smooth_alpha * tv
        rs = (1.0 - smooth_alpha) * rs + smooth_alpha * rv
        draw_joystick(frame_bgr, left_center, radius, ts, label=label_left)
        draw_joystick(frame_bgr, right_center, radius, rs, label=label_right)
        result.append(frame_bgr[:, :, ::-1].copy())
    return result


def _load_gt_video_rgb_for_viz(
    video_path: str,
    height: int,
    width: int,
    num_target_frames: int = None,
    target_frame_indices = None,
    target_fps: int = 24,
    source_fps: int = 30,
    start_seconds: float = 0.0,
) -> "list[np.ndarray]":
    """Load GT video frames via cv2, decode BGR->RGB, center-crop and resize. Returns list[H,W,3 uint8 RGB].

    Exactly one of num_target_frames or target_frame_indices must be provided.
    """
    import cv2

    assert (num_target_frames is None) != (target_frame_indices is None), (
        "exactly one of num_target_frames / target_frame_indices must be given"
    )

    if target_frame_indices is not None:
        target_indices = list(target_frame_indices)
    else:
        target_indices = list(range(int(num_target_frames)))

    start_src = int(round(start_seconds * source_fps))
    src_indices = [start_src + int(round(t * source_fps / target_fps)) for t in target_indices]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {video_path}")
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_indices = [min(idx, total_src - 1) for idx in src_indices]

    frames = []
    last_read_idx = -1
    cache = None
    import time as _time
    _t_loop = _time.time()
    _PROGRESS_EVERY = 100      # log every 100 frames only for long clips
    _show_progress = len(src_indices) >= 200
    for _i, idx in enumerate(src_indices):
        if idx != last_read_idx + 1 and idx != last_read_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if idx == last_read_idx and cache is not None:
            frames.append(cache)
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                if cache is None:
                    raise RuntimeError(f"cv2 read failed at idx={idx} for {video_path}")
                frames.append(cache.copy())
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cache = frame_rgb
            last_read_idx = idx
            frames.append(frame_rgb)
        if _show_progress and (_i + 1) % _PROGRESS_EVERY == 0:
            _elapsed = _time.time() - _t_loop
            _rate = (_i + 1) / max(_elapsed, 1e-6)
            _eta = (len(src_indices) - _i - 1) / max(_rate, 1e-6)
            print(f"[gt_load] {_i + 1}/{len(src_indices)} frames | {_elapsed:.1f}s elapsed | ETA {_eta:.1f}s", flush=True)
    cap.release()

    # center-crop to target aspect ratio then resize
    h_src, w_src = frames[0].shape[:2]
    target_ar = height / width
    src_ar = h_src / w_src
    if src_ar >= target_ar:
        new_h = int(w_src * target_ar)
        top = (h_src - new_h) // 2
        crop_box = (slice(top, top + new_h), slice(0, w_src))
    else:
        new_w = int(h_src / target_ar)
        left = (w_src - new_w) // 2
        crop_box = (slice(0, h_src), slice(left, left + new_w))

    out = []
    for f in frames:
        cropped = f[crop_box[0], crop_box[1]]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
        out.append(resized)
    return out


def combine_gt_pred_with_cam_viz(
    pred_video,
    gt_video_path: str,
    pose_path: str,
    height: int,
    width: int,
    num_frames: int,
    ref_seconds: float = 0.0,
    start_seconds: float = 0.0,
    latent_window_size: int = 9,
    vae_stride_t: int = 4,
    target_fps: int = 24,
    source_fps: int = 30,
    source_resolution=(1080, 1920),
    pose_type: str = "vipe",
    panel_label_gt: str = "GT",
    panel_label_pred: str = "Pred",
    warp_dump_dir: "str | None" = None,
    max_rotation_deg: float = 0.0,
    fallback_default_intrinsic: bool = False,
):
    """Build a side-by-side validation video: GT (left) | Pred (right) [| Warp (optional third column)], each with joystick overlay.

    Returns list[np.ndarray] uint8 RGB; use save_rgb_video to write to disk.
    """
    import numpy as np
    import time as _time

    pred_frames = _to_np_uint8_frames(pred_video)
    n_pred = len(pred_frames)

    # derive pose frame count to match pipe's latent section layout
    W = int(latent_window_size)
    stride = int(vae_stride_t)
    window_pix = (W - 1) * stride + 1
    num_secs = max(1, (int(num_frames) + window_pix - 1) // window_pix)
    ref_pix = max(0, int(round(ref_seconds * target_fps)))
    ref_lat = ((ref_pix - 1) // stride + 1) if ref_pix > 0 else 0
    needed_lat = ref_lat + num_secs * W
    pose_num_target_frames = (needed_lat - 1) * stride + 1

    _t = _time.time()
    print(f"[combine_viz] (1/5) loading pose npz ({pose_num_target_frames} frames) ...", flush=True)
    _Ks, c2ws_full = load_pose_for_v2v(
        pose_path,
        target_height=height,
        target_width=width,
        source_resolution=source_resolution,
        pose_type=pose_type,
        num_target_frames=pose_num_target_frames,
        target_fps=target_fps,
        source_fps=source_fps,
        start_seconds=start_seconds,
        max_rotation_deg=max_rotation_deg,
        fallback_default_intrinsic=fallback_default_intrinsic,   # npz without intrinsics (e.g. game) would KeyError and silently skip cam_viz
    )
    print(f"[combine_viz] (1/5) pose loaded: shape={tuple(c2ws_full.shape)} ({_time.time() - _t:.1f}s)", flush=True)

    # build per-section pose slices and corresponding GT frame indices
    pose_slices = []
    target_frame_indices = []
    for k in range(num_secs):
        s_lat = ref_lat + k * W
        s_pix = s_lat * stride
        e_pix = (s_lat + W - 1) * stride + 1
        e_pix = min(e_pix, c2ws_full.shape[0])
        pose_slices.append(c2ws_full[s_pix:e_pix])
        target_frame_indices.extend(range(s_pix, e_pix))
    c2ws_viz = torch.cat(pose_slices, dim=0)              # [num_secs * (W-1)*stride+1, 4, 4]

    _t = _time.time()
    print(f"[combine_viz] (2/5) loading GT frames from mp4 ({len(target_frame_indices)} frames, "
          f"cv2 per-frame seek+decode) ...", flush=True)
    gt_frames = _load_gt_video_rgb_for_viz(
        gt_video_path,
        height=height, width=width,
        target_frame_indices=target_frame_indices,
        target_fps=target_fps, source_fps=source_fps,
        start_seconds=start_seconds,
    )
    print(f"[combine_viz] (2/5) GT loaded: {len(gt_frames)} frames ({_time.time() - _t:.1f}s)", flush=True)

    n = min(n_pred, len(gt_frames), c2ws_viz.shape[0])
    gt_frames = gt_frames[:n]
    pred_frames_n = pred_frames[:n]
    c2ws_viz_n = c2ws_viz[:n]

    # joystick overlay (labels must be ASCII for Hershey font)
    _t = _time.time()
    print(f"[combine_viz] (3/5) joystick overlay on GT ({n} frames, per-frame cv2.putText/draw) ...", flush=True)
    gt_with_cam = add_joystick_overlay_from_c2ws(
        gt_frames, c2ws_viz_n,
        label_left=f"{panel_label_gt} Move", label_right=f"{panel_label_gt} Rot",
    )
    print(f"[combine_viz] (3/5) GT overlay done ({_time.time() - _t:.1f}s)", flush=True)

    _t = _time.time()
    print(f"[combine_viz] (4/5) joystick overlay on Pred ({n} frames) ...", flush=True)
    pred_with_cam = add_joystick_overlay_from_c2ws(
        pred_frames_n, c2ws_viz_n,
        label_left=f"{panel_label_pred} Move", label_right=f"{panel_label_pred} Rot",
    )
    print(f"[combine_viz] (4/5) Pred overlay done ({_time.time() - _t:.1f}s)", flush=True)

    # optional: load warp (chunk_*_warp.mp4) + visibility mask (chunk_*_vis.mp4) columns from dump dir.
    # 4-column layout: GT | Pred | Warp | Mask. Each extra column is independent (warp/mask loaded separately;
    # either missing -> that column dropped, falling back to 3 / 2 cols).
    def _load_aligned(kind, label):
        if warp_dump_dir is None:
            return None
        _t0 = _time.time()
        frames = _load_warp_frames_from_dump_dir(
            warp_dump_dir, target_height=height, target_width=width, kind=kind,
        )
        if frames is None:
            print(f"[combine_viz] ({label}) skip: dump_dir={warp_dump_dir} has no chunk_*_{kind}.mp4",
                  flush=True)
            return None
        if len(frames) < n:
            _pad_n = n - len(frames)
            frames = frames + [frames[-1]] * _pad_n
            print(f"[combine_viz] ({label}) loaded {len(frames) - _pad_n} frames, padded {_pad_n} "
                  f"to match pred n={n} ({_time.time() - _t0:.1f}s)", flush=True)
        else:
            frames = frames[:n]
            print(f"[combine_viz] ({label}) loaded {len(frames)} frames (truncated to pred n={n}, "
                  f"{_time.time() - _t0:.1f}s)", flush=True)
        return frames

    _warp_frames_aligned = _load_aligned("warp", "warp")
    _mask_frames_aligned = _load_aligned("vis", "mask")

    # concatenate columns horizontally: GT | Pred | [Warp] | [Mask]
    _t = _time.time()
    _ncol = 2 + (_warp_frames_aligned is not None) + (_mask_frames_aligned is not None)
    print(f"[combine_viz] (5/5) concatenating {_ncol} columns ({n} frames) ...", flush=True)
    combined = []
    for i in range(n):
        _parts = [gt_with_cam[i], pred_with_cam[i]]
        if _warp_frames_aligned is not None:
            _parts.append(_warp_frames_aligned[i])
        if _mask_frames_aligned is not None:
            _parts.append(_mask_frames_aligned[i])
        combined.append(np.concatenate(_parts, axis=1))
    print(f"[combine_viz] (5/5) concat done ({_time.time() - _t:.1f}s, {_ncol} cols)", flush=True)
    return combined


def _load_warp_frames_from_dump_dir(
    warp_dump_dir: str,
    target_height: int,
    target_width: int,
    kind: str = "warp",
) -> "list[np.ndarray] | None":
    """Read all chunk_NNN_<kind>.mp4 (kind='warp' or 'vis') from dump dir in order, return concatenated
    list[H,W,3] uint8 RGB or None. The visibility mask ('vis') is written grayscale but decodes as 3-ch."""
    import re
    from pathlib import Path

    import cv2

    warp_dir = Path(warp_dump_dir)
    if not warp_dir.is_dir():
        return None

    pattern = re.compile(rf"chunk_(\d+)_{re.escape(kind)}\.mp4$")
    chunks = []
    for f in sorted(warp_dir.iterdir()):
        m = pattern.match(f.name)
        if m:
            chunks.append((int(m.group(1)), f))
    if not chunks:
        return None
    chunks.sort(key=lambda x: x[0])

    all_frames = []
    for _ck_idx, mp4_path in chunks:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            continue
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[0] != target_height or frame_rgb.shape[1] != target_width:
                frame_rgb = cv2.resize(
                    frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            all_frames.append(frame_rgb)
        cap.release()
    return all_frames if all_frames else None


def _load_test_clip(
    video_path: str,
    num_target_frames: int,
    source_fps: int,
    target_fps: int,
    height: int,
    width: int,
    start_seconds: float = 0.0,
) -> torch.Tensor:
    """Load mp4, resample source_fps to target_fps, center-crop and resize. Returns [3, T, H, W] in [-1, 1] fp32 CPU."""
    from worldfoundry.core.media.codecs.video import (
        get_video_details,
        load_frames_from_video,
        resize_video_tensor_to_resolution,
    )

    total_src, _, _ = get_video_details(video_path)
    start_src = int(round(start_seconds * source_fps))
    src_indices = [start_src + round(i * source_fps / target_fps) for i in range(num_target_frames)]
    assert src_indices[-1] < total_src, (
        f"video {video_path} is too short: needs src frames >={src_indices[-1] + 1}, has {total_src}"
    )
    frames = load_frames_from_video(video_path, src_indices).float()  # [T, H, W, 3] in 0..255
    frames = (frames / 127.5) - 1.0
    video = resize_video_tensor_to_resolution(frames.permute(0, 3, 1, 2), (height, width))
    return video.permute(1, 0, 2, 3).contiguous()                   # [3, T, H, W]
