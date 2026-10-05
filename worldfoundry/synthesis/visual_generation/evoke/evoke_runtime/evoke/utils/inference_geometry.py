"""Inference camera transforms adapted from AlayaLab/Evoke (Apache-2.0)."""

import torch

def resolve_intrinsic_source_resolution(K_3x3, h_org, w_org, tag=""):
    """Pick the resolution K_3x3 was calibrated at, for feeding transform_intrinsic_for_crop_resize.

    A wrong declared size rescales fx/fy and shifts the principal point off center, so the size implied
    by the principal point wins when BOTH axes disagree by > 10% at a consistent ratio (the signature of
    a resolution mismatch). A single-axis disagreement is more likely a real off-center principal point,
    where overriding would corrupt a correct intrinsic. Normalized intrinsics carry no size (pp <= 1) and
    always keep the declared value. Must stay shared between the training and val/infer pose loaders.
    """
    try:
        h_inferred = int(round(float(K_3x3[1, 2]) * 2)) if float(K_3x3[1, 2]) > 1 else None
        w_inferred = int(round(float(K_3x3[0, 2]) * 2)) if float(K_3x3[0, 2]) > 1 else None
    except Exception:
        h_inferred, w_inferred = None, None

    # Orientation flip = the pose was solved on a rotated frame, so K AND c2w are in a rotated camera
    # frame and no choice of source resolution repairs it. Raise so training skips the sample
    # (ConfigAwareDataset retries on ValueError) and val/infer fails loudly instead of warping garbage.
    # Requires the inferred aspect to be the declared aspect inverted, so an off-center principal point
    # (whose inferred size is meaningless) is not mistaken for a rotation.
    if h_inferred is not None and w_inferred is not None and h_org and w_org:
        a_inferred, a_declared = w_inferred / h_inferred, h_org / w_org
        if (w_inferred > h_inferred) != (w_org > h_org) and abs(a_inferred - a_declared) <= 0.05 * a_declared:
            raise ValueError(
                f"[CamCtrl] {tag}: intrinsic calibrated at {h_inferred}x{w_inferred}, opposite orientation "
                f"to the declared {h_org}x{w_org} -- pose solved on a rotated frame, sample unusable")

    h_final = h_org if h_org is not None else (h_inferred if h_inferred is not None else 720)
    w_final = w_org if w_org is not None else (w_inferred if w_inferred is not None else 1280)

    off_h = h_org is not None and h_inferred is not None and abs(h_org - h_inferred) / max(h_inferred, 1) > 0.1
    off_w = w_org is not None and w_inferred is not None and abs(w_org - w_inferred) / max(w_inferred, 1) > 0.1
    if off_h and off_w:
        s_h, s_w = h_inferred / h_org, w_inferred / w_org
        if abs(s_h - s_w) <= 0.05 * max(s_h, s_w):
            print(f"[CamCtrl] {tag}: declared {h_org}x{w_org} overridden by intrinsic-derived "
                  f"{h_inferred}x{w_inferred} (both axes off > 10% at scale {s_h:.3f}/{s_w:.3f})")
            return h_inferred, w_inferred
    if off_h:
        print(f"[CamCtrl] WARN {tag}: declared source_h={h_org} differs from intrinsic-derived {h_inferred} by > 10%")
    if off_w:
        print(f"[CamCtrl] WARN {tag}: declared source_w={w_org} differs from intrinsic-derived {w_inferred} by > 10%")
    return h_final, w_final


def transform_intrinsic_for_crop_resize(K_3x3, h_org, w_org, h_target, w_target):
    """Transform a 3x3 intrinsic matrix from original resolution to training resolution via resize+center-crop.

    Handles both pixel-unit and normalized intrinsics (all values <=2 are treated as normalized).
    Returns a torch [4] tensor [fx, fy, cx, cy] in pixel units at the target resolution.
    """
    fx = float(K_3x3[0, 0])
    fy = float(K_3x3[1, 1])
    cx = float(K_3x3[0, 2])
    cy = float(K_3x3[1, 2])

    # auto-detect normalized intrinsic (e.g. VIPE output where fx=0.5 means focal=W/2)
    if abs(fx) <= 2.0 and abs(fy) <= 2.0 and abs(cx) <= 2.0 and abs(cy) <= 2.0:
        fx = fx * w_org
        cx = cx * w_org
        fy = fy * h_org
        cy = cy * h_org

    scale_h = h_target / h_org
    scale_w = w_target / w_org
    scale = max(scale_h, scale_w)

    h_resize = round(h_org * scale)
    w_resize = round(w_org * scale)

    # apply uniform scale
    fx_r = fx * scale
    fy_r = fy * scale
    cx_r = cx * scale
    cy_r = cy * scale

    # shift principal point for center crop
    crop_offset_x = (w_resize - w_target) / 2.0
    crop_offset_y = (h_resize - h_target) / 2.0
    cx_f = cx_r - crop_offset_x
    cy_f = cy_r - crop_offset_y

    return torch.tensor([fx_r, fy_r, cx_f, cy_f], dtype=torch.float32)
