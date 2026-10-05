"""MemoBench phase-aware CLIP metrics (V-D-R protocol).

Provenance: CLIP model loading and image/text embedding are delegated to the
shared in-tree helpers in
``worldfoundry.evaluation.tasks.metrics._shared.clip_embed`` (same
``openai:ViT-B-32`` backbone, cached bundle). This module keeps only the
MemoBench-specific protocol: BGR frame handling, phase index sampling, and the
V-D-R revisit/recovery formulas.
"""

import cv2
import numpy as np

from worldfoundry.evaluation.tasks.metrics._shared.clip_embed import (
    encode_clip_images,
    encode_clip_texts,
)

_CLIP_MODEL = "openai:ViT-B-32"


def _embed(img: np.ndarray, device: str = None) -> np.ndarray:
    """L2-normalized CLIP embedding of one BGR frame."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return encode_clip_images([rgb], model=_CLIP_MODEL, device=device)[0]


def _phase_indices(start: int, end: int, n: int, total: int) -> list:
    """Uniformly spaced indices in [start, end], clamped to [0, total-1]."""
    start = max(0, min(start, total - 1))
    end   = max(start, min(end, total - 1))
    if n <= 1 or start == end:
        return [start]
    return [start + int(i * (end - start) / (n - 1)) for i in range(n)]


def scene_consistency_score(frames, n_sample: int = 9, device: str = None,
                             start: int = 0, end: int = None) -> dict:
    """
    Mean CLIP cosine similarity between consecutive sampled frames within
    [start, end]. Catches sudden scene identity shifts or jump cuts.

    For phase-aware V-D-R evaluation call with start/end set to the V or R phase
    boundaries to avoid penalising the intentional camera motion in D phase.

    Returns:
      - scene_consistency  : mean consecutive cosine similarity
      - min_consecutive_sim: minimum consecutive similarity (flags sudden jumps)
    """
    N = frames.num_frames
    if end is None:
        end = N - 1

    indices = _phase_indices(start, end, n_sample, N)
    if len(indices) < 2:
        return {"scene_consistency": 1.0, "min_consecutive_sim": 1.0}

    embs = [_embed(frames.get(idx), device) for idx in indices]
    consecutive_sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]

    return {
        "scene_consistency":   round(float(np.mean(consecutive_sims)), 4),
        "min_consecutive_sim": round(float(np.min(consecutive_sims)),  4),
    }


def revisit_metrics(frames, h_start: int, r_start: int,
                    n_sample: int = 6, device: str = None) -> dict:
    """
    Phase-aware V-D-R metrics using CLIP similarity vs frame_0 as the anchor.

    Splits the video into three phases using the GT-derived keyframe boundaries:
      V phase: [0,       h_start]  — observer sees target
      D phase: [h_start, r_start]  — target is hidden (outside FOV)
      R phase: [r_start, N-1    ]  — target re-enters FOV

    Computes mean CLIP cosine similarity vs frame_0 for each phase:
      s_O, s_H, s_R

    RevisitRecovery:
      Measures how much the R phase recovered toward the V phase baseline
      after the D-phase drop.
        recovery = clip((s_R - s_H) / (s_O - s_H + ε), 0, 1)
        1.0 → R phase fully returns to V phase similarity level
        0.0 → R phase stays at D phase level (no semantic recovery)

    RevisitBridge:
      CLIP cosine similarity between frame[h_start] and frame[r_start].
      Measures whether there is an abrupt scene jump at the D→R boundary.
      Low value → identity reset or jump cut across the hidden interval.

    Returns:
      - revisit_recovery : float in [0, 1]
      - revisit_bridge   : float in [0, 1]
      - s_O, s_H, s_R    : per-phase mean similarities (diagnostic)
    """
    N = frames.num_frames
    h = max(1, min(h_start, N - 2))
    r = max(h + 1, min(r_start, N - 1))

    emb_0 = _embed(frames.get(0), device)

    def phase_mean_sim(p_start, p_end):
        idxs = _phase_indices(p_start, p_end, n_sample, N)
        sims = [float(np.dot(emb_0, _embed(frames.get(i), device))) for i in idxs]
        return float(np.mean(sims))

    s_O = phase_mean_sim(0, h)
    s_H = phase_mean_sim(h, r)
    s_R = phase_mean_sim(r, N - 1)

    # RevisitRecovery: how much of the D-phase drop was recovered in R phase
    drop = s_O - s_H
    if drop < 1e-4:
        # No meaningful drop in D phase — check R is at least as good as V phase
        recovery = 1.0 if s_R >= s_O - 0.01 else float(np.clip(s_R / (s_O + 1e-6), 0.0, 1.0))
    else:
        recovery = float(np.clip((s_R - s_H) / drop, 0.0, 1.0))

    # RevisitBridge: continuity at the D→R boundary
    emb_h_end   = _embed(frames.get(h), device)
    emb_r_start = _embed(frames.get(r), device)
    bridge = float(np.dot(emb_h_end, emb_r_start))

    # OcclusionDrop: D phase must actually move away from V phase baseline.
    # Low value → model never "left" target (D phase same as V phase) → Recovery invalid.
    occlusion_drop = float(np.clip((s_O - s_H) / (s_O + 1e-6), 0.0, 1.0))

    return {
        "revisit_recovery": round(recovery,       4),
        "revisit_bridge":   round(bridge,         4),
        "occlusion_drop":   round(occlusion_drop, 4),
        "s_O":              round(s_O, 4),
        "s_H":              round(s_H, 4),
        "s_R":              round(s_R, 4),
    }


def prompt_alignment_score(frames, prompt: str, n_sample: int = 5,
                           device: str = None) -> float:
    """
    CLIP text-image cosine similarity between the text prompt and sampled frames.
    Measures how well the generated video follows the input instruction.

    Returns mean cosine similarity over sampled frames (higher = better).
    """
    text_emb = encode_clip_texts([prompt], model=_CLIP_MODEL, device=device)[0]

    N = frames.num_frames
    indices = [int(i * (N - 1) / (n_sample - 1)) for i in range(n_sample)]

    sims = [float(np.dot(text_emb, _embed(frames.get(idx), device))) for idx in indices]
    return round(float(np.mean(sims)), 4)


def clip_frame_similarity(img_a: np.ndarray, img_b: np.ndarray,
                           device: str = None) -> float:
    """
    CLIP cosine similarity between two BGR frames.
    Used for GT-grounded R-phase comparison: generated frame vs GT frame.
    Returns a value in [0, 1].
    """
    emb_a = _embed(img_a, device)
    emb_b = _embed(img_b, device)
    return float(np.clip(float(np.dot(emb_a, emb_b)), 0.0, 1.0))
