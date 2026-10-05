"""GEO FrameBank: keyframe store with top-K retrieval for Pi3X warp history."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FrameBankEntry:
    frame: torch.Tensor          # [3, H, W] CPU, 8-bit quantized
    c2w: torch.Tensor            # [4, 4] CPU
    chunk_idx: int
    pixel_idx: int               # global pixel index used for recency sorting
    # Optional Pi3X geometry cache; None by default.
    cached_geometry: Optional[dict] = None


class FrameBank:
    """Append-only keyframe store with top-K retrieval."""

    def __init__(self, max_size: Optional[int] = None):
        """Args:
            max_size: capacity cap; None or <=0 means unlimited; excess entries evicted FIFO.
        """
        self.entries: list[FrameBankEntry] = []
        self.max_size: Optional[int] = (
            int(max_size) if (max_size is not None and int(max_size) > 0) else None
        )

    # ---------------- ingest ----------------

    def add(self, frame: torch.Tensor, c2w: torch.Tensor, chunk_idx: int, pixel_idx: int) -> None:
        """Add a frame: detach, move to CPU, 8-bit quantize, FIFO evict if over max_size."""
        self.entries.append(FrameBankEntry(
            frame=self._quantize_8bit(frame.detach().float().cpu()),
            c2w=c2w.detach().float().cpu(),
            chunk_idx=int(chunk_idx),
            pixel_idx=int(pixel_idx),
        ))
        if self.max_size is not None and len(self.entries) > self.max_size:
            # Evict oldest entries to stay within max_size.
            self.entries = self.entries[-self.max_size:]

    # ---------------- query ----------------

    def retrieve(
        self,
        target_c2w_window: torch.Tensor,     # [W, 4, 4] target pose window for current chunk
        anchor_frame: torch.Tensor,          # [3, H, W] render origin for this chunk
        anchor_c2w: torch.Tensor,            # [4, 4]
        # Legacy API: top_k acts as select_k.
        top_k: int = 8,
        # Split API: nearby by recency + select by metric.
        nearby_k: int = 0,                   # number of most-recent frames by pixel_idx
        select_k: Optional[int] = None,       # number of frames chosen by metric; defaults to top_k
        metric: str = "v1",                   # "v1" / "v2" / "v3"
        metric_kwargs: Optional[dict] = None, # per-metric keyword args
        # V1-only heuristics kept for backward compat.
        diversity_min_dist: float = 0.0,     # skip candidates closer than this to already-selected
        time_decay_weight: float = 0.0,      # penalize recent frames by this weight
    ) -> list[torch.Tensor]:
        """Return [nearby + selected + anchor] RGB tensors; anchor is always last.

        Args:
            nearby_k: most-recent frames by pixel_idx (not scored by metric)
            select_k: frames chosen from remaining bank by metric; None falls back to top_k
            metric: scoring variant "v1", "v2", or "v3"
            metric_kwargs: extra args forwarded to the chosen scoring method
        """
        if select_k is None:
            select_k = int(top_k)
        select_k = int(select_k)
        nearby_k = int(nearby_k)
        metric_kwargs = metric_kwargs or {}

        anchor_quant = self._quantize_8bit(anchor_frame.detach().float().cpu())
        target_c2w_window = target_c2w_window.detach().cpu()

        if len(self.entries) == 0:
            return [anchor_quant]

        # 1) Pick nearby_k most-recent entries; skip entries coinciding with anchor pose.
        anchor_c2w_cpu = anchor_c2w.detach().cpu().float() if torch.is_tensor(anchor_c2w) else None
        entries_by_recency = sorted(self.entries, key=lambda e: -e.pixel_idx)
        nearby = []
        if nearby_k > 0:
            for e in entries_by_recency:
                if len(nearby) >= nearby_k:
                    break
                # Skip entry if it duplicates the anchor pose.
                if anchor_c2w_cpu is not None:
                    if float((e.c2w - anchor_c2w_cpu).norm()) < 1e-3:
                        continue
                nearby.append(e)
        nearby_ids = set(id(e) for e in nearby)

        # 2) Score remaining entries by metric and greedily select up to select_k.
        remaining = [e for e in self.entries if id(e) not in nearby_ids]
        selected: list[FrameBankEntry] = []
        if select_k > 0 and len(remaining) > 0:
            max_pixel_idx = max(e.pixel_idx for e in remaining) if len(remaining) > 0 else 0
            scored = []
            for e in remaining:
                if metric == "v3":
                    base = self._score_v3(target_c2w_window, e.c2w, **metric_kwargs)
                elif metric == "v2":
                    base = self._score_v2(target_c2w_window, e.c2w, **metric_kwargs)
                else:  # v1 default
                    base = self._score_v1(target_c2w_window, e.c2w)
                penalty = 0.0
                if float(time_decay_weight) > 0.0 and max_pixel_idx > 0:
                    recency = (float(max_pixel_idx) - float(e.pixel_idx)) / float(max_pixel_idx)
                    penalty = float(time_decay_weight) * (1.0 - recency)
                scored.append((base - penalty, e))
            scored.sort(key=lambda x: -x[0])

            for score, entry in scored:
                if len(selected) >= select_k:
                    break
                if float(diversity_min_dist) > 0.0 and len(selected) > 0:
                    too_close = any(
                        float((entry.c2w[:3, 3] - s.c2w[:3, 3]).norm()) < float(diversity_min_dist)
                        for s in selected
                    )
                    if too_close:
                        continue
                selected.append(entry)

        # 3) Concatenate [nearby + selected + anchor]; anchor is always last.
        result = [e.frame for e in nearby] + [e.frame for e in selected] + [anchor_quant]
        assert torch.equal(result[-1], anchor_quant), (
            "anchor MUST be last: Pi3X treats the final list entry as the source pose"
        )
        return result

    # ---------------- scoring ----------------

    @staticmethod
    def _score_v1(target_c2w_window: torch.Tensor, candidate_c2w: torch.Tensor) -> float:
        """Score = forward-direction cosine similarity minus 0.1 * position distance."""
        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            target = target_c2w_window[-1]
        elif target_c2w_window.ndim == 2:
            target = target_c2w_window
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos = target[:3, 3]
        target_dir = -target[:3, 2]

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]

        dir_sim = float((target_dir @ cand_dir).clamp(-1, 1))
        dist = float((target_pos - cand_pos).norm())
        return dir_sim - 0.1 * dist

    @staticmethod
    def _score_v2(
        target_c2w_window: torch.Tensor,
        candidate_c2w: torch.Tensor,
        dir_w: float = 0.4,
        facing_w: float = 0.3,
        dist_w: float = 0.3,
        dist_sigma: float = 2.0,
    ) -> float:
        """Score = weighted sum of direction similarity, candidate-facing-target, and Gaussian distance."""
        # Compute mean position and direction over the target window.
        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            tw = target_c2w_window
        elif target_c2w_window.ndim == 2:
            tw = target_c2w_window.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos_mean = tw[:, :3, 3].mean(0)                          # [3]
        target_dirs_mean = (-tw[:, :3, 2]).mean(0)                       # [3]
        target_dirs_mean = target_dirs_mean / (target_dirs_mean.norm() + 1e-8)

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]

        # Component 1: direction similarity between candidate and mean target direction.
        dir_sim = float((target_dirs_mean @ cand_dir).clamp(-1, 1))

        # Component 2: how much candidate faces toward the target region.
        to_target = target_pos_mean - cand_pos
        to_target_norm = float(to_target.norm())
        if to_target_norm > 1e-6:
            facing_target = float((cand_dir @ (to_target / (to_target_norm + 1e-8))).clamp(-1, 1))
        else:
            facing_target = 1.0   # same position counts as facing

        # Component 3: Gaussian distance decay.
        dist_score = float(math.exp(-(to_target_norm ** 2) / (2.0 * float(dist_sigma) ** 2)))

        return float(dir_w) * dir_sim + float(facing_w) * facing_target + float(dist_w) * dist_score

    @staticmethod
    def _score_v3(
        target_c2w_window: torch.Tensor,
        candidate_c2w: torch.Tensor,
        depth: float = 5.0,
        fov_rad: float = math.radians(60.0),
    ) -> float:
        """Score = circle-circle overlap area / (pi*r^2) at canonical depth as a view-overlap proxy.

        Args:
            depth: canonical scene depth in metres
            fov_rad: camera field of view in radians
        """
        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            tw = target_c2w_window
        elif target_c2w_window.ndim == 2:
            tw = target_c2w_window.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos_mean = tw[:, :3, 3].mean(0)
        target_dirs_mean = (-tw[:, :3, 2]).mean(0)
        target_dirs_mean = target_dirs_mean / (target_dirs_mean.norm() + 1e-8)

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]
        cand_dir = cand_dir / (cand_dir.norm() + 1e-8)

        # Project each view to a footprint circle center at canonical depth.
        target_footprint = target_pos_mean + target_dirs_mean * float(depth)
        cand_footprint = cand_pos + cand_dir * float(depth)

        # Circle radius from FOV at the given depth.
        r = float(depth) * math.tan(float(fov_rad) / 2.0)
        d = float((target_footprint - cand_footprint).norm())

        # Analytic circle-circle overlap area.
        if d >= 2.0 * r:
            return 0.0
        if d <= 0.0 or r <= 1e-6:
            return 1.0
        # area = 2r²·acos(d/2r) - (d/2)·sqrt(4r²-d²)
        x = d / (2.0 * r)
        x = max(-1.0, min(1.0, x))   # clamp to avoid acos domain error
        area = 2.0 * r ** 2 * math.acos(x) - (d / 2.0) * math.sqrt(max(0.0, 4.0 * r ** 2 - d ** 2))
        iou = area / (math.pi * r ** 2)
        return float(iou)

    # Backward-compat alias: _frustum_score == _score_v1
    @staticmethod
    def _frustum_score(target_c2w_window: torch.Tensor, candidate_c2w: torch.Tensor) -> float:
        """Deprecated alias for _score_v1."""
        return FrameBank._score_v1(target_c2w_window, candidate_c2w)

    # ---------------- helpers ----------------

    @staticmethod
    def _quantize_8bit(frame01: torch.Tensor) -> torch.Tensor:
        """Simulate MP4 quantization: map [-1,1] -> 256 levels -> [-1,1]."""
        v = (frame01 / 2.0 + 0.5).clamp(0, 1)
        v = (v * 255.0).round() / 255.0
        return v * 2.0 - 1.0

    def __len__(self) -> int:
        return len(self.entries)
