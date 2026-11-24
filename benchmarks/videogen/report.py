# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Markdown leaderboard formatter for ``benchmarks.videogen`` JSON receipts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def format_markdown_leaderboard(
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_e2e_s: float | None = None,
    baseline_world_size: int | None = None,
) -> str:
    """Format benchmark result dicts as a GitHub-flavored Markdown table."""
    if not records:
        return "_No benchmark records supplied._\n"

    ref_e2e = baseline_e2e_s
    ref_world = baseline_world_size
    if ref_e2e is None or ref_world is None:
        for row in records:
            e2e = row.get("median_e2e_s")
            if isinstance(e2e, (int, float)) and e2e > 0:
                if ref_e2e is None:
                    ref_e2e = float(e2e)
                if ref_world is None:
                    ref_world = max(int(row.get("world_size", 1)), 1)
                break
    if ref_world is None:
        ref_world = 1

    lines = [
        "| Run | GPUs | Topology | E2E (s) | Speedup | Scale Eff. | Text (s) | DiT (s) | VAE (s) | Peak VRAM | PSNR (dB) | SSIM | Rel-L2 | Gate |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        label = str(row.get("label", row.get("profile", "run")))
        world = int(row.get("world_size", 1))
        topo = str(row.get("topology_label", "1GPU" if world == 1 else f"{world}GPU"))
        if "error" in row:
            lines.append(f"| {label} | {world} | {topo} | ERROR | - | - | - | - | - | - | - | - | - | FAIL |")
            continue
        e2e = float(row.get("median_e2e_s", 0.0))
        text_s = float(row.get("median_text_s", 0.0))
        dit_s = float(row.get("median_dit_s", 0.0))
        vae_s = float(row.get("median_vae_s", 0.0))
        vram = float(row.get("peak_alloc_gib", 0.0))
        speedup = (ref_e2e / e2e) if (ref_e2e and e2e > 0) else 1.0
        scale_factor = world / ref_world
        scale_eff = f"{(speedup / scale_factor) * 100.0:.1f}%" if world != ref_world and scale_factor > 0 else "-"

        q = row.get("quality") or {}
        if q.get("reference") is True:
            psnr_s, ssim_s, l2_s, gate = "REF", "REF", "REF", "REF"
        elif "video_psnr_db" in q:
            raw_psnr = q["video_psnr_db"]
            psnr_s = "inf" if raw_psnr == "infinity" else f"{float(raw_psnr):.2f}"
            ssim_s = f"{float(q.get('video_ssim', 0.0)):.4f}"
            l2_s = f"{float(q.get('latent_relative_l2', 0.0)):.4f}"
            gate = "PASS" if q.get("passed") else "APPROX"
        else:
            psnr_s, ssim_s, l2_s, gate = "-", "-", "-", "-"

        lines.append(
            f"| {label} | {world} | {topo} | {e2e:.2f} | {speedup:.2f}x | {scale_eff} | "
            f"{text_s:.2f} | {dit_s:.2f} | {vae_s:.2f} | {vram:.1f} GiB | "
            f"{psnr_s} | {ssim_s} | {l2_s} | {gate} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Format videogen benchmark JSON files as Markdown.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSON files produced by benchmarks.videogen.runner.")
    parser.add_argument("--baseline-e2e", type=float, default=None, help="Optional reference E2E seconds for speedup.")
    args = parser.parse_args(argv)

    expanded_paths: list[Path] = []
    for path in args.inputs:
        if path.is_dir():
            expanded_paths.extend(sorted(path.glob("*.json")))
        else:
            expanded_paths.append(path)

    records: list[Mapping[str, Any]] = []
    for path in expanded_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(payload)
        elif isinstance(payload, Mapping):
            records.append(payload)
    print(format_markdown_leaderboard(records, baseline_e2e_s=args.baseline_e2e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
