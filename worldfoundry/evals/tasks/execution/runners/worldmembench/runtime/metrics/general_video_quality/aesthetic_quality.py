# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""OpenAI CLIP ViT-L/14 + LAION aesthetic quality."""

from pathlib import Path

from .common import FrameQualityMetric


class AestheticQualityMetric(FrameQualityMetric):
    name = "aesthetic_quality"
    description = "CLIP ViT-L/14 + LAION aesthetic quality (stride 4, raw / 10)."
    raw_key = "raw_score"
    normalizer = 10.0

    def _backend(self, context):
        from .aesthetic import LaionAestheticBackend

        config = context.metric_config(self.name)
        clip_path = (
            Path(config.get("clip_checkpoint") or (context.cache_dir / "metrics" / self.name / "ViT-L-14.pt"))
            .expanduser()
            .resolve()
        )
        aesthetic_path = (
            Path(
                config.get("aesthetic_checkpoint")
                or (context.cache_dir / "metrics" / self.name / "sa_0_4_vit_l_14_linear.pth")
            )
            .expanduser()
            .resolve()
        )
        batch_size = int(config.get("batch_size", 32))
        return context.get_resource(
            f"laion_aesthetic:{clip_path}:{aesthetic_path}:{context.device}:{batch_size}",
            lambda: LaionAestheticBackend(
                clip_checkpoint=clip_path,
                aesthetic_checkpoint=aesthetic_path,
                device=context.device,
                batch_size=batch_size,
            ),
        )
