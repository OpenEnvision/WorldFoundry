# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""MUSIQ-SPAQ imaging quality, unchanged from the reference calculation."""

from pathlib import Path

from .common import FrameQualityMetric


class ImagingQualityMetric(FrameQualityMetric):
    name = "imaging_quality"
    description = "MUSIQ-SPAQ imaging quality (stride 4, MOS / 100)."
    raw_key = "raw_mos"
    normalizer = 100.0

    def _backend(self, context):
        from .musiq import MusiqSpaqBackend

        config = context.metric_config(self.name)
        checkpoint = (
            Path(
                config.get("checkpoint") or (context.cache_dir / "metrics" / self.name / "musiq_spaq_ckpt-358bb6af.pth")
            )
            .expanduser()
            .resolve()
        )
        batch_size = int(config.get("batch_size", 8))
        return context.get_resource(
            f"musiq_spaq:{checkpoint}:{context.device}:{batch_size}",
            lambda: MusiqSpaqBackend(checkpoint=checkpoint, device=context.device, batch_size=batch_size),
        )
