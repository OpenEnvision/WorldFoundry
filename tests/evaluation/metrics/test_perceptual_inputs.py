"""Real CPU backend checks; requires the distribution_metrics dependencies."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from worldfoundry.evaluation.tasks.metrics import (
    compute_fsim,
    compute_lpips,
    compute_ms_ssim,
    compute_psnr,
    compute_ssim,
)
from worldfoundry.evaluation.tasks.metrics.fsim.wrapper import _fsim_fn


class PerceptualInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        weights = tempfile.TemporaryDirectory(prefix="metric-weights-")
        cls.addClassCleanup(weights.cleanup)
        previous_hub = torch.hub.get_dir()
        torch.hub.set_dir(weights.name)
        cls.addClassCleanup(torch.hub.set_dir, previous_hub)
        cls.reference = np.random.default_rng(7).integers(20, 230, size=(192, 192, 3), dtype=np.uint8)
        cls.generated = (cls.reference.astype(np.float32) * 0.75 + 40).astype(np.uint8)
        cls.ref_tensor = torch.from_numpy(cls.reference).permute(2, 0, 1).unsqueeze(0).float()
        cls.gen_tensor = torch.from_numpy(cls.generated).permute(2, 0, 1).unsqueeze(0).float()

    def test_metric_values_match_native_backends_across_input_ranges(self):
        cases = (
            (compute_psnr, PeakSignalNoiseRatio(data_range=255)),
            (compute_ssim, StructuralSimilarityIndexMeasure(data_range=255)),
            (compute_ms_ssim, MultiScaleStructuralSimilarityIndexMeasure(data_range=255)),
            (compute_fsim, lambda r, g: _fsim_fn()(r, g, data_range=255)),
        )
        for compute, native in cases:
            with torch.no_grad():
                expected = native(self.ref_tensor, self.gen_tensor).item()
            for normalized in (False, True):
                with self.subTest(metric=compute.__name__, normalized=normalized):
                    ref = self.reference.astype(np.float32) / 255 if normalized else self.reference
                    gen = self.generated.astype(np.float32) / 255 if normalized else self.generated
                    self.assertAlmostEqual(compute(ref, gen, device="cpu"), expected, places=4)
                    self.assertAlmostEqual(
                        compute(ref, gen, device="cpu", data_range=1 if normalized else 255), expected, places=4
                    )

    def test_dark_uint8_pixels_use_the_uint8_range(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        near_black = np.ones_like(black)
        self.assertAlmostEqual(compute_psnr(black, near_black, device="cpu"), 20 * np.log10(255), places=4)

    def test_explicit_data_range_uses_original_pixel_units(self):
        mse = np.mean((self.reference.astype(np.float64) - self.generated) ** 2)
        expected = 10 * np.log10(510**2 / mse)
        self.assertAlmostEqual(
            compute_psnr(self.reference, self.generated, data_range=510, device="cpu"), expected, places=4
        )
        self.assertAlmostEqual(
            compute_psnr(self.reference / 255, self.generated / 255, data_range=2, device="cpu"), expected, places=4
        )

    def test_dark_float_pair_uses_one_scale(self):
        ref = np.ones((192, 192, 3), dtype=np.float32)
        gen = np.full_like(ref, 2)
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
        gen_tensor = torch.from_numpy(gen).permute(2, 0, 1).unsqueeze(0)
        cases = (
            (compute_psnr, PeakSignalNoiseRatio(data_range=255)),
            (compute_ssim, StructuralSimilarityIndexMeasure(data_range=255)),
            (compute_ms_ssim, MultiScaleStructuralSimilarityIndexMeasure(data_range=255)),
            (compute_fsim, lambda r, g: _fsim_fn()(r, g, data_range=255)),
        )
        for compute, native in cases:
            with self.subTest(metric=compute.__name__), torch.no_grad():
                expected = native(ref_tensor, gen_tensor).item()
                self.assertAlmostEqual(compute(ref, gen, device="cpu"), expected, places=4)
                self.assertAlmostEqual(compute(ref, gen, data_range=255, device="cpu"), expected, places=4)

        native_lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
        with torch.no_grad():
            expected_lpips = native_lpips(ref_tensor / 255, gen_tensor / 255).item()
        self.assertAlmostEqual(compute_lpips(ref, gen, net_type="squeeze", device="cpu"), expected_lpips, places=6)

    def test_lpips_matches_native_normalized_inputs(self):
        native = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True)
        with torch.no_grad():
            expected = native(self.ref_tensor / 255, self.gen_tensor / 255).item()
        self.assertGreater(expected, 0)
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                ref = self.reference.astype(np.float32) / 255 if normalized else self.reference
                gen = self.generated.astype(np.float32) / 255 if normalized else self.generated
                self.assertAlmostEqual(compute_lpips(ref, gen, net_type="squeeze", device="cpu"), expected, places=6)


if __name__ == "__main__":
    unittest.main()
