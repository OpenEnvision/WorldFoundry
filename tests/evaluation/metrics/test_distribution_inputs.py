"""Directory and explicit image-list inputs must follow the same metric pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from worldfoundry.evaluation.tasks.metrics import (
    compute_distribution_metrics,
    compute_fid,
    compute_inception_score,
    compute_kid,
    compute_mind,
    compute_precision_recall,
)
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity import metrics as backend
from worldfoundry.evaluation.tasks.metrics._shared.vendor.torch_fidelity.feature_extractor_base import (
    FeatureExtractorBase,
)


class PixelFeatures(FeatureExtractorBase):
    @staticmethod
    def get_provided_features_list():
        return ("2048", "logits_unbiased")

    def forward(self, images):
        features = images.float().mean((2, 3))[:, :2] / 255
        return tuple(features for _ in self.features_list)


class DistributionInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="distribution-tests-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        rng = np.random.default_rng(42)
        self.reference, self.generated = self.root / "reference", self.root / "generated"
        for directory in (self.reference, self.generated):
            directory.mkdir()
            for i, channels in enumerate((3, 4, 1, 3)):
                pixels = rng.integers(0, 256, size=(10, 12, channels), dtype=np.uint8)
                Image.fromarray(pixels.squeeze(-1) if channels == 1 else pixels).save(directory / f"{i}.png")

    def test_path_lists_match_directories_for_each_torch_fidelity_metric(self):
        cases = (
            (compute_fid, {}),
            (compute_kid, {"kid_subsets": 2, "kid_subset_size": 3}),
            (compute_precision_recall, {"prc_neighborhood": 1}),
            (compute_mind, {"mind_num_projections": 8}),
            (compute_distribution_metrics, {"metrics": ("fid", "kid"), "kid_subsets": 2, "kid_subset_size": 3}),
            (compute_inception_score, {"splits": 2}),
        )
        options = dict(cuda=False, batch_size=2, verbose=False, cache=False, save_cpu_ram=True)
        for compute, extra in cases:
            for crop_size in (0, 8):
                with self.subTest(metric=compute.__name__, crop_size=crop_size):
                    inputs = (
                        (self.reference,) if compute is compute_inception_score else (self.reference, self.generated)
                    )
                    path_lists = tuple(tuple(sorted(path.glob("*.png"))) for path in inputs)
                    with patch.object(
                        backend,
                        "create_feature_extractor",
                        side_effect=lambda name, layers, **kwargs: PixelFeatures(name, layers),
                    ):
                        expected = compute(*inputs, **options, **extra, samples_resize_and_crop=crop_size)
                        actual = compute(*path_lists, **options, **extra, samples_resize_and_crop=crop_size)
                    self.assertEqual(actual, expected)

    def test_swav_lists_match_directory_statistics(self):
        from worldfoundry.evaluation.tasks.metrics.fid.swav import _swav_fid_module

        module = _swav_fid_module()
        native_statistics = module.calculate_activation_statistics

        class MeanEncoder(torch.nn.Module):
            def forward(self, images):
                return images.mean((2, 3))[:, :2]

        def small_statistics(files, model, batch_size, dims, device):
            return native_statistics(files, model, batch_size, 2, device)

        with (
            patch.object(module, "resnet50", return_value=MeanEncoder()),
            patch.object(module, "cpu_count", return_value=0),
            patch.object(module, "calculate_activation_statistics", side_effect=small_statistics),
        ):
            for max_size in ("all", "3"):
                with self.subTest(max_size=max_size):
                    options = dict(feature_extractor="swav-resnet50", cuda=False, batch_size=2, max_size=max_size)
                    expected = compute_fid(self.reference, self.generated, **options)
                    actual = compute_fid(sorted(self.reference.glob("*.png")), self.generated, **options)
                    self.assertAlmostEqual(actual, expected)

    def test_precision_recall_follow_known_asymmetric_manifolds(self):
        # At k=1, all generated points are in the reference manifold, while
        # only reference point 0 is in the generated manifold.
        for directory, values in ((self.reference, (0, 10, 100, 110)), (self.generated, (0, 1, 2, 3))):
            for i, value in enumerate(values):
                Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(directory / f"{i}.png")
        options = dict(cuda=False, batch_size=2, verbose=False, cache=False, save_cpu_ram=True, prc_neighborhood=1)
        with patch.object(
            backend,
            "create_feature_extractor",
            side_effect=lambda name, layers, **kwargs: PixelFeatures(name, layers),
        ):
            for compute, extra in (
                (compute_precision_recall, {}),
                (compute_distribution_metrics, {"metrics": ("prc",)}),
            ):
                with self.subTest(metric=compute.__name__):
                    result = compute(self.reference, self.generated, **options, **extra)
                    self.assertEqual(result, {"precision": 1.0, "recall": 0.25, "f_score": 0.4})

    def test_bundled_inception_score_uses_generated_images(self):
        for directory, colors in (
            (self.reference, ((255, 0, 0),) * 4),
            (self.generated, ((255, 0, 0), (0, 255, 0)) * 2),
        ):
            for i, color in enumerate(colors):
                pixels = np.broadcast_to(np.array(color, dtype=np.uint8), (8, 8, 3)).copy()
                Image.fromarray(pixels).save(directory / f"{i}.png")
        options = dict(cuda=False, batch_size=2, verbose=False, cache=False, save_cpu_ram=True)
        # PixelFeatures gives logits [1, 0] and [0, 1]. Balanced generated
        # classes have this analytic IS, whereas the one-class reference has IS=1.
        p = np.exp(1) / (np.exp(1) + 1)
        expected = np.exp(p * np.log(2 * p) + (1 - p) * np.log(2 * (1 - p)))
        with patch.object(
            backend,
            "create_feature_extractor",
            side_effect=lambda name, layers, **kwargs: PixelFeatures(name, layers),
        ):
            standalone = compute_inception_score(self.generated, splits=1, **options)
            fid = compute_fid(self.reference, self.generated, **options)
            for metrics in (("isc",), ("fid", "isc")):
                with self.subTest(metrics=metrics):
                    result = compute_distribution_metrics(
                        self.reference, self.generated, metrics=metrics, isc_splits=1, **options
                    )
                    self.assertAlmostEqual(result["inception_score_mean"], expected)
                    self.assertEqual(result["inception_score_mean"], standalone["inception_score_mean"])
                    if "fid" in metrics:
                        self.assertAlmostEqual(result["frechet_inception_distance"], fid)


if __name__ == "__main__":
    unittest.main()
