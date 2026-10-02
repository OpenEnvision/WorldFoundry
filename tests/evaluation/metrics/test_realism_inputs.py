"""Realism input adapters must preserve the backend's per-image definition."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from worldfoundry.evaluation.tasks.metrics.improved_precision_recall import wrapper


class MeanEncoder(torch.nn.Module):
    def forward(self, images):
        return images.mean((2, 3))


class RealismInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="realism-tests-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.reference = self.root / "reference.npz"
        features = np.array([[-2, -1, 0], [0, 0, 0], [1, 1, 1], [2, 0, -1]], dtype=np.float32)
        radii = np.array([0.5, 1, 1.5, 2], dtype=np.float32)
        np.savez(self.reference, feature=features, radii=radii)
        self.generated = self.root / "generated"
        self.generated.mkdir()
        self.images = []
        self.expected = []
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for i, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
            image = Image.new("RGB", (224, 224), color)
            image.save(self.generated / f"{i}.png")
            self.images.append(np.array(image))
            feature = (np.array(color) / 255 - mean) / std
            self.expected.append(float(np.max(radii / (np.linalg.norm(features - feature, axis=1) + 1e-6))))
        ipr_class = wrapper._ipr_class()
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(wrapper, "_ipr_class", return_value=lambda **kwargs: ipr_class(model=MeanEncoder(), **kwargs))
        )
        loader = wrapper._custom_loader
        patches.enter_context(
            patch.object(
                wrapper, "_custom_loader", side_effect=lambda *args, **kwargs: loader(*args, **kwargs, num_workers=0)
            )
        )

    def test_directory_scores_each_image_independently_of_batch_size(self):
        for batch_size in (1, 2, 50):
            with self.subTest(batch_size=batch_size):
                result = wrapper.compute_realism_score(
                    self.reference, self.generated, batch_size=batch_size, device="cpu"
                )
                np.testing.assert_allclose(np.sort(result["realism_scores"]), np.sort(self.expected), rtol=1e-5)
                self.assertAlmostEqual(result["mean_realism"], np.mean(self.expected), places=5)

    def test_files_and_arrays_match_the_same_per_image_scores(self):
        files = sorted(self.generated.glob("*.png"))
        for inputs in (files, self.images, [files[0], self.images[1], files[2]]):
            with self.subTest(input_type=type(inputs[0]).__name__):
                result = wrapper.compute_realism_score(self.reference, inputs, device="cpu")
                np.testing.assert_allclose(result["realism_scores"], self.expected, rtol=1e-5)
        single_file = wrapper.compute_realism_score(self.reference, files[0], device="cpu")
        np.testing.assert_allclose(single_file["realism_scores"], self.expected[:1], rtol=1e-5)
        single_array = wrapper.compute_realism_score(self.reference, self.images[0], device="cpu")
        self.assertAlmostEqual(single_array, self.expected[0], places=5)


if __name__ == "__main__":
    unittest.main()
