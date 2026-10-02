"""Numerical and memory regressions for paired VS similarity."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("numpy"), "VS similarity requires numpy")
class VSSimilarityTests(unittest.TestCase):
    def test_paired_and_unpaired_match_the_full_matrix(self):
        import numpy as np

        from worldfoundry.evaluation.tasks.metrics.vs_similarity.wrapper import (
            compute_vs_similarity,
            compute_vs_similarity_matrix,
        )

        rng = np.random.default_rng(42)
        images, texts = rng.normal(size=(32, 16)), rng.normal(size=(32, 16))
        images[0] = 0
        matrix = compute_vs_similarity_matrix(images, texts)
        self.assertAlmostEqual(compute_vs_similarity(images, texts), float(np.mean(np.diag(matrix))))
        self.assertAlmostEqual(compute_vs_similarity(images, texts, paired=False), float(np.mean(matrix)))
        self.assertAlmostEqual(compute_vs_similarity(images[1], texts[1]), matrix[1, 1])

    def test_large_paired_input_does_not_build_pairwise_scores(self):
        import numpy as np

        from worldfoundry.evaluation.tasks.metrics.vs_similarity import wrapper

        features = np.ones((10000, 16))
        with patch.object(wrapper, "compute_vs_similarity_matrix", side_effect=AssertionError("quadratic allocation")):
            self.assertAlmostEqual(wrapper.compute_vs_similarity(features, features), 1)

    def test_misaligned_pairs_are_rejected(self):
        import numpy as np

        from worldfoundry.evaluation.tasks.metrics.vs_similarity.wrapper import compute_vs_similarity

        for shape in ((3, 4), (2, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                compute_vs_similarity(np.ones((2, 4)), np.ones(shape))


if __name__ == "__main__":
    unittest.main()
